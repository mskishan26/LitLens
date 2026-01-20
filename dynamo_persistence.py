"""
DynamoDB Persistence Layer for RAG Pipeline
============================================

Clean, DynamoDB-native design with:
- TraceBuilder: Accumulates stages in-memory, single write on completion
- DynamoPersistence: Handles all DB operations
- Coupled messages: Each turn stores query + answer together
- Single ID: message_id serves as both message and trace identifier
- Save-at-start pattern: Trace created with status="running", updated on completion

Tables:
- chat-table: Turns (user query + assistant answer + metadata, one item per Q&A)
- trace-table: Pipeline traces (stages, timings, debug data)  
- chat-metadata-table: Chat list per user (for sidebar)

Key Design:
- chat-table: PK=ChatId, SK=MessageId (timestamp-prefixed for sorting)
- trace-table: PK=ChatId, SK=MessageId (same ID, trace is linked to message)
- metadata-table: PK=UserId, SK=ChatId, GSI on last_message_date

Storage Philosophy:
- Store STRUCTURED data (answer, sources, hallucination as separate fields)
- Frontend renders the final display format
- This allows re-rendering when UI changes without data migration

Abandoned Request Handling:
- Trace is saved immediately when pipeline starts (status="running")
- On completion, trace is updated to status="success" or "failed"
- If user closes tab mid-generation, trace remains with status="running"
- This gives visibility into abandoned requests

Usage:
    persistence = DynamoPersistence.from_oidc(role_arn="arn:aws:iam::...")
    
    # In pipeline (start):
    trace = TraceBuilder(chat_id, user_id, query)
    await persistence.save_trace_start(trace)  # Save immediately
    
    # ... pipeline runs, trace.add_stage() called ...
    
    # On completion:
    trace.complete()
    await persistence.save_turn(
        trace=trace,
        answer="The generated answer text...",
        sources=[{"text": "...", "metadata": {...}}],
        hallucination_result={...},
    )
"""

import json
import uuid
import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal

try:
    import boto3
    from boto3.dynamodb.conditions import Key
except ImportError:
    boto3 = None


# =============================================================================
# Utilities
# =============================================================================

def _decimal_serializer(obj):
    """JSON serializer for Decimal types from DynamoDB."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _to_decimal(value: Optional[float]) -> Optional[Decimal]:
    """Convert float to Decimal for DynamoDB (avoids float precision issues)."""
    if value is None:
        return None
    return Decimal(str(round(value, 2)))


def _clean_item(item: Dict) -> Dict:
    """Remove None values (DynamoDB doesn't accept them)."""
    return {k: v for k, v in item.items() if v is not None}


def _generate_id(prefix: str = "") -> str:
    """Generate a sortable ID with timestamp prefix."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:17]  # YYYYMMDDHHMMSSmmm
    rand = uuid.uuid4().hex[:8]
    return f"{prefix}{ts}_{rand}" if prefix else f"{ts}_{rand}"


# =============================================================================
# TraceBuilder - Accumulates pipeline stages, writes once
# =============================================================================

@dataclass
class TraceBuilder:
    """
    Accumulates trace data during pipeline execution.
    
    Note: message_id is used as both the message identifier AND trace identifier.
    There is no separate trace_id - this simplifies lookups and reduces redundancy.
    
    Stages tracked:
    - bm25: BM25 search results
    - embedding_paper: Paper-level embedding search
    - embedding_chunk: Chunk-level embedding search  
    - hybrid_fusion: Fusion of BM25 + embedding results
    - reranker: Reranked chunks
    - generator: LLM generation metrics
    - hallucination: Claim verification results (if enabled)
    
    Usage:
        trace = TraceBuilder(chat_id="chat-123", user_id="user-456", ...)
        await persistence.save_trace_start(trace)  # Save immediately with status="running"
        
        trace.add_stage("bm25", {"results": [...], "total": 50}, duration_ms=45.0)
        trace.add_stage("embedding_paper", {...}, duration_ms=120.0)
        # ... pipeline runs ...
        
        trace.complete()  # or trace.fail("reranker", "OOM error")
        await persistence.save_turn(trace, answer, sources, ...)  # Final save
    """
    chat_id: str
    user_id: str
    query: str
    is_anonymous: bool = False  # Whether this is an anonymous user
    # message_id serves as both message ID and trace ID
    message_id: str = field(default_factory=lambda: _generate_id("msg_"))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Accumulated during pipeline
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Final state
    status: str = "running"  # running | success | failed
    error: Optional[Dict[str, str]] = None  # {"stage": "reranker", "message": "OOM"}
    finished_at: Optional[str] = None
    
    # Summary metrics (populated from stages)
    papers_retrieved: int = 0
    chunks_retrieved: int = 0
    chunks_reranked: int = 0
    grounding_ratio: Optional[float] = None

    def add_stage(
        self, 
        name: str, 
        data: Dict[str, Any], 
        duration_ms: float,
        **metrics
    ) -> None:
        """
        Add a pipeline stage result.
        
        Args:
            name: Stage name (bm25, embedding_paper, embedding_chunk, 
                  hybrid_fusion, reranker, generator, hallucination)
            data: Stage output data (will be stored as-is)
            duration_ms: How long the stage took
            **metrics: Additional metrics to update (papers_retrieved, etc.)
        """
        self.stages[name] = {
            "data": data,
            "duration_ms": round(duration_ms, 2),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Update summary metrics from kwargs
        for key in ["papers_retrieved", "chunks_retrieved", "chunks_reranked", "grounding_ratio"]:
            if key in metrics:
                setattr(self, key, metrics[key])

    def fail(self, stage: str, message: str) -> None:
        """Mark trace as failed at a specific stage."""
        self.status = "failed"
        self.error = {"stage": stage, "message": message}
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def complete(self) -> None:
        """Mark trace as successfully completed."""
        self.status = "success"
        self.finished_at = datetime.now(timezone.utc).isoformat()

    @property
    def total_duration_ms(self) -> Optional[float]:
        """Calculate total duration from start to finish."""
        if not self.finished_at:
            return None
        started = datetime.fromisoformat(self.started_at)
        finished = datetime.fromisoformat(self.finished_at)
        return (finished - started).total_seconds() * 1000

    def to_trace_item(self) -> Dict[str, Any]:
        """
        Convert to DynamoDB item for trace-table.
        
        Uses MessageId as the sort key (no separate TraceId).
        Stores stages as JSON string (nested structure too deep for DynamoDB Map).
        Individual durations stored at top level for querying.
        """
        # Extract durations for top-level querying
        stage_durations = {}
        for name, stage in self.stages.items():
            stage_durations[f"{name}_duration_ms"] = _to_decimal(stage["duration_ms"])
        
        item = {
            "ChatId": self.chat_id,
            "MessageId": self.message_id,  # SK is now MessageId (same as chat-table)
            "UserId": self.user_id,
            "is_anonymous": self.is_anonymous,
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "total_duration_ms": _to_decimal(self.total_duration_ms),
            
            # Summary metrics (for quick filtering/display)
            "papers_retrieved": self.papers_retrieved,
            "chunks_retrieved": self.chunks_retrieved,
            "chunks_reranked": self.chunks_reranked,
            "grounding_ratio": _to_decimal(self.grounding_ratio),
            
            # Full stage data as JSON
            "stages": json.dumps(self.stages),
            
            # Individual durations at top level for querying
            **stage_durations,
        }
        
        if self.error:
            item["error"] = self.error
        
        return _clean_item(item)

    def to_initial_trace_item(self) -> Dict[str, Any]:
        """
        Convert to DynamoDB item for initial save (status="running").
        
        Called at the START of pipeline execution to create a record.
        This ensures abandoned requests are visible (they'll have status="running").
        """
        return _clean_item({
            "ChatId": self.chat_id,
            "MessageId": self.message_id,
            "UserId": self.user_id,
            "is_anonymous": self.is_anonymous,
            "query": self.query,
            "started_at": self.started_at,
            "status": "running",
            "stages": json.dumps({}),  # Empty initially
        })


# =============================================================================
# DynamoPersistence - All database operations
# =============================================================================

class DynamoPersistence:
    """
    DynamoDB persistence layer for RAG pipeline.
    
    Handles:
    - Saving turns (coupled user query + assistant answer)
    - Saving traces (with save-at-start pattern for abandoned request visibility)
    - Updating chat metadata
    - Querying chat history
    - Ownership verification for security
    """
    
    def __init__(
        self,
        dynamodb_resource,
        chat_table_name: str = "chat-table",
        trace_table_name: str = "trace-table",
        metadata_table_name: str = "chat-metadata-table",
    ):
        self.dynamodb = dynamodb_resource
        self.chat_table = self.dynamodb.Table(chat_table_name)
        self.trace_table = self.dynamodb.Table(trace_table_name)
        self.metadata_table = self.dynamodb.Table(metadata_table_name)

    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_oidc(
        cls,
        role_arn: str,
        region: str = "us-east-2",
        **table_names,
    ) -> "DynamoPersistence":
        """
        Create using Modal's OIDC for credential-free AWS access.
        
        Args:
            role_arn: IAM role ARN that Modal can assume
            region: AWS region
        """
        identity_token = os.environ.get("MODAL_IDENTITY_TOKEN")
        if not identity_token:
            raise RuntimeError(
                "MODAL_IDENTITY_TOKEN not found. This must run inside Modal."
            )
        
        sts = boto3.client("sts", region_name=region)
        creds = sts.assume_role_with_web_identity(
            RoleArn=role_arn,
            RoleSessionName=f"modal-rag-{uuid.uuid4().hex[:8]}",
            WebIdentityToken=identity_token,
            DurationSeconds=3600,
        )["Credentials"]
        
        session = boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=region,
        )
        
        return cls(dynamodb_resource=session.resource("dynamodb"), **table_names)

    @classmethod
    def from_session(
        cls,
        session: "boto3.Session",
        **table_names,
    ) -> "DynamoPersistence":
        """Create using an existing boto3 session (for local dev with SSO)."""
        return cls(dynamodb_resource=session.resource("dynamodb"), **table_names)

    # =========================================================================
    # Write Operations
    # =========================================================================

    async def save_trace_start(self, trace: TraceBuilder) -> None:
        """
        Save initial trace record with status="running".
        
        Called at the START of pipeline execution. This ensures that if the
        user closes the tab mid-generation, we still have a record of the
        abandoned request.
        
        The trace will be updated with full data on completion via save_turn().
        """
        await self._put_item(self.trace_table, trace.to_initial_trace_item())

    async def save_turn(
        self,
        trace: TraceBuilder,
        answer: str,
        sources: List[Dict[str, Any]],
        hallucination_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Save a complete Q&A turn (query + answer + trace + metadata).
        
        This is the main entry point after pipeline completion.
        Stores data in STRUCTURED format for frontend rendering flexibility.
        
        Note: This UPDATES the trace that was created by save_trace_start().
        
        Args:
            trace: Completed TraceBuilder with all stages
            answer: The raw generated answer (without formatting)
            sources: List of source dicts [{text, metadata, score}, ...]
            hallucination_result: Optional hallucination check results
        
        Returns:
            {"message_id": "..."} - message_id is also the trace identifier
        """
        if trace.status == "running":
            trace.complete()
        
        # Write all three in parallel (trace is updated, not created)
        await asyncio.gather(
            self._save_message(trace, answer, sources, hallucination_result),
            self._update_trace(trace),  # Update existing trace
            self._update_metadata(trace),
        )
        
        return {"message_id": trace.message_id}

    async def _save_message(
        self,
        trace: TraceBuilder,
        answer: str,
        sources: List[Dict],
        hallucination_result: Optional[Dict],
    ) -> None:
        """
        Save a coupled message (user query + assistant answer in one item).
        
        Stores structured data for frontend rendering flexibility:
        - query: The user's question
        - answer: Raw generated text (no formatting)
        - sources: Source documents used
        - hallucination: Verification results (if checked)
        """
        item = {
            "ChatId": trace.chat_id,
            "MessageId": trace.message_id,
            "UserId": trace.user_id,
            "is_anonymous": trace.is_anonymous,
            
            # Core content (STRUCTURED, not pre-rendered)
            "query": trace.query,
            "answer": answer,
            
            # Metadata
            "timestamp": trace.started_at,
            "duration_ms": _to_decimal(trace.total_duration_ms),
            "status": trace.status,
        }
        
        # Sources - stored as JSON for citation display
        # Format: [{text, metadata: {title, file_path, ...}, score}, ...]
        if sources:
            # Store minimal source info for display (not full text)
            display_sources = [
                {
                    "title": s.get("metadata", {}).get("title") or 
                             s.get("metadata", {}).get("file_path", "Unknown"),
                    "file_path": s.get("metadata", {}).get("file_path"),
                    "score": s.get("score"),
                    "chunk_id": s.get("metadata", {}).get("chroma_id"),
                }
                for s in sources
            ]
            item["sources"] = json.dumps(display_sources)
        
        # Hallucination results - stored as JSON
        # Format: {grounding_ratio, num_claims, num_grounded, unsupported_claims, verifications}
        if hallucination_result:
            item["hallucination"] = json.dumps({
                "grounding_ratio": hallucination_result.get("grounding_ratio"),
                "num_claims": hallucination_result.get("num_claims"),
                "num_grounded": hallucination_result.get("num_grounded"),
                "unsupported_claims": hallucination_result.get("unsupported_claims", []),
                # Optionally include full verifications for detailed view
                "verifications": hallucination_result.get("verifications", []),
            })
        
        # Error info if failed
        if trace.error:
            item["error"] = json.dumps(trace.error)
        
        await self._put_item(self.chat_table, _clean_item(item))

    async def _update_trace(self, trace: TraceBuilder) -> None:
        """
        Update the pipeline trace with final data.
        
        This updates the trace created by save_trace_start() with:
        - All stage data
        - Final status (success/failed)
        - Duration metrics
        """
        await self._put_item(self.trace_table, trace.to_trace_item())

    async def _update_metadata(self, trace: TraceBuilder, title: Optional[str] = None) -> None:
        """
        Update chat metadata (upsert).
        
        Creates entry if first message, updates last_message_date otherwise.
        Title is set only on first message (if_not_exists).
        """
        timestamp = trace.finished_at or trace.started_at
        chat_title = title or trace.query[:100]
        
        loop = asyncio.get_event_loop()
        
        def update():
            self.metadata_table.update_item(
                Key={"UserId": trace.user_id, "ChatId": trace.chat_id},
                UpdateExpression=(
                    "SET #title = if_not_exists(#title, :title), "
                    "last_message_date = :ts, "
                    "updated_at = :ts, "
                    "is_anonymous = :is_anon"
                ),
                ExpressionAttributeNames={"#title": "title"},
                ExpressionAttributeValues={
                    ":title": chat_title,
                    ":ts": timestamp,
                    ":is_anon": trace.is_anonymous,
                },
            )
        
        await loop.run_in_executor(None, update)

    async def update_chat_title(
        self,
        user_id: str,
        chat_id: str,
        title: str,
    ) -> bool:
        """
        Update the title of a chat.
        
        Returns True if successful, False if chat doesn't exist or user doesn't own it.
        """
        loop = asyncio.get_event_loop()
        
        def update():
            try:
                # Use a conditional update to verify ownership
                self.metadata_table.update_item(
                    Key={"UserId": user_id, "ChatId": chat_id},
                    UpdateExpression="SET #title = :title, updated_at = :ts",
                    ConditionExpression="attribute_exists(UserId)",  # Chat must exist for this user
                    ExpressionAttributeNames={"#title": "title"},
                    ExpressionAttributeValues={
                        ":title": title[:200],  # Truncate to reasonable length
                        ":ts": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return True
            except self.dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
                return False
        
        return await loop.run_in_executor(None, update)

    async def _put_item(self, table, item: Dict) -> None:
        """Async wrapper for put_item."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: table.put_item(Item=item))

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def get_user_chats(
        self,
        user_id: str,
        limit: int = 20,
    ) -> List[Dict]:
        """Get user's recent chats (for sidebar)."""
        loop = asyncio.get_event_loop()
        
        def query():
            response = self.metadata_table.query(
                IndexName="LastUsedIndex",
                KeyConditionExpression=Key("UserId").eq(user_id),
                ScanIndexForward=False,  # Newest first
                Limit=limit,
            )
            return response.get("Items", [])
        
        return await loop.run_in_executor(None, query)

    async def get_chat_messages(
        self,
        chat_id: str,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get messages for a chat (for loading conversation history).
        
        Returns messages in chronological order with parsed JSON fields.
        Frontend uses this data to render the conversation.
        """
        loop = asyncio.get_event_loop()
        
        def query():
            response = self.chat_table.query(
                KeyConditionExpression=Key("ChatId").eq(chat_id),
                ScanIndexForward=True,  # Chronological order
                Limit=limit,
            )
            items = response.get("Items", [])
            
            # Parse JSON fields for frontend consumption
            for item in items:
                for field in ["sources", "hallucination", "error"]:
                    if field in item and isinstance(item[field], str):
                        try:
                            item[field] = json.loads(item[field])
                        except json.JSONDecodeError:
                            pass
            
            return items
        
        return await loop.run_in_executor(None, query)

    async def get_chat_owner(self, chat_id: str) -> Optional[str]:
        """
        Get the owner (user_id) of a chat.
        
        Used for authorization checks before returning traces.
        Returns None if chat doesn't exist.
        """
        loop = asyncio.get_event_loop()
        
        def query():
            # Query chat-table for any message in this chat to get UserId
            response = self.chat_table.query(
                KeyConditionExpression=Key("ChatId").eq(chat_id),
                Limit=1,
                ProjectionExpression="UserId",
            )
            items = response.get("Items", [])
            if items:
                return items[0].get("UserId")
            return None
        
        return await loop.run_in_executor(None, query)

    async def verify_chat_ownership(
        self,
        chat_id: str,
        user_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify that a user owns a chat.
        
        Returns:
            (is_owner, actual_owner_id)
            - (True, user_id) if user owns the chat
            - (False, actual_owner) if chat exists but user doesn't own it
            - (False, None) if chat doesn't exist
        """
        owner = await self.get_chat_owner(chat_id)
        if owner is None:
            return (False, None)
        return (owner == user_id, owner)

    async def get_trace(
        self,
        chat_id: str,
        message_id: str,
    ) -> Optional[Dict]:
        """
        Get a trace by chat_id and message_id.
        
        Note: message_id is now also the trace identifier (no separate trace_id).
        """
        loop = asyncio.get_event_loop()
        
        def get():
            response = self.trace_table.get_item(
                Key={"ChatId": chat_id, "MessageId": message_id}
            )
            item = response.get("Item")
            
            if item and "stages" in item and isinstance(item["stages"], str):
                item["stages"] = json.loads(item["stages"])
            
            return item
        
        return await loop.run_in_executor(None, get)

    async def get_trace_with_auth(
        self,
        chat_id: str,
        message_id: str,
        requesting_user_id: str,
        is_anonymous: bool = False,
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Get a trace with ownership verification.
        
        For security:
        - If the chat belongs to the requesting user, return the trace
        - If the user is anonymous, they can only access their own traces
          (identified by matching user_id - Firebase anon UID)
        - If the chat belongs to someone else, return auth error
        
        Args:
            chat_id: The chat containing the message
            message_id: The message/trace ID
            requesting_user_id: The user making the request (Firebase UID)
            is_anonymous: Whether the requesting user is anonymous
        
        Returns:
            (trace_data, error_message)
            - (trace, None) on success
            - (None, "not_found") if trace doesn't exist
            - (None, "unauthorized") if user doesn't own the chat
        """
        # Verify ownership
        is_owner, actual_owner = await self.verify_chat_ownership(chat_id, requesting_user_id)
        
        if actual_owner is None:
            # Chat doesn't exist in chat-table yet
            # This could happen if we're fetching trace for a request that's still running
            # (trace exists but message not saved yet)
            trace = await self.get_trace(chat_id, message_id)
            if not trace:
                return (None, "not_found")
            # Check if the trace belongs to this user
            if trace.get("UserId") == requesting_user_id:
                return (trace, None)
            return (None, "unauthorized")
        
        if not is_owner:
            return (None, "unauthorized")
        
        # User owns the chat, get the trace
        trace = await self.get_trace(chat_id, message_id)
        if not trace:
            return (None, "not_found")
        
        return (trace, None)

    async def get_message_count(self, chat_id: str) -> int:
        """
        Get the number of messages in a chat.
        
        Used to determine if we should offer title generation.
        """
        loop = asyncio.get_event_loop()
        
        def count():
            response = self.chat_table.query(
                KeyConditionExpression=Key("ChatId").eq(chat_id),
                Select="COUNT",
            )
            return response.get("Count", 0)
        
        return await loop.run_in_executor(None, count)

    async def get_first_queries(
        self,
        chat_id: str,
        n: int = 2,
    ) -> List[str]:
        """
        Get the first N queries from a chat.
        
        Used for generating a chat title.
        """
        loop = asyncio.get_event_loop()
        
        def query():
            response = self.chat_table.query(
                KeyConditionExpression=Key("ChatId").eq(chat_id),
                ScanIndexForward=True,  # Chronological
                Limit=n,
                ProjectionExpression="query",
            )
            return [item.get("query", "") for item in response.get("Items", [])]
        
        return await loop.run_in_executor(None, query)

    async def get_abandoned_traces(
        self,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get traces that were abandoned (status still "running").
        
        Useful for monitoring and debugging.
        Note: This requires a scan, so use sparingly.
        """
        loop = asyncio.get_event_loop()
        
        def scan():
            response = self.trace_table.scan(
                FilterExpression="status = :status",
                ExpressionAttributeValues={":status": "running"},
                Limit=limit,
            )
            return response.get("Items", [])
        
        return await loop.run_in_executor(None, scan)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    async def demo():
        print("DynamoDB Persistence Demo")
        print("=" * 50)
        
        # Create a trace (note: no separate trace_id)
        trace = TraceBuilder(
            chat_id="chat-demo-123",
            user_id="user-demo",
            query="What is matrix suppression in biostatistics?",
            is_anonymous=False,
        )
        
        print(f"Message ID (also trace ID): {trace.message_id}")
        print(f"Initial status: {trace.status}")
        
        print("\n--- Initial Trace Item (saved at start) ---")
        initial_item = trace.to_initial_trace_item()
        print(json.dumps(initial_item, indent=2, default=_decimal_serializer))
        
        # Simulate pipeline stages
        trace.add_stage(
            "bm25",
            {
                "results": [
                    {"file_path": "paper1.pdf", "score": 12.5},
                    {"file_path": "paper2.pdf", "score": 10.2},
                ],
                "total": 50,
            },
            duration_ms=45.0,
            papers_retrieved=50,
        )
        
        trace.add_stage(
            "embedding_paper",
            {
                "results": [{"chroma_id": "emb-1", "distance": 0.15}],
                "total": 50,
            },
            duration_ms=120.0,
        )
        
        trace.add_stage(
            "hybrid_fusion",
            {
                "selected_papers": ["paper1.pdf", "paper2.pdf"],
                "scores": {"paper1.pdf": 0.92, "paper2.pdf": 0.85},
            },
            duration_ms=2.0,
        )
        
        trace.add_stage(
            "embedding_chunk",
            {
                "results": [{"chroma_id": "chunk-1", "distance": 0.12}],
                "total": 100,
            },
            duration_ms=80.0,
            chunks_retrieved=100,
        )
        
        trace.add_stage(
            "reranker",
            {
                "results": {
                    "chunk-1": {"rank": 1, "score": 0.95},
                    "chunk-2": {"rank": 2, "score": 0.88},
                }
            },
            duration_ms=1500.0,
            chunks_reranked=10,
        )
        
        trace.add_stage(
            "generator",
            {
                "answer_length": 500,
                "completion_tokens": 150,
                "ttft_ms": 85.0,
            },
            duration_ms=3000.0,
        )
        
        trace.add_stage(
            "hallucination",
            {
                "verifications": [
                    {"claim": "Matrix suppression reduces variance", "is_grounded": True, "max_score": 0.92},
                    {"claim": "Discovered in 1985", "is_grounded": False, "max_score": 0.23},
                ],
                "grounding_ratio": 0.5,
                "unsupported_claims": ["Discovered in 1985"],
            },
            duration_ms=800.0,
            grounding_ratio=0.5,
        )
        
        trace.complete()
        
        print("\n--- Final Trace Item (saved on completion) ---")
        trace_item = trace.to_trace_item()
        # Pretty print without stages (too long)
        display_item = {k: v for k, v in trace_item.items() if k != "stages"}
        print(json.dumps(display_item, indent=2, default=_decimal_serializer))
        print(f"\n  stages: <{len(trace.stages)} stages captured>")
        for name, stage in trace.stages.items():
            print(f"    - {name}: {stage['duration_ms']:.1f}ms")
        
        print(f"\n--- Summary ---")
        print(f"Message/Trace ID: {trace.message_id}")
        print(f"Total duration: {trace.total_duration_ms:.0f}ms")
        print(f"Status: {trace.status}")
        print(f"Papers: {trace.papers_retrieved}, Chunks: {trace.chunks_retrieved}, Reranked: {trace.chunks_reranked}")
        print(f"Grounding ratio: {trace.grounding_ratio}")
    
    import asyncio
    asyncio.run(demo())
