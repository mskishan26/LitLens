"""
Request Tracer - Pipeline Observability
========================================

Captures detailed trace data for each RAG pipeline request.
Stores traces in SQLite for durability and easy querying.

Features:
- Per-request trace capture with req_id and conversation_id
- Stage-by-stage metrics (BM25, embedding, rerank, generation, hallucination)
- SQLite storage with automatic cleanup of old traces
- JSON serialization for complex data structures

Usage:
    tracer = RequestTracer(db_path="traces.db")
    
    # Start a new trace
    trace_id = tracer.start_trace(conversation_id="conv-123")
    
    # Capture stages
    tracer.capture_bm25(trace_id, results, scores, duration_ms)
    tracer.capture_embedding_paper(trace_id, results, duration_ms)
    tracer.capture_embedding_chunk(trace_id, results, duration_ms)
    tracer.capture_reranker(trace_id, results, duration_ms)
    tracer.capture_generator(trace_id, answer, duration_ms)
    tracer.capture_hallucination(trace_id, verifications, grounding_ratio, duration_ms)
    
    # Finish
    tracer.finish_trace(trace_id, success=True)
"""

import sqlite3
import json
import uuid
import time
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager

from utils.logger import get_logger, get_request_context

logger = get_logger(__name__)


class RequestTracer:
    """
    Captures and stores detailed trace data for RAG pipeline requests.
    Uses SQLite for persistent storage with automatic old trace cleanup.
    """
    
    def __init__(
        self,
        db_path: str = "traces/request_traces.db",
        retention_days: int = 30,
        max_traces_per_conversation: int = 1000
    ):
        """
        Initialize the request tracer.
        
        Args:
            db_path: Path to SQLite database file.
            retention_days: Days to retain traces before cleanup.
            max_traces_per_conversation: Max traces to keep per conversation.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.max_traces_per_conversation = max_traces_per_conversation
        
        # Thread-local connections
        self._local = threading.local()
        
        # Initialize schema
        self._init_schema()
        
        logger.info(f"RequestTracer initialized with DB at {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn
    
    @contextmanager
    def _cursor(self):
        """Context manager for database cursor with auto-commit."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._cursor() as cursor:
            # Main traces table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    req_id TEXT,
                    query TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    success INTEGER,
                    error_message TEXT,
                    total_duration_ms REAL,
                    
                    -- Stage data (JSON)
                    bm25_data TEXT,
                    embedding_paper_data TEXT,
                    embedding_chunk_data TEXT,
                    hybrid_fusion_data TEXT,
                    reranker_data TEXT,
                    generator_data TEXT,
                    hallucination_data TEXT,
                    
                    -- Stage durations
                    bm25_duration_ms REAL,
                    embedding_paper_duration_ms REAL,
                    embedding_chunk_duration_ms REAL,
                    reranker_duration_ms REAL,
                    generator_duration_ms REAL,
                    hallucination_duration_ms REAL,
                    
                    -- Summary metrics
                    papers_retrieved INTEGER,
                    chunks_retrieved INTEGER,
                    chunks_reranked INTEGER,
                    grounding_ratio REAL,
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_conversation 
                ON traces(conversation_id, started_at DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_req_id 
                ON traces(req_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_created 
                ON traces(created_at)
            """)
            
            logger.debug("Database schema initialized")
    
    def start_trace(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        req_id: Optional[str] = None
    ) -> str:
        """
        Start a new trace for a request.
        
        Args:
            query: The user's query.
            conversation_id: Chat session ID.
            req_id: Request ID (auto-generated if not provided).
        
        Returns:
            The trace_id for this trace.
        """
        # Get context from logger if not provided
        ctx = get_request_context()
        conversation_id = conversation_id or ctx.get('conversation_id')
        req_id = req_id or ctx.get('req_id') or str(uuid.uuid4())[:8]
        
        trace_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO traces (trace_id, conversation_id, req_id, query, started_at)
                VALUES (?, ?, ?, ?, ?)
            """, (trace_id, conversation_id, req_id, query, started_at))
        
        logger.debug(f"Started trace {trace_id} for req {req_id}")
        return trace_id
    
    def capture_bm25(
        self,
        trace_id: str,
        results: List[Tuple[str, float]],
        duration_ms: float
    ) -> None:
        """
        Capture BM25 search results.
        
        Args:
            trace_id: The trace ID.
            results: List of (file_path, score) tuples.
            duration_ms: Stage duration in milliseconds.
        """
        data = {
            "results": [
                {"file_path": fp, "score": round(score, 4)}
                for fp, score in results[:50]  # Limit stored results
            ],
            "total_results": len(results)
        }
        
        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE traces 
                SET bm25_data = ?, bm25_duration_ms = ?
                WHERE trace_id = ?
            """, (json.dumps(data), duration_ms, trace_id))
    
    def capture_embedding_paper(
        self,
        trace_id: str,
        results: List[List],
        duration_ms: float
    ) -> None:
        """
        Capture embedding search results for papers.
        
        Args:
            trace_id: The trace ID.
            results: Processed tracer format [[chroma_id, score], ...].
            duration_ms: Stage duration in milliseconds.
        """
        data = {
            "results": [
                {"chroma_id": r[0], "distance": r[1]}
                for r in results[:50]
            ],
            "total_results": len(results)
        }
        
        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE traces 
                SET embedding_paper_data = ?, embedding_paper_duration_ms = ?
                WHERE trace_id = ?
            """, (json.dumps(data), duration_ms, trace_id))
    
    def capture_embedding_chunk(
        self,
        trace_id: str,
        results: List[List],
        duration_ms: float
    ) -> None:
        """
        Capture embedding search results for chunks.
        
        Args:
            trace_id: The trace ID.
            results: Processed tracer format [[chroma_id, score], ...].
            duration_ms: Stage duration in milliseconds.
        """
        data = {
            "results": [
                {"chroma_id": r[0], "distance": r[1]}
                for r in results[:100]
            ],
            "total_results": len(results)
        }
        
        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE traces 
                SET embedding_chunk_data = ?, embedding_chunk_duration_ms = ?,
                    chunks_retrieved = ?
                WHERE trace_id = ?
            """, (json.dumps(data), duration_ms, len(results), trace_id))
    
    def capture_hybrid_fusion(
        self,
        trace_id: str,
        selected_papers: List[str],
        fusion_scores: Dict[str, float]
    ) -> None:
        """
        Capture hybrid fusion results.
        
        Args:
            trace_id: The trace ID.
            selected_papers: List of selected paper file paths.
            fusion_scores: Dict of file_path -> fusion score.
        """
        data = {
            "selected_papers": list(selected_papers),
            "fusion_scores": {
                fp: round(score, 4) 
                for fp, score in sorted(fusion_scores.items(), key=lambda x: -x[1])[:20]
            }
        }
        
        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE traces 
                SET hybrid_fusion_data = ?, papers_retrieved = ?
                WHERE trace_id = ?
            """, (json.dumps(data), len(selected_papers), trace_id))
    
    def capture_reranker(
        self,
        trace_id: str,
        results: Dict[str, Dict],
        duration_ms: float
    ) -> None:
        """
        Capture reranker results.
        
        Args:
            trace_id: The trace ID.
            results: Dict from reranker_tracer function.
            duration_ms: Stage duration in milliseconds.
        """
        data = {
            "results": results,
            "total_reranked": len(results)
        }
        
        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE traces 
                SET reranker_data = ?, reranker_duration_ms = ?,
                    chunks_reranked = ?
                WHERE trace_id = ?
            """, (json.dumps(data), duration_ms, len(results), trace_id))
    
    def capture_generator(
        self,
        trace_id: str,
        answer: str,
        duration_ms: float,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        ttft_ms: Optional[float] = None
    ) -> None:
        """
        Capture generator output.
        
        Args:
            trace_id: The trace ID.
            answer: The generated answer text.
            duration_ms: Stage duration in milliseconds.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            ttft_ms: Time to first token in milliseconds.
        """
        data = {
            "answer": answer,
            "answer_length": len(answer),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_ms": ttft_ms
        }
        
        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE traces 
                SET generator_data = ?, generator_duration_ms = ?
                WHERE trace_id = ?
            """, (json.dumps(data), duration_ms, trace_id))
    
    def capture_hallucination(
        self,
        trace_id: str,
        verifications: List[Dict],
        grounding_ratio: float,
        unsupported_claims: List[str],
        duration_ms: float
    ) -> None:
        """
        Capture hallucination check results.
        
        Args:
            trace_id: The trace ID.
            verifications: List of verification dicts from HallucinationChecker.
            grounding_ratio: Ratio of grounded claims.
            unsupported_claims: List of unsupported claim texts.
            duration_ms: Stage duration in milliseconds.
        """
        data = {
            "verifications": verifications,
            "grounding_ratio": round(grounding_ratio, 4),
            "unsupported_claims": unsupported_claims,
            "num_claims": len(verifications),
            "num_grounded": sum(1 for v in verifications if v.get("is_grounded", False))
        }
        
        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE traces 
                SET hallucination_data = ?, hallucination_duration_ms = ?,
                    grounding_ratio = ?
                WHERE trace_id = ?
            """, (json.dumps(data), duration_ms, grounding_ratio, trace_id))
    
    def finish_trace(
        self,
        trace_id: str,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """
        Finish a trace and calculate total duration.
        
        Args:
            trace_id: The trace ID.
            success: Whether the request succeeded.
            error_message: Error message if failed.
        """
        finished_at = datetime.now(timezone.utc).isoformat()
        
        with self._cursor() as cursor:
            # Get start time to calculate duration
            cursor.execute(
                "SELECT started_at FROM traces WHERE trace_id = ?",
                (trace_id,)
            )
            row = cursor.fetchone()
            
            total_duration_ms = None
            if row:
                started_at = datetime.fromisoformat(row['started_at'])
                finished = datetime.fromisoformat(finished_at)
                total_duration_ms = (finished - started_at).total_seconds() * 1000
            
            cursor.execute("""
                UPDATE traces 
                SET finished_at = ?, success = ?, error_message = ?, total_duration_ms = ?
                WHERE trace_id = ?
            """, (finished_at, 1 if success else 0, error_message, total_duration_ms, trace_id))
        
        logger.debug(f"Finished trace {trace_id}, success={success}")
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace by ID."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_dict(row)
    
    def get_traces_by_conversation(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent traces for a conversation."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT * FROM traces 
                WHERE conversation_id = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (conversation_id, limit))
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a trace (without full data)."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT trace_id, conversation_id, req_id, query, started_at, 
                       finished_at, success, total_duration_ms,
                       papers_retrieved, chunks_retrieved, chunks_reranked, grounding_ratio,
                       bm25_duration_ms, embedding_paper_duration_ms, embedding_chunk_duration_ms,
                       reranker_duration_ms, generator_duration_ms, hallucination_duration_ms
                FROM traces WHERE trace_id = ?
            """, (trace_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return dict(row)
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary with parsed JSON."""
        result = dict(row)
        
        # Parse JSON fields
        json_fields = [
            'bm25_data', 'embedding_paper_data', 'embedding_chunk_data',
            'hybrid_fusion_data', 'reranker_data', 'generator_data', 'hallucination_data'
        ]
        
        for field in json_fields:
            if result.get(field):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    pass
        
        return result
    
    def cleanup_old_traces(self) -> int:
        """
        Remove traces older than retention_days.
        
        Returns:
            Number of traces deleted.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        
        with self._cursor() as cursor:
            cursor.execute(
                "DELETE FROM traces WHERE created_at < ?",
                (cutoff,)
            )
            deleted = cursor.rowcount
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old traces")
            # Vacuum to reclaim space
            self._get_connection().execute("VACUUM")
        
        return deleted
    
    def close(self) -> None:
        """Close database connections."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Singleton instance for convenience
_default_tracer: Optional[RequestTracer] = None


def get_tracer(db_path: str = "traces/request_traces.db") -> RequestTracer:
    """Get or create the default tracer instance."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = RequestTracer(db_path=db_path)
    return _default_tracer


if __name__ == "__main__":
    # Demo
    tracer = RequestTracer(db_path="traces/test_traces.db")
    
    # Start a trace
    trace_id = tracer.start_trace(
        query="What is matrix suppression?",
        conversation_id="conv-123",
        req_id="req-abc"
    )
    print(f"Started trace: {trace_id}")
    
    # Capture stages
    tracer.capture_bm25(
        trace_id,
        results=[("paper1.pdf", 12.5), ("paper2.pdf", 10.2)],
        duration_ms=45.0
    )
    
    tracer.capture_embedding_paper(
        trace_id,
        results=[["chunk-1", 0.85], ["chunk-2", 0.72]],
        duration_ms=120.0
    )
    
    tracer.capture_hybrid_fusion(
        trace_id,
        selected_papers=["paper1.pdf", "paper2.pdf"],
        fusion_scores={"paper1.pdf": 0.9, "paper2.pdf": 0.8}
    )
    
    tracer.capture_reranker(
        trace_id,
        results={
            "chunk-1": {"rank": 1, "rerank_score": 0.95, "original_rank": 2},
            "chunk-2": {"rank": 2, "rerank_score": 0.88, "original_rank": 1}
        },
        duration_ms=1500.0
    )
    
    tracer.capture_generator(
        trace_id,
        answer="Matrix suppression is a phenomenon...",
        duration_ms=3000.0,
        completion_tokens=50
    )
    
    tracer.capture_hallucination(
        trace_id,
        verifications=[
            {"claim": "Matrix suppression exists", "is_grounded": True, "max_score": 0.9},
            {"claim": "Discovered in 1985", "is_grounded": False, "max_score": 0.2}
        ],
        grounding_ratio=0.5,
        unsupported_claims=["Discovered in 1985"],
        duration_ms=800.0
    )
    
    tracer.finish_trace(trace_id, success=True)
    
    # Retrieve and print
    trace = tracer.get_trace(trace_id)
    print(f"\nTrace summary:")
    print(f"  Duration: {trace['total_duration_ms']:.0f}ms")
    print(f"  Papers: {trace['papers_retrieved']}")
    print(f"  Chunks: {trace['chunks_retrieved']}")
    print(f"  Grounding: {trace['grounding_ratio']:.0%}")
    
    tracer.close()