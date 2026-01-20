"""
RAG Backend Lite - Modal CPU-only Mock Service
===============================================

Lightweight version of the RAG backend that:
- Runs on CPU only (no GPU, no L4)
- Keeps DynamoDB persistence fully functional
- Mocks the /query endpoint with lorem ipsum streaming
- Maintains identical API contract for frontend compatibility

Use case: 
- Development and testing without GPU costs
- Testing DynamoDB persistence and chat management
- Frontend integration testing

Endpoints (identical contract to full backend):
- POST /query - Mock RAG query (streaming lorem ipsum)
- GET /health - Health check
- GET /chats - List user's chats
- GET /chats/{chat_id}/messages - Get chat history
- GET /chats/{chat_id}/messages/{message_id}/trace - Get trace for a message
- POST /chats/{chat_id}/generate-title - Generate a mock title
"""

import modal
from fastapi import FastAPI, HTTPException, Header, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os
import asyncio
import random
import time

# =============================================================================
# Configuration
# =============================================================================

LOCAL_CODE_DIR = "."
MODAL_REMOTE_CODE_DIR = "/root/app"

# No GPU, longer idle timeout since it's cheap
CONTAINER_IDLE_TIMEOUT = 1800  # 30 minutes

# Security
API_KEY_NAME = "X-Service-Token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# =============================================================================
# Modal Image (CPU-only, minimal dependencies)
# =============================================================================

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi",
        "pydantic>=2.0",
        "boto3",
    )
    .env({
        "ENV": "dev",
        "PYTHONPATH": MODAL_REMOTE_CODE_DIR,
        "SERVICE_AUTH_TOKEN": "dev-secret-123",  # Override in Modal Secrets
        "USE_DYNAMODB": "true",
        "AWS_REGION": "us-east-2",
        "AWS_ROLE_ARN": "arn:aws:iam::649489225731:role/modal-litlens-role",
    })
    .add_local_dir(
        LOCAL_CODE_DIR,
        remote_path=MODAL_REMOTE_CODE_DIR,
        ignore=[".git/", "__pycache__/", ".ipynb_checkpoints/",
                "litlens_inference/", ".files/", ".chainlit/",
                "logs/", "traces/", "*.pyc", "aws/",
                # Ignore GPU-heavy modules
                "inference/", "models/"],
    )
)

# =============================================================================
# Modal App
# =============================================================================

app = modal.App("rag-backend-lite")


class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = "default-session"
    enable_hallucination_check: Optional[bool] = False
    stream: Optional[bool] = True


class GenerateTitleRequest(BaseModel):
    """Request body for title generation."""
    queries: list[str]


# =============================================================================
# Mock Data Generators
# =============================================================================

LOREM_SENTENCES = [
    "The analysis reveals significant correlations between the variables under study.",
    "Previous research has established foundational frameworks for understanding this phenomenon.",
    "Statistical methods employed include regression analysis and hypothesis testing.",
    "The findings suggest a moderate effect size with implications for clinical practice.",
    "Further investigation is warranted to validate these preliminary observations.",
    "The methodology adheres to established protocols in the field of biostatistics.",
    "Cross-validation techniques were applied to ensure model robustness.",
    "The confidence intervals indicate reasonable precision in the estimates.",
    "Longitudinal data analysis reveals temporal patterns of interest.",
    "The study population was selected using stratified random sampling.",
    "Causal inference methods were employed to address confounding variables.",
    "The results are consistent with theoretical predictions from prior work.",
    "Sensitivity analyses confirm the stability of the primary findings.",
    "The sample size provides adequate statistical power for the main hypotheses.",
    "Subgroup analyses reveal heterogeneous effects across demographic categories.",
]

MOCK_SOURCES = [
    {
        "text": "The fundamental principles of causal inference require careful consideration of confounding variables and selection bias.",
        "metadata": {
            "title": "Introduction to Causal Inference Methods",
            "file_path": "papers/causal_inference_intro.pdf",
            "chroma_id": "chunk_mock_001",
        },
        "score": 0.92,
    },
    {
        "text": "Regression analysis provides a flexible framework for modeling relationships between dependent and independent variables.",
        "metadata": {
            "title": "Statistical Methods in Biomedical Research",
            "file_path": "papers/biostats_methods.pdf",
            "chroma_id": "chunk_mock_002",
        },
        "score": 0.87,
    },
    {
        "text": "The propensity score methodology offers a powerful approach to addressing confounding in observational studies.",
        "metadata": {
            "title": "Propensity Score Methods: A Comprehensive Review",
            "file_path": "papers/propensity_scores.pdf",
            "chroma_id": "chunk_mock_003",
        },
        "score": 0.84,
    },
]


def generate_mock_answer(query: str, num_sentences: int = 5) -> str:
    """Generate a mock answer based on the query."""
    # Use query length to seed randomness for reproducibility in testing
    random.seed(len(query))
    selected = random.sample(LOREM_SENTENCES, min(num_sentences, len(LOREM_SENTENCES)))
    random.seed()  # Reset seed
    return " ".join(selected)


def generate_mock_title(queries: list[str]) -> str:
    """Generate a mock title from queries."""
    if not queries:
        return "New Chat"
    
    # Extract key words from first query
    first_query = queries[0]
    words = first_query.split()[:6]
    
    # Simple title generation
    if len(words) >= 3:
        return " ".join(words[:5]).title()
    return first_query[:50]


# =============================================================================
# RAG Lite Service
# =============================================================================

@app.cls(
    image=image,
    # No GPU!
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    timeout=300,
    min_containers=0,
    max_containers=2,
    # secrets=[modal.Secret.from_name("litlens-config")],
)
@modal.concurrent(max_inputs=20)  # Higher concurrency since no GPU bottleneck
class RAGServiceLite:
    """Modal class hosting mock RAG pipeline + FastAPI (CPU only)."""
    
    persistence = None

    @modal.enter()
    async def startup(self):
        """Initialize persistence only (no ML models)."""
        import sys
        sys.path.insert(0, MODAL_REMOTE_CODE_DIR)
        
        print("🚀 Starting RAG Backend Lite (CPU-only mock service)")
        
        # Initialize persistence
        use_dynamodb = os.environ.get("USE_DYNAMODB", "false").lower() == "true"
        aws_role_arn = os.environ.get("AWS_ROLE_ARN")
        aws_region = os.environ.get("AWS_REGION", "us-east-2")
        
        if use_dynamodb and aws_role_arn:
            print(f"Initializing DynamoDB with OIDC...")
            print(f"  Role ARN: {aws_role_arn}")
            print(f"  Region: {aws_region}")
            
            try:
                from dynamo_persistence import DynamoPersistence
                self.persistence = DynamoPersistence.from_oidc(
                    role_arn=aws_role_arn,
                    region=aws_region,
                )
                print("✓ DynamoDB initialized")
            except Exception as e:
                print(f"ERROR: DynamoDB init failed: {e}")
                self.persistence = None
        else:
            if use_dynamodb:
                print("WARNING: USE_DYNAMODB=true but AWS_ROLE_ARN not set")
            print("Running without persistence")
        
        print("✓ RAG Backend Lite ready (mock mode)")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI application."""
        
        web_app = FastAPI(
            title="RAG Backend Lite API",
            version="2.1.0-lite",
            description="CPU-only mock service for development and testing",
        )
        
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        service = self

        async def verify_token(api_key: str = Security(api_key_header)):
            """Verify the service token (authenticates the frontend app)."""
            expected = os.environ.get("SERVICE_AUTH_TOKEN")
            if api_key != expected:
                raise HTTPException(status_code=403, detail="Invalid credentials")
            return api_key

        def parse_anonymous_header(x_user_anonymous: Optional[str] = Header(None)) -> bool:
            """Parse the X-User-Anonymous header."""
            if x_user_anonymous is None:
                return False
            return x_user_anonymous.lower() == "true"

        # =================================================================
        # Core Endpoints
        # =================================================================

        @web_app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "mode": "lite",
                "pipeline_ready": True,  # Always ready in mock mode
                "persistence": "dynamodb" if service.persistence else "none",
                "queue_depth": 0,
            }

        @web_app.post("/query")
        async def query_endpoint(
            request: QueryRequest,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """
            Mock RAG query - streams lorem ipsum response.
            
            Maintains identical API contract to full backend.
            """
            from dynamo_persistence import TraceBuilder, _generate_id
            
            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)
            
            print(f"[MOCK] Query: user={user_id}, anon={is_anonymous}, conv={request.conversation_id}")

            message_id = _generate_id("msg_")

            if request.stream:
                async def stream():
                    trace = None
                    try:
                        start_time = time.perf_counter()
                        
                        # Create and save trace start if persistence available
                        if service.persistence:
                            trace = TraceBuilder(
                                chat_id=request.conversation_id,
                                user_id=user_id,
                                query=request.question,
                                is_anonymous=is_anonymous,
                                message_id=message_id,
                            )
                            await service.persistence.save_trace_start(trace)
                        
                        # Simulate retrieval delay
                        await asyncio.sleep(0.1)
                        
                        # Emit status event
                        yield f"data: {json.dumps({'type': 'status', 'stage': 'retrieval_complete', 'papers_found': 3, 'chunks_reranked': 5})}\n\n"
                        
                        # Emit context (sources)
                        sources_for_frontend = [
                            {
                                "text": s["text"],
                                "metadata": s["metadata"],
                                "score": s["score"],
                            }
                            for s in MOCK_SOURCES
                        ]
                        yield f"data: {json.dumps({'type': 'context', 'data': sources_for_frontend})}\n\n"
                        
                        # Add mock retrieval stages to trace
                        if trace:
                            trace.add_stage("bm25", {"total": 50, "mock": True}, 45.0, papers_retrieved=50)
                            trace.add_stage("embedding_paper", {"total": 50, "mock": True}, 120.0)
                            trace.add_stage("hybrid_fusion", {"mock": True}, 2.0)
                            trace.add_stage("embedding_chunk", {"total": 100, "mock": True}, 80.0, chunks_retrieved=100)
                            trace.add_stage("reranker", {"mock": True}, 150.0, chunks_reranked=5)
                        
                        # Stream mock answer tokens
                        mock_answer = generate_mock_answer(request.question, num_sentences=5)
                        words = mock_answer.split()
                        
                        accumulated_answer = ""
                        for i, word in enumerate(words):
                            token = word + (" " if i < len(words) - 1 else "")
                            accumulated_answer += token
                            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                            # Simulate generation speed (30-80ms per token)
                            await asyncio.sleep(random.uniform(0.03, 0.08))
                        
                        # Add generator stage to trace
                        if trace:
                            trace.add_stage("generator", {"answer_length": len(accumulated_answer), "mock": True}, 500.0)
                        
                        # Hallucination check (if enabled)
                        hal_result = None
                        if request.enable_hallucination_check:
                            await asyncio.sleep(0.2)
                            hal_result = {
                                "type": "hallucination",
                                "grounding_ratio": 0.85,
                                "num_claims": 5,
                                "num_grounded": 4,
                                "unsupported_claims": ["One claim was not fully supported"],
                                "verifications": [],
                            }
                            yield f"data: {json.dumps(hal_result)}\n\n"
                            
                            if trace:
                                trace.add_stage("hallucination", {"mock": True, "grounding_ratio": 0.85}, 200.0, grounding_ratio=0.85)
                        
                        total_duration = (time.perf_counter() - start_time) * 1000
                        
                        # Save to DynamoDB if available
                        if service.persistence and trace:
                            trace.complete()
                            await service.persistence.save_turn(
                                trace=trace,
                                answer=accumulated_answer,
                                sources=sources_for_frontend,
                                hallucination_result=hal_result,
                            )
                            print(f"[MOCK] Saved to DynamoDB: message_id={message_id}")
                        
                        # Done event
                        yield f"data: {json.dumps({'type': 'done', 'message_id': message_id, 'chat_id': request.conversation_id, 'total_duration_ms': total_duration})}\n\n"
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        if trace:
                            trace.fail("mock_generation", str(e))
                            if service.persistence:
                                await service.persistence.save_turn(
                                    trace=trace,
                                    answer="",
                                    sources=[],
                                    hallucination_result=None,
                                )
                        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "X-Message-Id": message_id,
                    },
                )
            else:
                # Non-streaming response
                mock_answer = generate_mock_answer(request.question, num_sentences=5)
                
                # Save to DynamoDB if available
                if service.persistence:
                    trace = TraceBuilder(
                        chat_id=request.conversation_id,
                        user_id=user_id,
                        query=request.question,
                        is_anonymous=is_anonymous,
                        message_id=message_id,
                    )
                    trace.add_stage("generator", {"mock": True}, 100.0)
                    trace.complete()
                    
                    await service.persistence.save_turn(
                        trace=trace,
                        answer=mock_answer,
                        sources=MOCK_SOURCES,
                        hallucination_result=None,
                    )
                
                return {
                    "answer": mock_answer,
                    "message_id": message_id,
                    "chat_id": request.conversation_id,
                    "sources": MOCK_SOURCES,
                    "total_duration_ms": 150.0,
                }

        # =================================================================
        # Chat History Endpoints (unchanged from full backend)
        # =================================================================

        @web_app.get("/chats")
        async def list_chats(
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
            limit: int = 20,
        ):
            """List user's recent chats (for sidebar)."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)
            
            if is_anonymous or user_id == "anonymous":
                return {
                    "chats": [],
                    "message": "Sign in to view chat history",
                }
            
            chats = await service.persistence.get_user_chats(user_id=user_id, limit=limit)
            return {"chats": chats}

        @web_app.get("/chats/{chat_id}/messages")
        async def get_messages(
            chat_id: str,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            limit: int = 100,
        ):
            """Get all messages for a chat."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            user_id = x_user_id or "anonymous"
            
            is_owner, actual_owner = await service.persistence.verify_chat_ownership(
                chat_id, user_id
            )
            
            if actual_owner is not None and not is_owner:
                raise HTTPException(status_code=403, detail="Access denied")
            
            messages = await service.persistence.get_chat_messages(
                chat_id=chat_id, limit=limit
            )
            return {"chat_id": chat_id, "messages": messages}

        # =================================================================
        # Trace Endpoint
        # =================================================================

        @web_app.get("/chats/{chat_id}/messages/{message_id}/trace")
        async def get_message_trace(
            chat_id: str,
            message_id: str,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """Get the trace for a specific message."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)
            
            trace, error = await service.persistence.get_trace_with_auth(
                chat_id=chat_id,
                message_id=message_id,
                requesting_user_id=user_id,
                is_anonymous=is_anonymous,
            )
            
            if error == "not_found":
                raise HTTPException(status_code=404, detail="Trace not found")
            elif error == "unauthorized":
                raise HTTPException(status_code=403, detail="Access denied")
            
            return trace

        # =================================================================
        # Chat Title Generation (mock)
        # =================================================================

        @web_app.post("/chats/{chat_id}/generate-title")
        async def generate_chat_title(
            chat_id: str,
            request: GenerateTitleRequest,
            token: str = Depends(verify_token),
            x_user_id: Optional[str] = Header(None),
            x_user_anonymous: Optional[str] = Header(None),
        ):
            """Generate a mock title for a chat."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            user_id = x_user_id or "anonymous"
            is_anonymous = parse_anonymous_header(x_user_anonymous)
            
            if is_anonymous or user_id == "anonymous":
                raise HTTPException(
                    status_code=403,
                    detail="Sign in to generate chat titles"
                )
            
            is_owner, actual_owner = await service.persistence.verify_chat_ownership(
                chat_id, user_id
            )
            
            if actual_owner is None:
                raise HTTPException(status_code=404, detail="Chat not found")
            
            if not is_owner:
                raise HTTPException(status_code=403, detail="Access denied")
            
            if not request.queries:
                raise HTTPException(status_code=400, detail="No queries provided")
            
            # Generate mock title
            title = generate_mock_title(request.queries[:3])
            
            # Update the title in metadata table
            success = await service.persistence.update_chat_title(
                user_id=user_id,
                chat_id=chat_id,
                title=title,
            )
            
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to update chat title"
                )
            
            return {
                "chat_id": chat_id,
                "title": title,
                "generated": True,
                "mock": True,
            }

        # =================================================================
        # Admin/Debug Endpoints
        # =================================================================

        @web_app.get("/admin/abandoned-traces")
        async def get_abandoned_traces(
            token: str = Depends(verify_token),
            limit: int = 50,
        ):
            """Get traces that were abandoned (status still 'running')."""
            if not service.persistence:
                raise HTTPException(status_code=501, detail="Persistence not configured")
            
            traces = await service.persistence.get_abandoned_traces(limit=limit)
            return {
                "count": len(traces),
                "traces": traces,
            }

        return web_app