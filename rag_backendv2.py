"""
RAG Backend - Modal GPU Service
"""
import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os

# CONFIGURATION
LOCAL_CODE_DIR = "."
MODAL_REMOTE_CODE_DIR = "/root/app"
MODEL_VOLUME_NAME = "model-weights-vol"
DATA_VOLUME_NAME = "data-storage-vol"

GPU_TYPE = "L4"
CONTAINER_IDLE_TIMEOUT = 900 # 15 minutes

def download_nltk():
    import nltk
    nltk.download("punkt_tab")

# IMAGE DEFINITION
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm", extra_options="--torch-backend=cu128")
    .uv_pip_install(
        "sentence-transformers==5.2.0",
        "chromadb==1.4.0",
        "nltk==3.9.2",
        "rank-bm25",
        "accelerate",
        "fastapi",
        "pydantic>=2.0",
    )
    .run_function(download_nltk)
    .env({
        "ENV": "prod",
        "HF_HOME": "/models",
        "PYTHONPATH": MODAL_REMOTE_CODE_DIR,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir(
        LOCAL_CODE_DIR,
        remote_path=MODAL_REMOTE_CODE_DIR,
        ignore=[".git/", "__pycache__/", ".ipynb_checkpoints/",
                "litlens_inference/", ".files/", ".chainlit/",
                "logs/", "traces/", "*.pyc"]
    )
)

# MODAL APP & VOLUMES
app = modal.App("rag-backend-service")
model_vol = modal.Volume.from_name(MODEL_VOLUME_NAME)
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME)

# REQUEST MODELS
class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = "default-session"
    enable_hallucination_check: Optional[bool] = False
    stream: Optional[bool] = True

# RAG SERVICE
@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/models": model_vol,
        "/data": data_vol,
    },
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    # allow_concurrent_inputs=10, # we need to understand how to use modal.concurrent decorator, also need to define the rules of scaling properly 
    timeout=600,
    min_containers=0,
    max_containers=1,
)
@modal.concurrent(max_inputs=10)
class RAGService:
    """
    Modal class hosting RAG pipeline + FastAPI.
    """
    pipeline = None

    @modal.enter()
    async def startup(self):
        """Initialize pipeline when container starts."""
        import sys
        sys.path.insert(0, MODAL_REMOTE_CODE_DIR)
        
        from utils.logger import configure_logging
        configure_logging({
            "logging": {"environment": "production", "level": "INFO"}
        })

        from chat_pipeline import RAGPipelineV2

        print("Initializing RAG Pipeline...")

        config_path = os.path.join(MODAL_REMOTE_CODE_DIR, "config.yaml")
        if not os.path.exists(config_path):
            config_path = None

        self.pipeline = RAGPipelineV2(config_path=config_path)
        await self.pipeline.initialize()

        print("RAG Pipeline Ready!")

    @modal.exit()
    async def shutdown(self):
        if self.pipeline:
            await self.pipeline.cleanup()
            print("Pipeline cleaned up successfully")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with endpoints that access self directly."""
        
        web_app = FastAPI(
            title="RAG Backend API",
            version="2.0.0"
        )

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Store reference to self for use in endpoints
        service = self

        @web_app.get("/health")
        async def health():
            """Health check."""
            try:
                return {
                    "status": "healthy",
                    "pipeline_ready": service.pipeline is not None,
                    "queue_depth": service.pipeline.generation_queue_depth if service.pipeline else 0
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}

        @web_app.post("/query")
        async def query_endpoint(request: QueryRequest):
            """Main query endpoint."""
            
            # Check pipeline is ready
            if service.pipeline is None:
                raise HTTPException(status_code=503, detail="Pipeline not initialized")

            if request.stream:
                async def event_generator():
                    try:
                        async for event in service.pipeline.answer_stream(
                            query=request.question,
                            conversation_id=request.conversation_id,
                            enable_hallucination_check=request.enable_hallucination_check
                        ):
                            yield f"data: {json.dumps(event)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        # Send error as SSE event before closing
                        error_event = {"type": "error", "message": str(e)}
                        yield f"data: {json.dumps(error_event)}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    event_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    }
                )
            else:
                # Non-streaming response
                try:
                    answer = ""
                    sources = []
                    trace_id = None
                    total_duration_ms = 0

                    async for event in service.pipeline.answer_stream(
                        query=request.question,
                        conversation_id=request.conversation_id,
                        enable_hallucination_check=request.enable_hallucination_check
                    ):
                        event_type = event.get("type")
                        if event_type == "token":
                            answer += event.get("content", "")
                        elif event_type == "context":
                            sources = event.get("data", [])
                        elif event_type == "done":
                            trace_id = event.get("trace_id")
                            total_duration_ms = event.get("total_duration_ms", 0)
                        elif event_type == "error":
                            raise HTTPException(status_code=500, detail=event.get("message"))

                    return {
                        "answer": answer,
                        "trace_id": trace_id,
                        "sources": sources,
                        "total_duration_ms": total_duration_ms
                    }
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

        @web_app.post("/query/stream")
        async def query_stream_endpoint(request: QueryRequest):
            """NDJSON streaming endpoint."""
            
            if service.pipeline is None:
                raise HTTPException(status_code=503, detail="Pipeline not initialized")

            async def ndjson_generator():
                try:
                    async for event in service.pipeline.answer_stream(
                        query=request.question,
                        conversation_id=request.conversation_id,
                        enable_hallucination_check=request.enable_hallucination_check
                    ):
                        yield json.dumps(event) + "\n"
                except Exception as e:
                    yield json.dumps({"type": "error", "message": str(e)}) + "\n"

            return StreamingResponse(
                ndjson_generator(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        return web_app