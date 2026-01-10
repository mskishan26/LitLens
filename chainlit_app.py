"""
Chainlit Frontend for RAG Backend (Production)
===============================================
Handles cold starts, retries, and deployment scenarios.

Run locally:
    chainlit run chainlit_app.py -w

Deploy:
    - Railway, Render, Fly.io, etc.
    - Set RAG_BACKEND_URL environment variable
"""

import chainlit as cl
import httpx
import json
import os
import asyncio

# =============================================================================
# CONFIGURATION
# =============================================================================

# Backend URL - use environment variable for deployment flexibility
RAG_BACKEND_URL = os.environ.get(
    "RAG_BACKEND_URL",
    # "https://mskishan26--rag-backend-service-ragservice-web-app-dev.modal.run"
    "https://mskishan26--litlens-backend-ragservice-web-app.modal.run"
)

# Timeouts
COLD_START_TIMEOUT = 420.0  # 7 min for cold start (model loading)
HEALTH_CHECK_TIMEOUT = 10.0  # Quick health check
STREAM_TIMEOUT = 240.0  # 3 min for actual streaming


# =============================================================================
# HELPERS
# =============================================================================

async def check_backend_health() -> dict:
    """Check if backend is ready, with cold-start awareness."""
    try:
        async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(f"{RAG_BACKEND_URL}/health")
            if response.status_code == 200:
                return response.json()
            return {"status": "error", "code": response.status_code}
    except httpx.TimeoutException:
        return {"status": "waking", "message": "Backend is waking up..."}
    except httpx.ConnectError:
        return {"status": "offline", "message": "Backend unreachable"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def wait_for_backend(status_msg: cl.Message, max_wait: int = 180) -> bool:
    """
    Wait for backend to become ready, updating status message.
    Returns True if ready, False if timeout.
    """
    start_time = asyncio.get_event_loop().time()
    check_interval = 5  # seconds between checks
    
    while (asyncio.get_event_loop().time() - start_time) < max_wait:
        health = await check_backend_health()
        
        if health.get("status") == "healthy" and health.get("pipeline_ready"):
            status_msg.content = "Backend ready!"
            await status_msg.update()
            return True
        
        elapsed = int(asyncio.get_event_loop().time() - start_time)
        status_msg.content = f"⏳ Waking up backend... ({elapsed}s)\n\n*First request spins up the GPU and loads models. This takes 1-2 minutes.*"
        await status_msg.update()
        
        await asyncio.sleep(check_interval)
    
    status_msg.content = "Backend failed to start. Please try again."
    await status_msg.update()
    return False

# SETTINGS CONFIGURATION
async def setup_settings():
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Switch(
                id="hallucination_check",
                label="Enable Hallucination Check",
                initial=False,
                description="Verifies generated answers against sources (slower)."
            )
        ]
    ).send()
    return settings

@cl.on_settings_update
async def on_settings_update(settings):
    # Update the session variable whenever the user toggles the switch
    cl.user_session.set("hallucination_check", settings["hallucination_check"])

# =============================================================================
# CHAINLIT HANDLERS
# =============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session."""
    cl.user_session.set("conversation_id", cl.user_session.get("id"))
    
    # Initialize settings (defaults to False)
    await setup_settings()
    cl.user_session.set("hallucination_check", False)
    
    # Quick health check
    health = await check_backend_health()
    
    if health.get("status") == "healthy" and health.get("pipeline_ready"):
        await cl.Message(
            content="Ready to answer questions about your papers!"
        ).send()
    else:
        await cl.Message(
            content="Welcome! The backend will wake up on your first question (takes ~1-2 min if cold)."
        ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages with cold-start awareness."""
    
    conversation_id = cl.user_session.get("conversation_id", "default")

    # Quick health check first
    health = await check_backend_health()
    
    # If backend seems asleep, show wake-up status
    if health.get("status") != "healthy" or not health.get("pipeline_ready"):
        status_msg = cl.Message(content="Waking up backend...")
        await status_msg.send()
    
    # Create response message for streaming
    response_msg = cl.Message(content="")
    await response_msg.send()
    
    sources = []
    
    # RETRIEVE THE TOGGLE STATE
    hallucination_check = cl.user_session.get("hallucination_check", False)
    
    try:
        # Use longer timeout for potential cold start
        async with httpx.AsyncClient(timeout=COLD_START_TIMEOUT) as client:
            # FIX: Manually build and send request to avoid context manager race conditions
            request_body = {
                "question": message.content,
                "conversation_id": conversation_id,
                "stream": True,
                "enable_hallucination_check": hallucination_check
            }
            
            req = client.build_request(
                "POST",
                f"{RAG_BACKEND_URL}/query",
                json=request_body,
                headers={"Accept": "text/event-stream"}
            )
            
            response = await client.send(req, stream=True)

            try:
                if response.status_code == 503:
                    response_msg.content = "Backend is starting up. Please wait 1-2 minutes and try again."
                    await response_msg.update()
                    return
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    response_msg.content = f"Error ({response.status_code}): {error_text.decode()[:200]}"
                    await response_msg.update()
                    return
                
                # Process SSE stream
                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    
                    if line.startswith("data: "):
                        data = line[6:]
                        
                        if data == "[DONE]":
                            break
                        
                        try:
                            event = json.loads(data)
                            event_type = event.get("type")
                            
                            if event_type == "token":
                                await response_msg.stream_token(event.get("content", ""))
                            
                            elif event_type == "status":
                                stage = event.get("stage", "")
                                if stage == "retrieval_complete":
                                    papers = event.get("papers_found", 0)
                                    chunks = event.get("chunks_reranked", 0)
                                    async with cl.Step(name="Retrieval") as step:
                                        step.output = f"Found {papers} papers, {chunks} relevant chunks"
                                
                                elif stage == "waiting_for_gpu":
                                    pos = event.get("queue_position", 0)
                                    async with cl.Step(name="Queue") as step:
                                        step.output = f"Position {pos} in GPU queue"
                            
                            elif event_type == "context":
                                sources = event.get("data", [])
                            
                            elif event_type == "hallucination":
                                ratio = event.get("grounding_ratio", 0)
                                num_claims = event.get("num_claims", 0)
                                num_grounded = event.get("num_grounded", 0)
                                async with cl.Step(name="✓ Fact Check") as step:
                                    step.output = f"{num_grounded}/{num_claims} claims grounded ({ratio:.0%})"
                            
                            elif event_type == "done":
                                duration = event.get("total_duration_ms", 0)
                                response_msg.content += f"\n\n---\n*Completed in {duration/1000:.1f}s*"
                                await response_msg.update()
                            
                            elif event_type == "error":
                                response_msg.content = f"Error: {event.get('message', 'Unknown error')}"
                                await response_msg.update()
                                # Don't just return, ensure we hit the finally block
                                break 
                        
                        except json.JSONDecodeError:
                            continue
            
            finally:
                # CRITICAL: Manually close the response to prevent "exit cancel scope" errors
                await response.aclose()
        
        # Add sources
        if sources:
            source_elements = []
            for i, src in enumerate(sources[:5]):
                metadata = src.get("metadata", {})
                title = metadata.get("title", metadata.get("file_path", f"Source {i+1}"))
                if "/" in title:
                    title = title.split("/")[-1].replace(".pdf", "").replace("_", " ")
                text = src.get("text", "")[:500]
                score = src.get("score", 0)
                
                source_elements.append(
                    cl.Text(
                        name=f"[{score:.2f}] {title[:50]}",
                        content=text,
                        display="side"
                    )
                )
            
            response_msg.elements = source_elements
            await response_msg.update()
    
    except httpx.TimeoutException:
        response_msg.content = "Request timed out.\n\nThe backend may be cold-starting (takes 1-2 min). Please try again in a moment."
        await response_msg.update()
    
    except httpx.ConnectError:
        response_msg.content = "Could not connect to backend. It may be starting up. Please try again in 1-2 minutes."
        await response_msg.update()
    
    except Exception as e:
        response_msg.content = f"Error: {str(e)}"
        await response_msg.update()