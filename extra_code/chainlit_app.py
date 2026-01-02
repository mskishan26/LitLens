"""
RAG Pipeline Chainlit Application
==================================

Full-featured Chainlit UI for the RAG pipeline with:
- Session persistence using JSONL files
- Chat history sidebar with resume functionality
- Streaming responses
- Source display
- Hallucination check results
- Debug information in expandable sections
- Feedback support

Run with: chainlit run chainlit_app.py -w
"""

import os
import sys
import json
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# -----------------------------------------------------------------------------
# Path Setup
# -----------------------------------------------------------------------------
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
# Add parent directory to path for imports (assumes script is in src/inference/)
PROJECT_ROOT = SCRIPT_DIR.parent
print(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print('added root path')

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
os.environ['HF_HOME'] = os.environ.get('HF_HOME', "/scratch/sathishbabu.ki/vllm_models/vllm/.cache/huggingface")
os.environ['RAG_CONSOLE_LOG_LEVEL'] = 'INFO'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Suppress transformers logs
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import chainlit as cl
import chainlit.data as cl_data
from chainlit.types import ThreadDict

# Import our modules - adjust path as needed
from utils.config_loader import load_config
from utils.logger import get_chat_logger

# Import data layer from same directory
from chainlit_data_layer import JSONLDataLayer

logger = get_chat_logger("chainlit_app")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
config = load_config()
SESSION_LOGS_DIR = Path(config['paths']['session_logs'])
SESSION_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize custom data layer
cl_data._data_layer = JSONLDataLayer(
    session_logs_dir=str(SESSION_LOGS_DIR)
)

# Global pipeline instance (shared across sessions for efficiency)
_pipeline = None


def get_pipeline():
    """Get or create the RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing RAG Pipeline...")
        from inference.chat_pipeline import RAGPipeline
        _pipeline = RAGPipeline(
            no_unload=True,  # Keep models loaded for better performance
            enable_hallucination_check=True
        )
    return _pipeline


# -----------------------------------------------------------------------------
# Authentication (Optional - comment out to disable)
# -----------------------------------------------------------------------------
#@cl.password_auth_callback
#def auth_callback(username: str, password: str) -> Optional[cl.User]:
#    """
#    Simple password authentication.
#    
#    For production, replace with proper authentication.
#    Default credentials: admin/admin or any username with password 'rag'
#    """
#    # Allow any user with password 'rag' for easy testing
#    if password == "rag" or (username == "admin" and password == "admin"):
#        return cl.User(
#            identifier=username,
#            metadata={"role": "user", "provider": "credentials"}
#        )
#    return None


# -----------------------------------------------------------------------------
# Chat Lifecycle Handlers
# -----------------------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat session."""
    # Generate thread ID
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("message_count", 0)
    
    # Create thread in data layer
    user = cl.user_session.get("user")
    user_id = user.identifier if user else None
    
    await cl_data._data_layer.update_thread(
        thread_id=thread_id,
        name='New Conversation',
        user_id=user_id
    )
    
    # Create empty thread file
    thread_file = SESSION_LOGS_DIR / f"thread_{thread_id}.jsonl"
    thread_file.touch()
    
    # Send welcome message
    welcome_msg = """# 🔬 RAG Research Assistant

Welcome! I'm your research assistant for biostatistics and causal inference papers.

**Features:**
- 📚 Searches through your paper collection
- 🎯 Uses hybrid retrieval (BM25 + embeddings)
- ✅ Verifies claims against source documents
- 📊 Shows sources and confidence scores

**Ask me anything about your research papers!**

---
*Tip: Your conversation will be saved and you can resume it later from the sidebar.*
"""
    
    await cl.Message(content=welcome_msg).send()
    
    logger.info(f"New chat session started: {thread_id}")


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Resume a previous chat session."""
    thread_id = thread.get("id")
    cl.user_session.set("thread_id", thread_id)
    
    # Count existing messages
    steps = thread.get("steps", [])
    message_count = len([s for s in steps if s.get("type") == "user_message"])
    cl.user_session.set("message_count", message_count)
    
    # Restore message history for context
    memory = []
    for step in steps:
        if step.get("type") == "user_message":
            memory.append({"role": "user", "content": step.get("output", "")})
        elif step.get("type") == "assistant_message":
            memory.append({"role": "assistant", "content": step.get("output", "")})
    
    cl.user_session.set("memory", memory)
    
    logger.info(f"Resumed chat session: {thread_id} with {message_count} messages")
    
    # Notify user
    await cl.Message(
        content=f"*Resumed conversation with {message_count} previous messages*"
    ).send()


@cl.on_stop
async def on_stop():
    """Handle user stopping generation."""
    logger.info("User stopped generation")


@cl.on_chat_end
async def on_chat_end():
    """Cleanup when chat ends."""
    thread_id = cl.user_session.get("thread_id")
    logger.info(f"Chat session ended: {thread_id}")


# -----------------------------------------------------------------------------
# Message Handler
# -----------------------------------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""
    thread_id = cl.user_session.get("thread_id")
    query = message.content
    
    # Increment message count
    msg_count = cl.user_session.get("message_count", 0) + 1
    cl.user_session.set("message_count", msg_count)
    
    # Update thread name from first message
    if msg_count == 1:
        name = query[:50] + ("..." if len(query) > 50 else "")
        await cl_data._data_layer.update_thread(thread_id, name=name)
    
    # Create entry for logging
    entry_id = f"{thread_id}_{msg_count:04d}"
    entry_timestamp = datetime.now().isoformat()
    
    # Get pipeline
    pipeline = get_pipeline()
    
    # Create response message with streaming
    response_msg = cl.Message(content="")
    await response_msg.send()
    
    # Variables to collect data
    sources = []
    hallucination_result = None
    accumulated_answer = ""
    
    try:
        # Show retrieval step
        async with cl.Step(name="🔍 Retrieving relevant passages", type="tool") as step:
            step.output = "Searching with BM25 and embeddings..."
        
        # Run pipeline with streaming
        stream = pipeline.answer_streaming(
            query=query,
            return_metadata=True,
            session_file=None
        )
        
        # Process stream
        for chunk in stream:
            if isinstance(chunk, str):
                accumulated_answer += chunk
                await response_msg.stream_token(chunk)
            elif isinstance(chunk, dict):
                if 'contexts' in chunk:
                    sources = chunk['contexts']
                elif 'hallucination_check' in chunk:
                    hallucination_result = chunk['hallucination_check']
        
        # Finalize response
        await response_msg.update()
        
        # Display sources
        if sources:
            sources_content = "\n\n---\n### 📚 Sources\n"
            for i, ctx in enumerate(sources, 1):
                if isinstance(ctx, dict):
                    meta = ctx.get('metadata', ctx)
                    title = meta.get('paper_title', 'Unknown Paper')
                    fname = Path(meta.get('file_path', '')).name
                    score = ctx.get('rerank_score', 0)
                    sources_content += f"\n**{i}. {title}**\n"
                    sources_content += f"   - File: `{fname}`\n"
                    sources_content += f"   - Relevance Score: `{score:.4f}`\n"
            
            await cl.Message(content=sources_content).send()
        
        # Display hallucination check results
        if hallucination_result:
            await display_hallucination_check(hallucination_result)
        
        # Build complete entry for logging
        entry = {
            'session_id': thread_id,
            'entry_id': entry_id,
            'timestamp': entry_timestamp,
            'query': query,
            'answer': accumulated_answer,
            'config': {
                'k_papers': pipeline.config['retrieval']['k_papers'],
                'm_chunks': pipeline.config['retrieval']['m_chunks'],
                'n_reranked': pipeline.config['retrieval']['n_reranked'],
            },
            'final_sources': [
                {
                    'rank': ctx.get('rank', i),
                    'paper_title': ctx.get('metadata', {}).get('paper_title', 'Unknown'),
                    'file_path': ctx.get('metadata', {}).get('file_path', ''),
                    'rerank_score': ctx.get('rerank_score', 0)
                }
                for i, ctx in enumerate(sources, 1)
            ] if sources else [],
            'hallucination_check': hallucination_result.to_dict() if hasattr(hallucination_result, 'to_dict') else hallucination_result,
            'status': 'success'
        }
        
        # Save to data layer
        await cl_data._data_layer.add_rag_entry(thread_id, entry)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        error_msg = f"\n\n⚠️ **Error:** {str(e)}"
        await response_msg.stream_token(error_msg)
        await response_msg.update()
        
        # Log error
        entry = {
            'session_id': thread_id,
            'entry_id': entry_id,
            'timestamp': entry_timestamp,
            'query': query,
            'answer': accumulated_answer + error_msg,
            'status': 'error',
            'error_message': str(e)
        }
        await cl_data._data_layer.add_rag_entry(thread_id, entry)


async def display_hallucination_check(result: Any):
    """Display hallucination check results."""
    if result is None:
        return
    
    # Handle both object and dict formats
    if hasattr(result, 'to_dict'):
        data = result.to_dict()
    else:
        data = result
    
    num_claims = data.get('num_claims', 0)
    num_grounded = data.get('num_grounded', 0)
    num_unsupported = data.get('num_unsupported', 0)
    grounding_ratio = data.get('grounding_ratio', 0)
    unsupported_claims = data.get('unsupported_claims', [])
    
    if num_claims == 0:
        return
    
    # Build content
    if num_unsupported > 0:
        content = f"""
---
### ⚠️ Grounding Check

**{num_grounded}/{num_claims}** claims are supported by source documents ({grounding_ratio:.0%})

#### Unsupported Claims:
"""
        for i, claim in enumerate(unsupported_claims, 1):
            display_claim = claim[:150] + "..." if len(claim) > 150 else claim
            content += f"\n{i}. {display_claim}\n"
        
        content += "\n*These claims could not be verified against the source documents.*"
        
    else:
        content = f"""
---
### ✅ Grounding Check

All **{num_claims}** claims are grounded in source documents.
"""
    
    await cl.Message(content=content).send()


# -----------------------------------------------------------------------------
# Starters (Suggested prompts)
# -----------------------------------------------------------------------------
@cl.set_starters
async def set_starters():
    """Set suggested starter prompts."""
    return [
        cl.Starter(
            label="Causal Inference",
            message="What are the key assumptions for causal inference from observational data?",
        ),
        cl.Starter(
            label="Mixed Effects Models",
            message="Explain the difference between fixed and random effects in linear mixed models.",
        ),
        cl.Starter(
            label="Survival Analysis",
            message="What is the Cox proportional hazards model and when should it be used?",
        ),
        cl.Starter(
            label="Multiple Testing",
            message="How does the Benjamini-Hochberg procedure control the false discovery rate?",
        ),
    ]


# -----------------------------------------------------------------------------
# Main entry point (for direct running)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Run with: chainlit run chainlit_app.py -w")