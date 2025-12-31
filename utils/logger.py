"""
Production RAG Logger (stdlib logging)
======================================

Unified structured logging for RAG pipeline using Python's stdlib logging.

Features:
- req_id (trace_id) per request for debugging single interactions
- conversation_id for grouping chat sessions  
- Component tagging (bm25, embedding, reranker, generator)
- Environment-aware output (JSON to stdout in prod, formatted file in dev)
- Thread-safe context propagation via contextvars
- No file proliferation (single stream, handlers cached)

Usage:
    from utils.logger import get_logger, set_request_context, clear_request_context
    
    # At request start (e.g., in FastAPI middleware)
    set_request_context(req_id="abc123", conversation_id="conv-456")
    
    # In any module
    logger = get_logger(__name__)
    logger.info("Retrieved chunks", extra={"count": 50, "duration_ms": 120})
    
    # At request end
    clear_request_context()
"""

import logging
import sys
import os
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from contextvars import ContextVar
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler


# =============================================================================
# Context Variables (Thread-safe request context)
# =============================================================================

_req_id: ContextVar[Optional[str]] = ContextVar('req_id', default=None)
_conversation_id: ContextVar[Optional[str]] = ContextVar('conversation_id', default=None)
_user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


def set_request_context(
    req_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> str:
    """
    Set request context for the current async task / thread.
    Call this at the start of each request (e.g., in FastAPI middleware).
    
    Args:
        req_id: Unique ID for this request. Auto-generated if None.
        conversation_id: ID for the chat session (persistent across messages).
        user_id: Optional user identifier.
    
    Returns:
        The req_id (generated or provided).
    """
    req_id = req_id or str(uuid.uuid4())[:8]  # Short UUID for readability
    _req_id.set(req_id)
    
    if conversation_id:
        _conversation_id.set(conversation_id)
    if user_id:
        _user_id.set(user_id)
    
    return req_id


def clear_request_context() -> None:
    """Clear request context. Call this at the end of each request."""
    _req_id.set(None)
    _conversation_id.set(None)
    _user_id.set(None)


def get_request_context() -> Dict[str, Optional[str]]:
    """Get current request context (for external access)."""
    return {
        'req_id': _req_id.get(),
        'conversation_id': _conversation_id.get(),
        'user_id': _user_id.get()
    }


# =============================================================================
# Environment Detection
# =============================================================================
# os.environ['ENV']='prod'
def _is_production() -> bool:
    """Check if running in production environment."""
    env = os.getenv('ENV', os.getenv('ENVIRONMENT', 'development')).lower()
    return env in ('production', 'prod')


def _get_log_level() -> int:
    """Get log level from environment or default to INFO."""
    level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
    return getattr(logging, level_name, logging.INFO)


# =============================================================================
# JSON Formatter (Production)
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    Structured JSON formatter for production.
    Outputs one JSON object per line for easy parsing by log aggregators.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'component': record.name,
            'message': record.getMessage(),
        }
        
        # Add request context if available
        req_id = _req_id.get()
        if req_id:
            log_data['req_id'] = req_id
        
        conv_id = _conversation_id.get()
        if conv_id:
            log_data['conversation_id'] = conv_id
        
        user_id = _user_id.get()
        if user_id:
            log_data['user_id'] = user_id
        
        # Add extra fields (metrics, counts, durations, etc.)
        if hasattr(record, '__dict__'):
            # Standard fields to exclude from extras
            standard_fields = {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'asctime', 'taskName'  # taskName is asyncio internal
            }
            extras = {
                k: v for k, v in record.__dict__.items()
                if k not in standard_fields and not k.startswith('_')
            }
            if extras:
                log_data['extra'] = extras
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


# =============================================================================
# Human-Readable Formatter (Development)
# =============================================================================

class DevFormatter(logging.Formatter):
    """
    Human-readable formatter for development.
    Includes req_id inline for easy grep.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Build context prefix
        context_parts = []
        
        req_id = _req_id.get()
        if req_id:
            context_parts.append(f"req:{req_id}")
        
        conv_id = _conversation_id.get()
        if conv_id:
            context_parts.append(f"conv:{conv_id[:8]}")
        
        context_str = f"[{' '.join(context_parts)}] " if context_parts else ""
        
        # Format timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Build message
        base_msg = f"{timestamp} | {record.name:<20} | {record.levelname:<5} | {context_str}{record.getMessage()}"
        
        # Add extras inline if present
        if hasattr(record, '__dict__'):
            standard_fields = {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'asctime', 'taskName'  # taskName is asyncio internal
            }
            extras = {
                k: v for k, v in record.__dict__.items()
                if k not in standard_fields and not k.startswith('_')
            }
            if extras:
                extras_str = ' | '.join(f"{k}={v}" for k, v in extras.items())
                base_msg += f" | {extras_str}"
        
        # Add exception if present
        if record.exc_info:
            base_msg += f"\n{self.formatException(record.exc_info)}"
        
        return base_msg


# =============================================================================
# Logger Setup (Singleton pattern)
# =============================================================================

_root_logger_configured = False
_handler_cache: Dict[str, logging.Handler] = {}


def _configure_root_logger() -> None:
    """
    Configure the root RAG logger once.
    All child loggers inherit this configuration.
    """
    global _root_logger_configured
    
    if _root_logger_configured:
        return
    
    # Get the root logger for our application
    root = logging.getLogger('rag')
    root.setLevel(_get_log_level())
    
    # Prevent propagation to root logger (avoid duplicate logs)
    root.propagate = False
    
    # Clear any existing handlers
    root.handlers.clear()
    
    if _is_production():
        # Production: JSON to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
    else:
        # Development: Human-readable to stderr + optional file
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(DevFormatter())
        
        # Optional: Also write to file in development
        log_dir = os.getenv('RAG_LOG_DIR')
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            # Fixed base name; rotation handles the dating and cleanup
            file_path = log_path / "rag.log"
            
            # Rotate at midnight
            file_handler = TimedRotatingFileHandler(
                file_path, 
                when='midnight', 
                interval=1, 
                backupCount=90, 
                encoding='utf-8'
            )
            file_handler.setFormatter(DevFormatter())
            root.addHandler(file_handler)
    
    root.addHandler(handler)
    _root_logger_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a component.
    
    Args:
        name: Component name (usually __name__). Will be prefixed with 'rag.'.
    
    Returns:
        Configured logger instance.
    
    Example:
        logger = get_logger(__name__)  # e.g., 'rag.inference.bm25_search'
        logger.info("Search complete", extra={"count": 30, "duration_ms": 45})
    """
    # Ensure root logger is configured
    _configure_root_logger()
    
    # Normalize name to be under 'rag' namespace
    if not name.startswith('rag.'):
        # Convert module paths like 'inference.bm25_search' to 'rag.bm25_search'
        # or '__main__' to 'rag.main'
        short_name = name.split('.')[-1] if '.' in name else name
        short_name = short_name.replace('__', '').replace('_', '.')
        name = f'rag.{short_name}'
    
    return logging.getLogger(name)


# =============================================================================
# Convenience Functions
# =============================================================================

def log_stage_start(logger: logging.Logger, stage: str, **kwargs) -> None:
    """Log the start of a pipeline stage with standard format."""
    logger.info(f"Stage {stage} started", extra={'stage': stage, 'event': 'stage_start', **kwargs})


def log_stage_end(logger: logging.Logger, stage: str, duration_ms: float, **kwargs) -> None:
    """Log the end of a pipeline stage with timing."""
    logger.info(
        f"Stage {stage} completed in {duration_ms:.1f}ms",
        extra={'stage': stage, 'event': 'stage_end', 'duration_ms': duration_ms, **kwargs}
    )


def log_retrieval_metrics(
    logger: logging.Logger,
    stage: str,
    count: int,
    duration_ms: float,
    top_score: Optional[float] = None,
    **kwargs
) -> None:
    """Log retrieval stage metrics."""
    extra = {
        'stage': stage,
        'event': 'retrieval',
        'count': count,
        'duration_ms': duration_ms,
    }
    if top_score is not None:
        extra['top_score'] = top_score
    extra.update(kwargs)
    
    logger.info(f"Retrieved {count} items", extra=extra)


def log_generation_metrics(
    logger: logging.Logger,
    duration_ms: float,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    ttft_ms: Optional[float] = None,
    **kwargs
) -> None:
    """Log generation stage metrics."""
    extra = {
        'stage': 'generation',
        'event': 'generation',
        'duration_ms': duration_ms,
    }
    if prompt_tokens is not None:
        extra['prompt_tokens'] = prompt_tokens
    if completion_tokens is not None:
        extra['completion_tokens'] = completion_tokens
    if ttft_ms is not None:
        extra['ttft_ms'] = ttft_ms
    extra.update(kwargs)
    
    tokens_info = f", {completion_tokens} tokens" if completion_tokens else ""
    logger.info(f"Generation completed in {duration_ms:.1f}ms{tokens_info}", extra=extra)


def log_request_summary(
    logger: logging.Logger,
    total_duration_ms: float,
    stages: Dict[str, float],
    success: bool = True,
    **kwargs
) -> None:
    """
    Log the "golden log" summary at the end of a request.
    This is the structured payload for analytics.
    """
    extra = {
        'event': 'request_completed',
        'success': success,
        'total_duration_ms': total_duration_ms,
        'latency': stages,
        **kwargs
    }
    
    status = "completed" if success else "failed"
    logger.info(f"Request {status} in {total_duration_ms:.1f}ms", extra=extra)


# =============================================================================
# FastAPI Middleware Helper
# =============================================================================

def create_request_context_middleware():
    """
    Create FastAPI middleware for automatic request context management.
    
    Usage:
        from fastapi import FastAPI
        from utils.logger import create_request_context_middleware
        
        app = FastAPI()
        app.middleware("http")(create_request_context_middleware())
    """
    async def middleware(request, call_next):
        # Extract or generate IDs
        req_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())[:8]
        conversation_id = request.headers.get('X-Conversation-ID')
        user_id = request.headers.get('X-User-ID')
        
        # Set context
        set_request_context(req_id=req_id, conversation_id=conversation_id, user_id=user_id)
        
        try:
            response = await call_next(request)
            # Add req_id to response headers for client correlation
            response.headers['X-Request-ID'] = req_id
            return response
        finally:
            clear_request_context()
    
    return middleware


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == '__main__':
    # Demo: Development mode
    print("=== Development Mode Demo ===\n")
    
    # Simulate a request
    req_id = set_request_context(conversation_id="chat-session-001")
    
    # Get loggers for different components
    bm25_logger = get_logger('inference.bm25_search')
    emb_logger = get_logger('inference.embedding_search')
    rerank_logger = get_logger('inference.reranker')
    gen_logger = get_logger('inference.generator')
    
    # Simulate pipeline stages
    log_stage_start(bm25_logger, "bm25")
    log_retrieval_metrics(bm25_logger, "bm25", count=30, duration_ms=45.2, top_score=12.5)
    
    log_stage_start(emb_logger, "embedding")
    log_retrieval_metrics(emb_logger, "embedding", count=50, duration_ms=120.5, top_score=0.89)
    
    log_stage_start(rerank_logger, "rerank")
    log_retrieval_metrics(rerank_logger, "rerank", count=5, duration_ms=1830.0, top_score=0.94)
    
    log_stage_start(gen_logger, "generation")
    log_generation_metrics(gen_logger, duration_ms=3200.5, prompt_tokens=540, completion_tokens=120, ttft_ms=250.0)
    
    # Final summary
    main_logger = get_logger('pipeline')
    log_request_summary(
        main_logger,
        total_duration_ms=5200.0,
        stages={'bm25': 45.2, 'embedding': 120.5, 'rerank': 1830.0, 'generation': 3200.5},
        chunks_retrieved=50,
        chunks_reranked=5
    )
    
    clear_request_context()
    
    print("\n=== Production Mode Demo (set ENV=production to see JSON) ===")