"""
Production RAG Logger (stdlib logging) - V2
============================================

Changes from V1:
- Configuration via config dict instead of env vars
- Must call configure_logging(config) before using loggers
- Supports environment, log_level, and log_dir from config

Usage:
    from utils.logger import configure_logging, get_logger
    from utils.config_loader import load_config
    
    config = load_config()
    configure_logging(config)  # Call ONCE at startup
    
    logger = get_logger(__name__)
    logger.info("Ready")
"""

import logging
import sys
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
    """Set request context for the current async task / thread."""
    req_id = req_id or str(uuid.uuid4())[:8]
    _req_id.set(req_id)
    
    if conversation_id:
        _conversation_id.set(conversation_id)
    if user_id:
        _user_id.set(user_id)
    
    return req_id


def clear_request_context() -> None:
    """Clear request context."""
    _req_id.set(None)
    _conversation_id.set(None)
    _user_id.set(None)


def get_request_context() -> Dict[str, Optional[str]]:
    """Get current request context."""
    return {
        'req_id': _req_id.get(),
        'conversation_id': _conversation_id.get(),
        'user_id': _user_id.get()
    }


# =============================================================================
# Configuration State
# =============================================================================

class _LogConfig:
    """Internal configuration state."""
    is_production: bool = False
    log_level: int = logging.INFO
    log_dir: Optional[Path] = None
    configured: bool = False

_config = _LogConfig()


# =============================================================================
# JSON Formatter (Production)
# =============================================================================

class JSONFormatter(logging.Formatter):
    """Structured JSON formatter for production."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'component': record.name,
            'message': record.getMessage(),
        }
        
        if req_id := _req_id.get():
            log_data['req_id'] = req_id
        if conv_id := _conversation_id.get():
            log_data['conversation_id'] = conv_id
        if user_id := _user_id.get():
            log_data['user_id'] = user_id
        
        # Add extra fields
        standard_fields = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'message', 'asctime', 'taskName'
        }
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_fields and not k.startswith('_')
        }
        if extras:
            log_data['extra'] = extras
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


# =============================================================================
# Human-Readable Formatter (Development)
# =============================================================================

class DevFormatter(logging.Formatter):
    """Human-readable formatter for development."""
    
    def format(self, record: logging.LogRecord) -> str:
        context_parts = []
        
        if req_id := _req_id.get():
            context_parts.append(f"req:{req_id}")
        if conv_id := _conversation_id.get():
            context_parts.append(f"conv:{conv_id[:8]}")
        
        context_str = f"[{' '.join(context_parts)}] " if context_parts else ""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        base_msg = f"{timestamp} | {record.name:<20} | {record.levelname:<5} | {context_str}{record.getMessage()}"
        
        # Add extras inline
        standard_fields = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'message', 'asctime', 'taskName'
        }
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_fields and not k.startswith('_')
        }
        if extras:
            extras_str = ' | '.join(f"{k}={v}" for k, v in extras.items())
            base_msg += f" | {extras_str}"
        
        if record.exc_info:
            base_msg += f"\n{self.formatException(record.exc_info)}"
        
        return base_msg


# =============================================================================
# Main Configuration Function
# =============================================================================

def configure_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging from config dict. Call ONCE at application startup,
    BEFORE calling get_logger().
    
    Expected config structure:
        logging:
          environment: "development"  # or "production" / "prod"
          level: "INFO"               # DEBUG, INFO, WARNING, ERROR
        paths:
          logs: "/path/to/logs"       # optional, for file logging
    
    Args:
        config: Loaded configuration dictionary
    """
    global _config
    
    if _config.configured:
        return  # Already configured
    
    # Extract logging config
    log_config = config.get('logging', {})
    
    # Environment
    env = log_config.get('environment', 'development').lower()
    _config.is_production = env in ('production', 'prod')
    
    # Log level
    level_name = log_config.get('level', 'INFO').upper()
    _config.log_level = getattr(logging, level_name, logging.INFO)
    
    # Log directory (from paths section)
    paths = config.get('paths', {})
    if log_dir := paths.get('logs'):
        _config.log_dir = Path(log_dir)
        _config.log_dir.mkdir(parents=True, exist_ok=True)
    
    _config.configured = True
    
    # Now configure the root logger
    _setup_root_logger()


def _setup_root_logger() -> None:
    """Internal: Set up the root 'rag' logger with configured settings."""
    root = logging.getLogger('rag')
    root.setLevel(_config.log_level)
    root.propagate = False
    root.handlers.clear()
    
    if _config.is_production:
        # Production: JSON to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
    else:
        # Development: Human-readable to stderr
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(DevFormatter())
        
        # Also write to file if log_dir is configured
        if _config.log_dir:
            file_path = _config.log_dir / "rag.log"
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


# =============================================================================
# Logger Factory
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a component.
    
    IMPORTANT: Call configure_logging(config) before first use.
    If not configured, will use sensible defaults (dev mode, INFO level, no file).
    """
    # Auto-configure with defaults if not done yet
    if not _config.configured:
        # Fallback: configure with empty config (dev defaults)
        configure_logging({})
    
    # Normalize name under 'rag' namespace
    if not name.startswith('rag.'):
        short_name = name.split('.')[-1] if '.' in name else name
        short_name = short_name.replace('__', '').replace('_', '.')
        name = f'rag.{short_name}'
    
    return logging.getLogger(name)


# =============================================================================
# Convenience Functions (unchanged from V1)
# =============================================================================

def log_stage_start(logger: logging.Logger, stage: str, **kwargs) -> None:
    """Log the start of a pipeline stage."""
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
    """Log the summary at the end of a request."""
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
    """Create FastAPI middleware for automatic request context management."""
    async def middleware(request, call_next):
        req_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())[:8]
        conversation_id = request.headers.get('X-Conversation-ID')
        user_id = request.headers.get('X-User-ID')
        
        set_request_context(req_id=req_id, conversation_id=conversation_id, user_id=user_id)
        
        try:
            response = await call_next(request)
            response.headers['X-Request-ID'] = req_id
            return response
        finally:
            clear_request_context()
    
    return middleware