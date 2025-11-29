# /src/utils/logger.py
import logging
import os
from pathlib import Path
from datetime import datetime
import glob

def setup_logger(name: str, log_type: str = 'general', log_level=logging.INFO, max_files: int = None):
    """
    Set up a logger with different behaviors based on log_type.
    
    Args:
        name: Logger name (usually __name__)
        log_type: 'ingestion' or 'chat' or 'general'
        log_level: Logging level
        max_files: For ingestion, max number of log files to keep (None = keep all)
    
    Returns:
        configured logger
    """
    # Get log directory
    log_base = os.getenv('RAG_LOG_DIR')
    if log_base is None:
        log_base = Path.home() / 'logs' / 'rag_project'
    else:
        log_base = Path(log_base)
    
    # Create separate subdirectories for different log types
    if log_type == 'ingestion':
        log_dir = log_base / 'ingestion'
    elif log_type == 'chat':
        log_dir = log_base / 'chat'
    else:
        log_dir = log_base / 'general'
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{name}_{timestamp}.log'
    
    # Handle rotation for ingestion logs
    if log_type == 'ingestion' and max_files is not None:
        _cleanup_old_logs(log_dir, name, max_files)
    
    # Set up logger
    logger = logging.getLogger(f"{log_type}.{name}")
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Logs writing to: {log_file}")
    
    return logger


def _cleanup_old_logs(log_dir: Path, logger_name: str, max_files: int):
    """
    Keep only the most recent max_files log files for a given logger.
    """
    # Find all log files for this logger
    pattern = str(log_dir / f'{logger_name}_*.log')
    log_files = sorted(glob.glob(pattern))
    
    # Delete oldest files if we exceed max_files
    if len(log_files) >= max_files:
        files_to_delete = log_files[:-max_files + 1]  # Keep max_files - 1, add current
        for old_file in files_to_delete:
            try:
                os.remove(old_file)
                print(f"Deleted old log file: {old_file}")
            except OSError as e:
                print(f"Error deleting {old_file}: {e}")


# Convenience functions
def get_ingestion_logger(name: str, max_files: int = 5):
    """Get a logger for ingestion pipeline with rotation."""
    return setup_logger(name, log_type='ingestion', max_files=max_files)


def get_chat_logger(name: str):
    """Get a logger for chat pipeline (keeps all logs)."""
    return setup_logger(name, log_type='chat')