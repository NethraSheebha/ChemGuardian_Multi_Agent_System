"""Structured logging configuration with JSON support"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields from record
        if hasattr(record, "agent_name"):
            log_data["agent_name"] = record.agent_name
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "data_id"):
            log_data["data_id"] = record.data_id
            
        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Custom text formatter for human-readable logging"""
    
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_dir: str = "logs"
) -> None:
    """
    Configure logging for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('json' or 'text')
        log_dir: Directory for log files
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    
    # File handler
    log_file = log_path / f"chemical_monitoring_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level))
    
    # Set formatter
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
        
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log startup message
    root_logger.info(
        f"Logging configured: level={level}, format={format_type}, log_dir={log_dir}"
    )


class LogContext:
    """Context manager for enriched logging"""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize log context
        
        Args:
            logger: Logger instance
            **kwargs: Additional fields to add to log records
        """
        self.logger = logger
        self.extra = kwargs
        self.old_factory = None
        
    def __enter__(self):
        """Enter context and add extra fields to log records"""
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.extra.items():
                setattr(record, key, value)
            return record
            
        logging.setLogRecordFactory(record_factory)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original log record factory"""
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)
