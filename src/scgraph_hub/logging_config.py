"""Centralized logging configuration for Single-Cell Graph Hub."""

import logging
import logging.config
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    json_format: bool = False,
    max_bytes: int = 10_000_000,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Setup comprehensive logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional log file name
        log_dir: Optional log directory (defaults to ~/.scgraph_hub/logs)
        enable_console: Whether to enable console logging
        json_format: Whether to use JSON format for structured logging
        max_bytes: Maximum bytes per log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Determine log directory
    if log_dir is None:
        log_dir = Path.home() / ".scgraph_hub" / "logs"
    else:
        log_dir = Path(log_dir)
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Default log file name
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"scgraph_hub_{timestamp}.log"
    
    log_path = log_dir / log_file
    
    # Create formatters
    if json_format:
        log_format = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}',
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "console": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {},
            "root": {
                "level": level,
                "handlers": []
            }
        }
        
        # File handler with JSON format
        log_format["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_path),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "formatter": "json",
            "level": level
        }
        log_format["root"]["handlers"].append("file")
        
        # Console handler with readable format
        if enable_console:
            log_format["handlers"]["console"] = {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "console",
                "level": level
            }
            log_format["root"]["handlers"].append("console")
            
    else:
        log_format = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "simple": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {},
            "root": {
                "level": level,
                "handlers": []
            }
        }
        
        # File handler with detailed format
        log_format["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_path),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "formatter": "detailed",
            "level": level
        }
        log_format["root"]["handlers"].append("file")
        
        # Console handler with simple format
        if enable_console:
            log_format["handlers"]["console"] = {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "simple",
                "level": level
            }
            log_format["root"]["handlers"].append("console")
    
    # Apply logging configuration
    logging.config.dictConfig(log_format)
    
    # Get logger
    logger = logging.getLogger("scgraph_hub")
    logger.info(f"Logging initialized. Level: {level}, Log file: {log_path}")
    
    return logger


def get_logger(name: str = "scgraph_hub") -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ContextualLogger:
    """Logger wrapper that adds contextual information to log messages."""
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
    
    def _format_message(self, message: str) -> str:
        """Format message with context."""
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        return f"[{context_str}] {message}"
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(self._format_message(message), **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(self._format_message(message), **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(self._format_message(message), **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.logger.error(self._format_message(message), **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self.logger.critical(self._format_message(message), **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with context."""
        self.logger.exception(self._format_message(message), **kwargs)


def get_contextual_logger(context: Dict[str, Any], name: str = "scgraph_hub") -> ContextualLogger:
    """Get a contextual logger with predefined context.
    
    Args:
        context: Context dictionary to include in all log messages
        name: Logger name
        
    Returns:
        Contextual logger instance
    """
    logger = get_logger(name)
    return ContextualLogger(logger, context)


class LoggingMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(f"scgraph_hub.{self.__class__.__name__}")
    
    def get_contextual_logger(self, **context) -> ContextualLogger:
        """Get contextual logger for this class."""
        full_context = {"class": self.__class__.__name__, **context}
        return get_contextual_logger(full_context)


# Configure logging on import if not already configured
if not logging.getLogger("scgraph_hub").handlers:
    # Only set up basic logging if environment variable is set
    if os.getenv("SCGRAPH_AUTO_LOGGING", "false").lower() == "true":
        setup_logging(
            level=os.getenv("SCGRAPH_LOG_LEVEL", "INFO"),
            enable_console=os.getenv("SCGRAPH_LOG_CONSOLE", "true").lower() == "true"
        )