"""
Unified Logging System for CAD Vector Database

Features:
- Structured logging with JSON formatter
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Log rotation (daily, size-based)
- Console and file handlers
- Request ID tracking for API calls
- Performance logging
"""
import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import traceback


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        if hasattr(record, 'duration_ms'):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, 'extra_data'):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors"""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    log_dir: Optional[str] = "logs",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup a logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Enable file logging
        log_to_console: Enable console logging
        json_format: Use JSON format for file logs
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler with rotation
    if log_to_file:
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = log_path / f"{name}.log"
        else:
            log_file = f"{name}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        if json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    # Console handler with colors
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredConsoleFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding context information"""
    
    def process(self, msg, kwargs):
        """Add extra context to log records"""
        extra = kwargs.get('extra', {})
        
        # Add context from adapter
        if self.extra:
            extra.update(self.extra)
        
        kwargs['extra'] = extra
        return msg, kwargs


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Get or create a logger
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for setup_logger
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name, **kwargs)
    
    return logger


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with parameters and timing
    
    Usage:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Calling {func_name}", extra={
                'extra_data': {
                    'args': str(args)[:200],  # Truncate long args
                    'kwargs': str(kwargs)[:200]
                }
            })
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds() * 1000
                logger.debug(f"Completed {func_name}", extra={
                    'duration_ms': duration
                })
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(f"Error in {func_name}: {e}", exc_info=True, extra={
                    'duration_ms': duration
                })
                raise
        
        return wrapper
    return decorator


# Create default loggers
api_logger = get_logger('cad_vectordb.api', level='INFO', json_format=True)
index_logger = get_logger('cad_vectordb.index', level='INFO', json_format=True)
retrieval_logger = get_logger('cad_vectordb.retrieval', level='INFO', json_format=True)
db_logger = get_logger('cad_vectordb.database', level='INFO', json_format=True)
security_logger = get_logger('cad_vectordb.security', level='WARNING', json_format=True)


if __name__ == "__main__":
    # Test logging system
    test_logger = setup_logger('test', log_dir='logs', level='DEBUG', json_format=True)
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    # Test with extra context
    test_logger.info("API request received", extra={
        'request_id': '12345',
        'user_id': 'user123',
        'extra_data': {'endpoint': '/search', 'method': 'POST'}
    })
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception:
        test_logger.error("Exception occurred", exc_info=True)
    
    print("\n✓ Logging test complete. Check logs/ directory.")
