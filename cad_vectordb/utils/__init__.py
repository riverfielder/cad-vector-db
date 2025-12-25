"""
Utilities for CAD Vector Database

Modules:
- config: Configuration management with environment variables
- logger: Unified logging system
- security: Authentication, rate limiting, input validation
- exceptions: Custom exception classes
- env: Environment variable loading
"""

from .config import get_config, Config, reload_config
from .logger import (
    get_logger, 
    setup_logger, 
    api_logger, 
    index_logger, 
    retrieval_logger,
    db_logger,
    security_logger
)
from .security import (
    require_auth,
    rate_limit,
    PathValidator,
    InputValidator,
    key_manager,
    rate_limiter
)
from .exceptions import (
    VectorDBException,
    IndexNotFoundError,
    IndexNotLoadedError,
    InvalidIndexTypeError,
    VectorNotFoundError,
    InvalidVectorError,
    DimensionMismatchError,
    QueryFileNotFoundError,
    InvalidPathError,
    SnapshotError,
    CompressionError,
    CacheError,
    DatabaseError,
    AuthenticationError,
    RateLimitExceededError,
    ValidationError,
    ConfigurationError
)
from .env import load_env

__all__ = [
    # Config
    'get_config',
    'Config',
    'reload_config',
    
    # Logger
    'get_logger',
    'setup_logger',
    'api_logger',
    'index_logger',
    'retrieval_logger',
    'db_logger',
    'security_logger',
    
    # Security
    'require_auth',
    'rate_limit',
    'PathValidator',
    'InputValidator',
    'key_manager',
    'rate_limiter',
    
    # Exceptions
    'VectorDBException',
    'IndexNotFoundError',
    'IndexNotLoadedError',
    'InvalidIndexTypeError',
    'VectorNotFoundError',
    'InvalidVectorError',
    'DimensionMismatchError',
    'QueryFileNotFoundError',
    'InvalidPathError',
    'SnapshotError',
    'CompressionError',
    'CacheError',
    'DatabaseError',
    'AuthenticationError',
    'RateLimitExceededError',
    'ValidationError',
    'ConfigurationError',
    
    # Env
    'load_env',
]
