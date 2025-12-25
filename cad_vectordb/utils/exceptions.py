"""
Custom Exceptions for CAD Vector Database

Provides specific exception classes for better error handling
and debugging.
"""
from typing import Optional, Dict, Any


class VectorDBException(Exception):
    """Base exception for Vector Database"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'details': self.details
        }


class IndexNotFoundError(VectorDBException):
    """Index file not found"""
    pass


class IndexNotLoadedError(VectorDBException):
    """Index not loaded in memory"""
    pass


class InvalidIndexTypeError(VectorDBException):
    """Invalid FAISS index type"""
    pass


class VectorNotFoundError(VectorDBException):
    """Vector ID not found in index"""
    pass


class InvalidVectorError(VectorDBException):
    """Vector data is invalid"""
    pass


class DimensionMismatchError(VectorDBException):
    """Vector dimension doesn't match index"""
    pass


class QueryFileNotFoundError(VectorDBException):
    """Query file not found"""
    pass


class InvalidPathError(VectorDBException):
    """Invalid or unsafe file path"""
    pass


class SnapshotError(VectorDBException):
    """Snapshot creation or restoration failed"""
    pass


class CompressionError(VectorDBException):
    """Compression operation failed"""
    pass


class CacheError(VectorDBException):
    """Cache operation failed"""
    pass


class DatabaseError(VectorDBException):
    """Database operation failed"""
    pass


class AuthenticationError(VectorDBException):
    """Authentication failed"""
    pass


class RateLimitExceededError(VectorDBException):
    """Rate limit exceeded"""
    pass


class ValidationError(VectorDBException):
    """Input validation failed"""
    pass


class ConfigurationError(VectorDBException):
    """Configuration error"""
    pass


# Error code mapping
ERROR_CODES = {
    IndexNotFoundError: 'INDEX_NOT_FOUND',
    IndexNotLoadedError: 'INDEX_NOT_LOADED',
    InvalidIndexTypeError: 'INVALID_INDEX_TYPE',
    VectorNotFoundError: 'VECTOR_NOT_FOUND',
    InvalidVectorError: 'INVALID_VECTOR',
    DimensionMismatchError: 'DIMENSION_MISMATCH',
    QueryFileNotFoundError: 'QUERY_FILE_NOT_FOUND',
    InvalidPathError: 'INVALID_PATH',
    SnapshotError: 'SNAPSHOT_ERROR',
    CompressionError: 'COMPRESSION_ERROR',
    CacheError: 'CACHE_ERROR',
    DatabaseError: 'DATABASE_ERROR',
    AuthenticationError: 'AUTHENTICATION_ERROR',
    RateLimitExceededError: 'RATE_LIMIT_EXCEEDED',
    ValidationError: 'VALIDATION_ERROR',
    ConfigurationError: 'CONFIGURATION_ERROR',
}


def get_error_code(exception: Exception) -> str:
    """Get error code for exception"""
    return ERROR_CODES.get(type(exception), 'UNKNOWN_ERROR')


if __name__ == "__main__":
    # Test exceptions
    try:
        raise IndexNotFoundError(
            "Index not found at path",
            details={'path': '/data/index', 'index_name': 'default'}
        )
    except IndexNotFoundError as e:
        print(f"✓ Exception: {e}")
        print(f"✓ Code: {get_error_code(e)}")
        print(f"✓ Dict: {e.to_dict()}")
