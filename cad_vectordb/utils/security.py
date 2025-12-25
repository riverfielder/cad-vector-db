"""
Security and Authentication for CAD Vector Database API

Features:
- API key authentication
- Rate limiting
- Input validation and sanitization
- Path traversal prevention
- CORS configuration
- Request ID tracking
"""
import hashlib
import secrets
import time
from typing import Optional, Dict, List, Callable
from pathlib import Path
from functools import wraps
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status, Header
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, validator, Field
import re


# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class SecurityError(Exception):
    """Base class for security errors"""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed"""
    pass


class RateLimitError(SecurityError):
    """Rate limit exceeded"""
    pass


class ValidationError(SecurityError):
    """Input validation failed"""
    pass


class APIKeyManager:
    """Manage API keys"""
    
    def __init__(self):
        self.keys: Dict[str, Dict] = {}
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from environment or file"""
        import os
        
        # Load from environment variable
        env_key = os.getenv('API_KEY')
        if env_key:
            key_hash = self._hash_key(env_key)
            self.keys[key_hash] = {
                'name': 'default',
                'created_at': datetime.now().isoformat(),
                'enabled': True
            }
    
    def _hash_key(self, key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def generate_key(self, name: str = "default") -> str:
        """Generate a new API key"""
        key = secrets.token_urlsafe(32)
        key_hash = self._hash_key(key)
        
        self.keys[key_hash] = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'enabled': True
        }
        
        return key
    
    def validate_key(self, key: str) -> bool:
        """Validate an API key"""
        if not key:
            return False
        
        key_hash = self._hash_key(key)
        key_info = self.keys.get(key_hash)
        
        return key_info is not None and key_info.get('enabled', False)
    
    def revoke_key(self, key: str):
        """Revoke an API key"""
        key_hash = self._hash_key(key)
        if key_hash in self.keys:
            self.keys[key_hash]['enabled'] = False


# Global key manager
key_manager = APIKeyManager()


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests: int = 100, period: int = 60):
        """
        Initialize rate limiter
        
        Args:
            requests: Maximum requests per period
            period: Time period in seconds
        """
        self.requests = requests
        self.period = period
        self.buckets: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        cutoff = now - self.period
        
        # Clean old requests
        self.buckets[identifier] = [
            req_time for req_time in self.buckets[identifier]
            if req_time > cutoff
        ]
        
        # Check limit
        if len(self.buckets[identifier]) >= self.requests:
            return False
        
        # Add current request
        self.buckets[identifier].append(now)
        return True
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier"""
        now = time.time()
        cutoff = now - self.period
        
        recent_requests = [
            req_time for req_time in self.buckets[identifier]
            if req_time > cutoff
        ]
        
        return max(0, self.requests - len(recent_requests))


# Global rate limiter
rate_limiter = RateLimiter()


class PathValidator:
    """Validate file paths to prevent traversal attacks"""
    
    @staticmethod
    def is_safe_path(base_dir: str, path: str) -> bool:
        """
        Check if path is within base directory
        
        Args:
            base_dir: Base directory
            path: Path to validate
        
        Returns:
            True if path is safe
        """
        try:
            base = Path(base_dir).resolve()
            target = Path(path).resolve()
            
            # Check if target is within base
            return target.is_relative_to(base)
        except Exception:
            return False
    
    @staticmethod
    def sanitize_path(path: str) -> str:
        """
        Sanitize file path
        
        Args:
            path: Path to sanitize
        
        Returns:
            Sanitized path
        """
        # Remove null bytes
        path = path.replace('\0', '')
        
        # Normalize path
        path = Path(path).as_posix()
        
        # Remove leading slashes (prevent absolute paths)
        path = path.lstrip('/')
        
        return path
    
    @staticmethod
    def validate_extension(path: str, allowed_extensions: List[str]) -> bool:
        """
        Validate file extension
        
        Args:
            path: File path
            allowed_extensions: List of allowed extensions (e.g., ['.h5', '.json'])
        
        Returns:
            True if extension is allowed
        """
        ext = Path(path).suffix.lower()
        return ext in allowed_extensions


class InputValidator:
    """Validate and sanitize user inputs"""
    
    @staticmethod
    def validate_id(id_str: str) -> bool:
        """
        Validate ID string format
        
        Args:
            id_str: ID string to validate
        
        Returns:
            True if valid
        """
        # Allow alphanumeric, underscore, hyphen, slash
        pattern = r'^[a-zA-Z0-9_/-]+\.(h5|json)$'
        return bool(re.match(pattern, id_str))
    
    @staticmethod
    def validate_k_value(k: int, max_k: int = 1000) -> bool:
        """
        Validate k parameter for search
        
        Args:
            k: Number of results
            max_k: Maximum allowed k
        
        Returns:
            True if valid
        """
        return 1 <= k <= max_k
    
    @staticmethod
    def validate_text_query(query: str, max_length: int = 500) -> bool:
        """
        Validate text query
        
        Args:
            query: Query text
            max_length: Maximum query length
        
        Returns:
            True if valid
        """
        return 0 < len(query.strip()) <= max_length
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """
        Sanitize text input
        
        Args:
            text: Input text
        
        Returns:
            Sanitized text
        """
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()


# Request validation models with built-in validation
class SafeSearchRequest(BaseModel):
    """Safe search request with validation"""
    query_file_path: str = Field(..., max_length=500)
    k: int = Field(20, ge=1, le=1000)
    stage1_topn: int = Field(100, ge=10, le=10000)
    fusion_method: str = Field("weighted", pattern="^(weighted|rrf|borda)$")
    alpha: float = Field(0.6, ge=0.0, le=1.0)
    beta: float = Field(0.4, ge=0.0, le=1.0)
    
    @validator('query_file_path')
    def validate_path(cls, v):
        """Validate file path"""
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid path: path traversal detected")
        if not v.endswith('.h5'):
            raise ValueError("Invalid file extension: must be .h5")
        return PathValidator.sanitize_path(v)
    
    @validator('alpha', 'beta')
    def validate_weights(cls, v, values):
        """Validate fusion weights"""
        if 'alpha' in values and 'beta' in values:
            if abs(values['alpha'] + v - 1.0) > 0.01:
                raise ValueError("Fusion weights must sum to 1.0")
        return v


class SafeSemanticSearchRequest(BaseModel):
    """Safe semantic search request"""
    query_text: str = Field(..., min_length=1, max_length=500)
    k: int = Field(20, ge=1, le=1000)
    encoder_type: str = Field("sentence-transformer", pattern="^(sentence-transformer|clip|bm25)$")
    
    @validator('query_text')
    def sanitize_query(cls, v):
        """Sanitize query text"""
        return InputValidator.sanitize_text(v)


def require_auth(func: Callable):
    """
    Decorator to require API key authentication
    
    Usage:
        @require_auth
        async def my_endpoint(...):
            pass
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get request from kwargs
        request = kwargs.get('request') or next((arg for arg in args if isinstance(arg, Request)), None)
        
        if not request:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Request object not found"
            )
        
        # Check if auth is enabled
        from .config import get_config
        config = get_config()
        
        if not config.server.enable_auth:
            return await func(*args, **kwargs)
        
        # Get API key from header
        api_key = request.headers.get('x-api-key')
        
        if not api_key or not key_manager.validate_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        return await func(*args, **kwargs)
    
    return wrapper


def rate_limit(func: Callable):
    """
    Decorator to apply rate limiting
    
    Usage:
        @rate_limit
        async def my_endpoint(...):
            pass
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get request from kwargs
        request = kwargs.get('request') or next((arg for arg in args if isinstance(arg, Request)), None)
        
        if not request:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Request object not found"
            )
        
        # Check if rate limiting is enabled
        from .config import get_config
        config = get_config()
        
        if not config.server.rate_limit_enabled:
            return await func(*args, **kwargs)
        
        # Use client IP as identifier
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        if not rate_limiter.is_allowed(client_ip):
            remaining = rate_limiter.get_remaining(client_ip)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again later. Remaining: {remaining}",
                headers={
                    "X-RateLimit-Limit": str(config.server.rate_limit_requests),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(int(time.time() + config.server.rate_limit_period))
                }
            )
        
        return await func(*args, **kwargs)
    
    return wrapper


def generate_request_id() -> str:
    """Generate unique request ID"""
    return secrets.token_hex(16)


if __name__ == "__main__":
    # Test security features
    print("Testing security features...")
    
    # Test API key
    key = key_manager.generate_key("test")
    print(f"\n✓ Generated API key: {key[:16]}...")
    print(f"✓ Valid: {key_manager.validate_key(key)}")
    
    key_manager.revoke_key(key)
    print(f"✓ Revoked: {not key_manager.validate_key(key)}")
    
    # Test rate limiter
    limiter = RateLimiter(requests=5, period=10)
    for i in range(7):
        allowed = limiter.is_allowed("test_client")
        print(f"✓ Request {i+1}: {'allowed' if allowed else 'blocked'}")
    
    # Test path validation
    print(f"\n✓ Safe path: {PathValidator.is_safe_path('/data', '/data/file.h5')}")
    print(f"✓ Unsafe path: {not PathValidator.is_safe_path('/data', '/etc/passwd')}")
    
    # Test input validation
    print(f"\n✓ Valid ID: {InputValidator.validate_id('0001/file.h5')}")
    print(f"✓ Invalid ID: {not InputValidator.validate_id('../../etc/passwd')}")
    
    print("\n✅ Security tests complete")
