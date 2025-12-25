"""
Tests for Security and Authentication Features

Run with: python -m pytest tests/test_security.py -v
"""
import pytest
import os
from cad_vectordb.utils.security import (
    APIKeyManager, RateLimiter, PathValidator, InputValidator
)


class TestAPIKeyManager:
    """Test API key management"""
    
    def test_generate_key(self):
        """Test key generation"""
        manager = APIKeyManager()
        key = manager.generate_key("test")
        
        assert len(key) > 0
        assert manager.validate_key(key)
    
    def test_validate_invalid_key(self):
        """Test validation of invalid key"""
        manager = APIKeyManager()
        assert not manager.validate_key("invalid-key")
    
    def test_revoke_key(self):
        """Test key revocation"""
        manager = APIKeyManager()
        key = manager.generate_key("test")
        
        assert manager.validate_key(key)
        
        manager.revoke_key(key)
        assert not manager.validate_key(key)


class TestRateLimiter:
    """Test rate limiting"""
    
    def test_rate_limit_allows_within_limit(self):
        """Test that requests within limit are allowed"""
        limiter = RateLimiter(requests=5, period=10)
        
        for i in range(5):
            assert limiter.is_allowed("client1")
    
    def test_rate_limit_blocks_over_limit(self):
        """Test that requests over limit are blocked"""
        limiter = RateLimiter(requests=5, period=10)
        
        # Use up the limit
        for i in range(5):
            limiter.is_allowed("client2")
        
        # This should be blocked
        assert not limiter.is_allowed("client2")
    
    def test_rate_limit_per_client(self):
        """Test that rate limit is per client"""
        limiter = RateLimiter(requests=5, period=10)
        
        # Client 1 uses limit
        for i in range(5):
            limiter.is_allowed("client1")
        
        # Client 2 should still be allowed
        assert limiter.is_allowed("client2")
    
    def test_get_remaining(self):
        """Test getting remaining requests"""
        limiter = RateLimiter(requests=5, period=10)
        
        limiter.is_allowed("client3")
        remaining = limiter.get_remaining("client3")
        
        assert remaining == 4


class TestPathValidator:
    """Test path validation"""
    
    def test_safe_path_within_base(self):
        """Test validation of safe path within base directory"""
        assert PathValidator.is_safe_path("/data", "/data/file.h5")
        assert PathValidator.is_safe_path("/data", "/data/subdir/file.h5")
    
    def test_unsafe_path_outside_base(self):
        """Test detection of path traversal"""
        assert not PathValidator.is_safe_path("/data", "/etc/passwd")
        assert not PathValidator.is_safe_path("/data", "/data/../etc/passwd")
    
    def test_sanitize_path(self):
        """Test path sanitization"""
        assert PathValidator.sanitize_path("path/to/file.h5") == "path/to/file.h5"
        assert PathValidator.sanitize_path("/absolute/path") == "absolute/path"
        assert PathValidator.sanitize_path("path\x00/file") == "path/file"
    
    def test_validate_extension(self):
        """Test file extension validation"""
        allowed = ['.h5', '.json']
        
        assert PathValidator.validate_extension("file.h5", allowed)
        assert PathValidator.validate_extension("file.json", allowed)
        assert not PathValidator.validate_extension("file.txt", allowed)
        assert not PathValidator.validate_extension("file.H5", allowed)  # Case sensitive


class TestInputValidator:
    """Test input validation"""
    
    def test_validate_id(self):
        """Test ID validation"""
        assert InputValidator.validate_id("0001/file.h5")
        assert InputValidator.validate_id("subset_1/data.json")
        assert not InputValidator.validate_id("../../etc/passwd")
        assert not InputValidator.validate_id("file.txt")
    
    def test_validate_k_value(self):
        """Test k parameter validation"""
        assert InputValidator.validate_k_value(1)
        assert InputValidator.validate_k_value(100)
        assert InputValidator.validate_k_value(1000)
        assert not InputValidator.validate_k_value(0)
        assert not InputValidator.validate_k_value(1001)
        assert not InputValidator.validate_k_value(-1)
    
    def test_validate_text_query(self):
        """Test text query validation"""
        assert InputValidator.validate_text_query("valid query")
        assert InputValidator.validate_text_query("a" * 500)
        assert not InputValidator.validate_text_query("")
        assert not InputValidator.validate_text_query("   ")
        assert not InputValidator.validate_text_query("a" * 501)
    
    def test_sanitize_text(self):
        """Test text sanitization"""
        assert InputValidator.sanitize_text("  hello   world  ") == "hello world"
        assert InputValidator.sanitize_text("hello\nworld") == "hello world"
        # Control characters should be removed
        text_with_control = "hello\x00\x01world"
        assert "\x00" not in InputValidator.sanitize_text(text_with_control)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
