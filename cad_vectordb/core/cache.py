"""
Caching System for Vector Database

Implements multi-level caching to accelerate queries:
- Level 1: In-memory LRU cache (query results)
- Level 2: Redis distributed cache (optional)
- Cache warming and invalidation strategies

Author: riverfielder
Date: 2025-01-25
"""

import hashlib
import json
import time
from typing import Any, Optional, Dict, List, Tuple
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    total_queries: int = 0
    hit_rate: float = 0.0
    avg_hit_time_ms: float = 0.0
    avg_miss_time_ms: float = 0.0
    memory_usage_mb: float = 0.0


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation
    
    Thread-safe in-memory cache with automatic eviction of least
    recently used items when capacity is reached.
    """
    
    def __init__(self, capacity: int = 1000, ttl: int = 3600):
        """
        Initialize LRU cache
        
        Args:
            capacity: Maximum number of items to cache
            ttl: Time-to-live in seconds (0 = no expiration)
        """
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.hit_times = []
        self.miss_times = []
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry has expired"""
        if self.ttl == 0:
            return False
        
        if key not in self.timestamps:
            return True
        
        age = time.time() - self.timestamps[key]
        return age > self.ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        start_time = time.time()
        
        if key not in self.cache or self._is_expired(key):
            self.misses += 1
            elapsed = (time.time() - start_time) * 1000
            self.miss_times.append(elapsed)
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        value = self.cache[key]
        
        self.hits += 1
        elapsed = (time.time() - start_time) * 1000
        self.hit_times.append(elapsed)
        
        return value
    
    def put(self, key: str, value: Any):
        """
        Put value into cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Update existing key
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            self.timestamps[key] = time.time()
            return
        
        # Add new key
        self.cache[key] = value
        self.timestamps[key] = time.time()
        
        # Evict least recently used if over capacity
        if len(self.cache) > self.capacity:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
    
    def delete(self, key: str):
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.timestamps.clear()
        self.hits = 0
        self.misses = 0
        self.hit_times.clear()
        self.miss_times.clear()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        avg_hit_time = np.mean(self.hit_times) if self.hit_times else 0.0
        avg_miss_time = np.mean(self.miss_times) if self.miss_times else 0.0
        
        # Estimate memory usage (rough approximation)
        memory_mb = len(self.cache) * 0.01  # ~10KB per entry estimate
        
        return CacheStats(
            hits=self.hits,
            misses=self.misses,
            total_queries=total,
            hit_rate=hit_rate,
            avg_hit_time_ms=avg_hit_time,
            avg_miss_time_ms=avg_miss_time,
            memory_usage_mb=memory_mb
        )
    
    def __len__(self) -> int:
        return len(self.cache)


class RedisCache:
    """
    Redis-based distributed cache
    
    Optional Redis backend for multi-instance deployments.
    Falls back gracefully if Redis is not available.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl: int = 3600,
        enabled: bool = True
    ):
        """
        Initialize Redis cache
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl: Time-to-live in seconds
            enabled: Enable Redis caching
        """
        self.ttl = ttl
        self.enabled = enabled
        self.redis_client = None
        
        if not enabled:
            return
        
        try:
            import redis
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False
            )
            # Test connection
            self.redis_client.ping()
            print(f"✓ Connected to Redis at {host}:{port}")
        except ImportError:
            print("⚠ Redis not installed. Install with: pip install redis")
            self.enabled = False
        except Exception as e:
            print(f"⚠ Could not connect to Redis: {e}")
            self.enabled = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        if not self.enabled or self.redis_client is None:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value is not None:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Redis get error: {e}")
            return None
    
    def put(self, key: str, value: Any):
        """Put value into Redis"""
        if not self.enabled or self.redis_client is None:
            return
        
        try:
            serialized = json.dumps(value, default=str)
            self.redis_client.setex(key, self.ttl, serialized)
        except Exception as e:
            print(f"Redis put error: {e}")
    
    def delete(self, key: str):
        """Delete key from Redis"""
        if not self.enabled or self.redis_client is None:
            return
        
        try:
            self.redis_client.delete(key)
        except Exception as e:
            print(f"Redis delete error: {e}")
    
    def clear(self):
        """Clear all keys in current database"""
        if not self.enabled or self.redis_client is None:
            return
        
        try:
            self.redis_client.flushdb()
        except Exception as e:
            print(f"Redis clear error: {e}")


class QueryCache:
    """
    Multi-level cache for query results
    
    Combines in-memory LRU cache with optional Redis backend
    for scalable caching across multiple instances.
    """
    
    def __init__(
        self,
        lru_capacity: int = 1000,
        lru_ttl: int = 3600,
        use_redis: bool = False,
        redis_config: Optional[Dict] = None,
        verbose: bool = True
    ):
        """
        Initialize query cache
        
        Args:
            lru_capacity: LRU cache capacity
            lru_ttl: LRU cache TTL in seconds
            use_redis: Enable Redis backend
            redis_config: Redis configuration dict
            verbose: Print debug messages
        """
        self.verbose = verbose
        
        # Level 1: In-memory LRU cache
        self.lru = LRUCache(capacity=lru_capacity, ttl=lru_ttl)
        
        # Level 2: Redis cache (optional)
        self.redis = None
        if use_redis:
            redis_config = redis_config or {}
            self.redis = RedisCache(**redis_config)
        
        if self.verbose:
            print(f"✓ QueryCache initialized")
            print(f"  - LRU: capacity={lru_capacity}, ttl={lru_ttl}s")
            print(f"  - Redis: {'enabled' if self.redis and self.redis.enabled else 'disabled'}")
    
    def _make_key(
        self,
        query_vec: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        Generate cache key from query parameters
        
        Args:
            query_vec: Query vector
            query_text: Query text
            k: Number of results
            filters: Filter parameters
            **kwargs: Additional parameters
        
        Returns:
            Cache key string
        """
        key_parts = []
        
        # Hash query vector
        if query_vec is not None:
            vec_hash = hashlib.md5(query_vec.tobytes()).hexdigest()
            key_parts.append(f"vec:{vec_hash}")
        
        # Hash query text
        if query_text is not None:
            text_hash = hashlib.md5(query_text.encode()).hexdigest()
            key_parts.append(f"text:{text_hash}")
        
        # Add k
        key_parts.append(f"k:{k}")
        
        # Hash filters
        if filters:
            filter_str = json.dumps(filters, sort_keys=True)
            filter_hash = hashlib.md5(filter_str.encode()).hexdigest()
            key_parts.append(f"filters:{filter_hash}")
        
        # Hash other kwargs
        if kwargs:
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
            kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()
            key_parts.append(f"kwargs:{kwargs_hash}")
        
        return "|".join(key_parts)
    
    def get(
        self,
        query_vec: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> Optional[List[Dict]]:
        """
        Get cached query results
        
        Returns:
            Cached results or None if not found
        """
        key = self._make_key(query_vec, query_text, k, filters, **kwargs)
        
        # Try L1 cache (LRU)
        result = self.lru.get(key)
        if result is not None:
            if self.verbose:
                print(f"  ✓ L1 cache hit")
            return result
        
        # Try L2 cache (Redis)
        if self.redis and self.redis.enabled:
            result = self.redis.get(key)
            if result is not None:
                # Populate L1 cache
                self.lru.put(key, result)
                if self.verbose:
                    print(f"  ✓ L2 cache hit (Redis)")
                return result
        
        if self.verbose:
            print(f"  ✗ Cache miss")
        
        return None
    
    def put(
        self,
        results: List[Dict],
        query_vec: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ):
        """
        Cache query results
        
        Args:
            results: Query results to cache
            query_vec: Query vector
            query_text: Query text
            k: Number of results
            filters: Filter parameters
            **kwargs: Additional parameters
        """
        key = self._make_key(query_vec, query_text, k, filters, **kwargs)
        
        # Put in L1 cache
        self.lru.put(key, results)
        
        # Put in L2 cache
        if self.redis and self.redis.enabled:
            self.redis.put(key, results)
        
        if self.verbose:
            print(f"  ✓ Cached results")
    
    def invalidate(
        self,
        query_vec: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ):
        """Invalidate specific cache entry"""
        key = self._make_key(query_vec, query_text, k, filters, **kwargs)
        
        self.lru.delete(key)
        
        if self.redis and self.redis.enabled:
            self.redis.delete(key)
    
    def clear(self):
        """Clear all caches"""
        self.lru.clear()
        
        if self.redis and self.redis.enabled:
            self.redis.clear()
        
        if self.verbose:
            print("✓ All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        lru_stats = self.lru.get_stats()
        
        stats = {
            "lru": {
                "hits": lru_stats.hits,
                "misses": lru_stats.misses,
                "total_queries": lru_stats.total_queries,
                "hit_rate": lru_stats.hit_rate,
                "avg_hit_time_ms": lru_stats.avg_hit_time_ms,
                "avg_miss_time_ms": lru_stats.avg_miss_time_ms,
                "memory_usage_mb": lru_stats.memory_usage_mb,
                "size": len(self.lru)
            },
            "redis": {
                "enabled": self.redis is not None and self.redis.enabled
            }
        }
        
        return stats
    
    def warm_cache(
        self,
        query_samples: List[Tuple[np.ndarray, int]],
        search_fn: callable
    ):
        """
        Warm up cache with common queries
        
        Args:
            query_samples: List of (query_vector, k) tuples
            search_fn: Function to execute search: fn(query_vec, k) -> results
        """
        if self.verbose:
            print(f"Warming cache with {len(query_samples)} queries...")
        
        for i, (query_vec, k) in enumerate(query_samples):
            results = search_fn(query_vec, k)
            self.put(results, query_vec=query_vec, k=k)
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(query_samples)}")
        
        if self.verbose:
            print(f"✓ Cache warmed with {len(query_samples)} entries")
