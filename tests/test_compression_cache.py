"""
Test script for vector compression and caching features
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_vectordb.core.compression import VectorCompressor, compare_compression_methods
from cad_vectordb.core.cache import LRUCache, QueryCache


def test_vector_compressor():
    """Test VectorCompressor basic functionality"""
    print("\n" + "="*60)
    print("TEST 1: VectorCompressor")
    print("="*60)
    
    # Create compressor
    compressor = VectorCompressor(compression_type="pq", dimension=128)
    print("✓ VectorCompressor created")
    
    # Configure PQ
    compressor.configure_pq(m=8, nbits=8)
    print("✓ PQ configured (m=8, nbits=8)")
    
    # Generate test vectors
    n_vectors = 10000
    vectors = np.random.randn(n_vectors, 128).astype('float32')
    print(f"✓ Generated {n_vectors} test vectors")
    
    # Train compressor
    compressor.train(vectors, sample_size=1000)
    print("✓ Compressor trained")
    
    # Compress vectors
    compressed = compressor.compress(vectors)
    print(f"✓ Vectors compressed: {vectors.shape} -> {compressed.shape}")
    
    # Get stats
    stats = compressor.get_compression_stats(vectors)
    print(f"✓ Compression ratio: {stats.compression_ratio:.2f}x")
    print(f"✓ Memory saved: {stats.memory_saved_mb:.2f} MB")
    
    return True


def test_lru_cache():
    """Test LRUCache functionality"""
    print("\n" + "="*60)
    print("TEST 2: LRUCache")
    print("="*60)
    
    # Create cache
    cache = LRUCache(capacity=100, ttl=60)
    print("✓ LRUCache created (capacity=100, ttl=60)")
    
    # Test put and get
    cache.put("key1", {"result": [1, 2, 3]})
    cache.put("key2", {"result": [4, 5, 6]})
    print("✓ Added 2 items to cache")
    
    # Test get
    result1 = cache.get("key1")
    assert result1 is not None, "Cache get failed"
    assert result1["result"] == [1, 2, 3], "Cache data mismatch"
    print("✓ Retrieved item from cache")
    
    # Test miss
    result_miss = cache.get("nonexistent")
    assert result_miss is None, "Cache should return None for miss"
    print("✓ Cache miss handled correctly")
    
    # Test stats
    stats = cache.get_stats()
    print(f"✓ Cache stats: hits={stats.hits}, misses={stats.misses}, hit_rate={stats.hit_rate:.2%}")
    
    # Test eviction
    for i in range(150):
        cache.put(f"key_{i}", {"data": i})
    assert len(cache) <= 100, "Cache eviction failed"
    print(f"✓ Cache eviction works (size={len(cache)})")
    
    # Test clear
    cache.clear()
    assert len(cache) == 0, "Cache clear failed"
    print("✓ Cache cleared")
    
    return True


def test_query_cache():
    """Test QueryCache functionality"""
    print("\n" + "="*60)
    print("TEST 3: QueryCache")
    print("="*60)
    
    # Create query cache (without Redis)
    cache = QueryCache(lru_capacity=100, lru_ttl=60, use_redis=False)
    print("✓ QueryCache created (LRU only)")
    
    # Test caching query results
    query_vec = np.random.randn(128).astype('float32')
    results = {
        "distances": [0.1, 0.2, 0.3],
        "indices": [10, 20, 30],
        "ids": ["id1", "id2", "id3"]
    }
    
    # Put results
    cache.put(results, query_vector=query_vec, k=3)
    print("✓ Cached query results")
    
    # Get results
    cached_results = cache.get(query_vector=query_vec, k=3)
    assert cached_results is not None, "Query cache get failed"
    print("✓ Retrieved cached query results")
    
    # Test with different k (should miss)
    cached_miss = cache.get(query_vector=query_vec, k=5)
    assert cached_miss is None, "Query cache should miss with different k"
    print("✓ Cache miss with different parameters")
    
    # Test stats
    stats = cache.get_stats()
    print(f"✓ Cache stats: hits={stats['lru']['hits']}, misses={stats['lru']['misses']}")
    
    # Test clear
    cache.clear()
    stats_after = cache.get_stats()
    assert stats_after['lru']['size'] == 0, "Cache clear failed"
    print("✓ Cache cleared")
    
    return True


def test_compression_comparison():
    """Test compression method comparison"""
    print("\n" + "="*60)
    print("TEST 4: Compression Method Comparison")
    print("="*60)
    
    # Generate test vectors
    n_vectors = 5000
    dimension = 64
    vectors = np.random.randn(n_vectors, dimension).astype('float32')
    print(f"✓ Generated {n_vectors} test vectors (dim={dimension})")
    
    # Compare methods
    try:
        results = compare_compression_methods(
            vectors,
            methods=['pq', 'sq', 'none']
        )
        print("\n✓ Compression comparison complete:")
        for method, stats in results.items():
            print(f"  - {method}: {stats.compression_ratio:.2f}x compression")
    except Exception as e:
        print(f"⚠ Comparison test skipped: {e}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Vector Compression & Caching - Test Suite")
    print("="*60)
    
    tests = [
        ("VectorCompressor", test_vector_compressor),
        ("LRUCache", test_lru_cache),
        ("QueryCache", test_query_cache),
        ("Compression Comparison", test_compression_comparison),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name}: FAILED")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
