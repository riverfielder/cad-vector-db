"""
Example: Vector Compression and Caching

Demonstrates the use of vector compression and caching features
to optimize memory usage and query performance.

Author: riverfielder
Date: 2025-01-25
"""

import os
import sys
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_vectordb.core.index import IndexManager
from cad_vectordb.core.compression import VectorCompressor, compare_compression_methods
from cad_vectordb.core.cache import QueryCache


def demo_compression():
    """Demonstrate vector compression"""
    print("=" * 70)
    print("DEMO 1: Vector Compression")
    print("=" * 70)
    
    # Initialize index manager with compression
    index_manager = IndexManager(
        index_dir="./data/index_compressed",
        enable_compression=True,
        compression_type="pq",
        verbose=True
    )
    
    # Load existing index (or build one first)
    try:
        index_manager.load_index("default")
        print(f"\n✓ Loaded index with {len(index_manager.ids)} vectors")
    except Exception as e:
        print(f"⚠ Could not load index: {e}")
        print("  Please build an index first using scripts/build_index.py")
        return
    
    # Get baseline stats
    print("\n" + "-" * 70)
    print("Baseline (No Compression)")
    print("-" * 70)
    n_vectors = index_manager.index.ntotal
    dimension = index_manager.index.d
    baseline_size_mb = n_vectors * dimension * 4 / 1024 / 1024
    print(f"  Vectors: {n_vectors}")
    print(f"  Dimension: {dimension}")
    print(f"  Memory usage: {baseline_size_mb:.2f} MB")
    
    # Enable compression
    print("\n" + "-" * 70)
    print("Enabling Product Quantization (PQ)")
    print("-" * 70)
    stats = index_manager.enable_vector_compression(
        compression_type="pq",
        train_samples=min(100000, n_vectors)
    )
    
    # Compare compression methods
    print("\n" + "-" * 70)
    print("Comparing Compression Methods")
    print("-" * 70)
    
    # Get sample vectors
    sample_size = min(1000, n_vectors)
    vectors = np.zeros((sample_size, dimension), dtype=np.float32)
    for i in range(sample_size):
        vectors[i] = index_manager.index.reconstruct(i)
    
    comparison = compare_compression_methods(
        vectors,
        methods=["none", "sq", "pq"]
    )
    
    # Rebuild with compression
    print("\n" + "-" * 70)
    print("Rebuilding Index with Compression")
    print("-" * 70)
    
    rebuild_stats = index_manager.rebuild_with_compression(
        compression_type="pq",
        index_type="IVF",
        nlist=100
    )
    
    print(f"\n✓ Compression complete!")
    print(f"  Original: {baseline_size_mb:.2f} MB")
    print(f"  Compressed: {baseline_size_mb / rebuild_stats.compression_ratio:.2f} MB")
    print(f"  Saved: {rebuild_stats.memory_saved_mb:.2f} MB ({rebuild_stats.compression_ratio:.1f}x)")


def demo_caching():
    """Demonstrate query caching"""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Query Caching")
    print("=" * 70)
    
    # Initialize index manager with caching
    index_manager = IndexManager(
        index_dir="./data/index",
        enable_cache=True,
        cache_capacity=1000,
        verbose=True
    )
    
    # Load index
    try:
        index_manager.load_index("default")
        print(f"\n✓ Loaded index with {len(index_manager.ids)} vectors")
    except Exception as e:
        print(f"⚠ Could not load index: {e}")
        return
    
    dimension = index_manager.index.d
    
    # Generate test queries
    n_queries = 100
    query_vectors = np.random.randn(n_queries, dimension).astype(np.float32)
    for i in range(n_queries):
        query_vectors[i] = query_vectors[i] / (np.linalg.norm(query_vectors[i]) + 1e-8)
    
    # Test 1: Cold cache (no caching)
    print("\n" + "-" * 70)
    print("Test 1: Without Cache (Cold)")
    print("-" * 70)
    
    start_time = time.time()
    for query_vec in query_vectors:
        D, I = index_manager.index.search(query_vec.reshape(1, -1), 10)
    cold_time = time.time() - start_time
    
    print(f"  Queries: {n_queries}")
    print(f"  Total time: {cold_time:.3f}s")
    print(f"  Avg time: {cold_time / n_queries * 1000:.2f}ms")
    print(f"  QPS: {n_queries / cold_time:.1f}")
    
    # Enable cache
    index_manager.enable_query_cache(capacity=1000, ttl=3600)
    
    # Test 2: Warm cache
    print("\n" + "-" * 70)
    print("Test 2: Warming Cache")
    print("-" * 70)
    
    index_manager.warm_cache(n_samples=50)
    
    # Test 3: With cache
    print("\n" + "-" * 70)
    print("Test 3: With Cache (Warm)")
    print("-" * 70)
    
    # Simulate repeated queries (cache hits)
    repeated_queries = np.random.choice(n_queries, size=n_queries, replace=True)
    
    start_time = time.time()
    cache_hits = 0
    
    for idx in repeated_queries:
        query_vec = query_vectors[idx]
        
        # Try to get from cache
        cached_result = index_manager.cache.get(query_vec=query_vec, k=10)
        
        if cached_result is not None:
            cache_hits += 1
        else:
            # Cache miss - perform search and cache result
            D, I = index_manager.index.search(query_vec.reshape(1, -1), 10)
            results = []
            for i, d in zip(I[0], D[0]):
                if i < len(index_manager.ids):
                    results.append({
                        "id": index_manager.ids[i],
                        "distance": float(d)
                    })
            index_manager.cache.put(results, query_vec=query_vec, k=10)
    
    warm_time = time.time() - start_time
    
    print(f"  Queries: {n_queries}")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Hit rate: {cache_hits / n_queries * 100:.1f}%")
    print(f"  Total time: {warm_time:.3f}s")
    print(f"  Avg time: {warm_time / n_queries * 1000:.2f}ms")
    print(f"  QPS: {n_queries / warm_time:.1f}")
    
    # Performance improvement
    speedup = cold_time / warm_time
    print(f"\n  ⚡ Speedup: {speedup:.1f}x faster with cache!")
    
    # Cache statistics
    print("\n" + "-" * 70)
    print("Cache Statistics")
    print("-" * 70)
    
    stats = index_manager.get_cache_stats()
    lru_stats = stats['lru']
    
    print(f"  Total queries: {lru_stats['total_queries']}")
    print(f"  Hits: {lru_stats['hits']}")
    print(f"  Misses: {lru_stats['misses']}")
    print(f"  Hit rate: {lru_stats['hit_rate'] * 100:.1f}%")
    print(f"  Avg hit time: {lru_stats['avg_hit_time_ms']:.2f}ms")
    print(f"  Avg miss time: {lru_stats['avg_miss_time_ms']:.2f}ms")
    print(f"  Cache size: {lru_stats['size']} entries")
    print(f"  Memory usage: {lru_stats['memory_usage_mb']:.2f} MB")


def demo_combined():
    """Demonstrate combined compression + caching"""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Compression + Caching (Combined)")
    print("=" * 70)
    
    # Initialize with both features
    index_manager = IndexManager(
        index_dir="./data/index_optimized",
        enable_compression=True,
        compression_type="pq",
        enable_cache=True,
        cache_capacity=1000,
        verbose=True
    )
    
    print("\n✓ Initialized with:")
    print("  - Product Quantization (PQ) compression")
    print("  - LRU query cache (capacity=1000)")
    
    try:
        index_manager.load_index("default")
        print(f"\n✓ Loaded index with {len(index_manager.ids)} vectors")
    except Exception as e:
        print(f"⚠ Could not load index: {e}")
        return
    
    # Enable compression
    print("\n" + "-" * 70)
    print("Applying Compression")
    print("-" * 70)
    
    comp_stats = index_manager.enable_vector_compression(compression_type="pq")
    
    # Warm cache
    print("\n" + "-" * 70)
    print("Warming Cache")
    print("-" * 70)
    
    index_manager.warm_cache(n_samples=100)
    
    # Summary
    print("\n" + "-" * 70)
    print("Optimization Summary")
    print("-" * 70)
    
    comp_info = index_manager.get_compression_stats()
    cache_info = index_manager.get_cache_stats()
    
    print(f"\n📊 Compression:")
    if comp_info:
        print(f"  Type: {comp_info['compression_type'].upper()}")
        print(f"  Ratio: {comp_info['compression_ratio']:.2f}x")
        print(f"  Saved: {comp_info['memory_saved_mb']:.1f} MB")
    
    print(f"\n🚀 Cache:")
    if cache_info.get('enabled'):
        print(f"  Size: {cache_info['lru']['size']} entries")
        print(f"  Hit rate: {cache_info['lru']['hit_rate'] * 100:.1f}%")
        print(f"  Memory: {cache_info['lru']['memory_usage_mb']:.2f} MB")
    
    print(f"\n✅ System optimized for:")
    print(f"  - Lower memory footprint ({comp_info['compression_ratio']:.1f}x smaller)")
    print(f"  - Faster query response (cache-accelerated)")
    print(f"  - Scalable to larger datasets")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 70)
    print("Vector Compression & Caching Examples")
    print("=" * 70)
    print("\nThis demo showcases:")
    print("  1. Vector compression (PQ/SQ) for memory optimization")
    print("  2. Query caching (LRU) for performance optimization")
    print("  3. Combined optimization strategies")
    print("\n" + "=" * 70)
    
    # Run demos
    demo_compression()
    demo_caching()
    demo_combined()
    
    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
