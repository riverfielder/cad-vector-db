"""Performance benchmark for search operations

Tests various aspects of search performance:
- Single query latency
- Batch query throughput  
- Different fusion methods
- Index types comparison
- Query complexity impact
"""
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from cad_vectordb.core.index import IndexManager
from cad_vectordb.core.retrieval import TwoStageRetrieval
from cad_vectordb.core.feature import load_macro_vec


class SearchBenchmark:
    """Benchmark search performance"""
    
    def __init__(self, index_manager: IndexManager):
        """Initialize benchmark
        
        Args:
            index_manager: Loaded index manager
        """
        self.index_manager = index_manager
        self.retrieval = TwoStageRetrieval(
            index_manager.index,
            index_manager.ids,
            index_manager.metadata
        )
        self.results = []
    
    def benchmark_single_query(self,
                              query_paths: List[str],
                              k: int = 20,
                              stage1_topn: int = 100,
                              fusion_method: str = "weighted") -> Dict:
        """Benchmark single query latency
        
        Args:
            query_paths: List of query file paths
            k: Number of results to return
            stage1_topn: Stage 1 candidate size
            fusion_method: Fusion method
            
        Returns:
            results: Dict with benchmark results
        """
        latencies = []
        
        print(f"🔍 Benchmarking {len(query_paths)} single queries...")
        
        for query_path in query_paths:
            query_vec = load_macro_vec(query_path)
            
            start = time.time()
            results = self.retrieval.search(
                query_vec,
                query_path,
                k=k,
                stage1_topn=stage1_topn,
                fusion_method=fusion_method
            )
            elapsed = time.time() - start
            
            latencies.append(elapsed)
        
        stats = {
            "test": "single_query",
            "num_queries": len(query_paths),
            "k": k,
            "stage1_topn": stage1_topn,
            "fusion_method": fusion_method,
            "avg_latency": np.mean(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "qps": 1.0 / np.mean(latencies),
        }
        
        self.results.append(stats)
        self._print_stats(stats)
        return stats
    
    def benchmark_batch_query(self,
                            query_paths: List[str],
                            k: int = 20,
                            parallel: bool = False,
                            max_workers: int = 8) -> Dict:
        """Benchmark batch query throughput
        
        Args:
            query_paths: List of query file paths
            k: Number of results to return
            parallel: Use parallel processing
            max_workers: Number of parallel workers
            
        Returns:
            results: Dict with benchmark results
        """
        print(f"🔍 Benchmarking batch query (parallel={parallel})...")
        
        start = time.time()
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for query_path in query_paths:
                    query_vec = load_macro_vec(query_path)
                    future = executor.submit(
                        self.retrieval.search,
                        query_vec, query_path, k=k
                    )
                    futures.append(future)
                
                # Wait for all to complete
                for future in as_completed(futures):
                    _ = future.result()
        else:
            for query_path in query_paths:
                query_vec = load_macro_vec(query_path)
                _ = self.retrieval.search(query_vec, query_path, k=k)
        
        elapsed = time.time() - start
        
        stats = {
            "test": "batch_query",
            "num_queries": len(query_paths),
            "k": k,
            "parallel": parallel,
            "max_workers": max_workers if parallel else 1,
            "total_time": elapsed,
            "avg_time_per_query": elapsed / len(query_paths),
            "qps": len(query_paths) / elapsed,
        }
        
        self.results.append(stats)
        self._print_stats(stats)
        return stats
    
    def benchmark_fusion_methods(self,
                                query_paths: List[str],
                                k: int = 20,
                                stage1_topn: int = 100) -> Dict:
        """Compare different fusion methods
        
        Args:
            query_paths: List of query file paths
            k: Number of results to return
            stage1_topn: Stage 1 candidate size
            
        Returns:
            results: Dict with comparison results
        """
        fusion_methods = ["weighted", "rrf", "borda"]
        results_by_method = {}
        
        print(f"🔍 Benchmarking fusion methods...")
        
        for method in fusion_methods:
            latencies = []
            
            for query_path in query_paths:
                query_vec = load_macro_vec(query_path)
                
                start = time.time()
                _ = self.retrieval.search(
                    query_vec,
                    query_path,
                    k=k,
                    stage1_topn=stage1_topn,
                    fusion_method=method
                )
                elapsed = time.time() - start
                latencies.append(elapsed)
            
            results_by_method[method] = {
                "avg_latency": np.mean(latencies),
                "p95_latency": np.percentile(latencies, 95),
                "qps": 1.0 / np.mean(latencies),
            }
            
            print(f"  {method:12s}: {np.mean(latencies)*1000:.2f}ms avg, "
                  f"{1.0/np.mean(latencies):.1f} QPS")
        
        stats = {
            "test": "fusion_methods",
            "num_queries": len(query_paths),
            "k": k,
            "stage1_topn": stage1_topn,
            "methods": results_by_method,
        }
        
        self.results.append(stats)
        return stats
    
    def benchmark_k_values(self,
                          query_paths: List[str],
                          k_values: List[int] = [5, 10, 20, 50, 100]) -> Dict:
        """Benchmark impact of k (result size)
        
        Args:
            query_paths: List of query file paths
            k_values: List of k values to test
            
        Returns:
            results: Dict with comparison results
        """
        results_by_k = {}
        
        print(f"🔍 Benchmarking different k values...")
        
        for k in k_values:
            latencies = []
            
            for query_path in query_paths:
                query_vec = load_macro_vec(query_path)
                
                start = time.time()
                _ = self.retrieval.search(query_vec, query_path, k=k)
                elapsed = time.time() - start
                latencies.append(elapsed)
            
            results_by_k[str(k)] = {
                "avg_latency": np.mean(latencies),
                "qps": 1.0 / np.mean(latencies),
            }
            
            print(f"  k={k:3d}: {np.mean(latencies)*1000:.2f}ms avg, "
                  f"{1.0/np.mean(latencies):.1f} QPS")
        
        stats = {
            "test": "k_values",
            "num_queries": len(query_paths),
            "k_values": k_values,
            "results": results_by_k,
        }
        
        self.results.append(stats)
        return stats
    
    def benchmark_stage1_topn(self,
                            query_paths: List[str],
                            topn_values: List[int] = [50, 100, 200, 500]) -> Dict:
        """Benchmark impact of stage1_topn
        
        Args:
            query_paths: List of query file paths
            topn_values: List of stage1_topn values to test
            
        Returns:
            results: Dict with comparison results
        """
        results_by_topn = {}
        
        print(f"🔍 Benchmarking different stage1_topn values...")
        
        for topn in topn_values:
            if topn > len(self.index_manager.ids):
                print(f"  Skipping topn={topn} (exceeds index size)")
                continue
            
            latencies = []
            
            for query_path in query_paths:
                query_vec = load_macro_vec(query_path)
                
                start = time.time()
                _ = self.retrieval.search(
                    query_vec,
                    query_path,
                    k=20,
                    stage1_topn=topn
                )
                elapsed = time.time() - start
                latencies.append(elapsed)
            
            results_by_topn[str(topn)] = {
                "avg_latency": np.mean(latencies),
                "qps": 1.0 / np.mean(latencies),
            }
            
            print(f"  topn={topn:4d}: {np.mean(latencies)*1000:.2f}ms avg, "
                  f"{1.0/np.mean(latencies):.1f} QPS")
        
        stats = {
            "test": "stage1_topn",
            "num_queries": len(query_paths),
            "topn_values": topn_values,
            "results": results_by_topn,
        }
        
        self.results.append(stats)
        return stats
    
    def run_full_benchmark(self,
                          query_paths: List[str],
                          output_file: Optional[str] = None) -> Dict:
        """Run complete benchmark suite
        
        Args:
            query_paths: List of query file paths
            output_file: Optional file to save results
            
        Returns:
            all_results: Dict with all benchmark results
        """
        print("=" * 60)
        print("🚀 Running Full Benchmark Suite")
        print("=" * 60)
        print(f"Index: {len(self.index_manager.ids)} vectors")
        print(f"Queries: {len(query_paths)}")
        print()
        
        # 1. Single query latency
        self.benchmark_single_query(query_paths[:50], k=20)
        print()
        
        # 2. Batch query (sequential vs parallel)
        self.benchmark_batch_query(query_paths[:50], k=20, parallel=False)
        self.benchmark_batch_query(query_paths[:50], k=20, parallel=True)
        print()
        
        # 3. Fusion methods
        self.benchmark_fusion_methods(query_paths[:20], k=20)
        print()
        
        # 4. K values
        self.benchmark_k_values(query_paths[:20], k_values=[5, 10, 20, 50, 100])
        print()
        
        # 5. Stage1 topn
        self.benchmark_stage1_topn(query_paths[:20], topn_values=[50, 100, 200])
        print()
        
        summary = {
            "index_stats": self.index_manager.get_stats(),
            "num_test_queries": len(query_paths),
            "benchmark_results": self.results,
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✅ Results saved to {output_file}")
        
        return summary
    
    def _print_stats(self, stats: Dict):
        """Pretty print statistics"""
        if stats["test"] == "single_query":
            print(f"  Avg latency: {stats['avg_latency']*1000:.2f}ms")
            print(f"  P50: {stats['p50_latency']*1000:.2f}ms, "
                  f"P95: {stats['p95_latency']*1000:.2f}ms, "
                  f"P99: {stats['p99_latency']*1000:.2f}ms")
            print(f"  QPS: {stats['qps']:.1f}")
        elif stats["test"] == "batch_query":
            print(f"  Total time: {stats['total_time']:.3f}s")
            print(f"  Avg per query: {stats['avg_time_per_query']*1000:.2f}ms")
            print(f"  QPS: {stats['qps']:.1f}")


def main():
    """Run benchmark from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search performance benchmark")
    parser.add_argument('--index-dir', required=True, help='Index directory')
    parser.add_argument('--index-name', default='default', help='Index name')
    parser.add_argument('--query-dir', required=True, help='Directory with query .h5 files')
    parser.add_argument('--num-queries', type=int, default=100, help='Number of queries to test')
    parser.add_argument('--output', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Load index
    print("Loading index...")
    index_mgr = IndexManager(args.index_dir)
    index_mgr.load_index(args.index_name)
    
    # Collect query files
    query_dir = Path(args.query_dir)
    query_paths = sorted([str(f) for f in query_dir.rglob("*.h5")])[:args.num_queries]
    
    if len(query_paths) == 0:
        print(f"Error: No query files found in {args.query_dir}")
        return
    
    print(f"Found {len(query_paths)} query files\n")
    
    # Run benchmark
    benchmark = SearchBenchmark(index_mgr)
    results = benchmark.run_full_benchmark(query_paths, args.output)
    
    print("\n" + "=" * 60)
    print("✅ Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
