"""Example: Batch search operations

Demonstrates:
- Sequential batch search
- Parallel batch search
- Performance comparison
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cad_vectordb.core.index import IndexManager
from cad_vectordb.core.retrieval import TwoStageRetrieval
from cad_vectordb.core.feature import load_macro_vec
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    print("=" * 60)
    print("Batch Search Example")
    print("=" * 60)
    
    # 1. Load index
    print("\n1. Loading index...")
    manager = IndexManager("./data/indexes")
    manager.load_index("default")
    
    retrieval = TwoStageRetrieval(
        manager.index,
        manager.ids,
        manager.metadata
    )
    
    # 2. Prepare queries
    data_dir = Path("../WHUCAD-main/data/vec/0000")
    query_paths = sorted([str(f) for f in data_dir.glob("*.h5")])[:50]
    print(f"\n2. Prepared {len(query_paths)} queries")
    
    # 3. Sequential batch search
    print("\n3. Sequential batch search...")
    start = time.time()
    results_seq = []
    for query_path in query_paths:
        query_vec = load_macro_vec(query_path)
        results = retrieval.search(query_vec, query_path, k=10)
        results_seq.append(results)
    elapsed_seq = time.time() - start
    
    print(f"   Time: {elapsed_seq:.3f}s")
    print(f"   QPS: {len(query_paths)/elapsed_seq:.1f}")
    print(f"   Avg per query: {elapsed_seq/len(query_paths)*1000:.2f}ms")
    
    # 4. Parallel batch search
    print("\n4. Parallel batch search...")
    start = time.time()
    results_par = []
    
    def process_query(query_path):
        query_vec = load_macro_vec(query_path)
        return retrieval.search(query_vec, query_path, k=10)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_query, qp) for qp in query_paths]
        for future in as_completed(futures):
            results_par.append(future.result())
    
    elapsed_par = time.time() - start
    
    print(f"   Time: {elapsed_par:.3f}s")
    print(f"   QPS: {len(query_paths)/elapsed_par:.1f}")
    print(f"   Avg per query: {elapsed_par/len(query_paths)*1000:.2f}ms")
    
    # 5. Compare
    print("\n5. Performance comparison:")
    speedup = elapsed_seq / elapsed_par
    print(f"   Speedup: {speedup:.2f}x")
    if speedup > 1:
        print(f"   ✅ Parallel is {speedup:.1f}x faster")
    else:
        print(f"   ⚠️  Sequential is {1/speedup:.1f}x faster (Python GIL limitation)")
    
    # 6. Verify results consistency
    print("\n6. Verifying results...")
    matches = 0
    for seq, par in zip(results_seq, results_par):
        if len(seq) > 0 and len(par) > 0:
            if seq[0]['id'] == par[0]['id']:
                matches += 1
    
    print(f"   Top-1 matches: {matches}/{len(query_paths)} ({matches/len(query_paths)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ Batch search demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
