"""Example: Basic search operations

Demonstrates:
- Loading an index
- Simple search
- Search with filtering
- Explainable search
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cad_vectordb.core.index import IndexManager
from cad_vectordb.core.retrieval import TwoStageRetrieval
from cad_vectordb.core.feature import load_macro_vec


def main():
    print("=" * 60)
    print("Basic Search Example")
    print("=" * 60)
    
    # 1. Load index
    print("\n1. Loading index...")
    manager = IndexManager("./data/indexes")
    manager.load_index("default")
    print(f"   Loaded {len(manager.ids)} vectors")
    
    # 2. Initialize retrieval
    retrieval = TwoStageRetrieval(manager)
    
    # 3. Prepare query
    query_path = "../WHUCAD-main/data/vec/0000/00000000.h5"
    query_vec = load_macro_vec(query_path)
    print(f"\n2. Query: {query_path}")
    print(f"   Sequence length: {len(query_vec)}")
    
    # 4. Simple search
    print("\n3. Simple search (top-10)...")
    results = retrieval.search(
        query_vec,
        query_path,
        k=10,
        stage1_topn=100
    )
    
    print(f"   Found {len(results)} results:")
    for i, result in enumerate(results[:5], 1):
        print(f"   {i}. {result['id']}: score={result['score']:.4f}, "
              f"subset={result['metadata']['subset']}")
    
    # 5. Search with metadata filtering
    print("\n4. Search with metadata filtering (subset=0000)...")
    results_filtered = retrieval.search(
        query_vec,
        query_path,
        k=10,
        filters={"subset": "0000", "min_seq_len": 50}
    )
    
    print(f"   Found {len(results_filtered)} filtered results:")
    for i, result in enumerate(results_filtered[:3], 1):
        print(f"   {i}. {result['id']}: score={result['score']:.4f}, "
              f"seq_len={result['metadata']['seq_len']}")
    
    # 6. Explainable search
    print("\n5. Explainable search...")
    results, explanation = retrieval.search(
        query_vec,
        query_path,
        k=5,
        explainable=True
    )
    
    print(f"   Top match: {explanation['top_match']['id']}")
    print(f"   Final score: {explanation['final_score']:.4f}")
    print(f"   Stage 1 similarity: {explanation['stage1_similarity']:.4f}")
    print(f"   Stage 2 similarity: {explanation['stage2_similarity']:.4f}")
    
    if 'contributions' in explanation:
        print(f"   Stage 1 contribution: {explanation['contributions']['stage1_percentage']:.1f}%")
        print(f"   Stage 2 contribution: {explanation['contributions']['stage2_percentage']:.1f}%")
    
    # 7. Compare fusion methods
    print("\n6. Comparing fusion methods...")
    for method in ["weighted", "rrf", "borda"]:
        results = retrieval.search(
            query_vec,
            query_path,
            k=5,
            fusion_method=method
        )
        top_score = results[0]['score'] if results else 0
        print(f"   {method:12s}: top score = {top_score:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Basic search demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
