"""Example: Semantic Search for CAD Vector Database

This example demonstrates how to use the semantic search feature to find
CAD models using natural language queries.

Features demonstrated:
1. Basic semantic search with text queries
2. Multilingual queries (Chinese and English)
3. Different embedding models (Sentence-BERT, CLIP, BM25)
4. Hybrid search combining text and vector queries
5. Batch semantic search
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_vectordb.core.index import IndexManager
from cad_vectordb.core.retrieval import TwoStageRetrieval
from cad_vectordb.core.text_encoder import create_text_encoder
from config import INDEX_DIR


def example_1_basic_semantic_search():
    """Example 1: Basic semantic search"""
    print("=" * 80)
    print("Example 1: Basic Semantic Search")
    print("=" * 80)
    
    # Load index
    print("\n1. Loading index...")
    index_manager = IndexManager(INDEX_DIR)
    index_manager.load_index()
    print(f"   Loaded {len(index_manager.ids)} vectors")
    
    # Initialize retrieval system
    retrieval = TwoStageRetrieval(index_manager)
    
    # Create text encoder (multilingual Sentence-BERT)
    print("\n2. Creating text encoder...")
    encoder = create_text_encoder(
        encoder_type='sentence-transformer',
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        device='cpu',
        use_cache=True
    )
    print(f"   Encoder dimension: {encoder.dimension}")
    
    # Semantic search with English query
    print("\n3. Searching with English query...")
    query_text = "cylindrical mechanical part"
    results = retrieval.semantic_search(
        query_text=query_text,
        text_encoder=encoder,
        k=5
    )
    
    print(f"\n   Query: '{query_text}'")
    print(f"   Found {len(results)} results:\n")
    for i, res in enumerate(results, 1):
        print(f"   {i}. ID: {res['id']}")
        print(f"      Score: {res['score']:.4f}")
        print(f"      Subset: {res['metadata'].get('subset', 'unknown')}")
        print(f"      Seq Length: {res['metadata'].get('seq_len', 0)}")
        print()


def example_2_multilingual_search():
    """Example 2: Multilingual semantic search"""
    print("=" * 80)
    print("Example 2: Multilingual Semantic Search (Chinese + English)")
    print("=" * 80)
    
    # Load index and retrieval system
    index_manager = IndexManager(INDEX_DIR)
    index_manager.load_index()
    retrieval = TwoStageRetrieval(index_manager)
    
    # Create multilingual encoder
    encoder = create_text_encoder('sentence-transformer', device='cpu')
    
    # Test queries in different languages
    queries = [
        "圆柱形零件",                           # Chinese: cylindrical part
        "cylindrical part",                    # English
        "找一个有螺纹的轴",                    # Chinese: find a threaded shaft
        "mechanical component with holes",     # English
    ]
    
    print("\nSearching with multilingual queries...\n")
    for query in queries:
        print(f"Query: '{query}'")
        results = retrieval.semantic_search(
            query_text=query,
            text_encoder=encoder,
            k=3
        )
        
        if results:
            top = results[0]
            print(f"  → Top match: {top['id']} (score: {top['score']:.4f})")
        else:
            print(f"  → No results found")
        print()


def example_3_explainable_semantic_search():
    """Example 3: Semantic search with explanations"""
    print("=" * 80)
    print("Example 3: Explainable Semantic Search")
    print("=" * 80)
    
    index_manager = IndexManager(INDEX_DIR)
    index_manager.load_index()
    retrieval = TwoStageRetrieval(index_manager)
    encoder = create_text_encoder('sentence-transformer', device='cpu')
    
    query_text = "mechanical gear component"
    print(f"\nQuery: '{query_text}'")
    print("\nSearching with explanations...\n")
    
    results, explanation = retrieval.semantic_search(
        query_text=query_text,
        text_encoder=encoder,
        k=5,
        explainable=True
    )
    
    # Display explanation
    print("Explanation:")
    print(f"  Query Text: {explanation['query_text']}")
    print(f"  Search Type: {explanation['search_type']}")
    print(f"  Encoder: {explanation['encoder_type']}")
    print(f"\n  Top Match:")
    print(f"    ID: {explanation['top_match']['id']}")
    print(f"    Score: {explanation['top_match']['score']:.4f}")
    print(f"    Subset: {explanation['top_match']['subset']}")
    print(f"\n  Interpretation: {explanation['interpretation']}")
    
    if 'recommendation' in explanation:
        print(f"  Recommendation: {explanation['recommendation']}")
    
    print(f"\n  All results:")
    for i, res in enumerate(results, 1):
        print(f"    {i}. {res['id']} - {res['score']:.4f}")


def example_4_hybrid_search():
    """Example 4: Hybrid search (text + CAD vector)"""
    print("=" * 80)
    print("Example 4: Hybrid Search (Semantic + Vector)")
    print("=" * 80)
    
    index_manager = IndexManager(INDEX_DIR)
    index_manager.load_index()
    retrieval = TwoStageRetrieval(index_manager)
    encoder = create_text_encoder('sentence-transformer', device='cpu')
    
    # For demonstration, we'll use a query from the dataset
    from cad_vectordb.core.feature import load_macro_vec
    query_file = Path("data/vec/0000/00000000.h5")
    
    if not query_file.exists():
        print(f"\nSkipping hybrid search demo: query file not found")
        print(f"Please provide a valid query file path")
        return
    
    query_vec = load_macro_vec(str(query_file))
    query_text = "cylindrical part with features"
    
    print(f"\nQuery Text: '{query_text}'")
    print(f"Query Vector: {query_file}")
    print("\nPerforming hybrid search...\n")
    
    results = retrieval.hybrid_search(
        query_text=query_text,
        text_encoder=encoder,
        query_vec=query_vec,
        query_file_path=str(query_file),
        k=5,
        semantic_weight=0.5,
        vector_weight=0.5
    )
    
    print("Results (combining semantic + vector similarity):\n")
    for i, res in enumerate(results, 1):
        print(f"{i}. ID: {res['id']}")
        print(f"   Fused Score: {res['score']:.4f}")
        print(f"   Semantic Score: {res['semantic_score']:.4f}")
        print(f"   Vector Score: {res['vector_score']:.4f}")
        print()


def example_5_different_encoders():
    """Example 5: Comparing different text encoders"""
    print("=" * 80)
    print("Example 5: Comparing Different Text Encoders")
    print("=" * 80)
    
    index_manager = IndexManager(INDEX_DIR)
    index_manager.load_index()
    retrieval = TwoStageRetrieval(index_manager)
    
    query_text = "round mechanical part"
    
    # Test different encoders
    encoder_configs = [
        ('sentence-transformer', 'paraphrase-multilingual-MiniLM-L12-v2', 'Multilingual S-BERT'),
        ('bm25', None, 'BM25 (Sparse)'),
    ]
    
    print(f"\nQuery: '{query_text}'\n")
    
    for encoder_type, model_name, description in encoder_configs:
        print(f"Testing {description}...")
        try:
            encoder = create_text_encoder(
                encoder_type=encoder_type,
                model_name=model_name,
                device='cpu',
                use_cache=False
            )
            
            results = retrieval.semantic_search(
                query_text=query_text,
                text_encoder=encoder,
                k=3
            )
            
            if results:
                print(f"  Top result: {results[0]['id']} (score: {results[0]['score']:.4f})")
            else:
                print(f"  No results found")
                
        except Exception as e:
            print(f"  Error: {e}")
        print()


def example_6_batch_semantic_search():
    """Example 6: Batch semantic search"""
    print("=" * 80)
    print("Example 6: Batch Semantic Search")
    print("=" * 80)
    
    index_manager = IndexManager(INDEX_DIR)
    index_manager.load_index()
    retrieval = TwoStageRetrieval(index_manager)
    encoder = create_text_encoder('sentence-transformer', device='cpu')
    
    # Multiple queries
    queries = [
        "cylindrical part",
        "圆形零件",
        "component with holes",
        "mechanical gear",
        "threaded shaft"
    ]
    
    print(f"\nProcessing {len(queries)} queries...\n")
    
    for query in queries:
        results = retrieval.semantic_search(
            query_text=query,
            text_encoder=encoder,
            k=1  # Get only top result
        )
        
        if results:
            print(f"'{query}' → {results[0]['id']} ({results[0]['score']:.4f})")
        else:
            print(f"'{query}' → No results")


def example_7_metadata_filtering():
    """Example 7: Semantic search with metadata filtering"""
    print("=" * 80)
    print("Example 7: Semantic Search with Metadata Filtering")
    print("=" * 80)
    
    index_manager = IndexManager(INDEX_DIR)
    index_manager.load_index()
    retrieval = TwoStageRetrieval(index_manager)
    encoder = create_text_encoder('sentence-transformer', device='cpu')
    
    query_text = "cylindrical part"
    
    # Search without filters
    print(f"\nQuery: '{query_text}'")
    print("\n1. Without filters:")
    results = retrieval.semantic_search(
        query_text=query_text,
        text_encoder=encoder,
        k=5
    )
    print(f"   Found {len(results)} results")
    for res in results[:3]:
        print(f"   - {res['id']} (subset: {res['metadata'].get('subset', 'unknown')})")
    
    # Search with subset filter
    print("\n2. With subset filter (subset='0000'):")
    results_filtered = retrieval.semantic_search(
        query_text=query_text,
        text_encoder=encoder,
        k=5,
        filters={'subset': '0000'}
    )
    print(f"   Found {len(results_filtered)} results")
    for res in results_filtered[:3]:
        print(f"   - {res['id']} (subset: {res['metadata'].get('subset', 'unknown')})")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CAD Semantic Search Examples" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Run examples
    try:
        example_1_basic_semantic_search()
        input("\nPress Enter to continue to Example 2...\n")
        
        example_2_multilingual_search()
        input("\nPress Enter to continue to Example 3...\n")
        
        example_3_explainable_semantic_search()
        input("\nPress Enter to continue to Example 4...\n")
        
        example_4_hybrid_search()
        input("\nPress Enter to continue to Example 5...\n")
        
        example_5_different_encoders()
        input("\nPress Enter to continue to Example 6...\n")
        
        example_6_batch_semantic_search()
        input("\nPress Enter to continue to Example 7...\n")
        
        example_7_metadata_filtering()
        
        print("\n" + "=" * 80)
        print("All examples completed!")
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()
