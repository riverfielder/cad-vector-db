"""Command-line tools for CAD Vector Database

Provides CLI tools for:
- Building indexes
- Running searches
- Managing indexes
- Performance benchmarks
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_vectordb.core.index import IndexManager
from cad_vectordb.core.retrieval import TwoStageRetrieval
from cad_vectordb.core.feature import load_macro_vec
from config import WHUCAD_DATA_ROOT, INDEX_DIR, INDEX_TYPE


def build_index_cli():
    """Build FAISS index from data"""
    parser = argparse.ArgumentParser(description="Build FAISS index from WHUCAD data")
    parser.add_argument('--data-root', default=WHUCAD_DATA_ROOT, help='Root directory with .h5 files')
    parser.add_argument('--output-dir', default=INDEX_DIR, help='Output directory for index')
    parser.add_argument('--index-name', default='default', help='Index name')
    parser.add_argument('--index-type', default=INDEX_TYPE, 
                       choices=['Flat', 'IVFFlat', 'HNSW'],
                       help='FAISS index type')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples (None for all)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Show progress')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CAD Vector Database - Index Builder")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Index type: {args.index_type}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print()
    
    # Build index
    manager = IndexManager(args.output_dir)
    stats = manager.build_index(
        data_root=args.data_root,
        index_type=args.index_type,
        max_samples=args.max_samples,
        verbose=args.verbose
    )
    
    # Save index
    save_path = manager.save_index(args.index_name)
    
    print()
    print("=" * 60)
    print("✅ Index Build Complete")
    print("=" * 60)
    print(f"Index saved to: {save_path}")
    print(f"Total vectors: {stats['num_vectors']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Index type: {stats['index_type']}")
    print(f"Unique subsets: {stats['unique_subsets']}")


def search_cli():
    """Run search from command line"""
    parser = argparse.ArgumentParser(description="Search in CAD vector database")
    parser.add_argument('query', help='Path to query .h5 file')
    parser.add_argument('--index-dir', default=INDEX_DIR, help='Index directory')
    parser.add_argument('--index-name', default='default', help='Index name')
    parser.add_argument('-k', type=int, default=20, help='Number of results')
    parser.add_argument('--stage1-topn', type=int, default=100, 
                       help='Stage 1 candidates')
    parser.add_argument('--fusion', default='weighted',
                       choices=['weighted', 'rrf', 'borda'],
                       help='Fusion method')
    parser.add_argument('--explainable', action='store_true',
                       help='Show detailed explanations')
    
    args = parser.parse_args()
    
    # Load index
    print(f"Loading index '{args.index_name}'...")
    manager = IndexManager(args.index_dir)
    manager.load_index(args.index_name)
    
    # Initialize retrieval
    retrieval = TwoStageRetrieval(
        manager.index,
        manager.ids,
        manager.metadata
    )
    
    # Load query
    print(f"Query: {args.query}")
    query_vec = load_macro_vec(args.query)
    print(f"Sequence length: {len(query_vec)}")
    
    # Search
    print(f"\nSearching (k={args.k}, stage1_topn={args.stage1_topn}, fusion={args.fusion})...")
    
    if args.explainable:
        results, explanation = retrieval.search(
            query_vec,
            args.query,
            k=args.k,
            stage1_topn=args.stage1_topn,
            fusion_method=args.fusion,
            explainable=True
        )
        
        print("\n" + "=" * 60)
        print("🎯 Top Results with Explanations")
        print("=" * 60)
        
        print(f"\nTop Match: {explanation['top_match']['id']}")
        print(f"Final Score: {explanation['final_score']:.4f}")
        print(f"Stage 1 Similarity: {explanation['stage1_similarity']:.4f} - {explanation.get('stage1_interpretation', '')}")
        print(f"Stage 2 Similarity: {explanation['stage2_similarity']:.4f} - {explanation.get('stage2_interpretation', '')}")
        
        if 'contributions' in explanation:
            print(f"\nContributions:")
            print(f"  Stage 1: {explanation['contributions']['stage1_percentage']:.1f}%")
            print(f"  Stage 2: {explanation['contributions']['stage2_percentage']:.1f}%")
    else:
        results = retrieval.search(
            query_vec,
            args.query,
            k=args.k,
            stage1_topn=args.stage1_topn,
            fusion_method=args.fusion
        )
    
    # Display results
    print("\n" + "=" * 60)
    print(f"Top {len(results)} Results")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['id']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Stage1: {result['stage1_sim']:.4f}, Stage2: {result['stage2_sim']:.4f}")
        print(f"   Subset: {result['metadata']['subset']}, SeqLen: {result['metadata']['seq_len']}")


def list_indexes_cli():
    """List all available indexes"""
    parser = argparse.ArgumentParser(description="List available indexes")
    parser.add_argument('--index-dir', default=INDEX_DIR, help='Index directory')
    
    args = parser.parse_args()
    
    manager = IndexManager(args.index_dir)
    indexes = manager.list_available_indexes()
    
    if not indexes:
        print(f"No indexes found in {args.index_dir}")
        return
    
    print("=" * 60)
    print("Available Indexes")
    print("=" * 60)
    
    for idx_name in indexes:
        try:
            mgr = IndexManager(args.index_dir)
            mgr.load_index(idx_name)
            stats = mgr.get_stats()
            
            print(f"\n📦 {idx_name}")
            print(f"   Vectors: {stats['num_vectors']}")
            print(f"   Dimension: {stats['dimension']}")
            print(f"   Type: {stats['index_type']}")
            print(f"   Subsets: {stats['num_subsets']}")
        except Exception as e:
            print(f"\n❌ {idx_name}: Error loading - {e}")


if __name__ == '__main__':
    # Determine which command to run based on script name or first argument
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove command from argv
        
        if command == 'build':
            build_index_cli()
        elif command == 'search':
            search_cli()
        elif command == 'list':
            list_indexes_cli()
        else:
            print("Usage: python -m cad_vectordb.cli [build|search|list] [options]")
            sys.exit(1)
    else:
        print("Usage: python -m cad_vectordb.cli [build|search|list] [options]")
        print("\nCommands:")
        print("  build   - Build a new index")
        print("  search  - Search in an index")
        print("  list    - List available indexes")
        sys.exit(1)
