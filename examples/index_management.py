"""Example: Index management operations

Demonstrates:
- Building a new index
- Loading an existing index
- Adding vectors dynamically
- Removing vectors
- Index validation
- Listing available indexes
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cad_vectordb.core.index import IndexManager


def main():
    # Initialize index manager
    index_dir = "./data/indexes"
    manager = IndexManager(index_dir)
    
    print("=" * 60)
    print("Index Management Example")
    print("=" * 60)
    
    # 1. Build a new index
    print("\n1. Building new index...")
    data_root = "../WHUCAD-main/data/vec"
    stats = manager.build_index(
        data_root=data_root,
        index_type="Flat",
        max_samples=100,  # Limit for demo
        verbose=True
    )
    print(f"   Built index with {stats['num_vectors']} vectors")
    
    # 2. Save the index
    print("\n2. Saving index...")
    save_path = manager.save_index(name="demo_index")
    print(f"   Saved to: {save_path}")
    
    # 3. Get index statistics
    print("\n3. Index statistics:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 4. Validate index
    print("\n4. Validating index...")
    validation = manager.validate_index()
    if validation['valid']:
        print("   ✅ Index is valid")
    else:
        print(f"   ❌ Issues found: {validation['issues']}")
    
    # 5. Add new vectors
    print("\n5. Adding new vectors...")
    new_files = [
        f"{data_root}/0000/00000100.h5",
        f"{data_root}/0000/00000101.h5",
    ]
    num_added = manager.add_vectors(new_files, verbose=True)
    print(f"   Added {num_added} vectors")
    
    # 6. List available indexes
    print("\n6. Available indexes:")
    indexes = manager.list_available_indexes()
    for idx_name in indexes:
        print(f"   - {idx_name}")
    
    # 7. Load an existing index
    print("\n7. Loading existing index...")
    manager2 = IndexManager(index_dir)
    config = manager2.load_index("demo_index")
    print(f"   Loaded: {config['num_vectors']} vectors")
    
    # 8. Remove vectors
    print("\n8. Removing vectors...")
    ids_to_remove = [manager2.ids[0]]  # Remove first vector
    num_removed = manager2.remove_vectors(ids_to_remove, rebuild=True)
    print(f"   Removed {num_removed} vectors")
    
    # 9. Save updated index
    print("\n9. Saving updated index...")
    manager2.save_index(name="demo_index")
    
    print("\n" + "=" * 60)
    print("✅ Index management demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
