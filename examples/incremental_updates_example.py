"""
Example: Incremental Updates for CAD Vector Database

This example demonstrates the incremental update capabilities:
1. Adding new vectors
2. Updating existing vectors
3. Batch updates
4. Soft delete and restore
5. Index compaction
6. Snapshots and rollback
7. Change log tracking

Author: riverfielder
Date: 2024
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_vectordb.core.index import IndexManager


def create_test_h5_file(file_path: str, vector_dim: int = 256):
    """Create a test H5 file with random vector data"""
    import h5py
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        # Create random vector data
        macro_vec = np.random.randn(10, vector_dim).astype(np.float32)
        f.create_dataset('macro_vec', data=macro_vec)
    
    print(f"Created test H5 file: {file_path}")


def main():
    """Main example function"""
    
    # Initialize index manager with versioning enabled
    print("=" * 60)
    print("Example 1: Initialize IndexManager with Versioning")
    print("=" * 60)
    
    index_dir = "./data/example_index"
    os.makedirs(index_dir, exist_ok=True)
    
    index_manager = IndexManager(
        index_dir=index_dir,
        verbose=True,
        enable_versioning=True
    )
    
    # Build initial index
    print("\n" + "=" * 60)
    print("Example 2: Build Initial Index")
    print("=" * 60)
    
    # Create test data
    test_data_dir = "./data/test_vectors"
    test_files = []
    
    for i in range(5):
        file_path = os.path.join(test_data_dir, f"vec_{i:04d}.h5")
        create_test_h5_file(file_path)
        test_files.append((f"vec_{i:04d}", file_path))
    
    # Build index with initial data
    index_manager.build_index(test_files[:3])  # Start with first 3 vectors
    print(f"\nInitial index built with {len(index_manager.ids)} vectors")
    
    # Add new vectors
    print("\n" + "=" * 60)
    print("Example 3: Add New Vectors")
    print("=" * 60)
    
    index_manager.add_vectors([test_files[3], test_files[4]])
    print(f"Index now contains {len(index_manager.ids)} vectors")
    
    # Update existing vector
    print("\n" + "=" * 60)
    print("Example 4: Update Existing Vector")
    print("=" * 60)
    
    # Create new version of vec_0001
    updated_file = os.path.join(test_data_dir, "vec_0001_v2.h5")
    create_test_h5_file(updated_file)
    
    index_manager.update_vector("vec_0001", updated_file)
    print(f"Vector 'vec_0001' updated successfully")
    
    # Batch update
    print("\n" + "=" * 60)
    print("Example 5: Batch Update Multiple Vectors")
    print("=" * 60)
    
    # Create new versions
    batch_updates = []
    for i in [2, 3]:
        updated_file = os.path.join(test_data_dir, f"vec_{i:04d}_v2.h5")
        create_test_h5_file(updated_file)
        batch_updates.append({
            "id_str": f"vec_{i:04d}",
            "h5_path": updated_file
        })
    
    index_manager.batch_update(batch_updates)
    print(f"Batch updated {len(batch_updates)} vectors")
    
    # Create snapshot before deletion
    print("\n" + "=" * 60)
    print("Example 6: Create Snapshot (Version Control)")
    print("=" * 60)
    
    snapshot_name = index_manager.create_snapshot("before_deletion")
    print(f"Created snapshot: {snapshot_name}")
    
    # List snapshots
    snapshots = index_manager.list_snapshots()
    print(f"\nAvailable snapshots ({len(snapshots)}):")
    for snap in snapshots:
        print(f"  - {snap['name']} ({snap['timestamp']})")
        print(f"    Vectors: {snap['num_vectors']}, Deleted: {snap['num_deleted']}")
    
    # Soft delete vectors
    print("\n" + "=" * 60)
    print("Example 7: Soft Delete Vectors")
    print("=" * 60)
    
    ids_to_delete = ["vec_0002", "vec_0003"]
    index_manager.soft_delete(ids_to_delete)
    print(f"Soft deleted: {ids_to_delete}")
    
    # Check deleted IDs
    deleted_ids = index_manager.get_deleted_ids()
    print(f"Currently deleted IDs: {deleted_ids}")
    
    # Search (soft-deleted vectors excluded by default)
    print("\n" + "=" * 60)
    print("Example 8: Search (Soft-Deleted Excluded)")
    print("=" * 60)
    
    query_vec = np.random.randn(256).astype(np.float32)
    results = index_manager.search(query_vec, k=5, include_deleted=False)
    print(f"Search results (excluding deleted): {len(results)} items")
    for rank, (idx, dist, id_str) in enumerate(results, 1):
        print(f"  {rank}. {id_str} (distance: {dist:.4f})")
    
    # Restore vectors
    print("\n" + "=" * 60)
    print("Example 9: Restore Soft-Deleted Vectors")
    print("=" * 60)
    
    index_manager.restore(["vec_0002"])
    print("Restored 'vec_0002'")
    
    deleted_ids = index_manager.get_deleted_ids()
    print(f"Remaining deleted IDs: {deleted_ids}")
    
    # Compact index (permanently remove deleted)
    print("\n" + "=" * 60)
    print("Example 10: Compact Index (Remove Deleted)")
    print("=" * 60)
    
    print(f"Before compaction: {len(index_manager.ids)} total vectors, {len(deleted_ids)} deleted")
    index_manager.compact_index()
    print(f"After compaction: {len(index_manager.ids)} vectors")
    
    # View change log
    print("\n" + "=" * 60)
    print("Example 11: View Change Log (Audit Trail)")
    print("=" * 60)
    
    changelog = index_manager.get_change_log(limit=10)
    print(f"Recent changes ({len(changelog)} entries):")
    for entry in changelog[:5]:  # Show last 5 entries
        print(f"  [{entry['timestamp']}] {entry['operation']} - {entry['target']}")
        if entry['details']:
            print(f"    Details: {entry['details']}")
    
    # Restore from snapshot
    print("\n" + "=" * 60)
    print("Example 12: Restore from Snapshot (Rollback)")
    print("=" * 60)
    
    print(f"Current state: {len(index_manager.ids)} vectors")
    index_manager.restore_snapshot(snapshot_name)
    print(f"Restored to snapshot '{snapshot_name}'")
    print(f"After restore: {len(index_manager.ids)} vectors")
    
    # Check deleted IDs after restore
    deleted_ids = index_manager.get_deleted_ids()
    print(f"Deleted IDs after restore: {deleted_ids}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary: Incremental Update Features Demonstrated")
    print("=" * 60)
    print("""
✓ Add new vectors to existing index
✓ Update individual vectors
✓ Batch update multiple vectors
✓ Soft delete (mark as deleted without removing)
✓ Restore soft-deleted vectors
✓ Compact index (permanently remove deleted)
✓ Create snapshots for version control
✓ List available snapshots
✓ Restore from snapshots (rollback)
✓ View change log for audit trail
✓ Search with soft-delete filtering

Benefits:
- No downtime for updates
- Safe deletion with restore capability
- Version control with snapshots
- Full audit trail of changes
- Efficient index compaction
""")


if __name__ == "__main__":
    main()
