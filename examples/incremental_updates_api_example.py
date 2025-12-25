"""
Example: Incremental Updates via REST API

This example demonstrates using the incremental update features
through the REST API endpoints.

Requirements:
- Server must be running (python server/app.py)
- Test vector files in data/test_vectors/

Author: riverfielder
Date: 2024
"""

import requests
import json
import time
import os
import numpy as np
import h5py


# API configuration
API_BASE_URL = "http://localhost:8123"


def create_test_h5_file(file_path: str, vector_dim: int = 256):
    """Create a test H5 file with random vector data"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        macro_vec = np.random.randn(10, vector_dim).astype(np.float32)
        f.create_dataset('macro_vec', data=macro_vec)
    
    print(f"✓ Created test H5 file: {file_path}")


def print_response(response):
    """Pretty print API response"""
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def main():
    """Main API example function"""
    
    # Check API availability
    print("=" * 60)
    print("Step 1: Check API Availability")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print_response(response)
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API server")
        print("Please start the server with: python server/app.py")
        return
    
    # Prepare test data
    print("\n" + "=" * 60)
    print("Step 2: Prepare Test Data")
    print("=" * 60)
    
    test_data_dir = "./data/test_vectors"
    test_files = []
    
    for i in range(5):
        file_path = os.path.abspath(os.path.join(test_data_dir, f"api_vec_{i:04d}.h5"))
        create_test_h5_file(file_path)
        test_files.append((f"api_vec_{i:04d}", file_path))
    
    # Add new vectors
    print("\n" + "=" * 60)
    print("Step 3: Add New Vectors")
    print("=" * 60)
    
    for vec_id, h5_path in test_files[:3]:
        data = {
            "id_str": vec_id,
            "h5_path": h5_path
        }
        response = requests.post(f"{API_BASE_URL}/vectors/add", json=data)
        print(f"\n➤ Adding {vec_id}:")
        print_response(response)
        time.sleep(0.5)
    
    # Update existing vector
    print("\n" + "=" * 60)
    print("Step 4: Update Existing Vector")
    print("=" * 60)
    
    updated_file = os.path.abspath(os.path.join(test_data_dir, "api_vec_0000_v2.h5"))
    create_test_h5_file(updated_file)
    
    data = {"h5_path": updated_file}
    response = requests.put(f"{API_BASE_URL}/vectors/api_vec_0000", json=data)
    print(f"\n➤ Updating api_vec_0000:")
    print_response(response)
    
    # Batch update
    print("\n" + "=" * 60)
    print("Step 5: Batch Update Multiple Vectors")
    print("=" * 60)
    
    batch_updates = []
    for i in [1, 2]:
        updated_file = os.path.abspath(os.path.join(test_data_dir, f"api_vec_{i:04d}_v2.h5"))
        create_test_h5_file(updated_file)
        batch_updates.append({
            "id_str": f"api_vec_{i:04d}",
            "h5_path": updated_file
        })
    
    data = {"updates": batch_updates}
    response = requests.post(f"{API_BASE_URL}/vectors/batch-update", json=data)
    print(f"\n➤ Batch updating {len(batch_updates)} vectors:")
    print_response(response)
    
    # Create snapshot
    print("\n" + "=" * 60)
    print("Step 6: Create Snapshot")
    print("=" * 60)
    
    data = {"name": "api_test_snapshot"}
    response = requests.post(f"{API_BASE_URL}/index/snapshot", json=data)
    print(f"\n➤ Creating snapshot:")
    print_response(response)
    
    # List snapshots
    print("\n" + "=" * 60)
    print("Step 7: List Snapshots")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE_URL}/index/snapshots")
    print(f"\n➤ Available snapshots:")
    print_response(response)
    
    # Soft delete vectors
    print("\n" + "=" * 60)
    print("Step 8: Soft Delete Vectors")
    print("=" * 60)
    
    data = {"ids": ["api_vec_0001", "api_vec_0002"]}
    response = requests.delete(f"{API_BASE_URL}/vectors/soft", json=data)
    print(f"\n➤ Soft deleting vectors:")
    print_response(response)
    
    # Get deleted vectors
    print("\n" + "=" * 60)
    print("Step 9: Get Deleted Vectors")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE_URL}/vectors/deleted")
    print(f"\n➤ Currently deleted vectors:")
    print_response(response)
    
    # Restore vectors
    print("\n" + "=" * 60)
    print("Step 10: Restore Soft-Deleted Vectors")
    print("=" * 60)
    
    data = {"ids": ["api_vec_0001"]}
    response = requests.post(f"{API_BASE_URL}/vectors/restore", json=data)
    print(f"\n➤ Restoring vectors:")
    print_response(response)
    
    # Get deleted vectors again
    response = requests.get(f"{API_BASE_URL}/vectors/deleted")
    print(f"\n➤ Remaining deleted vectors:")
    print_response(response)
    
    # Get change log
    print("\n" + "=" * 60)
    print("Step 11: View Change Log")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE_URL}/index/changelog?limit=10")
    print(f"\n➤ Recent changes:")
    print_response(response)
    
    # Compact index
    print("\n" + "=" * 60)
    print("Step 12: Compact Index")
    print("=" * 60)
    
    response = requests.post(f"{API_BASE_URL}/index/compact")
    print(f"\n➤ Compacting index:")
    print_response(response)
    
    # Restore from snapshot
    print("\n" + "=" * 60)
    print("Step 13: Restore from Snapshot")
    print("=" * 60)
    
    response = requests.post(f"{API_BASE_URL}/index/snapshot/api_test_snapshot/restore")
    print(f"\n➤ Restoring from snapshot:")
    print_response(response)
    
    # Get stats
    print("\n" + "=" * 60)
    print("Step 14: Get Database Stats")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE_URL}/stats")
    print(f"\n➤ Database statistics:")
    print_response(response)
    
    # Summary
    print("\n" + "=" * 60)
    print("API Testing Complete!")
    print("=" * 60)
    print("""
✓ Tested all incremental update API endpoints
✓ Add vectors: POST /vectors/add
✓ Update vector: PUT /vectors/{id}
✓ Batch update: POST /vectors/batch-update
✓ Soft delete: DELETE /vectors/soft
✓ Restore: POST /vectors/restore
✓ Get deleted: GET /vectors/deleted
✓ Compact: POST /index/compact
✓ Create snapshot: POST /index/snapshot
✓ List snapshots: GET /index/snapshots
✓ Restore snapshot: POST /index/snapshot/{name}/restore
✓ Change log: GET /index/changelog
    """)


if __name__ == "__main__":
    main()
