# Incremental Updates Guide

## Overview

The CAD Vector Database now supports incremental updates, allowing you to modify the index without rebuilding from scratch. This feature is essential for production systems that require:

- **Zero downtime**: Add/update/delete vectors without service interruption
- **Safe deletion**: Soft delete with restore capability
- **Version control**: Create snapshots and rollback to previous states
- **Audit trail**: Track all changes with detailed change logs

---

## Features

### 1. Add New Vectors

Add new vectors to an existing index without rebuilding:

**Python API:**
```python
from cad_vectordb.core.index import IndexManager

index_manager = IndexManager("./data/index", verbose=True)
index_manager.load_index()

# Add single vector
index_manager.add_vectors([("vec_0100", "/path/to/vec_0100.h5")])

# Add multiple vectors
new_vectors = [
    ("vec_0101", "/path/to/vec_0101.h5"),
    ("vec_0102", "/path/to/vec_0102.h5"),
]
index_manager.add_vectors(new_vectors)
```

**REST API:**
```bash
curl -X POST http://localhost:8123/vectors/add \
  -H "Content-Type: application/json" \
  -d '{
    "id_str": "vec_0100",
    "h5_path": "/absolute/path/to/vec_0100.h5"
  }'
```

---

### 2. Update Existing Vectors

Update the feature vector of an existing entry:

**Python API:**
```python
# Update single vector
index_manager.update_vector("vec_0001", "/path/to/vec_0001_v2.h5")
```

**REST API:**
```bash
curl -X PUT http://localhost:8123/vectors/vec_0001 \
  -H "Content-Type: application/json" \
  -d '{
    "h5_path": "/absolute/path/to/vec_0001_v2.h5"
  }'
```

---

### 3. Batch Updates

Update multiple vectors efficiently in one operation:

**Python API:**
```python
updates = [
    {"id_str": "vec_0001", "h5_path": "/path/to/vec_0001_v2.h5"},
    {"id_str": "vec_0002", "h5_path": "/path/to/vec_0002_v2.h5"},
]
index_manager.batch_update(updates)
```

**REST API:**
```bash
curl -X POST http://localhost:8123/vectors/batch-update \
  -H "Content-Type: application/json" \
  -d '{
    "updates": [
      {"id_str": "vec_0001", "h5_path": "/path/to/vec_0001_v2.h5"},
      {"id_str": "vec_0002", "h5_path": "/path/to/vec_0002_v2.h5"}
    ]
  }'
```

---

### 4. Soft Delete

Mark vectors as deleted without physically removing them. Soft-deleted vectors are excluded from search results but can be restored.

**Python API:**
```python
# Soft delete vectors
index_manager.soft_delete(["vec_0001", "vec_0002"])

# Check deleted IDs
deleted_ids = index_manager.get_deleted_ids()
print(f"Deleted: {deleted_ids}")

# Search (soft-deleted excluded by default)
results = index_manager.search(query_vec, k=10, include_deleted=False)
```

**REST API:**
```bash
# Soft delete
curl -X DELETE http://localhost:8123/vectors/soft \
  -H "Content-Type: application/json" \
  -d '{
    "ids": ["vec_0001", "vec_0002"]
  }'

# Get deleted IDs
curl -X GET http://localhost:8123/vectors/deleted
```

---

### 5. Restore Soft-Deleted Vectors

Restore vectors that were previously soft-deleted:

**Python API:**
```python
# Restore vectors
index_manager.restore(["vec_0001"])
```

**REST API:**
```bash
curl -X POST http://localhost:8123/vectors/restore \
  -H "Content-Type: application/json" \
  -d '{
    "ids": ["vec_0001"]
  }'
```

---

### 6. Compact Index

Permanently remove soft-deleted vectors by rebuilding the index without them. This reclaims disk space and improves performance.

**Python API:**
```python
# Compact index (permanently remove deleted)
index_manager.compact_index()
```

**REST API:**
```bash
curl -X POST http://localhost:8123/index/compact
```

---

### 7. Snapshots (Version Control)

Create snapshots of the index state for version control and rollback:

**Python API:**
```python
# Create snapshot
snapshot_name = index_manager.create_snapshot("before_major_update")

# List snapshots
snapshots = index_manager.list_snapshots()
for snap in snapshots:
    print(f"{snap['name']} - {snap['timestamp']}")
    print(f"  Vectors: {snap['num_vectors']}, Deleted: {snap['num_deleted']}")

# Restore from snapshot
index_manager.restore_snapshot("before_major_update")
```

**REST API:**
```bash
# Create snapshot
curl -X POST http://localhost:8123/index/snapshot \
  -H "Content-Type: application/json" \
  -d '{"name": "before_major_update"}'

# List snapshots
curl -X GET http://localhost:8123/index/snapshots

# Restore snapshot
curl -X POST http://localhost:8123/index/snapshot/before_major_update/restore
```

**Snapshot Directory Structure:**
```
data/index/
├── _snapshots/
│   ├── snapshot_20240101_120000/
│   │   ├── index.faiss
│   │   ├── id_map.pkl
│   │   └── metadata.json
│   └── before_major_update/
│       ├── index.faiss
│       ├── id_map.pkl
│       └── metadata.json
├── index.faiss
└── id_map.pkl
```

---

### 8. Change Log (Audit Trail)

Track all operations with a detailed change log:

**Python API:**
```python
# Get change log
changelog = index_manager.get_change_log(limit=50)

for entry in changelog:
    print(f"[{entry['timestamp']}] {entry['operation']}")
    print(f"  Target: {entry['target']}")
    print(f"  Details: {entry['details']}")
```

**REST API:**
```bash
curl -X GET "http://localhost:8123/index/changelog?limit=50"
```

**Change Log Entry Format:**
```json
{
  "timestamp": "2024-01-01 12:00:00",
  "operation": "update_vector",
  "target": "vec_0001",
  "details": {"h5_path": "/path/to/vec_0001_v2.h5"}
}
```

---

## Implementation Details

### Soft Delete Mechanism

Since FAISS doesn't support direct deletion, we implement soft delete using an in-memory set:

```python
# Soft-deleted IDs stored in memory
self.deleted_ids = set()

# Filtered search
def search(self, query_vec, k, include_deleted=False):
    results = self.index.search(query_vec, k * 2)  # Over-fetch
    
    if not include_deleted:
        # Filter out soft-deleted IDs
        results = [r for r in results if r.id not in self.deleted_ids]
    
    return results[:k]
```

### Snapshot System

Snapshots are stored in a `_snapshots` subdirectory:

```python
snapshot_dir = os.path.join(self.index_dir, "_snapshots", snapshot_name)

# Save index files
shutil.copy2(index_file, snapshot_dir)
shutil.copy2(id_map_file, snapshot_dir)

# Save metadata
metadata = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "num_vectors": len(self.ids),
    "num_deleted": len(self.deleted_ids),
    "deleted_ids": list(self.deleted_ids)
}
```

### Change Log

Changes are tracked in memory (last 1000 entries) and persisted with the index:

```python
self.change_log = []  # Max 1000 entries

def _log_change(self, operation, target, details=None):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation,
        "target": target,
        "details": details
    }
    self.change_log.append(entry)
    
    # Keep only last 1000 entries
    if len(self.change_log) > 1000:
        self.change_log = self.change_log[-1000:]
```

---

## Best Practices

### 1. Use Soft Delete First

Always soft delete before permanently removing:

```python
# Bad: Direct remove (rebuilds entire index)
index_manager.remove_vectors(["vec_0001"])

# Good: Soft delete (instant, reversible)
index_manager.soft_delete(["vec_0001"])
# ... verify everything works ...
# Then compact when needed
index_manager.compact_index()
```

### 2. Create Snapshots Before Major Changes

```python
# Before bulk update
index_manager.create_snapshot("before_bulk_update")

# Perform updates
index_manager.batch_update(large_update_list)

# If something goes wrong
index_manager.restore_snapshot("before_bulk_update")
```

### 3. Batch Updates for Efficiency

```python
# Bad: Multiple individual updates
for vec_id, h5_path in updates:
    index_manager.update_vector(vec_id, h5_path)

# Good: Single batch update
index_manager.batch_update([
    {"id_str": vec_id, "h5_path": h5_path}
    for vec_id, h5_path in updates
])
```

### 4. Regular Compaction

Schedule periodic compaction to reclaim space:

```python
# Check if compaction is needed
deleted_ratio = len(index_manager.deleted_ids) / len(index_manager.ids)

if deleted_ratio > 0.1:  # More than 10% deleted
    index_manager.create_snapshot("before_compaction")
    index_manager.compact_index()
```

### 5. Monitor Change Log

Regularly review the change log for audit and debugging:

```python
# Get recent changes
recent_changes = index_manager.get_change_log(limit=100)

# Check for suspicious activity
for entry in recent_changes:
    if entry['operation'] == 'soft_delete':
        print(f"Deleted: {entry['target']} at {entry['timestamp']}")
```

---

## API Reference

### Python API

| Method | Description |
|--------|-------------|
| `add_vectors(vectors)` | Add new vectors to index |
| `update_vector(id_str, h5_path)` | Update existing vector |
| `batch_update(updates)` | Batch update multiple vectors |
| `soft_delete(ids)` | Mark vectors as deleted |
| `restore(ids)` | Restore soft-deleted vectors |
| `get_deleted_ids()` | Get list of deleted IDs |
| `compact_index()` | Permanently remove deleted |
| `create_snapshot(name)` | Create version snapshot |
| `list_snapshots()` | List all snapshots |
| `restore_snapshot(name)` | Rollback to snapshot |
| `get_change_log(limit)` | Get change history |
| `search(query, k, include_deleted)` | Search with soft-delete filter |

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vectors/add` | POST | Add new vector |
| `/vectors/{id}` | PUT | Update existing vector |
| `/vectors/batch-update` | POST | Batch update vectors |
| `/vectors/soft` | DELETE | Soft delete vectors |
| `/vectors/restore` | POST | Restore deleted vectors |
| `/vectors/deleted` | GET | Get deleted IDs |
| `/index/compact` | POST | Compact index |
| `/index/snapshot` | POST | Create snapshot |
| `/index/snapshots` | GET | List snapshots |
| `/index/snapshot/{name}/restore` | POST | Restore snapshot |
| `/index/changelog` | GET | Get change log |

---

## Examples

See the following example files:

- **`examples/incremental_updates_example.py`** - Python API usage
- **`examples/incremental_updates_api_example.py`** - REST API usage

Run examples:
```bash
# Python API example
python examples/incremental_updates_example.py

# REST API example (requires server running)
python server/app.py &
python examples/incremental_updates_api_example.py
```

---

## Troubleshooting

### Issue: Snapshot Restore Fails

**Symptom:** Error when restoring snapshot

**Solution:** Ensure the snapshot directory exists and contains all required files:
```python
snapshots = index_manager.list_snapshots()
print(snapshots)  # Check if snapshot exists
```

### Issue: Soft Delete Not Working

**Symptom:** Deleted vectors still appear in search results

**Solution:** Ensure `include_deleted=False` in search:
```python
# Correct
results = index_manager.search(query, k=10, include_deleted=False)
```

### Issue: Change Log Missing Entries

**Symptom:** Change log doesn't show all operations

**Solution:** Change log keeps only last 1000 entries. Create snapshots for long-term history.

---

## Performance Considerations

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Add vectors | O(n) | Linear in number of vectors added |
| Update vector | O(1) | Direct FAISS update |
| Batch update | O(n) | Linear in batch size |
| Soft delete | O(1) | Set operation |
| Restore | O(1) | Set operation |
| Compact | O(N) | Rebuilds entire index (N = total vectors) |
| Snapshot create | O(N) | Copies index files |
| Snapshot restore | O(N) | Copies index files |
| Search | O(log N + k) | Excludes soft-deleted at query time |

**Recommendations:**
- Batch operations when possible (100-1000 vectors per batch)
- Compact when deleted ratio > 10%
- Create snapshots before major operations
- Use soft delete for frequent changes

---

## Migration Guide

If you have an existing index without versioning:

```python
# Load existing index
index_manager = IndexManager("./data/index", enable_versioning=False)
index_manager.load_index()

# Enable versioning (will initialize tracking structures)
index_manager.enable_versioning = True
index_manager.deleted_ids = set()
index_manager.change_log = []

# Create first snapshot
index_manager.create_snapshot("migration_baseline")

# Save with versioning enabled
index_manager.save_index()
```

---

## Future Enhancements

Planned improvements for incremental updates:

1. **Persistent Change Log** - Store change log in database for unlimited history
2. **Automatic Snapshots** - Scheduled snapshot creation
3. **Snapshot Compression** - Compress old snapshots to save space
4. **Differential Snapshots** - Store only changes between snapshots
5. **Multi-Index Updates** - Update multiple indexes atomically

---

## References

- **FAISS Documentation**: https://github.com/facebookresearch/faiss/wiki
- **API Documentation**: See `server/app.py` for endpoint details
- **Examples**: See `examples/` directory

---

*Last Updated: 2024*
