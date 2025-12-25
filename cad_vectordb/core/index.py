"""Index management for CAD vector database

Supports:
- Building FAISS indexes from data
- Loading and saving indexes
- Adding/removing vectors dynamically (incremental updates)
- Updating existing vectors
- Index statistics and validation
- Multiple index management
- Version control and rollback
"""
import os
import json
import h5py
import numpy as np
import faiss
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
from datetime import datetime

from .feature import extract_feature
from .compression import VectorCompressor
from .cache import QueryCache


class IndexManager:
    """Manages FAISS index creation, loading, and modification with incremental updates"""
    
    def __init__(
        self,
        index_dir: str,
        enable_versioning: bool = False,
        enable_compression: bool = False,
        compression_type: str = "pq",
        enable_cache: bool = False,
        cache_capacity: int = 1000,
        verbose: bool = True
    ):
        """Initialize index manager
        
        Args:
            index_dir: Directory to store index files
            enable_versioning: Enable index versioning for rollback support
            enable_compression: Enable vector compression
            compression_type: Compression type ("pq", "sq", or "none")
            enable_cache: Enable query result caching
            cache_capacity: LRU cache capacity
            verbose: Print status messages
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.ids = []
        self.metadata = []
        self.config = {}
        self.enable_versioning = enable_versioning
        self.verbose = verbose
        
        # Track deleted IDs (for soft delete)
        self.deleted_ids = set()
        
        # Change log for auditing
        self.change_log = []
        
        # Compression support
        self.enable_compression = enable_compression
        self.compressor = None
        if enable_compression:
            self.compressor = VectorCompressor(
                compression_type=compression_type,
                verbose=verbose
            )
            if verbose:
                print(f"✓ Compression enabled: {compression_type}")
        
        # Caching support
        self.enable_cache = enable_cache
        self.cache = None
        if enable_cache:
            self.cache = QueryCache(
                lru_capacity=cache_capacity,
                verbose=verbose
            )
            if verbose:
                print(f"✓ Cache enabled: capacity={cache_capacity}")
        
    def build_index(self, 
                    data_root: str,
                    index_type: str = "Flat",
                    max_samples: Optional[int] = None,
                    verbose: bool = True) -> Dict:
        """Build FAISS index from h5 data files
        
        Args:
            data_root: Root directory containing subdirectories with .h5 files
            index_type: FAISS index type ("Flat", "IVFFlat", "HNSW")
            max_samples: Maximum number of samples to index (None for all)
            verbose: Show progress bars
            
        Returns:
            stats: Dict with build statistics
        """
        ids, features, metadata = self._load_vectors(data_root, max_samples, verbose)
        
        if len(ids) == 0:
            raise ValueError(f"No data found in {data_root}")
        
        # Build FAISS index
        d = features.shape[1]
        
        if index_type == "Flat":
            index = faiss.IndexFlatL2(d)
        elif index_type == "IVFFlat":
            nlist = min(100, len(ids) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(features)
        elif index_type == "HNSW":
            index = faiss.IndexHNSWFlat(d, 32)  # 32 neighbors
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        if verbose:
            print(f"Adding {len(features)} vectors to {index_type} index...")
        index.add(features)
        
        self.index = index
        self.ids = ids
        self.metadata = metadata
        self.config = {
            "index_type": index_type,
            "dimension": d,
            "num_vectors": len(ids),
            "data_root": str(data_root)
        }
        
        stats = {
            "num_vectors": len(ids),
            "dimension": d,
            "index_type": index_type,
            "unique_subsets": len(set(m['subset'] for m in metadata))
        }
        
        if verbose:
            print(f"✅ Index built: {stats}")
        
        return stats
    
    def _load_vectors(self, 
                     data_root: str,
                     max_samples: Optional[int] = None,
                     verbose: bool = True) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """Load vectors from h5 files"""
        ids, features, metadata = [], [], []
        data_root = Path(data_root)
        
        # Walk through all subdirectories
        subdirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
        
        pbar = tqdm(subdirs, desc="Loading subsets") if verbose else subdirs
        for subset_dir in pbar:
            subset = subset_dir.name
            files = sorted(subset_dir.glob("*.h5"))
            
            for h5_file in files:
                if max_samples and len(ids) >= max_samples:
                    break
                
                try:
                    with h5py.File(h5_file, 'r') as f:
                        vec = f['vec'][:]
                    
                    feat = extract_feature(vec)
                    
                    id_str = f"{subset}/{h5_file.name}"
                    ids.append(id_str)
                    features.append(feat)
                    metadata.append({
                        "id": id_str,
                        "file_path": str(h5_file),
                        "subset": subset,
                        "seq_len": len(vec)
                    })
                except Exception as e:
                    if verbose:
                        print(f"Error loading {h5_file}: {e}")
                    continue
            
            if max_samples and len(ids) >= max_samples:
                break
        
        features = np.array(features, dtype='float32')
        return ids, features, metadata
    
    def save_index(self, name: str = "default") -> str:
        """Save index to disk
        
        Args:
            name: Index name
            
        Returns:
            save_dir: Directory where index was saved
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        save_dir = self.index_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / "faiss_index.bin"))
        
        # Save IDs
        with open(save_dir / "id_map.json", 'w') as f:
            json.dump(self.ids, f)
        
        # Save metadata
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save config
        with open(save_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"✅ Index saved to {save_dir}")
        return str(save_dir)
    
    def load_index(self, name: str = "default") -> Dict:
        """Load index from disk
        
        Args:
            name: Index name
            
        Returns:
            config: Index configuration
        """
        load_dir = self.index_dir / name
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Index not found: {load_dir}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_dir / "faiss_index.bin"))
        
        # Load IDs
        with open(load_dir / "id_map.json", 'r') as f:
            self.ids = json.load(f)
        
        # Load metadata
        with open(load_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load config
        with open(load_dir / "config.json", 'r') as f:
            self.config = json.load(f)
        
        # Handle backward compatibility: add 'dimension' key if only 'feature_dim' exists
        if 'dimension' not in self.config and 'feature_dim' in self.config:
            self.config['dimension'] = self.config['feature_dim']
        
        dimension = self.config.get('dimension', 'unknown')
        print(f"✅ Loaded index: {len(self.ids)} vectors, dim={dimension}")
        return self.config
    
    def add_vectors(self, 
                    h5_paths: List[str],
                    verbose: bool = True) -> int:
        """Add new vectors to existing index
        
        Args:
            h5_paths: List of paths to .h5 files
            verbose: Show progress
            
        Returns:
            num_added: Number of vectors added
        """
        if self.index is None:
            raise ValueError("No index loaded. Build or load an index first.")
        
        new_ids, new_features, new_metadata = [], [], []
        
        pbar = tqdm(h5_paths, desc="Adding vectors") if verbose else h5_paths
        for h5_path in pbar:
            try:
                with h5py.File(h5_path, 'r') as f:
                    vec = f['vec'][:]
                
                feat = extract_feature(vec)
                
                # Extract subset and filename
                path_obj = Path(h5_path)
                subset = path_obj.parent.name
                id_str = f"{subset}/{path_obj.name}"
                
                # Check if already exists
                if id_str in self.ids:
                    if verbose:
                        print(f"Skipping duplicate: {id_str}")
                    continue
                
                new_ids.append(id_str)
                new_features.append(feat)
                new_metadata.append({
                    "id": id_str,
                    "file_path": str(h5_path),
                    "subset": subset,
                    "seq_len": len(vec)
                })
            except Exception as e:
                if verbose:
                    print(f"Error adding {h5_path}: {e}")
                continue
        
        if len(new_features) > 0:
            new_features = np.array(new_features, dtype='float32')
            self.index.add(new_features)
            self.ids.extend(new_ids)
            self.metadata.extend(new_metadata)
            self.config['num_vectors'] = len(self.ids)
        
        if verbose:
            print(f"✅ Added {len(new_features)} vectors (total: {len(self.ids)})")
        
        return len(new_features)
    
    def remove_vectors(self, ids_to_remove: List[str], rebuild: bool = True) -> int:
        """Remove vectors from index
        
        Note: FAISS doesn't support direct deletion, so we rebuild the index
        
        Args:
            ids_to_remove: List of IDs to remove
            rebuild: Whether to rebuild index after removal
            
        Returns:
            num_removed: Number of vectors removed
        """
        if self.index is None:
            raise ValueError("No index loaded")
        
        ids_set = set(ids_to_remove)
        
        # Filter out vectors to remove
        keep_indices = [i for i, id_str in enumerate(self.ids) if id_str not in ids_set]
        
        num_removed = len(self.ids) - len(keep_indices)
        
        if num_removed == 0:
            print("No vectors to remove")
            return 0
        
        # Update IDs and metadata
        self.ids = [self.ids[i] for i in keep_indices]
        self.metadata = [self.metadata[i] for i in keep_indices]
        
        if rebuild:
            # Rebuild index with remaining vectors
            # Extract features from metadata file paths
            features = []
            for meta in tqdm(self.metadata, desc="Rebuilding index"):
                try:
                    with h5py.File(meta['file_path'], 'r') as f:
                        vec = f['vec'][:]
                    feat = extract_feature(vec)
                    features.append(feat)
                except:
                    # If file not found, skip (will be removed)
                    continue
            
            # Create new index
            features = np.array(features, dtype='float32')
            d = features.shape[1]
            index_type = self.config.get('index_type', 'Flat')
            
            if index_type == "Flat":
                new_index = faiss.IndexFlatL2(d)
            elif index_type == "IVFFlat":
                nlist = min(100, len(features) // 10)
                quantizer = faiss.IndexFlatL2(d)
                new_index = faiss.IndexIVFFlat(quantizer, d, nlist)
                new_index.train(features)
            elif index_type == "HNSW":
                new_index = faiss.IndexHNSWFlat(d, 32)
            else:
                new_index = faiss.IndexFlatL2(d)
            
            new_index.add(features)
            self.index = new_index
            self.config['num_vectors'] = len(self.ids)
        
        print(f"✅ Removed {num_removed} vectors (remaining: {len(self.ids)})")
        return num_removed
    
    def get_stats(self) -> Dict:
        """Get index statistics
        
        Returns:
            stats: Dict with index statistics
        """
        if self.index is None:
            return {"status": "no_index_loaded"}
        
        # Handle both 'dimension' and 'feature_dim' keys for backward compatibility
        dimension = self.config.get('dimension') or self.config.get('feature_dim', 0)
        
        stats = {
            "num_vectors": len(self.ids),
            "dimension": dimension,
            "index_type": self.config.get('index_type', 'Unknown'),
            "num_subsets": len(set(m['subset'] for m in self.metadata)),
            "avg_seq_len": np.mean([m['seq_len'] for m in self.metadata]),
            "min_seq_len": min(m['seq_len'] for m in self.metadata) if self.metadata else 0,
            "max_seq_len": max(m['seq_len'] for m in self.metadata) if self.metadata else 0,
        }
        
        return stats
    
    def list_available_indexes(self) -> List[str]:
        """List all available indexes in index_dir
        
        Returns:
            index_names: List of index names
        """
        if not self.index_dir.exists():
            return []
        
        indexes = []
        for subdir in self.index_dir.iterdir():
            if subdir.is_dir() and (subdir / "faiss_index.bin").exists():
                indexes.append(subdir.name)
        
        return sorted(indexes)
    
    def validate_index(self) -> Dict:
        """Validate index integrity
        
        Returns:
            validation: Dict with validation results
        """
        if self.index is None:
            return {"valid": False, "error": "No index loaded"}
        
        issues = []
        
        # Check consistency
        if self.index.ntotal != len(self.ids):
            issues.append(f"Index vectors ({self.index.ntotal}) != ID count ({len(self.ids)})")
        
        if len(self.ids) != len(self.metadata):
            issues.append(f"ID count ({len(self.ids)}) != metadata count ({len(self.metadata)})")
        
        # Check for duplicate IDs
        id_set = set(self.ids)
        if len(id_set) != len(self.ids):
            issues.append(f"Duplicate IDs found: {len(self.ids) - len(id_set)} duplicates")
        
        # Check metadata consistency
        for i, (id_str, meta) in enumerate(zip(self.ids, self.metadata)):
            if meta['id'] != id_str:
                issues.append(f"ID mismatch at index {i}: {id_str} != {meta['id']}")
                break  # Only report first mismatch
        
        return {
            "valid": len(issues) == 0,
            "num_vectors": len(self.ids),
            "issues": issues
        }
    
    # ==================== Enhanced Incremental Update Methods ====================
    
    def update_vector(self, id_str: str, h5_path: str, verbose: bool = True) -> bool:
        """Update an existing vector in the index
        
        This operation removes the old vector and adds the new one.
        
        Args:
            id_str: ID of the vector to update
            h5_path: Path to new h5 file
            verbose: Show progress
            
        Returns:
            success: True if updated successfully
        """
        if self.index is None:
            raise ValueError("No index loaded")
        
        # Check if ID exists
        if id_str not in self.ids:
            if verbose:
                print(f"ID {id_str} not found, adding as new vector")
            return self.add_vectors([h5_path], verbose=verbose) > 0
        
        # Remove old vector
        self.remove_vectors([id_str], rebuild=False)
        
        # Add new vector
        added = self.add_vectors([h5_path], verbose=verbose)
        
        # Log change
        self._log_change("update", id_str, h5_path)
        
        if verbose:
            print(f"✅ Updated vector: {id_str}")
        
        return added > 0
    
    def batch_update(self, 
                    updates: List[Tuple[str, str]], 
                    verbose: bool = True) -> Dict[str, int]:
        """Batch update multiple vectors
        
        Args:
            updates: List of (id_str, h5_path) tuples
            verbose: Show progress
            
        Returns:
            stats: Dict with update statistics
        """
        if self.index is None:
            raise ValueError("No index loaded")
        
        stats = {"updated": 0, "added": 0, "failed": 0}
        
        pbar = tqdm(updates, desc="Batch updating") if verbose else updates
        for id_str, h5_path in pbar:
            try:
                if id_str in self.ids:
                    if self.update_vector(id_str, h5_path, verbose=False):
                        stats["updated"] += 1
                else:
                    if self.add_vectors([h5_path], verbose=False) > 0:
                        stats["added"] += 1
            except Exception as e:
                stats["failed"] += 1
                if verbose:
                    print(f"Failed to update {id_str}: {e}")
        
        if verbose:
            print(f"✅ Batch update complete: {stats}")
        
        return stats
    
    def soft_delete(self, ids_to_delete: List[str], verbose: bool = True) -> int:
        """Soft delete vectors (mark as deleted without removing)
        
        Soft-deleted vectors are excluded from search results but remain in index.
        Use compact_index() to permanently remove them.
        
        Args:
            ids_to_delete: List of IDs to soft delete
            verbose: Show progress
            
        Returns:
            num_deleted: Number of vectors soft deleted
        """
        if self.index is None:
            raise ValueError("No index loaded")
        
        count = 0
        for id_str in ids_to_delete:
            if id_str in self.ids and id_str not in self.deleted_ids:
                self.deleted_ids.add(id_str)
                count += 1
                self._log_change("soft_delete", id_str)
        
        if verbose:
            print(f"✅ Soft deleted {count} vectors (total deleted: {len(self.deleted_ids)})")
        
        return count
    
    def restore(self, ids_to_restore: List[str], verbose: bool = True) -> int:
        """Restore soft-deleted vectors
        
        Args:
            ids_to_restore: List of IDs to restore
            verbose: Show progress
            
        Returns:
            num_restored: Number of vectors restored
        """
        if self.index is None:
            raise ValueError("No index loaded")
        
        count = 0
        for id_str in ids_to_restore:
            if id_str in self.deleted_ids:
                self.deleted_ids.remove(id_str)
                count += 1
                self._log_change("restore", id_str)
        
        if verbose:
            print(f"✅ Restored {count} vectors")
        
        return count
    
    def compact_index(self, verbose: bool = True) -> int:
        """Permanently remove soft-deleted vectors and compact index
        
        This rebuilds the index without deleted vectors.
        
        Args:
            verbose: Show progress
            
        Returns:
            num_removed: Number of vectors permanently removed
        """
        if self.index is None:
            raise ValueError("No index loaded")
        
        if len(self.deleted_ids) == 0:
            if verbose:
                print("No deleted vectors to compact")
            return 0
        
        # Remove deleted vectors
        num_removed = self.remove_vectors(list(self.deleted_ids), rebuild=True)
        self.deleted_ids.clear()
        
        if verbose:
            print(f"✅ Compacted index: removed {num_removed} vectors")
        
        return num_removed
    
    def create_snapshot(self, snapshot_name: Optional[str] = None) -> str:
        """Create a snapshot of current index state
        
        Args:
            snapshot_name: Optional name for snapshot (auto-generated if None)
            
        Returns:
            snapshot_path: Path to snapshot directory
        """
        if self.index is None:
            raise ValueError("No index loaded")
        
        if snapshot_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"snapshot_{timestamp}"
        
        snapshot_dir = self.index_dir / "_snapshots" / snapshot_name
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current state
        current_name = self.config.get('index_name', 'default')
        current_dir = self.index_dir / current_name
        
        if current_dir.exists():
            # Copy all files
            for file in ["faiss_index.bin", "id_map.json", "metadata.json", "config.json"]:
                src = current_dir / file
                if src.exists():
                    shutil.copy2(src, snapshot_dir / file)
        
        # Save snapshot metadata
        snapshot_meta = {
            "snapshot_name": snapshot_name,
            "created_at": datetime.now().isoformat(),
            "num_vectors": len(self.ids),
            "num_deleted": len(self.deleted_ids)
        }
        
        with open(snapshot_dir / "snapshot_info.json", 'w') as f:
            json.dump(snapshot_meta, f, indent=2)
        
        print(f"✅ Created snapshot: {snapshot_name}")
        return str(snapshot_dir)
    
    def list_snapshots(self) -> List[Dict]:
        """List all available snapshots
        
        Returns:
            snapshots: List of snapshot info dicts
        """
        snapshots_dir = self.index_dir / "_snapshots"
        if not snapshots_dir.exists():
            return []
        
        snapshots = []
        for snapshot_dir in sorted(snapshots_dir.iterdir()):
            if snapshot_dir.is_dir():
                info_file = snapshot_dir / "snapshot_info.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    info['path'] = str(snapshot_dir)
                    snapshots.append(info)
        
        return snapshots
    
    def restore_snapshot(self, snapshot_name: str, verbose: bool = True) -> bool:
        """Restore index from a snapshot
        
        Args:
            snapshot_name: Name of snapshot to restore
            verbose: Show progress
            
        Returns:
            success: True if restored successfully
        """
        snapshot_dir = self.index_dir / "_snapshots" / snapshot_name
        
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot {snapshot_name} not found")
        
        # Load snapshot
        index = faiss.read_index(str(snapshot_dir / "faiss_index.bin"))
        
        with open(snapshot_dir / "id_map.json", 'r') as f:
            ids = json.load(f)
        
        with open(snapshot_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        with open(snapshot_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Replace current index
        self.index = index
        self.ids = ids
        self.metadata = metadata
        self.config = config
        self.deleted_ids.clear()
        
        self._log_change("restore_snapshot", snapshot_name)
        
        if verbose:
            print(f"✅ Restored snapshot: {snapshot_name} ({len(ids)} vectors)")
        
        return True
    
    def get_change_log(self, limit: Optional[int] = None) -> List[Dict]:
        """Get change log history
        
        Args:
            limit: Maximum number of entries to return (None for all)
            
        Returns:
            log_entries: List of change log dicts
        """
        if limit is None:
            return self.change_log
        return self.change_log[-limit:]
    
    def get_deleted_ids(self) -> List[str]:
        """Get list of soft-deleted IDs
        
        Returns:
            deleted_ids: List of soft-deleted IDs
        """
        return list(self.deleted_ids)
    
    def search(self, query_vector: np.ndarray, k: int = 10, 
               include_deleted: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Search index with soft-delete filtering
        
        Args:
            query_vector: Query vector (1, d) or (d,)
            k: Number of results
            include_deleted: Include soft-deleted vectors in results
            
        Returns:
            distances: Distance array
            indices: Index array
            ids: List of IDs (filtered)
        """
        if self.index is None:
            raise ValueError("No index loaded")
        
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search with extra results to account for deleted vectors
        search_k = k if include_deleted else min(k * 2, len(self.ids))
        D, I = self.index.search(query_vector, search_k)
        
        if include_deleted or len(self.deleted_ids) == 0:
            result_ids = [self.ids[i] for i in I[0] if i < len(self.ids)]
            return D[0], I[0], result_ids[:k]
        
        # Filter out deleted vectors
        filtered_distances = []
        filtered_indices = []
        filtered_ids = []
        
        for dist, idx in zip(D[0], I[0]):
            if idx >= len(self.ids):
                continue
            id_str = self.ids[idx]
            if id_str not in self.deleted_ids:
                filtered_distances.append(dist)
                filtered_indices.append(idx)
                filtered_ids.append(id_str)
            if len(filtered_ids) >= k:
                break
        
        return (np.array(filtered_distances), 
                np.array(filtered_indices), 
                filtered_ids)
    
    def enable_vector_compression(
        self,
        compression_type: str = "pq",
        train_samples: int = 100000
    ):
        """
        Enable vector compression on the index
        
        Args:
            compression_type: "pq" or "sq"
            train_samples: Number of samples for training quantizer
        """
        if self.index is None:
            raise RuntimeError("No index loaded")
        
        dimension = self.index.d
        
        # Initialize compressor
        self.compressor = VectorCompressor(
            compression_type=compression_type,
            dimension=dimension,
            verbose=self.verbose
        )
        
        # Get vectors for training
        n_vectors = self.index.ntotal
        if n_vectors > train_samples:
            # Sample vectors
            indices = np.random.choice(n_vectors, train_samples, replace=False)
            vectors = np.zeros((len(indices), dimension), dtype=np.float32)
            for i, idx in enumerate(indices):
                vectors[i] = self.index.reconstruct(int(idx))
        else:
            # Use all vectors
            vectors = np.zeros((n_vectors, dimension), dtype=np.float32)
            for i in range(n_vectors):
                vectors[i] = self.index.reconstruct(i)
        
        # Train compressor
        self.compressor.train(vectors)
        
        # Get compression stats
        stats = self.compressor.get_compression_stats(vectors)
        
        if self.verbose:
            print(f"\nCompression Statistics:")
            print(f"  Original size: {stats.original_size / 1024 / 1024:.1f} MB")
            print(f"  Compressed size: {stats.compressed_size / 1024 / 1024:.1f} MB")
            print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
            print(f"  Memory saved: {stats.memory_saved_mb:.1f} MB")
        
        self.enable_compression = True
        self.config['compression'] = {
            'type': compression_type,
            'ratio': stats.compression_ratio,
            'memory_saved_mb': stats.memory_saved_mb
        }
        
        return stats
    
    def rebuild_with_compression(
        self,
        compression_type: str = "pq",
        index_type: str = "IVF",
        nlist: int = 100
    ):
        """
        Rebuild index with compression
        
        Args:
            compression_type: "pq" or "sq"
            index_type: "IVF" or "HNSW"
            nlist: Number of clusters for IVF
        """
        if self.index is None:
            raise RuntimeError("No index loaded")
        
        if self.verbose:
            print(f"Rebuilding index with {compression_type} compression...")
        
        # Get all vectors
        n_vectors = self.index.ntotal
        dimension = self.index.d
        vectors = np.zeros((n_vectors, dimension), dtype=np.float32)
        
        for i in range(n_vectors):
            vectors[i] = self.index.reconstruct(i)
        
        # Initialize and train compressor
        self.compressor = VectorCompressor(
            compression_type=compression_type,
            dimension=dimension,
            verbose=self.verbose
        )
        
        self.compressor.train(vectors)
        
        # Create compressed index
        new_index = self.compressor.create_compressed_index(
            index_type=index_type,
            nlist=nlist
        )
        
        # Train and add vectors
        if hasattr(new_index, 'train') and not new_index.is_trained:
            new_index.train(vectors)
        
        new_index.add(vectors)
        
        # Replace old index
        old_index = self.index
        self.index = new_index
        
        # Update config
        self.config['index_type'] = f"{index_type}+{compression_type.upper()}"
        self.config['compression'] = {'type': compression_type}
        self.enable_compression = True
        
        # Get stats
        stats = self.compressor.get_compression_stats(vectors)
        
        if self.verbose:
            print(f"✓ Index rebuilt with compression")
            print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
            print(f"  Memory saved: {stats.memory_saved_mb:.1f} MB")
        
        self._log_change("rebuild_compressed", "index", {
            "compression_type": compression_type,
            "index_type": index_type,
            "compression_ratio": stats.compression_ratio
        })
        
        return stats
    
    def enable_query_cache(
        self,
        capacity: int = 1000,
        ttl: int = 3600,
        use_redis: bool = False
    ):
        """
        Enable query result caching
        
        Args:
            capacity: LRU cache capacity
            ttl: Time-to-live in seconds
            use_redis: Enable Redis backend
        """
        self.cache = QueryCache(
            lru_capacity=capacity,
            lru_ttl=ttl,
            use_redis=use_redis,
            verbose=self.verbose
        )
        
        self.enable_cache = True
        
        if self.verbose:
            print(f"✓ Query cache enabled (capacity={capacity}, ttl={ttl}s)")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if not self.enable_cache or self.cache is None:
            return {"enabled": False}
        
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear query cache"""
        if self.enable_cache and self.cache is not None:
            self.cache.clear()
            if self.verbose:
                print("✓ Cache cleared")
    
    def warm_cache(self, n_samples: int = 100):
        """
        Warm up cache with random queries
        
        Args:
            n_samples: Number of random queries to cache
        """
        if not self.enable_cache or self.cache is None:
            if self.verbose:
                print("⚠ Cache not enabled")
            return
        
        if self.index is None:
            raise RuntimeError("No index loaded")
        
        # Generate random query samples
        dimension = self.index.d
        query_samples = []
        
        for _ in range(n_samples):
            query_vec = np.random.randn(dimension).astype(np.float32)
            query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            k = np.random.choice([5, 10, 20])
            query_samples.append((query_vec, k))
        
        # Define search function
        def search_fn(query_vec, k):
            D, I = self.index.search(query_vec.reshape(1, -1), k)
            results = []
            for idx, dist in zip(I[0], D[0]):
                if idx < len(self.ids):
                    results.append({
                        "id": self.ids[idx],
                        "distance": float(dist),
                        "metadata": self.metadata[idx]
                    })
            return results
        
        # Warm cache
        self.cache.warm_cache(query_samples, search_fn)
    
    def get_compression_stats(self) -> Optional[Dict]:
        """Get compression statistics"""
        if not self.enable_compression or self.compressor is None:
            return None
        
        if self.index is None:
            return None
        
        # Get sample vectors
        n_vectors = self.index.ntotal
        dimension = self.index.d
        sample_size = min(1000, n_vectors)
        vectors = np.zeros((sample_size, dimension), dtype=np.float32)
        
        for i in range(sample_size):
            vectors[i] = self.index.reconstruct(i)
        
        stats = self.compressor.get_compression_stats(vectors)
        
        return {
            "compression_type": self.compressor.compression_type,
            "compression_ratio": stats.compression_ratio,
            "memory_saved_mb": stats.memory_saved_mb,
            "original_size_mb": stats.original_size / 1024 / 1024,
            "compressed_size_mb": stats.compressed_size / 1024 / 1024
        }
    
    def _log_change(self, operation: str, target: str, details: str = ""):
        """Log a change operation
        
        Args:
            operation: Type of operation (add, update, delete, etc.)
            target: Target ID or name
            details: Additional details
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "target": target,
            "details": details
        }
        self.change_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.change_log) > 1000:
            self.change_log = self.change_log[-1000:]

