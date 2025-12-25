"""Index management for CAD vector database

Supports:
- Building FAISS indexes from data
- Loading and saving indexes
- Adding/removing vectors dynamically
- Index statistics and validation
- Multiple index management
"""
import os
import json
import h5py
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from .feature import extract_feature


class IndexManager:
    """Manages FAISS index creation, loading, and modification"""
    
    def __init__(self, index_dir: str):
        """Initialize index manager
        
        Args:
            index_dir: Directory to store index files
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.ids = []
        self.metadata = []
        self.config = {}
        
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
