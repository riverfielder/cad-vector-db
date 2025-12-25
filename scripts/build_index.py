"""Ingest WHUCAD vec data and build FAISS index"""
import os
import sys
import json
import h5py
import numpy as np
import faiss
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


def extract_feature(vec):
    """Extract fixed-dim feature from variable-length vec
    
    Args:
        vec: (seq_len, 33) array, col 0 is command, col 1-32 are params
    
    Returns:
        feat: (32,) float32 feature vector
    """
    # Mean pooling over params (ignore command column)
    feat = vec[:, 1:].mean(axis=0).astype('float32')
    return feat


def load_vectors(data_root, max_samples=None, verbose=True):
    """Load and extract features from all h5 files
    
    Returns:
        ids: list of str, format "subset/filename.h5"
        features: (N, 32) float32 array
        metadata: list of dict with {id, file_path, subset, seq_len}
    """
    ids, features, metadata = [], [], []
    
    # Walk through all subdirectories
    subdirs = sorted([d for d in os.listdir(data_root) 
                      if os.path.isdir(os.path.join(data_root, d))])
    
    pbar = tqdm(subdirs, desc="Loading subsets") if verbose else subdirs
    for subset in pbar:
        subset_dir = os.path.join(data_root, subset)
        files = sorted([f for f in os.listdir(subset_dir) if f.endswith('.h5')])
        
        for filename in files:
            if max_samples and len(ids) >= max_samples:
                break
                
            file_path = os.path.join(subset_dir, filename)
            try:
                with h5py.File(file_path, 'r') as fp:
                    vec = fp['vec'][:]  # (seq_len, 33)
                
                feat = extract_feature(vec)
                item_id = f"{subset}/{filename}"
                
                ids.append(item_id)
                features.append(feat)
                metadata.append({
                    'id': item_id,
                    'file_path': file_path,
                    'subset': subset,
                    'seq_len': int(vec.shape[0])
                })
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if max_samples and len(ids) >= max_samples:
            break
    
    features = np.vstack(features)
    
    if verbose:
        print(f"Loaded {len(ids)} vectors, feature shape: {features.shape}")
    
    return ids, features, metadata


def build_faiss_index(features, index_type="HNSW", normalize=True):
    """Build FAISS index
    
    Args:
        features: (N, d) float32 array
        index_type: "HNSW", "IVF", or "IVFPQ"
        normalize: if True, L2 normalize (for cosine similarity)
    
    Returns:
        index: faiss index
    """
    N, d = features.shape
    
    if normalize:
        faiss.normalize_L2(features)
    
    if index_type == "HNSW":
        index = faiss.IndexHNSWFlat(d, HNSW_M)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = HNSW_EF_SEARCH
        print(f"Building HNSW index: M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}")
        
    elif index_type == "IVF":
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, IVF_NLIST)
        print(f"Training IVF index: nlist={IVF_NLIST}")
        index.train(features)
        index.nprobe = IVF_NPROBE
        
    elif index_type == "IVFPQ":
        quantizer = faiss.IndexFlatL2(d)
        m = 8  # number of subquantizers
        bits = 8  # bits per subquantizer
        index = faiss.IndexIVFPQ(quantizer, d, IVF_NLIST, m, bits)
        print(f"Training IVFPQ index: nlist={IVF_NLIST}, m={m}, bits={bits}")
        index.train(features)
        index.nprobe = IVF_NPROBE
        
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    print(f"Adding {N} vectors to index...")
    index.add(features)
    print(f"Index built successfully. Total vectors: {index.ntotal}")
    
    return index


def save_index_and_metadata(index, ids, metadata, output_dir):
    """Save index and ID mapping"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    index_path = os.path.join(output_dir, 'faiss_index.bin')
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")
    
    # Save ID mapping
    id_map_path = os.path.join(output_dir, 'id_map.json')
    with open(id_map_path, 'w') as f:
        json.dump({'ids': ids}, f)
    print(f"Saved ID mapping to {id_map_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    config_data = {
        'index_type': INDEX_TYPE,
        'feature_dim': FEATURE_DIM,
        'normalized': USE_NORMALIZATION,
        'hnsw_m': HNSW_M,
        'hnsw_ef_construction': HNSW_EF_CONSTRUCTION,
        'hnsw_ef_search': HNSW_EF_SEARCH,
        'ivf_nlist': IVF_NLIST,
        'ivf_nprobe': IVF_NPROBE,
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config to {config_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build FAISS index from WHUCAD vec data')
    parser.add_argument('--data_root', default=WHUCAD_DATA_ROOT, help='WHUCAD data/vec directory')
    parser.add_argument('--output_dir', default=INDEX_DIR, help='Output directory for index')
    parser.add_argument('--index_type', default=INDEX_TYPE, choices=['HNSW', 'IVF', 'IVFPQ'])
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to load (for testing)')
    parser.add_argument('--no_normalize', action='store_true', help='Do not normalize features')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CAD Vector Database - Ingestion & Indexing")
    print("=" * 60)
    
    # Load data
    print(f"\n[1/3] Loading vectors from {args.data_root}")
    ids, features, metadata = load_vectors(
        args.data_root, 
        max_samples=args.max_samples,
        verbose=True
    )
    
    # Build index
    print(f"\n[2/3] Building {args.index_type} index")
    index = build_faiss_index(
        features,
        index_type=args.index_type,
        normalize=not args.no_normalize
    )
    
    # Save
    print(f"\n[3/3] Saving index and metadata")
    save_index_and_metadata(index, ids, metadata, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✓ Ingestion complete!")
    print(f"  Total vectors: {len(ids)}")
    print(f"  Index saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
