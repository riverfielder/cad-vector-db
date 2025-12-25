#!/usr/bin/env python
"""Build FAISS index from WHUCAD dataset

This script builds a FAISS index from the WHUCAD CAD vector dataset.
It supports different index types (HNSW/IVF/IVFPQ) and can build partial
indices for testing.

Usage:
    # Quick test (500 samples)
    python scripts/build_index.py --max_samples 500 --output_dir data/index_test

    # Full index
    python scripts/build_index.py --output_dir data/index_full

    # Custom index type
    python scripts/build_index.py --index_type IVF --output_dir data/index_ivf
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_vectordb.core.feature import extract_feature, load_macro_vec
import config as default_config


def collect_vector_files(data_root: str, max_samples: int = None) -> List[Tuple[str, str]]:
    """Collect all vector files from WHUCAD dataset
    
    Args:
        data_root: Root directory of WHUCAD dataset
        max_samples: Maximum number of samples (None for all)
        
    Returns:
        List of (vector_id, file_path) tuples
    """
    data_path = Path(data_root)
    if not data_path.exists():
        raise ValueError(f"Data root does not exist: {data_root}")
    
    vector_files = []
    
    # Iterate through subdirectories (0000, 0001, etc.)
    for subset_dir in sorted(data_path.glob("*")):
        if not subset_dir.is_dir():
            continue
        
        # Collect H5 files in this subset
        for h5_file in sorted(subset_dir.glob("*.h5")):
            # Create vector ID: subset/filename
            vector_id = f"{subset_dir.name}/{h5_file.name}"
            vector_files.append((vector_id, str(h5_file)))
            
            if max_samples and len(vector_files) >= max_samples:
                return vector_files
    
    return vector_files


def extract_features_batch(vector_files: List[Tuple[str, str]], 
                           batch_size: int = 100) -> Tuple[np.ndarray, List[str]]:
    """Extract features from vector files in batches
    
    Args:
        vector_files: List of (vector_id, file_path) tuples
        batch_size: Batch size for progress reporting
        
    Returns:
        features: (N, D) float32 array
        ids: List of vector IDs
    """
    features_list = []
    ids_list = []
    failed = 0
    
    print(f"Extracting features from {len(vector_files)} files...")
    
    for i, (vector_id, file_path) in enumerate(vector_files):
        try:
            vec = load_macro_vec(file_path)
            if vec is not None:
                feature = extract_feature(vec)
                if feature is not None:
                    features_list.append(feature)
                    ids_list.append(vector_id)
                else:
                    failed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Failed to extract feature from {vector_id}: {e}")
            failed += 1
        
        # Progress reporting
        if (i + 1) % batch_size == 0:
            print(f"  Processed {i+1}/{len(vector_files)} files, {failed} failed")
    
    print(f"✓ Extracted {len(features_list)} features ({failed} failures)")
    
    if len(features_list) == 0:
        raise ValueError("No features extracted! Check data path and file format.")
    
    features = np.vstack(features_list).astype('float32')
    return features, ids_list


def build_faiss_index(features: np.ndarray, 
                      index_type: str,
                      normalize: bool = True,
                      hnsw_m: int = 32,
                      hnsw_ef_construction: int = 200,
                      ivf_nlist: int = 100) -> faiss.Index:
    """Build FAISS index
    
    Args:
        features: (N, D) float32 array
        index_type: 'HNSW', 'IVF', or 'IVFPQ'
        normalize: Whether to L2 normalize features
        hnsw_m: HNSW parameter
        hnsw_ef_construction: HNSW parameter
        ivf_nlist: IVF nlist parameter
        
    Returns:
        FAISS index
    """
    n, dim = features.shape
    print(f"\nBuilding {index_type} index for {n} vectors (dim={dim})...")
    
    # Normalize if requested
    if normalize:
        print("  Normalizing features...")
        faiss.normalize_L2(features)
    
    # Build index based on type
    if index_type == "HNSW":
        print(f"  HNSW parameters: M={hnsw_m}, ef_construction={hnsw_ef_construction}")
        index = faiss.IndexHNSWFlat(dim, hnsw_m)
        index.hnsw.efConstruction = hnsw_ef_construction
        
    elif index_type == "IVF":
        print(f"  IVF parameters: nlist={ivf_nlist}")
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, ivf_nlist)
        print("  Training IVF index...")
        index.train(features)
        
    elif index_type == "IVFPQ":
        print(f"  IVFPQ parameters: nlist={ivf_nlist}")
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, ivf_nlist, 8, 8)
        print("  Training IVFPQ index...")
        index.train(features)
        
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Add vectors
    print("  Adding vectors to index...")
    start = time.time()
    index.add(features)
    elapsed = time.time() - start
    print(f"  ✓ Added {index.ntotal} vectors in {elapsed:.2f}s")
    
    return index


def save_index(index: faiss.Index, 
               ids: List[str],
               output_dir: str,
               index_type: str,
               feature_dim: int,
               normalized: bool,
               **params):
    """Save index and metadata
    
    Args:
        index: FAISS index
        ids: List of vector IDs
        output_dir: Output directory
        index_type: Index type string
        feature_dim: Feature dimension
        normalized: Whether features are normalized
        **params: Additional index parameters
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving index to {output_dir}...")
    
    # Save FAISS index
    index_file = output_path / "index.faiss"
    faiss.write_index(index, str(index_file))
    print(f"  ✓ Saved FAISS index: {index_file}")
    
    # Save IDs
    ids_file = output_path / "ids.json"
    with open(ids_file, 'w') as f:
        json.dump(ids, f, indent=2)
    print(f"  ✓ Saved {len(ids)} IDs: {ids_file}")
    
    # Save config
    config = {
        "index_type": index_type,
        "feature_dim": feature_dim,
        "num_vectors": len(ids),
        "normalized": normalized,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **params
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config: {config_file}")
    
    print(f"\n✅ Index built successfully!")
    print(f"   Location: {output_dir}")
    print(f"   Vectors: {len(ids)}")
    print(f"   Type: {index_type}")
    print(f"   Dimension: {feature_dim}")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from WHUCAD dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 500 samples
  python scripts/build_index.py --max_samples 500 --output_dir data/index_test

  # Full index with default settings
  python scripts/build_index.py --output_dir data/index_full

  # Custom index type and parameters
  python scripts/build_index.py \\
      --index_type IVF \\
      --ivf_nlist 200 \\
      --output_dir data/index_ivf

  # HNSW index with custom parameters
  python scripts/build_index.py \\
      --index_type HNSW \\
      --hnsw_m 64 \\
      --hnsw_ef_construction 400 \\
      --output_dir data/index_hnsw
        """
    )
    
    # Data options
    parser.add_argument(
        '--data_root',
        default=default_config.WHUCAD_DATA_ROOT,
        help=f'Root directory of WHUCAD dataset (default: {default_config.WHUCAD_DATA_ROOT})'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to index (None for all)'
    )
    
    # Index options
    parser.add_argument(
        '--index_type',
        choices=['HNSW', 'IVF', 'IVFPQ'],
        default=default_config.INDEX_TYPE,
        help=f'Type of FAISS index (default: {default_config.INDEX_TYPE})'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=default_config.USE_NORMALIZATION,
        help='L2 normalize features (default: True for cosine similarity)'
    )
    
    # HNSW parameters
    parser.add_argument(
        '--hnsw_m',
        type=int,
        default=default_config.HNSW_M,
        help=f'HNSW M parameter (default: {default_config.HNSW_M})'
    )
    parser.add_argument(
        '--hnsw_ef_construction',
        type=int,
        default=default_config.HNSW_EF_CONSTRUCTION,
        help=f'HNSW ef_construction parameter (default: {default_config.HNSW_EF_CONSTRUCTION})'
    )
    
    # IVF parameters
    parser.add_argument(
        '--ivf_nlist',
        type=int,
        default=default_config.IVF_NLIST,
        help=f'IVF nlist parameter (default: {default_config.IVF_NLIST})'
    )
    
    # Output
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for index files'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("CAD Vector Database - Index Builder")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print(f"Index type: {args.index_type}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)
    
    # Step 1: Collect vector files
    print("\n[1/4] Collecting vector files...")
    vector_files = collect_vector_files(args.data_root, args.max_samples)
    print(f"  Found {len(vector_files)} files")
    
    # Step 2: Extract features
    print("\n[2/4] Extracting features...")
    features, ids = extract_features_batch(vector_files)
    
    # Step 3: Build index
    print("\n[3/4] Building FAISS index...")
    index = build_faiss_index(
        features,
        args.index_type,
        args.normalize,
        args.hnsw_m,
        args.hnsw_ef_construction,
        args.ivf_nlist
    )
    
    # Step 4: Save index
    print("\n[4/4] Saving index...")
    save_index(
        index,
        ids,
        args.output_dir,
        args.index_type,
        default_config.FEATURE_DIM,
        args.normalize,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        ivf_nlist=args.ivf_nlist
    )


if __name__ == "__main__":
    main()
