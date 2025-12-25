"""Two-stage retrieval with reranking and fusion"""
import os
import sys
import json
import h5py
import numpy as np
import faiss
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


def macro_distance(vec_a, vec_b, max_len=256):
    """Calculate fine-grained distance between two macro vectors
    
    Args:
        vec_a, vec_b: (seq_len, 33) arrays
        max_len: max sequence length to compare
    
    Returns:
        distance: float, lower is more similar
    """
    # Pad or truncate to same length
    L = min(vec_a.shape[0], vec_b.shape[0], max_len)
    
    if vec_a.shape[0] < L:
        pad_a = np.zeros((L - vec_a.shape[0], vec_a.shape[1]), dtype=vec_a.dtype)
        vec_a = np.vstack([vec_a, pad_a])
    else:
        vec_a = vec_a[:L]
    
    if vec_b.shape[0] < L:
        pad_b = np.zeros((L - vec_b.shape[0], vec_b.shape[1]), dtype=vec_b.dtype)
        vec_b = np.vstack([vec_b, pad_b])
    else:
        vec_b = vec_b[:L]
    
    # Command mismatch penalty (col 0)
    cmd_penalty = np.sum(vec_a[:, 0] != vec_b[:, 0]).astype(float)
    
    # Parameter L2 distance (col 1-32)
    param_l2 = np.linalg.norm(vec_a[:, 1:] - vec_b[:, 1:])
    
    # Normalize by sequence length
    distance = cmd_penalty + param_l2 / np.sqrt(L)
    
    return distance


def load_macro_vec(file_path):
    """Load macro vector from h5 file"""
    with h5py.File(file_path, 'r') as fp:
        vec = fp['vec'][:]
    return vec


def minmax_normalize(scores):
    """Min-max normalization to [0, 1]"""
    scores = np.array(scores)
    min_val, max_val = scores.min(), scores.max()
    if max_val - min_val < 1e-12:
        return np.ones_like(scores)
    return (scores - min_val) / (max_val - min_val)


def fusion_weighted(sim_stage1, sim_stage2, alpha=0.6, beta=0.4):
    """Weighted fusion of two similarity scores"""
    return alpha * sim_stage1 + beta * sim_stage2


def fusion_rrf(ranks_stage1, ranks_stage2, k=60):
    """Reciprocal Rank Fusion
    
    Args:
        ranks_stage1, ranks_stage2: list of (id, rank) tuples
        k: constant (default 60)
    
    Returns:
        fused_scores: dict {id: score}
    """
    scores = {}
    for item_id, rank in ranks_stage1:
        scores[item_id] = scores.get(item_id, 0) + 1.0 / (k + rank)
    for item_id, rank in ranks_stage2:
        scores[item_id] = scores.get(item_id, 0) + 1.0 / (k + rank)
    return scores


def two_stage_search(query_feat, query_file_path, index, ids, metadata, 
                      stage1_topn=100, stage2_topk=20, 
                      fusion_method="weighted", alpha=0.6, beta=0.4, rrf_k=60,
                      filters=None):
    """Two-stage retrieval with fusion and optional metadata filtering
    
    Args:
        query_feat: (32,) query feature
        query_file_path: path to query h5 file (for stage2 rerank)
        index: FAISS index
        ids: list of ids corresponding to index
        metadata: list of metadata dicts
        stage1_topn: number of candidates from ANN
        stage2_topk: final top-k results
        fusion_method: "weighted", "rrf", or "borda"
        alpha, beta: weights for weighted fusion
        rrf_k: constant for RRF
        filters: dict with optional filters:
            - subset: str or list of str (e.g., "0000" or ["0000", "0001"])
            - min_seq_len: int
            - max_seq_len: int
            - label: str or list of str
    
    Returns:
        results: list of dicts with {id, score, sim_stage1, sim_stage2, metadata}
    """
    # Apply pre-filtering if filters provided
    allowed_indices = None
    if filters:
        allowed_indices = apply_metadata_filters(metadata, filters)
        if not allowed_indices:
            print("Warning: No vectors match the filters, returning empty results")
            return []
        print(f"Filtered to {len(allowed_indices)} vectors (from {len(metadata)} total)")
    
    # Stage 1: ANN recall
    query_feat = query_feat.reshape(1, -1).astype('float32')
    if USE_NORMALIZATION:
        faiss.normalize_L2(query_feat)
    
    # Search with larger topn if filtering is applied
    search_topn = stage1_topn * 3 if allowed_indices else stage1_topn
    D_stage1, I_stage1 = index.search(query_feat, search_topn)
    D_stage1 = D_stage1[0]
    I_stage1 = I_stage1[0]
    
    # Apply filtering to Stage 1 results
    if allowed_indices:
        allowed_set = set(allowed_indices)
        filtered_results = [(d, idx) for d, idx in zip(D_stage1, I_stage1) if idx in allowed_set]
        
        if not filtered_results:
            print("Warning: No results match filters after Stage 1 retrieval")
            return []
        
        # Take top-N after filtering
        filtered_results = filtered_results[:stage1_topn]
        D_stage1 = np.array([d for d, _ in filtered_results])
        I_stage1 = np.array([idx for _, idx in filtered_results])
    
    # Convert distances to similarities (for cosine: 1 - dist/2; for L2: 1/(1+dist))
    if USE_NORMALIZATION:  # cosine
        sim_stage1_raw = 1.0 - D_stage1 / 2.0
    else:  # L2
        sim_stage1_raw = 1.0 / (1.0 + D_stage1)
    
    # Normalize stage1 scores
    sim_stage1_norm = minmax_normalize(sim_stage1_raw)
    
    # Stage 2: Rerank with macro vector distance
    query_vec = load_macro_vec(query_file_path)
    
    stage2_distances = []
    for idx in I_stage1:
        cand_id = ids[idx]
        cand_meta = metadata[idx]
        cand_file_path = cand_meta['file_path']
        
        try:
            cand_vec = load_macro_vec(cand_file_path)
            dist = macro_distance(query_vec, cand_vec)
            stage2_distances.append(dist)
        except Exception as e:
            print(f"Error reranking {cand_id}: {e}")
            stage2_distances.append(float('inf'))
    
    stage2_distances = np.array(stage2_distances)
    
    # Convert distances to similarities
    sim_stage2_raw = 1.0 / (1.0 + stage2_distances)
    sim_stage2_norm = minmax_normalize(sim_stage2_raw)
    
    # Fusion
    if fusion_method == "weighted":
        fused_scores = fusion_weighted(sim_stage1_norm, sim_stage2_norm, alpha, beta)
        
    elif fusion_method == "rrf":
        # Create rank lists
        ranks_stage1 = [(ids[I_stage1[i]], i) for i in range(len(I_stage1))]
        ranks_stage2 = [(ids[I_stage1[i]], i) for i in np.argsort(stage2_distances)]
        fused_scores_dict = fusion_rrf(ranks_stage1, ranks_stage2, rrf_k)
        fused_scores = np.array([fused_scores_dict.get(ids[idx], 0) for idx in I_stage1])
        
    elif fusion_method == "borda":
        # Borda count: N - rank
        N = len(I_stage1)
        borda_stage1 = np.array([N - i for i in range(N)])
        borda_stage2 = np.array([N - i for i in np.argsort(stage2_distances).argsort()])
        fused_scores = borda_stage1 + borda_stage2
        fused_scores = minmax_normalize(fused_scores)
        
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Sort by fused score and take top-k
    top_indices = np.argsort(-fused_scores)[:stage2_topk]
    
    results = []
    for i in top_indices:
        idx = I_stage1[i]
        result = {
            'id': ids[idx],
            'score': float(fused_scores[i]),
            'sim_stage1': float(sim_stage1_norm[i]),
            'sim_stage2': float(sim_stage2_norm[i]),
            'metadata': metadata[idx]
        }
        
        # Add filter match info if filters were applied
        if filters:
            result['filter_matched'] = True
        
        results.append(result)
    
    return results


def apply_metadata_filters(metadata, filters):
    """Apply metadata filters and return list of allowed indices
    
    Args:
        metadata: list of metadata dicts
        filters: dict with keys:
            - subset: str or list
            - min_seq_len: int
            - max_seq_len: int
            - label: str or list
    
    Returns:
        allowed_indices: list of indices that match filters
    """
    allowed_indices = []
    
    for idx, meta in enumerate(metadata):
        # Check subset filter
        if 'subset' in filters:
            subset_filter = filters['subset']
            if isinstance(subset_filter, str):
                subset_filter = [subset_filter]
            if meta.get('subset') not in subset_filter:
                continue
        
        # Check sequence length filters
        if 'min_seq_len' in filters:
            if meta.get('seq_len', 0) < filters['min_seq_len']:
                continue
        
        if 'max_seq_len' in filters:
            if meta.get('seq_len', float('inf')) > filters['max_seq_len']:
                continue
        
        # Check label filter
        if 'label' in filters:
            label_filter = filters['label']
            if isinstance(label_filter, str):
                label_filter = [label_filter]
            if meta.get('label') not in label_filter:
                continue
        
        # If all filters passed, add to allowed indices
        allowed_indices.append(idx)
    
    return allowed_indices


def load_index_and_metadata(index_dir):
    """Load FAISS index, ID map, and metadata"""
    index_path = os.path.join(index_dir, 'faiss_index.bin')
    id_map_path = os.path.join(index_dir, 'id_map.json')
    metadata_path = os.path.join(index_dir, 'metadata.json')
    
    index = faiss.read_index(index_path)
    
    with open(id_map_path, 'r') as f:
        ids = json.load(f)['ids']
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return index, ids, metadata


def main():
    """Test two-stage search"""
    import argparse
    from scripts.build_index import extract_feature
    
    parser = argparse.ArgumentParser(description='Test two-stage retrieval')
    parser.add_argument('--index_dir', default=INDEX_DIR, help='Index directory')
    parser.add_argument('--query_file', required=True, help='Query h5 file path')
    parser.add_argument('--topn', type=int, default=STAGE1_TOPN, help='Stage1 top-N')
    parser.add_argument('--topk', type=int, default=STAGE2_TOPK, help='Stage2 top-K')
    parser.add_argument('--fusion', default=FUSION_METHOD, choices=['weighted', 'rrf', 'borda'])
    
    args = parser.parse_args()
    
    print("Loading index...")
    index, ids, metadata = load_index_and_metadata(args.index_dir)
    
    print(f"Loading query from {args.query_file}")
    with h5py.File(args.query_file, 'r') as fp:
        query_vec = fp['vec'][:]
    query_feat = extract_feature(query_vec)
    
    print(f"Running two-stage search (topN={args.topn}, topK={args.topk}, fusion={args.fusion})")
    results = two_stage_search(
        query_feat, args.query_file, index, ids, metadata,
        stage1_topn=args.topn, stage2_topk=args.topk,
        fusion_method=args.fusion, alpha=FUSION_ALPHA, beta=FUSION_BETA, rrf_k=RRF_K
    )
    
    print(f"\nTop-{args.topk} results:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['id']}")
        print(f"   Score: {res['score']:.4f} (stage1: {res['sim_stage1']:.4f}, stage2: {res['sim_stage2']:.4f})")


if __name__ == '__main__':
    main()
