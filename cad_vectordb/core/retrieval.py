"""Two-stage retrieval with reranking and fusion

This module provides the core retrieval functionality including:
- Stage 1: Fast ANN search with FAISS
- Stage 2: Fine-grained macro distance reranking  
- Multiple fusion methods (weighted, RRF, Borda)
- Hybrid search with metadata filtering
- Explainable retrieval with similarity breakdown
"""
import h5py
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def macro_distance(vec_a, vec_b, max_len=256, return_details=False):
    """Calculate fine-grained distance between two macro vectors
    
    Args:
        vec_a, vec_b: (seq_len, 33) arrays
        max_len: max sequence length to compare
        return_details: if True, return detailed breakdown
    
    Returns:
        distance: float, lower is more similar
        details (optional): dict with breakdown if return_details=True
    """
    # Pad or truncate to same length
    L = min(vec_a.shape[0], vec_b.shape[0], max_len)
    original_len_a = vec_a.shape[0]
    original_len_b = vec_b.shape[0]
    
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
    cmd_matches = np.sum(vec_a[:, 0] == vec_b[:, 0])
    cmd_mismatches = L - cmd_matches
    cmd_penalty = float(cmd_mismatches)
    
    # Parameter L2 distance (col 1-32)
    param_diff = vec_a[:, 1:] - vec_b[:, 1:]
    param_l2 = np.linalg.norm(param_diff)
    param_l2_per_step = np.linalg.norm(param_diff, axis=1)
    
    # Normalize by sequence length
    distance = cmd_penalty + param_l2 / np.sqrt(L)
    
    if return_details:
        details = {
            'total_distance': float(distance),
            'cmd_penalty': float(cmd_penalty),
            'param_l2': float(param_l2),
            'normalized_param_l2': float(param_l2 / np.sqrt(L)),
            'sequence_length': int(L),
            'query_seq_len': int(original_len_a),
            'candidate_seq_len': int(original_len_b),
            'cmd_matches': int(cmd_matches),
            'cmd_mismatches': int(cmd_mismatches),
            'cmd_match_rate': float(cmd_matches / L) if L > 0 else 0.0,
            'avg_param_distance_per_step': float(np.mean(param_l2_per_step)),
            'max_param_distance_per_step': float(np.max(param_l2_per_step)),
        }
        return distance, details
    
    return distance


def minmax_normalize(scores):
    """Min-max normalization to [0, 1]"""
    scores = np.array(scores)
    min_val, max_val = scores.min(), scores.max()
    if max_val - min_val < 1e-12:
        return np.ones_like(scores)
    return (scores - min_val) / (max_val - min_val)


class TwoStageRetrieval:
    """Two-stage retrieval system with fusion and filtering"""
    
    def __init__(self, index, ids: List[str], metadata: List[Dict]):
        """Initialize retrieval system
        
        Args:
            index: FAISS index
            ids: List of IDs
            metadata: List of metadata dicts
        """
        self.index = index
        self.ids = ids
        self.metadata = metadata
        self.id_to_meta = {m['id']: m for m in metadata}
    
    def search(self,
              query_vec,
              query_file_path: str,
              k: int = 20,
              stage1_topn: int = 100,
              fusion_method: str = "weighted",
              alpha: float = 0.6,
              beta: float = 0.4,
              rrf_k: int = 60,
              filters: Optional[Dict] = None,
              explainable: bool = False) -> List[Dict]:
        """Two-stage search with fusion
        
        Args:
            query_vec: Query macro vector (seq_len, 33)
            query_file_path: Path to query h5 file
            k: Number of final results
            stage1_topn: Stage 1 candidate count
            fusion_method: "weighted", "rrf", or "borda"
            alpha, beta: Weights for weighted fusion
            rrf_k: Constant for RRF
            filters: Optional metadata filters
            explainable: Return detailed explanations
            
        Returns:
            results: List of result dicts
        """
        from .feature import extract_feature
        
        # Extract query feature for stage 1
        query_feat = extract_feature(query_vec)
        
        # Stage 1: ANN search
        stage1_topn = min(stage1_topn, len(self.ids))
        D, I = self.index.search(query_feat.reshape(1, -1), stage1_topn)
        
        # Get candidates
        candidates = []
        for idx, dist in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            
            candidate_id = self.ids[idx]
            candidate_meta = self.id_to_meta[candidate_id]
            
            # Apply filters
            if filters and not self._match_filters(candidate_meta, filters):
                continue
            
            candidates.append({
                'id': candidate_id,
                'stage1_dist': float(dist),
                'metadata': candidate_meta
            })
        
        if len(candidates) == 0:
            return []
        
        # Stage 2: Reranking with macro distance
        for cand in candidates:
            cand_vec = self._load_macro_vec(cand['metadata']['file_path'])
            cand['stage2_dist'] = macro_distance(query_vec, cand_vec)
        
        # Normalize distances to similarities [0, 1]
        stage1_dists = [c['stage1_dist'] for c in candidates]
        stage2_dists = [c['stage2_dist'] for c in candidates]
        
        stage1_sims = 1.0 - minmax_normalize(stage1_dists)
        stage2_sims = 1.0 - minmax_normalize(stage2_dists)
        
        # Fusion
        if fusion_method == "weighted":
            fused_scores = alpha * stage1_sims + beta * stage2_sims
        elif fusion_method == "rrf":
            fused_scores = self._fusion_rrf(stage1_sims, stage2_sims, rrf_k)
        elif fusion_method == "borda":
            fused_scores = self._fusion_borda(stage1_sims, stage2_sims)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Add scores to candidates
        for i, cand in enumerate(candidates):
            cand['stage1_sim'] = float(stage1_sims[i])
            cand['stage2_sim'] = float(stage2_sims[i])
            cand['score'] = float(fused_scores[i])
        
        # Sort by fused score and take top-k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        results = candidates[:k]
        
        # Add explanations if requested
        if explainable and len(results) > 0:
            explanation = self._generate_explanation(
                results[0], fusion_method, alpha, beta
            )
            return results, explanation
        
        return results
    
    def _match_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches filters"""
        if 'subset' in filters:
            subsets = filters['subset'] if isinstance(filters['subset'], list) else [filters['subset']]
            if metadata['subset'] not in subsets:
                return False
        
        if 'min_seq_len' in filters:
            if metadata['seq_len'] < filters['min_seq_len']:
                return False
        
        if 'max_seq_len' in filters:
            if metadata['seq_len'] > filters['max_seq_len']:
                return False
        
        return True
    
    def _load_macro_vec(self, file_path: str):
        """Load macro vector from file"""
        with h5py.File(file_path, 'r') as f:
            vec = f['vec'][:]
        return vec.astype('float32')
    
    def _fusion_rrf(self, stage1_sims, stage2_sims, k=60):
        """Reciprocal Rank Fusion"""
        # Convert similarities to ranks
        stage1_ranks = np.argsort(-stage1_sims)
        stage2_ranks = np.argsort(-stage2_sims)
        
        scores = np.zeros(len(stage1_sims))
        for rank, idx in enumerate(stage1_ranks):
            scores[idx] += 1.0 / (k + rank + 1)
        for rank, idx in enumerate(stage2_ranks):
            scores[idx] += 1.0 / (k + rank + 1)
        
        return scores
    
    def _fusion_borda(self, stage1_sims, stage2_sims):
        """Borda count fusion"""
        n = len(stage1_sims)
        stage1_ranks = np.argsort(-stage1_sims)
        stage2_ranks = np.argsort(-stage2_sims)
        
        scores = np.zeros(n)
        for rank, idx in enumerate(stage1_ranks):
            scores[idx] += (n - rank)
        for rank, idx in enumerate(stage2_ranks):
            scores[idx] += (n - rank)
        
        return scores / (2 * n)  # Normalize
    
    def _generate_explanation(self, top_result: Dict, fusion_method: str, alpha: float, beta: float) -> Dict:
        """Generate explanation for top result"""
        stage1_sim = top_result['stage1_sim']
        stage2_sim = top_result['stage2_sim']
        fused_score = top_result['score']
        
        explanation = {
            'top_match': {
                'id': top_result['id'],
                'score': fused_score,
                'subset': top_result['metadata']['subset'],
                'seq_len': top_result['metadata']['seq_len']
            },
            'fusion_method': fusion_method,
            'stage1_similarity': stage1_sim,
            'stage2_similarity': stage2_sim,
            'final_score': fused_score,
        }
        
        if fusion_method == 'weighted':
            contrib1 = alpha * stage1_sim
            contrib2 = beta * stage2_sim
            explanation['contributions'] = {
                'stage1_weight': alpha,
                'stage2_weight': beta,
                'stage1_contribution': contrib1,
                'stage2_contribution': contrib2,
                'stage1_percentage': (contrib1 / fused_score * 100) if fused_score > 0 else 0,
                'stage2_percentage': (contrib2 / fused_score * 100) if fused_score > 0 else 0,
            }
        
        # Add interpretations
        if stage1_sim > 0.9:
            explanation['stage1_interpretation'] = 'Excellent feature-level match'
        elif stage1_sim > 0.7:
            explanation['stage1_interpretation'] = 'Good feature-level match'
        else:
            explanation['stage1_interpretation'] = 'Moderate feature-level match'
        
        if stage2_sim > 0.9:
            explanation['stage2_interpretation'] = 'Excellent sequence-level match'
        elif stage2_sim > 0.7:
            explanation['stage2_interpretation'] = 'Good sequence-level match'
        else:
            explanation['stage2_interpretation'] = 'Moderate sequence-level match'
        
        return explanation
