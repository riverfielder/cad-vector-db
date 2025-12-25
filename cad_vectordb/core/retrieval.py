"""Two-stage retrieval with reranking and fusion

This module provides the core retrieval functionality including:
- Stage 1: Fast ANN search with FAISS
- Stage 2: Fine-grained macro distance reranking  
- Multiple fusion methods (weighted, RRF, Borda)
- Hybrid search with metadata filtering
- Explainable retrieval with similarity breakdown
- Semantic query support (text-to-CAD search)
"""
import h5py
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
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
    
    def __init__(self, index_or_manager, ids: Optional[List[str]] = None, metadata: Optional[List[Dict]] = None):
        """Initialize retrieval system
        
        Args:
            index_or_manager: Either IndexManager object or FAISS index
            ids: List of IDs (required if index_or_manager is FAISS index)
            metadata: List of metadata dicts (required if index_or_manager is FAISS index)
        """
        # Support both IndexManager and raw FAISS index
        if hasattr(index_or_manager, 'index'):  # It's an IndexManager
            self.index = index_or_manager.index
            self.ids = index_or_manager.ids
            self.metadata = index_or_manager.metadata
        else:  # It's a raw FAISS index
            if ids is None or metadata is None:
                raise ValueError("ids and metadata are required when passing raw FAISS index")
            self.index = index_or_manager
            self.ids = ids
            self.metadata = metadata
        
        self.id_to_meta = {m['id']: m for m in self.metadata}
    
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
    
    def semantic_search(self,
                       query_text: str,
                       text_encoder,
                       k: int = 20,
                       fusion_method: str = "cosine",
                       filters: Optional[Dict] = None,
                       explainable: bool = False) -> Union[List[Dict], Tuple[List[Dict], Dict]]:
        """Semantic search using text queries
        
        This method enables natural language search over CAD models. The text query
        is encoded into a vector and matched against the indexed features.
        
        Args:
            query_text: Natural language query (e.g., "圆柱形零件", "cylindrical part")
            text_encoder: TextEncoder instance for encoding query
            k: Number of results to return
            fusion_method: "cosine" (recommended) or "l2"
            filters: Optional metadata filters {"subset": "train", ...}
            explainable: Return detailed explanation of top result
            
        Returns:
            results: List of result dicts with scores and metadata
            explanation (optional): Explanation dict if explainable=True
            
        Examples:
            >>> from cad_vectordb.core.text_encoder import create_text_encoder
            >>> encoder = create_text_encoder('sentence-transformer')
            >>> results = retrieval.semantic_search("找一个圆柱形零件", encoder, k=10)
            >>> print(f"Top match: {results[0]['id']}, score: {results[0]['score']:.3f}")
        """
        # Encode query text to vector
        query_embedding = text_encoder.encode(query_text, is_query=True)
        
        # Ensure correct dimensionality
        if query_embedding.shape[0] != self.index.d:
            raise ValueError(
                f"Text encoder dimension ({query_embedding.shape[0]}) "
                f"does not match index dimension ({self.index.d}). "
                f"You may need to train a projection layer or use a compatible encoder."
            )
        
        # Search with FAISS
        topn = min(k * 10, len(self.ids))  # Over-sample for filtering
        D, I = self.index.search(query_embedding.reshape(1, -1), topn)
        
        # Convert distances to similarities
        if fusion_method == "cosine":
            # FAISS returns L2 distance, convert to cosine similarity
            # similarity = 1 / (1 + distance)
            similarities = 1.0 / (1.0 + D[0])
        elif fusion_method == "l2":
            # Invert L2 distance
            similarities = 1.0 / (1.0 + D[0])
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Collect results
        results = []
        for idx, score in zip(I[0], similarities):
            if idx < 0 or idx >= len(self.ids):
                continue
            
            cad_id = self.ids[idx]
            meta = self.id_to_meta.get(cad_id, {})
            
            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if meta.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            results.append({
                'id': cad_id,
                'score': float(score),
                'metadata': meta,
                'query_text': query_text,
                'search_type': 'semantic'
            })
        
        # Sort by score and limit to k
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:k]
        
        if not explainable:
            return results
        
        # Generate explanation for top result
        if results:
            explanation = self._generate_semantic_explanation(
                query_text=query_text,
                top_result=results[0],
                encoder=text_encoder
            )
            return results, explanation
        else:
            return results, {'message': 'No results found'}
    
    def _generate_semantic_explanation(self, query_text: str, top_result: Dict, encoder) -> Dict:
        """Generate explanation for semantic search result"""
        explanation = {
            'query_text': query_text,
            'top_match': {
                'id': top_result['id'],
                'score': top_result['score'],
                'subset': top_result['metadata'].get('subset', 'unknown'),
                'seq_len': top_result['metadata'].get('seq_len', 0)
            },
            'search_type': 'semantic',
            'encoder_type': type(encoder).__name__,
        }
        
        # Add score interpretation
        score = top_result['score']
        if score > 0.8:
            explanation['interpretation'] = 'Excellent semantic match'
        elif score > 0.6:
            explanation['interpretation'] = 'Good semantic match'
        elif score > 0.4:
            explanation['interpretation'] = 'Moderate semantic match'
        else:
            explanation['interpretation'] = 'Weak semantic match'
        
        # Add recommendation
        if score < 0.5:
            explanation['recommendation'] = (
                'Low score suggests the text query may not align well with '
                'the CAD feature space. Try more descriptive or specific queries.'
            )
        
        return explanation
    
    def hybrid_search(self,
                     query_text: str,
                     text_encoder,
                     query_vec: Optional[np.ndarray] = None,
                     query_file_path: Optional[str] = None,
                     k: int = 20,
                     semantic_weight: float = 0.5,
                     vector_weight: float = 0.5,
                     filters: Optional[Dict] = None) -> List[Dict]:
        """Hybrid search combining semantic and vector-based retrieval
        
        Performs both semantic (text-based) and vector-based search, then fuses
        the results using weighted score combination.
        
        Args:
            query_text: Natural language query
            text_encoder: TextEncoder instance
            query_vec: Optional CAD vector for vector-based search
            query_file_path: Optional path to query h5 file
            k: Number of results
            semantic_weight: Weight for semantic scores
            vector_weight: Weight for vector-based scores
            filters: Optional metadata filters
            
        Returns:
            results: Fused results combining both search modes
            
        Examples:
            >>> # Search with both text and CAD vector
            >>> results = retrieval.hybrid_search(
            ...     query_text="cylindrical part",
            ...     text_encoder=encoder,
            ...     query_vec=my_cad_vector,
            ...     query_file_path="query.h5",
            ...     k=10
            ... )
        """
        # Get semantic search results
        semantic_results = self.semantic_search(
            query_text=query_text,
            text_encoder=text_encoder,
            k=k * 2,  # Over-sample
            filters=filters
        )
        
        # Get vector search results if provided
        if query_vec is not None and query_file_path is not None:
            vector_results = self.search(
                query_vec=query_vec,
                query_file_path=query_file_path,
                k=k * 2,
                filters=filters
            )
        else:
            vector_results = []
        
        # Fuse results
        score_map = {}
        
        # Add semantic scores
        for res in semantic_results:
            cad_id = res['id']
            score_map[cad_id] = {
                'semantic_score': res['score'] * semantic_weight,
                'vector_score': 0.0,
                'metadata': res['metadata']
            }
        
        # Add vector scores
        for res in vector_results:
            cad_id = res['id']
            if cad_id not in score_map:
                score_map[cad_id] = {
                    'semantic_score': 0.0,
                    'vector_score': 0.0,
                    'metadata': res['metadata']
                }
            score_map[cad_id]['vector_score'] = res['score'] * vector_weight
        
        # Combine scores
        fused_results = []
        for cad_id, scores in score_map.items():
            fused_score = scores['semantic_score'] + scores['vector_score']
            fused_results.append({
                'id': cad_id,
                'score': fused_score,
                'semantic_score': scores['semantic_score'],
                'vector_score': scores['vector_score'],
                'metadata': scores['metadata'],
                'search_type': 'hybrid'
            })
        
        # Sort and return top k
        fused_results = sorted(fused_results, key=lambda x: x['score'], reverse=True)[:k]
        return fused_results
