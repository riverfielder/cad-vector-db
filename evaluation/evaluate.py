"""Evaluation script for CAD Vector Database"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *
from scripts.retrieval import load_index_and_metadata, two_stage_search, load_macro_vec
from scripts.build_index import extract_feature


def precision_at_k(retrieved, relevant, k):
    """Precision@K"""
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / k


def recall_at_k(retrieved, relevant, k):
    """Recall@K"""
    if len(relevant) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / len(relevant)


def average_precision(retrieved, relevant):
    """Average Precision"""
    if len(relevant) == 0:
        return 0.0
    
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(retrieved):
        if item in relevant:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / len(relevant)


def evaluate_retrieval(index, ids, metadata, queries, ground_truth,
                        k_values=[1, 5, 10, 20], fusion_method="weighted"):
    """Evaluate retrieval performance
    
    Args:
        index, ids, metadata: loaded index and metadata
        queries: list of (query_id, query_file_path) tuples
        ground_truth: dict {query_id: [relevant_ids]}
        k_values: list of K values for metrics
        fusion_method: fusion method
    
    Returns:
        metrics: dict with precision@k, recall@k, map, latencies
    """
    all_precisions = {k: [] for k in k_values}
    all_recalls = {k: [] for k in k_values}
    all_aps = []
    latencies = []
    
    print(f"Evaluating {len(queries)} queries...")
    
    for query_id, query_file in tqdm(queries):
        # Load query
        query_vec = load_macro_vec(query_file)
        query_feat = extract_feature(query_vec)
        
        # Search with timing
        start_time = time.time()
        results = two_stage_search(
            query_feat, query_file, index, ids, metadata,
            stage1_topn=STAGE1_TOPN, stage2_topk=max(k_values),
            fusion_method=fusion_method, alpha=FUSION_ALPHA, beta=FUSION_BETA
        )
        latency = (time.time() - start_time) * 1000  # ms
        latencies.append(latency)
        
        # Extract retrieved IDs
        retrieved = [r['id'] for r in results]
        
        # Get ground truth
        relevant = ground_truth.get(query_id, [])
        if len(relevant) == 0:
            continue
        
        # Compute metrics
        for k in k_values:
            all_precisions[k].append(precision_at_k(retrieved, relevant, k))
            all_recalls[k].append(recall_at_k(retrieved, relevant, k))
        
        all_aps.append(average_precision(retrieved, relevant))
    
    # Aggregate
    metrics = {}
    for k in k_values:
        metrics[f'precision@{k}'] = np.mean(all_precisions[k])
        metrics[f'recall@{k}'] = np.mean(all_recalls[k])
    
    metrics['map'] = np.mean(all_aps)
    metrics['latency_p50'] = np.percentile(latencies, 50)
    metrics['latency_p95'] = np.percentile(latencies, 95)
    metrics['latency_mean'] = np.mean(latencies)
    
    return metrics


def load_split_data(split_file):
    """Load train/val/test split from JSON
    
    Returns:
        split_data: dict with 'train', 'validation', 'test' keys
    """
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    return split_data


def build_ground_truth_by_subset(metadata):
    """Build weak ground truth: same subset = relevant
    
    Returns:
        ground_truth: dict {id: [relevant_ids]}
    """
    subset_groups = {}
    for meta in metadata:
        subset = meta['subset']
        if subset not in subset_groups:
            subset_groups[subset] = []
        subset_groups[subset].append(meta['id'])
    
    ground_truth = {}
    for meta in metadata:
        item_id = meta['id']
        subset = meta['subset']
        # Relevant = same subset, excluding self
        relevant = [x for x in subset_groups[subset] if x != item_id]
        ground_truth[item_id] = relevant
    
    return ground_truth


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CAD Vector Database')
    parser.add_argument('--index_dir', default=INDEX_DIR, help='Index directory')
    parser.add_argument('--split_file', default=None, 
                        help='Train/val/test split JSON (optional)')
    parser.add_argument('--n_queries', type=int, default=100, 
                        help='Number of test queries')
    parser.add_argument('--fusion', default=FUSION_METHOD, 
                        choices=['weighted', 'rrf', 'borda'])
    parser.add_argument('--output', default='evaluation/results.json', 
                        help='Output results JSON')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CAD Vector Database - Evaluation")
    print("=" * 60)
    
    # Load index
    print("\nLoading index...")
    index, ids, metadata = load_index_and_metadata(args.index_dir)
    print(f"Loaded {len(ids)} vectors")
    
    # Build ground truth (weak: same subset)
    print("\nBuilding ground truth (subset-based)...")
    ground_truth = build_ground_truth_by_subset(metadata)
    
    # Sample queries from test set
    print(f"\nSampling {args.n_queries} queries...")
    if args.split_file and os.path.exists(args.split_file):
        split_data = load_split_data(args.split_file)
        test_ids = split_data.get('test', [])
        # Convert to full IDs (subset/filename.h5)
        test_meta = [m for m in metadata if m['id'].split('/')[1].replace('.h5', '') in test_ids]
    else:
        test_meta = metadata
    
    np.random.seed(42)
    query_indices = np.random.choice(len(test_meta), min(args.n_queries, len(test_meta)), replace=False)
    queries = [(test_meta[i]['id'], test_meta[i]['file_path']) for i in query_indices]
    
    # Evaluate
    print(f"\nEvaluating with fusion={args.fusion}...")
    metrics = evaluate_retrieval(
        index, ids, metadata, queries, ground_truth,
        k_values=EVAL_K_VALUES, fusion_method=args.fusion
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for metric, value in sorted(metrics.items()):
        print(f"  {metric:20s}: {value:.4f}")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved results to {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
