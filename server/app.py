"""FastAPI server for CAD Vector Database"""
import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *
from scripts.retrieval import (
    load_index_and_metadata, two_stage_search, load_macro_vec
)
from scripts.build_index import extract_feature

# Initialize FastAPI
app = FastAPI(title="CAD Vector Database API", version="1.0.0")

# Global state
index = None
ids = None
metadata = None


class SearchRequest(BaseModel):
    query_file_path: str
    k: int = 20
    stage1_topn: int = 100
    fusion_method: str = "weighted"
    alpha: float = 0.6
    beta: float = 0.4
    # Hybrid search filters
    filters: Optional[Dict] = None  # {"subset": "0000", "min_seq_len": 10, "max_seq_len": 100}
    # Explainable retrieval
    explainable: bool = False  # Return detailed explanations for similarity scores


class SearchResult(BaseModel):
    id: str
    score: float
    sim_stage1: float
    sim_stage2: float
    metadata: Dict


class StatsResponse(BaseModel):
    total_vectors: int
    index_type: str
    feature_dim: int
    index_params: Dict


@app.on_event("startup")
async def startup_event():
    """Load index on startup"""
    global index, ids, metadata
    print("Loading FAISS index...")
    index, ids, metadata = load_index_and_metadata(INDEX_DIR)
    print(f"Loaded {len(ids)} vectors")


@app.get("/")
async def root():
    return {
        "message": "CAD Vector Database API",
        "version": "1.0.0",
        "endpoints": ["/search", "/stats", "/vectors/{id}"]
    }


@app.post("/search")
async def search(req: SearchRequest):
    """Two-stage search with fusion and optional metadata filtering (Hybrid Search)
    
    With explainable=true, returns detailed similarity breakdown and interpretations
    """
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Validate query file
    if not os.path.exists(req.query_file_path):
        raise HTTPException(status_code=404, detail=f"Query file not found: {req.query_file_path}")
    
    try:
        # Extract query feature
        query_vec = load_macro_vec(req.query_file_path)
        query_feat = extract_feature(query_vec)
        
        # Search with optional filters (Hybrid Search)
        results = two_stage_search(
            query_feat, req.query_file_path, index, ids, metadata,
            stage1_topn=req.stage1_topn, stage2_topk=req.k,
            fusion_method=req.fusion_method, alpha=req.alpha, beta=req.beta,
            filters=req.filters,  # Enable hybrid search
            explainable=req.explainable  # Enable explainable retrieval
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics"""
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Load config
    config_path = os.path.join(INDEX_DIR, 'config.json')
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return {
        "total_vectors": len(ids),
        "index_type": config_data.get('index_type', 'unknown'),
        "feature_dim": config_data.get('feature_dim', 32),
        "index_params": {
            "hnsw_m": config_data.get('hnsw_m'),
            "hnsw_ef_search": config_data.get('hnsw_ef_search'),
            "normalized": config_data.get('normalized', True)
        }
    }


@app.get("/vectors/{vector_id:path}")
async def get_vector(vector_id: str):
    """Get vector metadata by ID"""
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Find in metadata
    for meta in metadata:
        if meta['id'] == vector_id:
            return meta
    
    raise HTTPException(status_code=404, detail=f"Vector not found: {vector_id}")


@app.post("/search/visualize")
async def search_and_visualize(req: SearchRequest):
    """Search with explanations and generate HTML visualization"""
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Validate query file
    if not os.path.exists(req.query_file_path):
        raise HTTPException(status_code=404, detail=f"Query file not found: {req.query_file_path}")
    
    try:
        # Force explainable mode
        query_feat = load_macro_vec(req.query_file_path)
        query_feat = extract_feature(query_feat)
        
        # Search with explanations
        results = two_stage_search(
            query_feat, req.query_file_path, index, ids, metadata,
            stage1_topn=req.stage1_topn, stage2_topk=req.k,
            fusion_method=req.fusion_method, alpha=req.alpha, beta=req.beta,
            filters=req.filters,
            explainable=True  # Force explainable mode
        )
        
        # Generate HTML visualization
        from scripts.visualize_explanation import generate_html_visualization
        output_file = "explanation.html"
        generate_html_visualization(results, req.query_file_path, output_file)
        
        return {
            "status": "success",
            "visualization_file": output_file,
            "num_results": len(results),
            "message": f"Visualization saved to {output_file}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
