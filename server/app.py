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
from cad_vectordb.core.index import IndexManager
from cad_vectordb.core.retrieval import TwoStageRetrieval
from cad_vectordb.core.feature import extract_feature, load_macro_vec
from cad_vectordb.core.text_encoder import create_text_encoder

# Initialize FastAPI
app = FastAPI(title="CAD Vector Database API", version="1.0.0")

# Global state
index_manager = None
retrieval_system = None
text_encoder = None  # Lazy loading for semantic search


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


class BatchSearchRequest(BaseModel):
    query_file_paths: List[str]  # List of query file paths
    k: int = 20
    stage1_topn: int = 100
    fusion_method: str = "weighted"
    alpha: float = 0.6
    beta: float = 0.4
    filters: Optional[Dict] = None
    explainable: bool = False
    parallel: bool = True  # Use parallel processing


class SemanticSearchRequest(BaseModel):
    query_text: str  # Natural language query
    k: int = 20
    encoder_type: str = "sentence-transformer"  # 'sentence-transformer', 'clip', or 'bm25'
    model_name: Optional[str] = None  # Optional specific model name
    filters: Optional[Dict] = None
    explainable: bool = False


class HybridSearchRequest(BaseModel):
    query_text: str
    query_file_path: Optional[str] = None  # Optional CAD vector for hybrid search
    k: int = 20
    semantic_weight: float = 0.5
    vector_weight: float = 0.5
    encoder_type: str = "sentence-transformer"
    model_name: Optional[str] = None
    filters: Optional[Dict] = None


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
    global index_manager, retrieval_system
    print("Loading FAISS index...")
    index_manager = IndexManager(INDEX_DIR)
    index_manager.load_index()
    retrieval_system = TwoStageRetrieval(index_manager)
    print(f"Loaded {len(index_manager.ids)} vectors")


@app.get("/")
async def root():
    return {
        "message": "CAD Vector Database API",
        "version": "1.0.0",
        "endpoints": [
            "/search",
            "/search/semantic",
            "/search/hybrid",
            "/search/batch",
            "/search/visualize",
            "/stats",
            "/vectors/{id}"
        ]
    }


@app.post("/search")
async def search(req: SearchRequest):
    """Two-stage search with fusion and optional metadata filtering (Hybrid Search)
    
    With explainable=true, returns detailed similarity breakdown and interpretations
    """
    if retrieval_system is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Validate query file
    if not os.path.exists(req.query_file_path):
        raise HTTPException(status_code=404, detail=f"Query file not found: {req.query_file_path}")
    
    try:
        # Extract query feature
        query_vec = load_macro_vec(req.query_file_path)
        query_feat = extract_feature(query_vec)
        
        # Search with optional filters (Hybrid Search)
        results = retrieval_system.search(
            query_feat, req.query_file_path,
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
    if index_manager is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Load config
    config_path = os.path.join(INDEX_DIR, 'config.json')
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return {
        "total_vectors": len(index_manager.ids),
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
    if index_manager is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Find in metadata
    for meta in index_manager.metadata:
        if meta['id'] == vector_id:
            return meta
    
    raise HTTPException(status_code=404, detail=f"Vector not found: {vector_id}")


@app.post("/search/visualize")
async def search_and_visualize(req: SearchRequest):
    """Search with explanations and generate HTML visualization"""
    if retrieval_system is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Validate query file
    if not os.path.exists(req.query_file_path):
        raise HTTPException(status_code=404, detail=f"Query file not found: {req.query_file_path}")
    
    try:
        # Force explainable mode
        query_vec = load_macro_vec(req.query_file_path)
        query_feat = extract_feature(query_vec)
        
        # Search with explanations
        results = retrieval_system.search(
            query_feat, req.query_file_path,
            stage1_topn=req.stage1_topn, stage2_topk=req.k,
            fusion_method=req.fusion_method, alpha=req.alpha, beta=req.beta,
            filters=req.filters,
            explainable=True  # Force explainable mode
        )
        
        # Generate HTML visualization
        from cad_vectordb.utils.visualization import generate_html_visualization
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


@app.post("/search/batch")
async def batch_search(req: BatchSearchRequest):
    """Batch search for multiple queries with optional parallel processing"""
    if retrieval_system is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Validate query files
    missing_files = [path for path in req.query_file_paths if not os.path.exists(path)]
    if missing_files:
        raise HTTPException(
            status_code=404, 
            detail=f"Query files not found: {missing_files[:3]}" + 
                   (f" and {len(missing_files)-3} more" if len(missing_files) > 3 else "")
        )
    
    try:
        import time
        start_time = time.time()
        
        all_results = {}
        
        if req.parallel:
            # Parallel processing using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def process_single_query(query_path):
                try:
                    query_vec = load_macro_vec(query_path)
                    query_feat = extract_feature(query_vec)
                    
                    results = retrieval_system.search(
                        query_feat, query_path,
                        stage1_topn=req.stage1_topn, stage2_topk=req.k,
                        fusion_method=req.fusion_method, alpha=req.alpha, beta=req.beta,
                        filters=req.filters,
                        explainable=req.explainable
                    )
                    return query_path, results, None
                except Exception as e:
                    return query_path, None, str(e)
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(8, len(req.query_file_paths))) as executor:
                futures = {executor.submit(process_single_query, path): path 
                          for path in req.query_file_paths}
                
                for future in as_completed(futures):
                    query_path, results, error = future.result()
                    if error:
                        all_results[query_path] = {"error": error}
                    else:
                        all_results[query_path] = {
                            "results": results,
                            "num_results": len(results)
                        }
        else:
            # Sequential processing
            for query_path in req.query_file_paths:
                try:
                    query_vec = load_macro_vec(query_path)
                    query_feat = extract_feature(query_vec)
                    
                    results = retrieval_system.search(
                        query_feat, query_path,
                        stage1_topn=req.stage1_topn, stage2_topk=req.k,
                        fusion_method=req.fusion_method, alpha=req.alpha, beta=req.beta,
                        filters=req.filters,
                        explainable=req.explainable
                    )
                    
                    all_results[query_path] = {
                        "results": results,
                        "num_results": len(results)
                    }
                except Exception as e:
                    all_results[query_path] = {"error": str(e)}
        
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        successful = sum(1 for r in all_results.values() if "error" not in r)
        failed = len(all_results) - successful
        avg_time_per_query = elapsed_time / len(req.query_file_paths) if req.query_file_paths else 0
        
        return {
            "status": "success",
            "total_queries": len(req.query_file_paths),
            "successful": successful,
            "failed": failed,
            "elapsed_time": elapsed_time,
            "avg_time_per_query": avg_time_per_query,
            "parallel": req.parallel,
            "results": all_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch search error: {str(e)}")


@app.post("/search/semantic")
async def semantic_search(req: SemanticSearchRequest):
    """Semantic search using natural language queries
    
    Supports multiple languages (Chinese and English) and different embedding models.
    
    Example queries:
    - "圆柱形零件" (Chinese: cylindrical part)
    - "mechanical component with holes"
    - "找一个有螺纹的轴"
    """
    global text_encoder
    
    if retrieval_system is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    try:
        # Lazy load text encoder (avoid loading on startup if not needed)
        if text_encoder is None or type(text_encoder).__name__ != req.encoder_type:
            print(f"Loading text encoder: {req.encoder_type}")
            text_encoder = create_text_encoder(
                encoder_type=req.encoder_type,
                model_name=req.model_name,
                device="cpu",  # Use GPU if available: "cuda"
                use_cache=True
            )
            print(f"Text encoder loaded, dimension: {text_encoder.dimension}")
        
        # Perform semantic search
        if req.explainable:
            results, explanation = retrieval_system.semantic_search(
                query_text=req.query_text,
                text_encoder=text_encoder,
                k=req.k,
                filters=req.filters,
                explainable=True
            )
            return {
                "status": "success",
                "query_text": req.query_text,
                "num_results": len(results),
                "results": results,
                "explanation": explanation
            }
        else:
            results = retrieval_system.semantic_search(
                query_text=req.query_text,
                text_encoder=text_encoder,
                k=req.k,
                filters=req.filters,
                explainable=False
            )
            return {
                "status": "success",
                "query_text": req.query_text,
                "num_results": len(results),
                "results": results
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search error: {str(e)}")


@app.post("/search/hybrid")
async def hybrid_search(req: HybridSearchRequest):
    """Hybrid search combining semantic and vector-based retrieval
    
    Fuses results from both text-based and CAD vector-based search for better accuracy.
    """
    global text_encoder
    
    if retrieval_system is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    try:
        # Load text encoder if needed
        if text_encoder is None or type(text_encoder).__name__ != req.encoder_type:
            print(f"Loading text encoder: {req.encoder_type}")
            text_encoder = create_text_encoder(
                encoder_type=req.encoder_type,
                model_name=req.model_name,
                device="cpu",
                use_cache=True
            )
        
        # Load CAD vector if provided
        query_vec = None
        if req.query_file_path:
            if not os.path.exists(req.query_file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Query file not found: {req.query_file_path}"
                )
            query_vec = load_macro_vec(req.query_file_path)
        
        # Perform hybrid search
        results = retrieval_system.hybrid_search(
            query_text=req.query_text,
            text_encoder=text_encoder,
            query_vec=query_vec,
            query_file_path=req.query_file_path,
            k=req.k,
            semantic_weight=req.semantic_weight,
            vector_weight=req.vector_weight,
            filters=req.filters
        )
        
        return {
            "status": "success",
            "query_text": req.query_text,
            "query_file": req.query_file_path,
            "semantic_weight": req.semantic_weight,
            "vector_weight": req.vector_weight,
            "num_results": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
