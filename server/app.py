"""FastAPI server for CAD Vector Database"""
import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables first
from cad_vectordb.utils.env import load_env
load_env()

# Import new utils
from cad_vectordb.utils.config import get_config
from cad_vectordb.utils.logger import get_logger, api_logger
from cad_vectordb.utils.security import (
    require_auth, rate_limit, generate_request_id,
    PathValidator, InputValidator
)

# Import core modules
from cad_vectordb.core.index import IndexManager
from cad_vectordb.core.retrieval import TwoStageRetrieval
from cad_vectordb.core.feature import extract_feature, load_macro_vec
from cad_vectordb.core.text_encoder import create_text_encoder

# Get configuration
config = get_config()

# Initialize FastAPI
app = FastAPI(
    title="CAD Vector Database API",
    version="2.0.0",
    description="Production-ready vector database for CAD model retrieval"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class AddVectorRequest(BaseModel):
    """Add vector request model"""
    id_str: str
    h5_path: str


class UpdateVectorRequest(BaseModel):
    """Update vector request model"""
    h5_path: str


class BatchUpdateRequest(BaseModel):
    """Batch update request model"""
    updates: List[Dict[str, str]]  # List of {id_str, h5_path} dicts


class SoftDeleteRequest(BaseModel):
    """Soft delete request model"""
    ids: List[str]


class RestoreRequest(BaseModel):
    """Restore request model"""
    ids: List[str]


class CreateSnapshotRequest(BaseModel):
    """Create snapshot request model"""
    name: Optional[str] = None


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


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing"""
    request_id = generate_request_id()
    start_time = time.time()
    
    # Add request ID to state
    request.state.request_id = request_id
    
    # Log request
    api_logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={
            'request_id': request_id,
            'extra_data': {
                'method': request.method,
                'path': request.url.path,
                'client_ip': request.client.host if request.client else None
            }
        }
    )
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        api_logger.info(
            f"Request completed: {request.method} {request.url.path}",
            extra={
                'request_id': request_id,
                'duration_ms': duration_ms,
                'extra_data': {
                    'status_code': response.status_code
                }
            }
        )
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
        
        return response
    
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        api_logger.error(
            f"Request failed: {request.method} {request.url.path}",
            exc_info=True,
            extra={
                'request_id': request_id,
                'duration_ms': duration_ms
            }
        )
        raise


@app.on_event("startup")
async def startup_event():
    """Load index on startup"""
    global index_manager, retrieval_system
    
    api_logger.info("Starting CAD Vector Database API server")
    api_logger.info(f"Environment: {config.env.value}")
    api_logger.info(f"Loading index from: {config.paths.index_dir}")
    
    try:
        index_manager = IndexManager(config.paths.index_dir, verbose=False)
        index_manager.load_index()
        retrieval_system = TwoStageRetrieval(index_manager)
        
        api_logger.info(
            f"Index loaded successfully: {len(index_manager.ids)} vectors",
            extra={'extra_data': {'num_vectors': len(index_manager.ids)}}
        )
    except Exception as e:
        api_logger.error("Failed to load index", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    api_logger.info("Shutting down CAD Vector Database API server")


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
            "/vectors/{id}",
            "/vectors/add",
            "/vectors/{vector_id}",
            "/vectors/batch-update",
            "/vectors/soft",
            "/vectors/restore",
            "/vectors/deleted",
            "/index/compact",
            "/index/snapshot",
            "/index/snapshots",
            "/index/snapshot/{snapshot_name}/restore",
            "/index/changelog"
        ]
    }


@app.post("/search")
@rate_limit
async def search(req: SearchRequest, request: Request):
    """Two-stage search with fusion and optional metadata filtering (Hybrid Search)
    
    With explainable=true, returns detailed similarity breakdown and interpretations
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    if retrieval_system is None:
        api_logger.error("Search failed: index not loaded", extra={'request_id': request_id})
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Validate file path
    if not PathValidator.is_safe_path(config.paths.whucad_data_root, req.query_file_path):
        api_logger.warning(
            f"Unsafe path detected: {req.query_file_path}",
            extra={'request_id': request_id}
        )
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    # Validate query file exists
    if not os.path.exists(req.query_file_path):
        api_logger.warning(
            f"File not found: {req.query_file_path}",
            extra={'request_id': request_id}
        )
        raise HTTPException(status_code=404, detail=f"Query file not found: {req.query_file_path}")
    
    # Validate k value
    if not InputValidator.validate_k_value(req.k):
        raise HTTPException(status_code=400, detail="Invalid k value (must be 1-1000)")
    
    try:
        api_logger.info(
            f"Search request: k={req.k}, method={req.fusion_method}",
            extra={
                'request_id': request_id,
                'extra_data': {
                    'k': req.k,
                    'fusion_method': req.fusion_method,
                    'explainable': req.explainable
                }
            }
        )
        
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
        
        api_logger.info(
            f"Search completed: {len(results)} results",
            extra={
                'request_id': request_id,
                'extra_data': {'num_results': len(results)}
            }
        )
        
        return results
        
    except Exception as e:
        api_logger.error(
            f"Search error: {str(e)}",
            exc_info=True,
            extra={'request_id': request_id}
        )
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


@app.post("/vectors/add")
async def add_vector(req: AddVectorRequest):
    """
    Add a new vector to the index
    
    Parameters:
    - id_str: Unique identifier for the vector
    - h5_path: Path to the H5 file containing the vector
    """
    try:
        index_manager.add_vectors([(req.id_str, req.h5_path)])
        return {
            "status": "success",
            "message": f"Vector {req.id_str} added successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Add vector error: {str(e)}")


@app.put("/vectors/{vector_id}")
async def update_vector(vector_id: str, req: UpdateVectorRequest):
    """
    Update an existing vector
    
    Parameters:
    - vector_id: ID of the vector to update
    - h5_path: Path to the H5 file containing the new vector
    """
    try:
        index_manager.update_vector(vector_id, req.h5_path)
        return {
            "status": "success",
            "message": f"Vector {vector_id} updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update vector error: {str(e)}")


@app.post("/vectors/batch-update")
async def batch_update(req: BatchUpdateRequest):
    """
    Batch update multiple vectors
    
    Parameters:
    - updates: List of update operations, each with id_str and h5_path
    """
    try:
        index_manager.batch_update(req.updates)
        return {
            "status": "success",
            "message": f"{len(req.updates)} vectors updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch update error: {str(e)}")


@app.delete("/vectors/soft")
async def soft_delete(req: SoftDeleteRequest):
    """
    Soft delete vectors (mark as deleted without removing from index)
    
    Parameters:
    - ids: List of vector IDs to soft delete
    """
    try:
        index_manager.soft_delete(req.ids)
        return {
            "status": "success",
            "message": f"{len(req.ids)} vectors soft deleted successfully",
            "deleted_ids": req.ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Soft delete error: {str(e)}")


@app.post("/vectors/restore")
async def restore_vectors(req: RestoreRequest):
    """
    Restore soft-deleted vectors
    
    Parameters:
    - ids: List of vector IDs to restore
    """
    try:
        index_manager.restore(req.ids)
        return {
            "status": "success",
            "message": f"{len(req.ids)} vectors restored successfully",
            "restored_ids": req.ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore error: {str(e)}")


@app.get("/vectors/deleted")
async def get_deleted_vectors():
    """Get list of soft-deleted vector IDs"""
    try:
        deleted_ids = index_manager.get_deleted_ids()
        return {
            "status": "success",
            "deleted_count": len(deleted_ids),
            "deleted_ids": list(deleted_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get deleted IDs error: {str(e)}")


@app.post("/index/compact")
async def compact_index():
    """
    Compact the index by permanently removing soft-deleted vectors
    This will rebuild the index without deleted vectors
    """
    try:
        index_manager.compact_index()
        return {
            "status": "success",
            "message": "Index compacted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compact index error: {str(e)}")


@app.post("/index/snapshot")
async def create_snapshot(req: CreateSnapshotRequest):
    """
    Create a snapshot of the current index state
    
    Parameters:
    - name: Optional snapshot name (auto-generated if not provided)
    """
    try:
        snapshot_name = index_manager.create_snapshot(req.name)
        return {
            "status": "success",
            "message": "Snapshot created successfully",
            "snapshot_name": snapshot_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Create snapshot error: {str(e)}")


@app.get("/index/snapshots")
async def list_snapshots():
    """List all available snapshots"""
    try:
        snapshots = index_manager.list_snapshots()
        return {
            "status": "success",
            "snapshot_count": len(snapshots),
            "snapshots": snapshots
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List snapshots error: {str(e)}")


@app.post("/index/snapshot/{snapshot_name}/restore")
async def restore_snapshot(snapshot_name: str):
    """
    Restore the index to a previous snapshot
    
    Parameters:
    - snapshot_name: Name of the snapshot to restore
    """
    try:
        index_manager.restore_snapshot(snapshot_name)
        return {
            "status": "success",
            "message": f"Index restored to snapshot: {snapshot_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore snapshot error: {str(e)}")


@app.get("/index/changelog")
async def get_changelog(limit: int = 100):
    """
    Get the change log of index operations
    
    Parameters:
    - limit: Maximum number of log entries to return (default: 100)
    """
    try:
        changelog = index_manager.get_change_log(limit)
        return {
            "status": "success",
            "log_count": len(changelog),
            "changelog": changelog
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get changelog error: {str(e)}")


@app.post("/index/compress")
async def compress_index(compression_type: str = "pq"):
    """
    Enable compression on the index
    
    Parameters:
    - compression_type: "pq" (Product Quantization) or "sq" (Scalar Quantization)
    """
    try:
        stats = index_manager.enable_vector_compression(
            compression_type=compression_type
        )
        return {
            "status": "success",
            "message": f"Compression enabled: {compression_type}",
            "compression_ratio": stats.compression_ratio,
            "memory_saved_mb": stats.memory_saved_mb,
            "original_size_mb": stats.original_size / 1024 / 1024,
            "compressed_size_mb": stats.compressed_size / 1024 / 1024
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression error: {str(e)}")


@app.post("/index/rebuild-compressed")
async def rebuild_compressed(
    compression_type: str = "pq",
    index_type: str = "IVF",
    nlist: int = 100
):
    """
    Rebuild index with compression
    
    Parameters:
    - compression_type: "pq" or "sq"
    - index_type: "IVF" or "HNSW"
    - nlist: Number of clusters for IVF
    """
    try:
        stats = index_manager.rebuild_with_compression(
            compression_type=compression_type,
            index_type=index_type,
            nlist=nlist
        )
        return {
            "status": "success",
            "message": "Index rebuilt with compression",
            "compression_ratio": stats.compression_ratio,
            "memory_saved_mb": stats.memory_saved_mb
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild error: {str(e)}")


@app.get("/index/compression-stats")
async def get_compression_stats():
    """Get compression statistics"""
    try:
        stats = index_manager.get_compression_stats()
        if stats is None:
            return {
                "status": "success",
                "compression_enabled": False,
                "message": "Compression not enabled"
            }
        return {
            "status": "success",
            "compression_enabled": True,
            **stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get stats error: {str(e)}")


@app.post("/cache/enable")
async def enable_cache(capacity: int = 1000, ttl: int = 3600, use_redis: bool = False):
    """
    Enable query result caching
    
    Parameters:
    - capacity: LRU cache capacity
    - ttl: Time-to-live in seconds
    - use_redis: Enable Redis backend
    """
    try:
        index_manager.enable_query_cache(
            capacity=capacity,
            ttl=ttl,
            use_redis=use_redis
        )
        return {
            "status": "success",
            "message": "Cache enabled",
            "capacity": capacity,
            "ttl": ttl,
            "redis_enabled": use_redis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enable cache error: {str(e)}")


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = index_manager.get_cache_stats()
        return {
            "status": "success",
            **stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get cache stats error: {str(e)}")


@app.post("/cache/clear")
async def clear_cache():
    """Clear query cache"""
    try:
        index_manager.clear_cache()
        return {
            "status": "success",
            "message": "Cache cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear cache error: {str(e)}")


@app.post("/cache/warm")
async def warm_cache(n_samples: int = 100):
    """
    Warm up cache with random queries
    
    Parameters:
    - n_samples: Number of random queries to cache
    """
    try:
        index_manager.warm_cache(n_samples=n_samples)
        return {
            "status": "success",
            "message": f"Cache warmed with {n_samples} samples"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warm cache error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
