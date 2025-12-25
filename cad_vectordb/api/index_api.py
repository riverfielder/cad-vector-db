"""Index Management API Endpoints

Provides REST API for index management operations:
- Create/build new indexes
- List available indexes
- Get index statistics
- Add vectors to index
- Remove vectors from index
- Validate index integrity
"""
import os
from typing import List, Optional
from fastapi import HTTPException
from pydantic import BaseModel

from .app import app
from ..core.index import IndexManager


# Global index managers (one per index name)
index_managers = {}


class BuildIndexRequest(BaseModel):
    name: str  # Index name
    data_root: str  # Path to data directory
    index_type: str = "Flat"  # "Flat", "IVFFlat", "HNSW"
    max_samples: Optional[int] = None  # Limit number of samples


class AddVectorsRequest(BaseModel):
    index_name: str
    file_paths: List[str]  # List of .h5 files to add


class RemoveVectorsRequest(BaseModel):
    index_name: str
    ids: List[str]  # List of IDs to remove
    rebuild: bool = True


@app.post("/index/build")
async def build_index(req: BuildIndexRequest):
    """Build a new index from data"""
    try:
        manager = IndexManager(INDEX_DIR)
        stats = manager.build_index(
            data_root=req.data_root,
            index_type=req.index_type,
            max_samples=req.max_samples,
            verbose=False
        )
        
        # Save index
        save_path = manager.save_index(req.name)
        
        # Cache manager
        index_managers[req.name] = manager
        
        return {
            "status": "success",
            "index_name": req.name,
            "save_path": save_path,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Build error: {str(e)}")


@app.get("/index/list")
async def list_indexes():
    """List all available indexes"""
    manager = IndexManager(INDEX_DIR)
    indexes = manager.list_available_indexes()
    
    # Get stats for each index
    index_info = []
    for idx_name in indexes:
        try:
            mgr = IndexManager(INDEX_DIR)
            mgr.load_index(idx_name)
            stats = mgr.get_stats()
            index_info.append({
                "name": idx_name,
                "num_vectors": stats['num_vectors'],
                "dimension": stats['dimension'],
                "index_type": stats['index_type']
            })
        except:
            index_info.append({
                "name": idx_name,
                "status": "error_loading"
            })
    
    return {
        "indexes": index_info,
        "total": len(indexes)
    }


@app.get("/index/{name}/stats")
async def get_index_stats(name: str):
    """Get detailed statistics for an index"""
    try:
        if name in index_managers:
            manager = index_managers[name]
        else:
            manager = IndexManager(INDEX_DIR)
            manager.load_index(name)
            index_managers[name] = manager
        
        stats = manager.get_stats()
        return {
            "index_name": name,
            "statistics": stats
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Index not found: {name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/add")
async def add_vectors(req: AddVectorsRequest):
    """Add vectors to an existing index"""
    try:
        if req.index_name in index_managers:
            manager = index_managers[req.index_name]
        else:
            manager = IndexManager(INDEX_DIR)
            manager.load_index(req.index_name)
            index_managers[req.index_name] = manager
        
        num_added = manager.add_vectors(req.file_paths, verbose=False)
        
        # Save updated index
        manager.save_index(req.index_name)
        
        return {
            "status": "success",
            "index_name": req.index_name,
            "num_added": num_added,
            "total_vectors": len(manager.ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Add error: {str(e)}")


@app.post("/index/remove")
async def remove_vectors(req: RemoveVectorsRequest):
    """Remove vectors from an index"""
    try:
        if req.index_name in index_managers:
            manager = index_managers[req.index_name]
        else:
            manager = IndexManager(INDEX_DIR)
            manager.load_index(req.index_name)
            index_managers[req.index_name] = manager
        
        num_removed = manager.remove_vectors(req.ids, rebuild=req.rebuild)
        
        # Save updated index
        manager.save_index(req.index_name)
        
        return {
            "status": "success",
            "index_name": req.index_name,
            "num_removed": num_removed,
            "total_vectors": len(manager.ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Remove error: {str(e)}")


@app.get("/index/{name}/validate")
async def validate_index(name: str):
    """Validate index integrity"""
    try:
        if name in index_managers:
            manager = index_managers[name]
        else:
            manager = IndexManager(INDEX_DIR)
            manager.load_index(name)
            index_managers[name] = manager
        
        validation = manager.validate_index()
        
        return {
            "index_name": name,
            "validation": validation
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Index not found: {name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
