"""CAD Vector Database - A vector database system for 3D CAD models

This package provides:
- FAISS-based vector indexing and search
- Two-stage retrieval with fusion
- OceanBase metadata database integration
- Hybrid search with metadata filtering
- Explainable retrieval with similarity breakdown
- Batch search capabilities
- Index management
"""

__version__ = "1.0.0"
__author__ = "He Tian"

from .core.index import IndexManager
from .core.retrieval import TwoStageRetrieval
from .core.feature import extract_feature, load_macro_vec

__all__ = [
    "IndexManager",
    "TwoStageRetrieval", 
    "extract_feature",
    "load_macro_vec",
]
