"""Configuration for CAD Vector Database"""
import os

# Paths
WHUCAD_DATA_ROOT = "/Users/he.tian/bs/WHUCAD-main/data/vec"
INDEX_DIR = "data/index"
METADATA_DB_PATH = "data/metadata.db"

# Vector extraction
FEATURE_DIM = 32  # vec[:,1:].mean(axis=0) -> 32-dim
USE_NORMALIZATION = True  # L2 normalize for cosine similarity

# FAISS index parameters
INDEX_TYPE = "HNSW"  # or "IVF" or "IVFPQ"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

IVF_NLIST = 100  # for IVF-based indices
IVF_NPROBE = 16

# Two-stage retrieval
STAGE1_TOPN = 100  # ANN recall candidates
STAGE2_TOPK = 20   # Final reranked results

# Fusion
FUSION_METHOD = "weighted"  # "weighted", "rrf", "borda"
FUSION_ALPHA = 0.6  # weight for stage1 score
FUSION_BETA = 0.4   # weight for stage2 score
RRF_K = 60

# Server
API_HOST = "127.0.0.1"
API_PORT = 8000

# Database
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "cad_vector_db"
DB_USER = "postgres"
DB_PASSWORD = ""

# Evaluation
EVAL_METRICS = ["precision", "recall", "map", "latency"]
EVAL_K_VALUES = [1, 5, 10, 20]
