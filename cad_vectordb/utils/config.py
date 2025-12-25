"""
Configuration Management for CAD Vector Database

Features:
- Environment-based configuration (dev/test/prod)
- Environment variables support
- Secrets management (passwords, API keys)
- Configuration validation
- Type safety
"""
import os
from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class PathConfig:
    """Path configuration"""
    whucad_data_root: str = field(default_factory=lambda: os.getenv(
        'WHUCAD_DATA_ROOT',
        '/Users/he.tian/bs/WHUCAD-main/data/vec'
    ))
    index_dir: str = field(default_factory=lambda: os.getenv(
        'INDEX_DIR',
        'data/index_test'
    ))
    metadata_db_path: str = field(default_factory=lambda: os.getenv(
        'METADATA_DB_PATH',
        'data/metadata.db'
    ))
    log_dir: str = field(default_factory=lambda: os.getenv(
        'LOG_DIR',
        'logs'
    ))


@dataclass
class VectorConfig:
    """Vector extraction configuration"""
    feature_dim: int = field(default_factory=lambda: int(os.getenv('FEATURE_DIM', '32')))
    use_normalization: bool = field(default_factory=lambda: os.getenv(
        'USE_NORMALIZATION', 'true'
    ).lower() == 'true')


@dataclass
class IndexConfig:
    """FAISS index configuration"""
    index_type: str = field(default_factory=lambda: os.getenv('INDEX_TYPE', 'HNSW'))
    
    # HNSW parameters
    hnsw_m: int = field(default_factory=lambda: int(os.getenv('HNSW_M', '32')))
    hnsw_ef_construction: int = field(default_factory=lambda: int(os.getenv(
        'HNSW_EF_CONSTRUCTION', '200'
    )))
    hnsw_ef_search: int = field(default_factory=lambda: int(os.getenv(
        'HNSW_EF_SEARCH', '128'
    )))
    
    # IVF parameters
    ivf_nlist: int = field(default_factory=lambda: int(os.getenv('IVF_NLIST', '100')))
    ivf_nprobe: int = field(default_factory=lambda: int(os.getenv('IVF_NPROBE', '16')))


@dataclass
class RetrievalConfig:
    """Two-stage retrieval configuration"""
    stage1_topn: int = field(default_factory=lambda: int(os.getenv('STAGE1_TOPN', '100')))
    stage2_topk: int = field(default_factory=lambda: int(os.getenv('STAGE2_TOPK', '20')))
    
    # Fusion parameters
    fusion_method: str = field(default_factory=lambda: os.getenv(
        'FUSION_METHOD', 'weighted'
    ))
    fusion_alpha: float = field(default_factory=lambda: float(os.getenv(
        'FUSION_ALPHA', '0.6'
    )))
    fusion_beta: float = field(default_factory=lambda: float(os.getenv(
        'FUSION_BETA', '0.4'
    )))
    rrf_k: int = field(default_factory=lambda: int(os.getenv('RRF_K', '60')))


@dataclass
class ServerConfig:
    """API server configuration"""
    host: str = field(default_factory=lambda: os.getenv('API_HOST', '127.0.0.1'))
    port: int = field(default_factory=lambda: int(os.getenv('API_PORT', '8000')))
    workers: int = field(default_factory=lambda: int(os.getenv('API_WORKERS', '1')))
    reload: bool = field(default_factory=lambda: os.getenv(
        'API_RELOAD', 'false'
    ).lower() == 'true')
    
    # Security
    enable_auth: bool = field(default_factory=lambda: os.getenv(
        'ENABLE_AUTH', 'false'
    ).lower() == 'true')
    api_key: Optional[str] = field(default_factory=lambda: os.getenv('API_KEY'))
    allowed_origins: List[str] = field(default_factory=lambda: [
        origin.strip() 
        for origin in os.getenv('ALLOWED_ORIGINS', '*').split(',')
    ])
    
    # Rate limiting
    rate_limit_enabled: bool = field(default_factory=lambda: os.getenv(
        'RATE_LIMIT_ENABLED', 'false'
    ).lower() == 'true')
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv(
        'RATE_LIMIT_REQUESTS', '100'
    )))
    rate_limit_period: int = field(default_factory=lambda: int(os.getenv(
        'RATE_LIMIT_PERIOD', '60'
    )))


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = field(default_factory=lambda: os.getenv('DB_HOST', '127.0.0.1'))
    port: int = field(default_factory=lambda: int(os.getenv('DB_PORT', '2881')))
    name: str = field(default_factory=lambda: os.getenv('DB_NAME', 'cad_vector_db'))
    user: str = field(default_factory=lambda: os.getenv('DB_USER', 'root@test'))
    password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD', ''))
    
    # Connection pool
    pool_size: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '5')))
    pool_recycle: int = field(default_factory=lambda: int(os.getenv(
        'DB_POOL_RECYCLE', '3600'
    )))


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    format: str = field(default_factory=lambda: os.getenv('LOG_FORMAT', 'json'))
    to_file: bool = field(default_factory=lambda: os.getenv(
        'LOG_TO_FILE', 'true'
    ).lower() == 'true')
    to_console: bool = field(default_factory=lambda: os.getenv(
        'LOG_TO_CONSOLE', 'true'
    ).lower() == 'true')
    max_bytes: int = field(default_factory=lambda: int(os.getenv(
        'LOG_MAX_BYTES', str(10 * 1024 * 1024)  # 10MB
    )))
    backup_count: int = field(default_factory=lambda: int(os.getenv(
        'LOG_BACKUP_COUNT', '5'
    )))


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str] = field(default_factory=lambda: os.getenv(
        'EVAL_METRICS', 'precision,recall,map,latency'
    ).split(','))
    k_values: List[int] = field(default_factory=lambda: [
        int(k) for k in os.getenv('EVAL_K_VALUES', '1,5,10,20').split(',')
    ])


class Config:
    """Main configuration class"""
    
    def __init__(self, env: Optional[Environment] = None):
        """
        Initialize configuration
        
        Args:
            env: Environment (development/testing/production)
                 If None, reads from ENV environment variable
        """
        if env is None:
            env_str = os.getenv('ENV', 'development')
            self.env = Environment(env_str)
        else:
            self.env = env
        
        # Initialize all config sections
        self.paths = PathConfig()
        self.vector = VectorConfig()
        self.index = IndexConfig()
        self.retrieval = RetrievalConfig()
        self.server = ServerConfig()
        self.database = DatabaseConfig()
        self.logging = LoggingConfig()
        self.evaluation = EvaluationConfig()
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration"""
        errors = []
        
        # Validate paths exist or can be created
        try:
            Path(self.paths.log_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create log directory: {e}")
        
        # Validate index type
        valid_index_types = ['Flat', 'IVF', 'IVFFlat', 'HNSW', 'IVFPQ']
        if self.index.index_type not in valid_index_types:
            errors.append(f"Invalid index_type: {self.index.index_type}. "
                         f"Must be one of {valid_index_types}")
        
        # Validate fusion method
        valid_fusion_methods = ['weighted', 'rrf', 'borda']
        if self.retrieval.fusion_method not in valid_fusion_methods:
            errors.append(f"Invalid fusion_method: {self.retrieval.fusion_method}. "
                         f"Must be one of {valid_fusion_methods}")
        
        # Validate weights sum to 1.0 for weighted fusion
        if self.retrieval.fusion_method == 'weighted':
            weight_sum = self.retrieval.fusion_alpha + self.retrieval.fusion_beta
            if abs(weight_sum - 1.0) > 0.01:
                errors.append(f"Fusion weights must sum to 1.0, got {weight_sum}")
        
        # Validate port ranges
        if not (1024 <= self.server.port <= 65535):
            errors.append(f"Invalid server port: {self.server.port}")
        
        if not (1024 <= self.database.port <= 65535):
            errors.append(f"Invalid database port: {self.database.port}")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.level.upper() not in valid_log_levels:
            errors.append(f"Invalid log level: {self.logging.level}")
        
        # Check for security issues in production
        if self.env == Environment.PRODUCTION:
            if self.server.enable_auth and not self.server.api_key:
                errors.append("API_KEY must be set when ENABLE_AUTH is true in production")
            
            if not self.database.password:
                errors.append("DB_PASSWORD should be set in production")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'environment': self.env.value,
            'paths': self.paths.__dict__,
            'vector': self.vector.__dict__,
            'index': self.index.__dict__,
            'retrieval': self.retrieval.__dict__,
            'server': {
                k: v for k, v in self.server.__dict__.items()
                if k not in ['api_key']  # Don't expose secrets
            },
            'database': {
                k: v for k, v in self.database.__dict__.items()
                if k not in ['password']  # Don't expose secrets
            },
            'logging': self.logging.__dict__,
            'evaluation': self.evaluation.__dict__,
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Config(env={self.env.value})"


# Global config instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return config


def reload_config(env: Optional[Environment] = None) -> Config:
    """Reload configuration"""
    global config
    config = Config(env)
    return config


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration system...")
    
    # Test default config
    cfg = Config()
    print(f"\n✓ Environment: {cfg.env.value}")
    print(f"✓ Server: {cfg.server.host}:{cfg.server.port}")
    print(f"✓ Index type: {cfg.index.index_type}")
    print(f"✓ Log level: {cfg.logging.level}")
    
    # Test validation
    try:
        os.environ['INDEX_TYPE'] = 'InvalidType'
        bad_config = Config()
    except ValueError as e:
        print(f"\n✓ Validation caught error: {e}")
    finally:
        del os.environ['INDEX_TYPE']
    
    # Print full config (masked)
    import json
    print("\nFull configuration:")
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
