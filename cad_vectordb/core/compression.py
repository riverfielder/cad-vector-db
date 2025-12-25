"""
Vector Compression and Quantization Module

Implements various vector compression techniques to reduce memory usage
and improve search performance:
- Product Quantization (PQ)
- Scalar Quantization (SQ)
- Hybrid compression strategies

Author: riverfielder
Date: 2025-01-25
"""

import numpy as np
import faiss
import pickle
import os
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class CompressionStats:
    """Statistics for compression operation"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    memory_saved_mb: float
    num_vectors: int
    dimension: int


class VectorCompressor:
    """
    Vector compression using FAISS quantization techniques
    
    Supports:
    - Product Quantization (PQ): Splits vectors into subvectors and quantizes
    - Scalar Quantization (SQ): Quantizes each dimension independently
    - No compression (baseline)
    """
    
    def __init__(
        self,
        compression_type: str = "pq",
        dimension: int = 32,
        verbose: bool = True
    ):
        """
        Initialize vector compressor
        
        Args:
            compression_type: "pq", "sq", or "none"
            dimension: Feature vector dimension
            verbose: Print progress messages
        """
        self.compression_type = compression_type.lower()
        self.dimension = dimension
        self.verbose = verbose
        
        # PQ parameters
        self.pq_m = None  # Number of subquantizers
        self.pq_nbits = None  # Bits per subquantizer
        
        # SQ parameters
        self.sq_type = None  # SQ4/SQ6/SQ8
        
        # Trained quantizer
        self.quantizer = None
        self.is_trained = False
        
        if self.verbose:
            print(f"Initialized VectorCompressor: type={compression_type}, dim={dimension}")
    
    def configure_pq(self, m: int = 8, nbits: int = 8):
        """
        Configure Product Quantization parameters
        
        Args:
            m: Number of subquantizers (must divide dimension evenly)
            nbits: Bits per subquantizer (4, 6, or 8)
        
        Notes:
            - m=8, nbits=8: 8 bytes per vector (vs 128 bytes for 32D float32)
            - Compression ratio: 16x for 32D vectors
        """
        if self.dimension % m != 0:
            raise ValueError(f"Dimension {self.dimension} must be divisible by m={m}")
        
        self.pq_m = m
        self.pq_nbits = nbits
        
        if self.verbose:
            bytes_per_vector = m * nbits // 8
            compression_ratio = (self.dimension * 4) / bytes_per_vector
            print(f"PQ Config: m={m}, nbits={nbits}")
            print(f"  Bytes per vector: {bytes_per_vector}")
            print(f"  Compression ratio: {compression_ratio:.1f}x")
    
    def configure_sq(self, sq_type: str = "SQ8"):
        """
        Configure Scalar Quantization parameters
        
        Args:
            sq_type: "SQ4", "SQ6", or "SQ8"
        
        Notes:
            - SQ8: 1 byte per dimension (4x compression)
            - SQ6: 0.75 bytes per dimension (5.3x compression)
            - SQ4: 0.5 bytes per dimension (8x compression)
        """
        valid_types = ["SQ4", "SQ6", "SQ8"]
        if sq_type not in valid_types:
            raise ValueError(f"sq_type must be one of {valid_types}")
        
        self.sq_type = sq_type
        
        if self.verbose:
            bits = int(sq_type[2:])
            bytes_per_vector = self.dimension * bits / 8
            compression_ratio = (self.dimension * 4) / bytes_per_vector
            print(f"SQ Config: type={sq_type}")
            print(f"  Bytes per vector: {bytes_per_vector:.1f}")
            print(f"  Compression ratio: {compression_ratio:.1f}x")
    
    def train(self, vectors: np.ndarray, sample_size: int = 100000):
        """
        Train the quantizer on sample vectors
        
        Args:
            vectors: Training vectors (N, D)
            sample_size: Number of vectors to use for training
        """
        if self.compression_type == "none":
            self.is_trained = True
            if self.verbose:
                print("No compression: skipping training")
            return
        
        # Sample vectors for training
        n_vectors = len(vectors)
        if n_vectors > sample_size:
            indices = np.random.choice(n_vectors, sample_size, replace=False)
            train_vectors = vectors[indices]
        else:
            train_vectors = vectors
        
        train_vectors = train_vectors.astype(np.float32)
        
        if self.verbose:
            print(f"Training quantizer on {len(train_vectors)} vectors...")
        
        if self.compression_type == "pq":
            if self.pq_m is None:
                self.configure_pq()
            
            # Create PQ index
            self.quantizer = faiss.IndexPQ(self.dimension, self.pq_m, self.pq_nbits)
            self.quantizer.train(train_vectors)
            
        elif self.compression_type == "sq":
            if self.sq_type is None:
                self.configure_sq()
            
            # Create SQ index
            if self.sq_type == "SQ8":
                self.quantizer = faiss.IndexScalarQuantizer(
                    self.dimension, faiss.ScalarQuantizer.QT_8bit
                )
            elif self.sq_type == "SQ6":
                self.quantizer = faiss.IndexScalarQuantizer(
                    self.dimension, faiss.ScalarQuantizer.QT_6bit
                )
            elif self.sq_type == "SQ4":
                self.quantizer = faiss.IndexScalarQuantizer(
                    self.dimension, faiss.ScalarQuantizer.QT_4bit
                )
            
            self.quantizer.train(train_vectors)
        
        self.is_trained = True
        
        if self.verbose:
            print(f"✓ Quantizer trained successfully")
    
    def compress(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compress vectors using trained quantizer
        
        Args:
            vectors: Input vectors (N, D)
        
        Returns:
            Compressed vectors (codes)
        """
        if not self.is_trained:
            raise RuntimeError("Quantizer not trained. Call train() first.")
        
        if self.compression_type == "none":
            return vectors
        
        vectors = vectors.astype(np.float32)
        
        # Add vectors to quantizer and retrieve codes
        self.quantizer.add(vectors)
        
        # For PQ/SQ, we can get the codes directly
        if hasattr(self.quantizer, 'sa_code_size'):
            # This is a PQ or SQ index
            codes = faiss.vector_to_array(self.quantizer.codes)
            n_vectors = self.quantizer.ntotal
            code_size = self.quantizer.sa_code_size()
            codes = codes.reshape(n_vectors, code_size)
        else:
            # Fallback: just use the index
            codes = vectors
        
        if self.verbose:
            original_size = vectors.nbytes
            compressed_size = codes.nbytes if isinstance(codes, np.ndarray) else original_size
            ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            print(f"Compressed {len(vectors)} vectors: {ratio:.1f}x reduction")
        
        return codes
    
    def get_compression_stats(self, vectors: np.ndarray) -> CompressionStats:
        """
        Calculate compression statistics
        
        Args:
            vectors: Original vectors (N, D)
        
        Returns:
            CompressionStats object
        """
        n_vectors = len(vectors)
        original_size = vectors.nbytes
        
        if self.compression_type == "none":
            compressed_size = original_size
        elif self.compression_type == "pq":
            if self.pq_m is None:
                self.configure_pq()
            compressed_size = n_vectors * self.pq_m * self.pq_nbits // 8
        elif self.compression_type == "sq":
            if self.sq_type is None:
                self.configure_sq()
            bits = int(self.sq_type[2:])
            compressed_size = n_vectors * self.dimension * bits // 8
        
        compression_ratio = original_size / compressed_size
        memory_saved_mb = (original_size - compressed_size) / (1024 * 1024)
        
        return CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            memory_saved_mb=memory_saved_mb,
            num_vectors=n_vectors,
            dimension=self.dimension
        )
    
    def create_compressed_index(
        self,
        index_type: str = "IVF",
        nlist: int = 100
    ) -> faiss.Index:
        """
        Create a compressed FAISS index
        
        Args:
            index_type: "IVF" or "HNSW"
            nlist: Number of clusters for IVF
        
        Returns:
            FAISS index with compression
        """
        if not self.is_trained:
            raise RuntimeError("Quantizer not trained. Call train() first.")
        
        if self.compression_type == "pq":
            if index_type == "IVF":
                # IVF + PQ
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFPQ(
                    quantizer, self.dimension, nlist,
                    self.pq_m, self.pq_nbits
                )
            else:
                # PQ alone
                index = faiss.IndexPQ(self.dimension, self.pq_m, self.pq_nbits)
        
        elif self.compression_type == "sq":
            if index_type == "IVF":
                # IVF + SQ
                quantizer = faiss.IndexFlatL2(self.dimension)
                if self.sq_type == "SQ8":
                    qtype = faiss.ScalarQuantizer.QT_8bit
                elif self.sq_type == "SQ6":
                    qtype = faiss.ScalarQuantizer.QT_6bit
                else:
                    qtype = faiss.ScalarQuantizer.QT_4bit
                
                index = faiss.IndexIVFScalarQuantizer(
                    quantizer, self.dimension, nlist, qtype
                )
            else:
                # SQ alone
                if self.sq_type == "SQ8":
                    qtype = faiss.ScalarQuantizer.QT_8bit
                elif self.sq_type == "SQ6":
                    qtype = faiss.ScalarQuantizer.QT_6bit
                else:
                    qtype = faiss.ScalarQuantizer.QT_4bit
                
                index = faiss.IndexScalarQuantizer(self.dimension, qtype)
        
        else:  # "none"
            if index_type == "IVF":
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                index = faiss.IndexFlatL2(self.dimension)
        
        if self.verbose:
            print(f"✓ Created compressed {index_type} index")
        
        return index
    
    def save(self, filepath: str):
        """Save compressor state"""
        state = {
            'compression_type': self.compression_type,
            'dimension': self.dimension,
            'pq_m': self.pq_m,
            'pq_nbits': self.pq_nbits,
            'sq_type': self.sq_type,
            'is_trained': self.is_trained
        }
        
        # Save quantizer separately using FAISS
        if self.quantizer is not None:
            quantizer_path = filepath.replace('.pkl', '_quantizer.index')
            faiss.write_index(self.quantizer, quantizer_path)
            state['quantizer_path'] = quantizer_path
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        if self.verbose:
            print(f"✓ Saved compressor to {filepath}")
    
    def load(self, filepath: str):
        """Load compressor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.compression_type = state['compression_type']
        self.dimension = state['dimension']
        self.pq_m = state.get('pq_m')
        self.pq_nbits = state.get('pq_nbits')
        self.sq_type = state.get('sq_type')
        self.is_trained = state['is_trained']
        
        # Load quantizer
        if 'quantizer_path' in state and os.path.exists(state['quantizer_path']):
            self.quantizer = faiss.read_index(state['quantizer_path'])
        
        if self.verbose:
            print(f"✓ Loaded compressor from {filepath}")


def compare_compression_methods(
    vectors: np.ndarray,
    methods: list = None
) -> Dict[str, CompressionStats]:
    """
    Compare different compression methods
    
    Args:
        vectors: Test vectors (N, D)
        methods: List of compression types to compare
    
    Returns:
        Dictionary mapping method name to CompressionStats
    """
    if methods is None:
        methods = ["none", "sq", "pq"]
    
    dimension = vectors.shape[1]
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Compression Comparison on {len(vectors)} vectors ({dimension}D)")
    print(f"{'='*60}\n")
    
    for method in methods:
        compressor = VectorCompressor(
            compression_type=method,
            dimension=dimension,
            verbose=False
        )
        
        if method == "pq":
            compressor.configure_pq(m=8, nbits=8)
        elif method == "sq":
            compressor.configure_sq(sq_type="SQ8")
        
        compressor.train(vectors)
        stats = compressor.get_compression_stats(vectors)
        results[method] = stats
        
        print(f"{method.upper():10s}: {stats.compression_ratio:.2f}x compression "
              f"({stats.memory_saved_mb:.1f} MB saved)")
    
    print(f"\n{'='*60}\n")
    
    return results
