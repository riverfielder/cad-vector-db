"""Feature extraction utilities"""
import h5py
import numpy as np


def extract_feature(vec):
    """Extract fixed-dim feature from variable-length vec
    
    Args:
        vec: (seq_len, 33) array, col 0 is command, col 1-32 are params
    
    Returns:
        feat: (32,) float32 feature vector (mean pooled parameters)
    """
    # Mean pooling over params (ignore command column)
    feat = vec[:, 1:].mean(axis=0).astype('float32')
    return feat


def load_macro_vec(h5_path):
    """Load macro sequence vector from h5 file
    
    Args:
        h5_path: Path to .h5 file
        
    Returns:
        vec: (seq_len, 33) float32 array
    """
    with h5py.File(h5_path, 'r') as f:
        vec = f['vec'][:]
    return vec.astype('float32')


def batch_extract_features(vectors):
    """Extract features from multiple vectors
    
    Args:
        vectors: List of (seq_len, 33) arrays
        
    Returns:
        features: (N, 32) array of features
    """
    return np.array([extract_feature(vec) for vec in vectors], dtype='float32')
