"""Text Encoder for Semantic Query Support

Provides text-to-vector encoding capabilities for semantic search:
- Multiple encoder backends (CLIP, SentenceTransformer, BM25)
- Bilingual support (Chinese and English)
- Query cache for performance
- Easy integration with existing retrieval system
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional
import numpy as np
from pathlib import Path
import json
import hashlib


class BaseTextEncoder(ABC):
    """Abstract base class for text encoders"""
    
    @abstractmethod
    def encode(self, text: Union[str, List[str]], is_query: bool = True) -> np.ndarray:
        """Encode text to vector(s)
        
        Args:
            text: Single string or list of strings
            is_query: If True, optimize for query (vs document)
            
        Returns:
            np.ndarray: shape (dim,) for single text, or (n, dim) for list
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension"""
        pass


class CLIPTextEncoder(BaseTextEncoder):
    """CLIP text encoder for multi-modal semantic search
    
    Uses OpenAI's CLIP model to encode text into the same space as images.
    Good for: text-to-CAD-image search, multi-modal applications
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        """Initialize CLIP encoder
        
        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, etc)
            device: 'cpu' or 'cuda'
        """
        try:
            import clip
            import torch
        except ImportError:
            raise ImportError(
                "CLIP encoder requires: pip install ftfy regex tqdm "
                "git+https://github.com/openai/CLIP.git"
            )
        
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self._dim = self.model.visual.output_dim
        
    def encode(self, text: Union[str, List[str]], is_query: bool = True) -> np.ndarray:
        """Encode text with CLIP"""
        import torch
        import clip
        
        if isinstance(text, str):
            text = [text]
        
        # Tokenize and encode
        tokens = clip.tokenize(text).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)
            # Normalize embeddings (CLIP uses cosine similarity)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        result = embeddings.cpu().numpy()
        return result[0] if len(text) == 1 else result
    
    @property
    def dimension(self) -> int:
        return self._dim


class SentenceTransformerEncoder(BaseTextEncoder):
    """Sentence-BERT encoder for semantic text similarity
    
    Uses Sentence Transformers library with pre-trained models.
    Good for: pure text semantic search, multilingual support
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):
        """Initialize Sentence Transformer encoder
        
        Args:
            model_name: Sentence-BERT model name
                - paraphrase-multilingual-MiniLM-L12-v2 (multilingual, 384d)
                - paraphrase-multilingual-mpnet-base-v2 (multilingual, 768d)
                - all-MiniLM-L6-v2 (English, 384d, fast)
            device: 'cpu' or 'cuda'
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "SentenceTransformer encoder requires: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name, device=device)
        self._dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, text: Union[str, List[str]], is_query: bool = True) -> np.ndarray:
        """Encode text with Sentence Transformer"""
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        return embeddings[0] if len(text) == 1 else embeddings
    
    @property
    def dimension(self) -> int:
        return self._dim


class BM25TextEncoder(BaseTextEncoder):
    """BM25-based sparse text encoder
    
    Uses TF-IDF + BM25 weighting for text encoding.
    Good for: keyword-based search, low resource scenarios
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        tokenizer: str = "jieba",
        k1: float = 1.5,
        b: float = 0.75
    ):
        """Initialize BM25 encoder
        
        Args:
            vocab_size: Maximum vocabulary size
            tokenizer: 'jieba' (Chinese) or 'simple' (whitespace)
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.vocab_size = vocab_size
        self.tokenizer_type = tokenizer
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[int, float] = {}
        self._dim = vocab_size
        
        if tokenizer == "jieba":
            try:
                import jieba
                self.tokenizer = jieba.cut
            except ImportError:
                raise ImportError("Jieba tokenizer requires: pip install jieba")
        else:
            self.tokenizer = lambda x: x.lower().split()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return list(self.tokenizer(text))
    
    def _build_vocab(self, tokens: List[str]) -> List[int]:
        """Build vocabulary and convert tokens to IDs"""
        ids = []
        for token in tokens:
            if token not in self.vocab:
                if len(self.vocab) < self.vocab_size:
                    self.vocab[token] = len(self.vocab)
                else:
                    continue  # Skip OOV tokens
            ids.append(self.vocab[token])
        return ids
    
    def encode(self, text: Union[str, List[str]], is_query: bool = True) -> np.ndarray:
        """Encode text with BM25 (sparse vector)"""
        if isinstance(text, str):
            text = [text]
        
        embeddings = []
        for t in text:
            tokens = self._tokenize(t)
            token_ids = self._build_vocab(tokens)
            
            # Create sparse vector (TF-based)
            vec = np.zeros(self.vocab_size, dtype=np.float32)
            for tid in token_ids:
                vec[tid] += 1.0
            
            # Simple TF normalization (can be improved with BM25 scoring)
            if vec.sum() > 0:
                vec = vec / np.sqrt(vec.sum())  # L2 normalize
            
            embeddings.append(vec)
        
        result = np.array(embeddings)
        return result[0] if len(text) == 1 else result
    
    @property
    def dimension(self) -> int:
        return self._dim


class CachedTextEncoder:
    """Wrapper that adds caching to any text encoder
    
    Caches encoded results to avoid redundant computation.
    Useful for repeated queries in production.
    """
    
    def __init__(
        self,
        encoder: BaseTextEncoder,
        cache_size: int = 1000,
        cache_file: Optional[Path] = None
    ):
        """Initialize cached encoder
        
        Args:
            encoder: Base encoder to wrap
            cache_size: Max number of cached entries
            cache_file: Path to persistent cache file (optional)
        """
        self.encoder = encoder
        self.cache_size = cache_size
        self.cache_file = cache_file
        self.cache: Dict[str, np.ndarray] = {}
        
        if cache_file and cache_file.exists():
            self._load_cache()
    
    def _hash_text(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk"""
        if not self.cache_file.exists():
            return
        
        try:
            data = np.load(self.cache_file, allow_pickle=True)
            self.cache = data['cache'].item()
        except Exception as e:
            print(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk"""
        if not self.cache_file:
            return
        
        try:
            np.savez(self.cache_file, cache=self.cache)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def encode(self, text: Union[str, List[str]], is_query: bool = True) -> np.ndarray:
        """Encode with caching"""
        if isinstance(text, str):
            key = self._hash_text(text)
            if key in self.cache:
                return self.cache[key]
            
            result = self.encoder.encode(text, is_query)
            
            # Update cache (LRU-like)
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = result
            
            return result
        else:
            # Batch encoding - check each text
            results = []
            to_encode = []
            indices = []
            
            for i, t in enumerate(text):
                key = self._hash_text(t)
                if key in self.cache:
                    results.append(self.cache[key])
                else:
                    to_encode.append(t)
                    indices.append(i)
            
            if to_encode:
                new_embeddings = self.encoder.encode(to_encode, is_query)
                for i, emb in zip(indices, new_embeddings):
                    key = self._hash_text(text[i])
                    self.cache[key] = emb
                    results.insert(i, emb)
            
            return np.array(results)
    
    @property
    def dimension(self) -> int:
        return self.encoder.dimension
    
    def save(self):
        """Manually save cache"""
        self._save_cache()


def create_text_encoder(
    encoder_type: str = "sentence-transformer",
    model_name: Optional[str] = None,
    device: str = "cpu",
    use_cache: bool = True,
    cache_file: Optional[Path] = None,
    **kwargs
) -> Union[BaseTextEncoder, CachedTextEncoder]:
    """Factory function to create text encoders
    
    Args:
        encoder_type: 'clip', 'sentence-transformer', or 'bm25'
        model_name: Specific model name (uses default if None)
        device: 'cpu' or 'cuda'
        use_cache: Enable caching
        cache_file: Path to cache file
        **kwargs: Additional encoder-specific arguments
        
    Returns:
        Text encoder instance (potentially wrapped with cache)
        
    Examples:
        >>> # Multilingual Sentence-BERT (recommended for Chinese+English)
        >>> encoder = create_text_encoder('sentence-transformer')
        
        >>> # CLIP for multi-modal
        >>> encoder = create_text_encoder('clip', device='cuda')
        
        >>> # BM25 for lightweight keyword search
        >>> encoder = create_text_encoder('bm25', tokenizer='jieba')
    """
    # Create base encoder
    if encoder_type.lower() in ['clip']:
        if model_name is None:
            model_name = "ViT-B/32"
        encoder = CLIPTextEncoder(model_name, device)
        
    elif encoder_type.lower() in ['sentence-transformer', 'sbert', 'sentence_transformer']:
        if model_name is None:
            model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # Good for Chinese
        encoder = SentenceTransformerEncoder(model_name, device)
        
    elif encoder_type.lower() in ['bm25']:
        encoder = BM25TextEncoder(**kwargs)
        
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Choose from: 'clip', 'sentence-transformer', 'bm25'"
        )
    
    # Wrap with cache if requested
    if use_cache:
        encoder = CachedTextEncoder(encoder, cache_file=cache_file)
    
    return encoder


if __name__ == "__main__":
    # Demo: test different encoders
    texts = [
        "圆柱形零件",
        "cylindrical part",
        "Find a mechanical component with holes"
    ]
    
    print("Testing text encoders...\n")
    
    # Test Sentence Transformer (multilingual)
    print("1. Sentence Transformer (Multilingual)")
    try:
        encoder = create_text_encoder('sentence-transformer', device='cpu')
        embeddings = encoder.encode(texts)
        print(f"   Dimension: {encoder.dimension}")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Sample: {embeddings[0][:5]}...")
        print()
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Test BM25
    print("2. BM25 (Sparse)")
    try:
        encoder = create_text_encoder('bm25', tokenizer='simple', use_cache=False)
        embeddings = encoder.encode(texts)
        print(f"   Dimension: {encoder.dimension}")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Sparsity: {(embeddings[0] == 0).sum() / len(embeddings[0]):.2%} zeros")
        print()
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Test caching
    print("3. Testing cache")
    try:
        encoder = create_text_encoder('sentence-transformer', use_cache=True)
        import time
        
        # First call (no cache)
        start = time.time()
        _ = encoder.encode("test query")
        time1 = time.time() - start
        
        # Second call (cached)
        start = time.time()
        _ = encoder.encode("test query")
        time2 = time.time() - start
        
        print(f"   First call: {time1*1000:.2f}ms")
        print(f"   Cached call: {time2*1000:.2f}ms")
        print(f"   Speedup: {time1/time2:.1f}x")
    except Exception as e:
        print(f"   Error: {e}")
