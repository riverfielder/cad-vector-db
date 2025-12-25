# Semantic Query Support - User Guide

## Overview

语义查询支持 (Semantic Query Support) 让您可以使用自然语言文本来搜索CAD模型，无需提供CAD向量。该功能支持中英文双语，并提供多种文本编码器选择。

## Features

- **🌐 Multilingual Support**: 支持中文和英文查询
- **🔄 Multiple Encoders**: Sentence-BERT, CLIP, BM25
- **🔍 Hybrid Search**: 结合文本和向量搜索
- **📊 Explainable Results**: 详细的相似度解释
- **⚡ Fast & Cached**: 查询缓存加速重复搜索
- **🎯 Metadata Filtering**: 按subset等元数据过滤

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies (already installed)
pip install sentence-transformers  # For multilingual text encoding

# Optional: for CLIP support
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# Optional: for Chinese tokenization in BM25
pip install jieba
```

### 2. Basic Usage (Python)

```python
from cad_vectordb.core.index import IndexManager
from cad_vectordb.core.retrieval import TwoStageRetrieval
from cad_vectordb.core.text_encoder import create_text_encoder

# Load index
index_manager = IndexManager("index")
index_manager.load_index()

# Initialize retrieval system
retrieval = TwoStageRetrieval(index_manager)

# Create text encoder
encoder = create_text_encoder('sentence-transformer', device='cpu')

# Semantic search
results = retrieval.semantic_search(
    query_text="圆柱形零件",  # "cylindrical part" in Chinese
    text_encoder=encoder,
    k=10
)

# Display results
for res in results:
    print(f"{res['id']}: {res['score']:.4f}")
```

### 3. Basic Usage (REST API)

```bash
# Start the API server
python server/app.py

# Send semantic search request
curl -X POST "http://localhost:8000/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "cylindrical mechanical part",
    "k": 10,
    "encoder_type": "sentence-transformer"
  }'
```

## Text Encoders

### 1. Sentence-BERT (Recommended)

**Best for**: General semantic search, multilingual support

```python
encoder = create_text_encoder(
    encoder_type='sentence-transformer',
    model_name='paraphrase-multilingual-MiniLM-L12-v2',
    device='cpu'
)
```

**Features**:
- ✅ Excellent Chinese + English support
- ✅ Fast inference (384 dimensions)
- ✅ Good semantic understanding
- ✅ Pre-trained on large text corpus

**Available Models**:
- `paraphrase-multilingual-MiniLM-L12-v2` (384d, fast, multilingual)
- `paraphrase-multilingual-mpnet-base-v2` (768d, better quality, multilingual)
- `all-MiniLM-L6-v2` (384d, fastest, English only)

### 2. CLIP

**Best for**: Multi-modal search (text + images)

```python
encoder = create_text_encoder(
    encoder_type='clip',
    model_name='ViT-B/32',
    device='cpu'
)
```

**Features**:
- ✅ Joint text-image embedding space
- ✅ Can link text to CAD renderings
- ⚠️ Requires more GPU memory
- ⚠️ English-focused (limited Chinese support)

**Available Models**:
- `ViT-B/32` (512d, balanced)
- `ViT-B/16` (512d, better quality)
- `ViT-L/14` (768d, best quality, slow)

### 3. BM25

**Best for**: Lightweight keyword-based search

```python
encoder = create_text_encoder(
    encoder_type='bm25',
    tokenizer='jieba',  # 'jieba' for Chinese, 'simple' for English
    vocab_size=10000
)
```

**Features**:
- ✅ No neural network, fast
- ✅ Small memory footprint
- ✅ Good for exact keyword matching
- ⚠️ Limited semantic understanding

## API Endpoints

### POST /search/semantic

Semantic search using natural language queries.

**Request**:
```json
{
  "query_text": "圆柱形机械零件",
  "k": 20,
  "encoder_type": "sentence-transformer",
  "model_name": null,
  "filters": {"subset": "0000"},
  "explainable": false
}
```

**Response**:
```json
{
  "status": "success",
  "query_text": "圆柱形机械零件",
  "num_results": 20,
  "results": [
    {
      "id": "00000123",
      "score": 0.8567,
      "metadata": {"subset": "0000", "seq_len": 45},
      "query_text": "圆柱形机械零件",
      "search_type": "semantic"
    },
    ...
  ]
}
```

**With Explanation** (`explainable=true`):
```json
{
  "status": "success",
  "results": [...],
  "explanation": {
    "query_text": "圆柱形机械零件",
    "top_match": {
      "id": "00000123",
      "score": 0.8567,
      "subset": "0000"
    },
    "interpretation": "Excellent semantic match",
    "encoder_type": "SentenceTransformerEncoder"
  }
}
```

### POST /search/hybrid

Hybrid search combining semantic and vector-based retrieval.

**Request**:
```json
{
  "query_text": "cylindrical part",
  "query_file_path": "data/vec/0000/00000000.h5",
  "k": 20,
  "semantic_weight": 0.5,
  "vector_weight": 0.5,
  "encoder_type": "sentence-transformer"
}
```

**Response**:
```json
{
  "status": "success",
  "num_results": 20,
  "results": [
    {
      "id": "00000123",
      "score": 0.8234,
      "semantic_score": 0.4117,
      "vector_score": 0.4117,
      "metadata": {...},
      "search_type": "hybrid"
    },
    ...
  ]
}
```

## Use Cases

### 1. Natural Language CAD Search

```python
# Chinese query
results = retrieval.semantic_search(
    query_text="找一个圆柱形的轴，带螺纹",
    text_encoder=encoder,
    k=10
)

# English query
results = retrieval.semantic_search(
    query_text="find a cylindrical shaft with threads",
    text_encoder=encoder,
    k=10
)
```

### 2. Metadata-Filtered Semantic Search

```python
# Search only in specific subset
results = retrieval.semantic_search(
    query_text="mechanical gear",
    text_encoder=encoder,
    k=10,
    filters={"subset": "0001"}
)
```

### 3. Explainable Semantic Search

```python
# Get detailed similarity explanation
results, explanation = retrieval.semantic_search(
    query_text="component with holes",
    text_encoder=encoder,
    k=10,
    explainable=True
)

print(explanation['interpretation'])
# Output: "Excellent semantic match"
```

### 4. Hybrid Text + Vector Search

```python
# Best of both worlds
results = retrieval.hybrid_search(
    query_text="cylindrical part",
    text_encoder=encoder,
    query_vec=my_cad_vector,
    query_file_path="query.h5",
    k=10,
    semantic_weight=0.6,  # Favor text matching
    vector_weight=0.4
)
```

### 5. Batch Semantic Queries

```python
queries = [
    "圆形零件",
    "cylindrical part",
    "mechanical gear",
    "threaded shaft"
]

for query in queries:
    results = retrieval.semantic_search(
        query_text=query,
        text_encoder=encoder,
        k=5
    )
    print(f"{query}: {results[0]['id']}")
```

## Performance Optimization

### 1. Enable Query Caching

```python
# Caching speeds up repeated queries
encoder = create_text_encoder(
    'sentence-transformer',
    use_cache=True,
    cache_file=Path("cache/text_embeddings.npz")
)

# First query: ~100ms
encoder.encode("cylindrical part")

# Cached query: ~1ms (100x faster)
encoder.encode("cylindrical part")
```

### 2. Use GPU Acceleration

```python
# Much faster on GPU
encoder = create_text_encoder(
    'sentence-transformer',
    device='cuda'  # Requires CUDA-enabled GPU
)
```

### 3. Choose Lightweight Models

```python
# Fast multilingual model (384d)
encoder = create_text_encoder(
    'sentence-transformer',
    model_name='paraphrase-multilingual-MiniLM-L12-v2'
)

# For English-only, even faster
encoder = create_text_encoder(
    'sentence-transformer',
    model_name='all-MiniLM-L6-v2'
)
```

## Query Examples

### Chinese Queries (中文查询)

```python
queries = [
    "圆柱形零件",           # Cylindrical part
    "带螺纹的轴",           # Threaded shaft
    "机械齿轮",             # Mechanical gear
    "有孔的零件",           # Part with holes
    "圆形法兰",             # Round flange
    "管道连接件",           # Pipe connector
]
```

### English Queries

```python
queries = [
    "cylindrical mechanical part",
    "threaded shaft component",
    "mechanical gear with teeth",
    "component with mounting holes",
    "round flange connector",
    "pipe fitting",
]
```

### Descriptive Queries

```python
# More detailed descriptions
queries = [
    "Find a cylindrical part with threads on one end",
    "找一个有多个安装孔的圆形法兰",
    "Mechanical gear with fine teeth",
    "Shaft with keyway and threads",
]
```

## Troubleshooting

### Issue: Dimension Mismatch Error

**Error**: `Text encoder dimension (384) does not match index dimension (256)`

**Solution**: 
The text encoder's output dimension must match your FAISS index dimension. Options:
1. Rebuild index with encoder dimension
2. Train a projection layer to map encoder output to index dimension
3. Use a different encoder with matching dimension

### Issue: Low Semantic Scores

**Problem**: All scores < 0.5

**Solutions**:
1. Use more descriptive queries
2. Try different encoder models
3. Consider hybrid search (text + vector)
4. Check if encoder is trained on relevant domain

### Issue: Slow First Query

**Problem**: First query takes 5-10 seconds

**Explanation**: Model loading time. Subsequent queries are fast.

**Solutions**:
1. Pre-load encoder on server startup
2. Enable caching
3. Use smaller models (MiniLM vs MPNet)

### Issue: Out of Memory

**Problem**: CUDA out of memory with CLIP

**Solutions**:
1. Use CPU instead: `device='cpu'`
2. Use smaller CLIP model: `ViT-B/32` instead of `ViT-L/14`
3. Process queries in smaller batches

## Architecture

```
┌─────────────────┐
│  Text Query     │ "圆柱形零件"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Encoder   │ Sentence-BERT / CLIP / BM25
└────────┬────────┘
         │
         ▼ Embedding (384d / 512d / ...)
┌─────────────────┐
│  FAISS Index    │ Cosine Similarity Search
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Results        │ Top-K CAD Models
└─────────────────┘
```

## Next Steps

1. **Try Examples**:
   ```bash
   python examples/semantic_search_example.py
   python examples/semantic_search_api_example.py
   ```

2. **Explore Different Encoders**: Test which encoder works best for your queries

3. **Optimize Performance**: Enable caching, use GPU if available

4. **Integrate with Your Application**: Use REST API or Python SDK

5. **Train Custom Encoder** (Advanced): Fine-tune on your CAD domain data

## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Multilingual Models](https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models)

## Support

For issues or questions:
1. Check the examples in `examples/`
2. Review API documentation above
3. Examine error messages for hints
4. Test with different encoders and queries
