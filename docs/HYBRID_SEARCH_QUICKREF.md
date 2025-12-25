# 混合检索快速参考

## API 请求格式

```bash
POST /search
Content-Type: application/json

{
  "query_file_path": "<路径>",
  "k": 5,                      # Top-K 结果数
  "stage1_topn": 50,           # Stage 1 候选数
  "filters": {                 # 可选：元数据过滤器
    "subset": "0000",          # 子集（str 或 list）
    "min_seq_len": 8,          # 最小序列长度
    "max_seq_len": 50,         # 最大序列长度
    "label": "某标签"          # 标签（str 或 list）
  }
}
```

## 常用示例

### 1. 标准检索（无过滤）
```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{
  "query_file_path": "/path/to/query.h5",
  "k": 10
}'
```

### 2. 子集过滤
```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{
  "query_file_path": "/path/to/query.h5",
  "k": 10,
  "filters": {"subset": "0000"}
}'
```

### 3. 多子集过滤
```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{
  "query_file_path": "/path/to/query.h5",
  "k": 10,
  "filters": {"subset": ["0000", "0001", "0002"]}
}'
```

### 4. 序列长度范围
```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{
  "query_file_path": "/path/to/query.h5",
  "k": 10,
  "filters": {
    "min_seq_len": 10,
    "max_seq_len": 50
  }
}'
```

### 5. 多条件组合
```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{
  "query_file_path": "/path/to/query.h5",
  "k": 10,
  "filters": {
    "subset": "0000",
    "min_seq_len": 10,
    "max_seq_len": 50
  }
}'
```

## Python 客户端代码

```python
import requests

# API 端点
API_URL = "http://localhost:8000/search"

# 标准检索
def search(query_path, k=10):
    response = requests.post(API_URL, json={
        "query_file_path": query_path,
        "k": k
    })
    return response.json()

# 混合检索（带过滤）
def hybrid_search(query_path, k=10, filters=None):
    payload = {
        "query_file_path": query_path,
        "k": k
    }
    if filters:
        payload["filters"] = filters
    
    response = requests.post(API_URL, json=payload)
    return response.json()

# 使用示例
results = hybrid_search(
    query_path="/path/to/query.h5",
    k=5,
    filters={
        "subset": "0000",
        "min_seq_len": 8,
        "max_seq_len": 10
    }
)

for result in results:
    print(f"ID: {result['id']}, Score: {result['score']:.3f}")
    print(f"  Subset: {result['metadata']['subset']}, Seq Len: {result['metadata']['seq_len']}")
```

## JavaScript 客户端代码

```javascript
// 标准检索
async function search(queryPath, k = 10) {
  const response = await fetch('http://localhost:8000/search', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      query_file_path: queryPath,
      k: k
    })
  });
  return await response.json();
}

// 混合检索（带过滤）
async function hybridSearch(queryPath, k = 10, filters = null) {
  const payload = {
    query_file_path: queryPath,
    k: k
  };
  if (filters) {
    payload.filters = filters;
  }
  
  const response = await fetch('http://localhost:8000/search', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  return await response.json();
}

// 使用示例
const results = await hybridSearch('/path/to/query.h5', 5, {
  subset: '0000',
  min_seq_len: 8,
  max_seq_len: 10
});

results.forEach(result => {
  console.log(`ID: ${result.id}, Score: ${result.score.toFixed(3)}`);
  console.log(`  Subset: ${result.metadata.subset}, Seq Len: ${result.metadata.seq_len}`);
});
```

## 响应格式

```json
[
  {
    "id": "0000/00000123.h5",
    "score": 0.918,
    "sim_stage1": 0.863,
    "sim_stage2": 1.0,
    "metadata": {
      "id": "0000/00000123.h5",
      "file_path": "/full/path/to/file.h5",
      "subset": "0000",
      "seq_len": 10
    }
  },
  ...
]
```

## 过滤器参数速查

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `subset` | str \| list[str] | 子集过滤 | `"0000"` 或 `["0000", "0001"]` |
| `min_seq_len` | int | 最小序列长度 | `8` |
| `max_seq_len` | int | 最大序列长度 | `50` |
| `label` | str \| list[str] | 标签过滤 | `"label1"` 或 `["label1", "label2"]` |

**注意**:
- 所有过滤器都是可选的
- 多个过滤器使用 AND 逻辑
- `filters` 参数本身也是可选的

## 性能提示

1. **过滤比例**: 当过滤掉 > 80% 的数据时，响应时间最优
2. **stage1_topn**: 使用过滤时会自动扩大 3 倍，无需手动调整
3. **索引优化**: OceanBase 已在 subset, seq_len, label 字段上建立索引
4. **批量查询**: 使用多子集过滤比分次查询更高效

## 常见错误

### 1. 文件不存在
```json
{"detail": "Query file not found: /path/to/file.h5"}
```
**解决**: 使用绝对路径或检查文件是否存在

### 2. 过滤结果为空
返回结果少于 k 个或为空列表

**原因**: 过滤条件太严格，没有符合的向量

**解决**: 放宽过滤条件或检查元数据值

### 3. 响应超时
**原因**: 索引未加载或查询文件过大

**解决**: 等待索引加载完成（3-5秒）或检查服务器日志

## 更多信息

- **完整指南**: [docs/HYBRID_SEARCH_GUIDE.md](./HYBRID_SEARCH_GUIDE.md)
- **实现总结**: [docs/HYBRID_SEARCH_SUMMARY.md](./HYBRID_SEARCH_SUMMARY.md)
- **API 文档**: [docs/API_TEST_RESULTS.md](./API_TEST_RESULTS.md)
