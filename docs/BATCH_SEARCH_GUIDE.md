# 批量检索指南 (Batch Search Guide)

## 概述

批量检索功能允许在单个API请求中处理多个查询，简化客户端代码并提供统一的性能统计。

## 核心特性

### 1. 批量处理
- 一次请求处理多个查询文件
- 统一的参数配置
- 聚合的结果返回

### 2. 处理模式
- **顺序模式**：按顺序逐个处理查询（默认关闭并行）
- **并行模式**：使用线程池并行处理（适合I/O密集型场景）

### 3. 性能统计
- 总查询数量
- 成功/失败计数
- 总耗时和平均每查询耗时
- QPS（每秒查询数）

### 4. 错误处理
- 单个查询失败不影响其他查询
- 返回每个查询的详细错误信息
- 聚合统计所有成功和失败的查询

## API 接口

### 端点
```
POST /search/batch
```

### 请求参数

```python
{
    "query_file_paths": [str],  # 查询文件路径列表（必需）
    "k": int,                    # 返回Top-K结果（默认20）
    "stage1_topn": int,          # 两阶段检索第一阶段候选数（默认100）
    "fusion_method": str,        # 融合方法："weighted"/"rrf"/"borda"（默认weighted）
    "alpha": float,              # 第一阶段权重（默认0.6）
    "beta": float,               # 第二阶段权重（默认0.4）
    "filters": dict,             # 元数据过滤器（可选）
    "explainable": bool,         # 是否返回可解释性分析（默认false）
    "parallel": bool             # 是否使用并行处理（默认true）
}
```

### 响应格式

```json
{
    "status": "success",
    "total_queries": 50,
    "successful": 50,
    "failed": 0,
    "elapsed_time": 2.430,
    "avg_time_per_query": 0.049,
    "parallel": false,
    "results": {
        "/path/to/query1.h5": {
            "status": "success",
            "results": [...],
            "explanation": {...}  // 仅在explainable=true时
        },
        "/path/to/query2.h5": {
            "status": "error",
            "error": "Query file not found"
        }
    }
}
```

## 使用示例

### 1. 基本批量检索

```python
import requests

# 准备查询文件列表
query_files = [
    "/path/to/queries/00000000.h5",
    "/path/to/queries/00000001.h5",
    "/path/to/queries/00000002.h5"
]

# 发送批量检索请求
response = requests.post(
    "http://localhost:8000/search/batch",
    json={
        "query_file_paths": query_files,
        "k": 10,
        "parallel": False  # 使用顺序模式
    }
)

result = response.json()
print(f"处理了 {result['total_queries']} 个查询")
print(f"成功: {result['successful']}, 失败: {result['failed']}")
print(f"平均耗时: {result['avg_time_per_query']:.3f}秒/查询")
```

### 2. 带混合检索的批量查询

```python
# 使用元数据过滤
response = requests.post(
    "http://localhost:8000/search/batch",
    json={
        "query_file_paths": query_files,
        "k": 20,
        "filters": {
            "subset": "0000",      # 只在子集0000中搜索
            "min_seq_len": 50,     # 最小序列长度
            "max_seq_len": 200     # 最大序列长度
        },
        "parallel": True
    }
)
```

### 3. 带可解释性的批量查询

```python
# 启用可解释性分析
response = requests.post(
    "http://localhost:8000/search/batch",
    json={
        "query_file_paths": query_files,
        "k": 10,
        "stage1_topn": 100,
        "explainable": True,  # 返回相似度分解
        "parallel": False     # 复杂查询建议使用顺序模式
    },
    timeout=180  # 增加超时时间
)

# 访问第一个查询的解释
first_query = query_files[0]
explanation = result['results'][first_query]['explanation']
print(f"融合方法: {explanation['fusion_method']}")
print(f"最佳匹配: {explanation['top_match']['id']}")
```

### 4. 完整的性能测试脚本

```python
#!/usr/bin/env python3
import requests
import time
from pathlib import Path

API_URL = "http://localhost:8000"

# 收集查询文件
query_files = []
data_dir = Path("/path/to/data/vec/0000")
for h5_file in sorted(data_dir.glob("*.h5"))[:100]:
    query_files.append(str(h5_file))

print(f"准备测试 {len(query_files)} 个查询")

# 测试顺序模式
print("\n🔄 测试顺序模式...")
start = time.time()
response = requests.post(
    f"{API_URL}/search/batch",
    json={
        "query_file_paths": query_files,
        "k": 10,
        "parallel": False
    },
    timeout=300
)
seq_time = time.time() - start
seq_result = response.json()

print(f"✅ 顺序模式: {seq_result['successful']} 查询成功")
print(f"   耗时: {seq_time:.3f}秒")
print(f"   QPS: {len(query_files)/seq_time:.1f} 查询/秒")

# 测试并行模式
time.sleep(2)
print("\n⚡ 测试并行模式...")
start = time.time()
response = requests.post(
    f"{API_URL}/search/batch",
    json={
        "query_file_paths": query_files,
        "k": 10,
        "parallel": True
    },
    timeout=300
)
par_time = time.time() - start
par_result = response.json()

print(f"✅ 并行模式: {par_result['successful']} 查询成功")
print(f"   耗时: {par_time:.3f}秒")
print(f"   QPS: {len(query_files)/par_time:.1f} 查询/秒")
print(f"   加速比: {seq_time/par_time:.2f}x")
```

## 性能特征

### 实测结果（500向量索引）

#### 简单查询（默认参数）
- **10个查询**
  - 顺序模式: 0.190秒 (52.6 QPS)
  - 并行模式: 0.192秒 (52.1 QPS)
  - 加速比: 0.99x

- **50个查询**
  - 顺序模式: 0.659秒 (75.8 QPS)
  - 并行模式: 0.797秒 (62.7 QPS)
  - 加速比: 0.83x

#### 复杂查询（两阶段检索 + 可解释性）
- **50个查询**
  - 顺序模式: 2.430秒 (19.6 QPS)
  - 并行模式: 2.729秒 (17.7 QPS)
  - 加速比: 0.90x

### 性能分析

当前实现使用 `ThreadPoolExecutor`（多线程），由于Python的全局解释器锁(GIL)限制：

1. **CPU密集型任务**：FAISS向量搜索和numpy计算无法真正并行化
2. **线程开销**：线程创建和管理的开销抵消了并行收益
3. **内存访问**：大量内存访问操作在GIL下串行执行

**结论**：批量检索主要作为**便捷性功能**，简化客户端代码和提供统一的错误处理，而不是性能优化功能。

### 使用建议

1. **小批量查询（< 10）**：使用顺序模式，避免线程开销
2. **中等批量（10-50）**：顺序模式通常更快
3. **大批量（> 50）**：可尝试并行模式，但效果有限
4. **复杂查询**：始终使用顺序模式（explainable=true）

## 错误处理

### 单个查询失败

```python
{
    "status": "success",  # 整体状态仍为success
    "total_queries": 3,
    "successful": 2,
    "failed": 1,
    "results": {
        "query1.h5": {"status": "success", "results": [...]},
        "query2.h5": {"status": "error", "error": "File not found"},
        "query3.h5": {"status": "success", "results": [...]}
    }
}
```

### 批量请求失败

```python
{
    "detail": "Batch search error: max_workers must be greater than 0"
}
```

## 高级用法

### 1. 动态文件收集

```python
from pathlib import Path

# 从目录收集所有.h5文件
data_dir = Path("/path/to/data")
query_files = [str(f) for f in data_dir.rglob("*.h5")][:100]

# 按子集分组批量查询
subsets = {}
for f in query_files:
    subset = f.split('/')[-2]  # 提取子集编号
    if subset not in subsets:
        subsets[subset] = []
    subsets[subset].append(f)

# 分批处理每个子集
for subset, files in subsets.items():
    response = requests.post(
        f"{API_URL}/search/batch",
        json={"query_file_paths": files, "k": 10}
    )
    print(f"子集 {subset}: {response.json()['successful']} 成功")
```

### 2. 结果聚合和分析

```python
# 收集所有Top-1结果
top1_results = {}
for query_path, query_result in result['results'].items():
    if query_result['status'] == 'success':
        top1 = query_result['results'][0]
        top1_results[query_path] = {
            'id': top1['id'],
            'score': top1['score']
        }

# 计算平均相似度
avg_score = sum(r['score'] for r in top1_results.values()) / len(top1_results)
print(f"平均Top-1相似度: {avg_score:.4f}")
```

### 3. 增量重试失败的查询

```python
# 第一次批量请求
response = requests.post(f"{API_URL}/search/batch", json={
    "query_file_paths": all_queries,
    "k": 10
})
result = response.json()

# 收集失败的查询
failed_queries = [
    path for path, res in result['results'].items()
    if res['status'] == 'error'
]

if failed_queries:
    print(f"重试 {len(failed_queries)} 个失败的查询...")
    retry_response = requests.post(f"{API_URL}/search/batch", json={
        "query_file_paths": failed_queries,
        "k": 10
    })
```

## 对比单次检索

### 批量检索的优势
1. **简化代码**：一次请求处理多个查询
2. **统一配置**：所有查询使用相同参数
3. **统计信息**：自动计算性能指标
4. **错误聚合**：集中处理所有错误

### 单次检索的优势
1. **灵活性**：每个查询可使用不同参数
2. **流式处理**：可逐个处理结果
3. **更低延迟**：单个查询立即返回
4. **资源控制**：更好的并发控制

## 未来优化方向

如需要真正的性能提升，可考虑：

1. **多进程处理**：使用 `ProcessPoolExecutor` 绕过GIL限制
2. **异步I/O**：使用 `asyncio` 处理I/O密集型操作
3. **GPU加速**：使用FAISS GPU版本进行并行搜索
4. **分布式索引**：将索引分片到多个节点
5. **查询缓存**：缓存常见查询的结果

## 故障排查

### 问题1：并行模式比顺序模式慢
**原因**：线程开销 + Python GIL限制  
**解决**：使用顺序模式（`parallel: false`）

### 问题2：超时错误
**原因**：查询数量太多或查询过于复杂  
**解决**：
- 减少批量大小
- 增加客户端超时时间
- 关闭可解释性分析（`explainable: false`）

### 问题3：部分查询失败
**原因**：文件不存在或文件格式错误  
**解决**：
- 检查 `results` 中的错误信息
- 验证所有文件路径
- 重试失败的查询

## 总结

批量检索功能提供了便捷的批量查询接口，适合：
- 批量评估和测试
- 简化客户端代码
- 统一的性能统计
- 集中式错误处理

对于性能敏感的应用，建议根据具体场景选择合适的处理模式，或考虑使用多进程/分布式方案。
