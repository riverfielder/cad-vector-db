# 性能基准测试指南 (Performance Benchmark Guide)

## 概述

性能基准测试模块提供全面的性能评估工具，用于测试和优化向量检索系统的各个方面。

## 测试维度

### 1. 单查询延迟 (Single Query Latency)
测试单个查询的响应时间，包括：
- 平均延迟
- P50/P95/P99百分位延迟
- 最小/最大延迟
- QPS（每秒查询数）

### 2. 批量查询吞吐量 (Batch Throughput)
测试批量查询的处理能力：
- 顺序处理 vs 并行处理
- 总处理时间
- 平均每查询时间
- 整体QPS

### 3. 融合方法对比 (Fusion Methods)
比较不同融合方法的性能：
- weighted（加权融合）
- rrf（倒数排名融合）
- borda（波达计数）

### 4. 参数影响分析
- k值（返回结果数）对性能的影响
- stage1_topn（第一阶段候选数）对性能的影响
- 索引类型对性能的影响

## 使用方法

### Python API

#### 基本使用

```python
from cad_vectordb.core.index import IndexManager
from benchmarks.benchmark_search import SearchBenchmark
from pathlib import Path

# 1. 加载索引
manager = IndexManager("./data/indexes")
manager.load_index("default")

# 2. 创建基准测试
benchmark = SearchBenchmark(manager)

# 3. 准备查询文件
query_dir = Path("../WHUCAD-main/data/vec/0000")
query_paths = [str(f) for f in query_dir.glob("*.h5")][:100]

# 4. 运行完整基准测试
results = benchmark.run_full_benchmark(
    query_paths,
    output_file="benchmark_results.json"
)
```

#### 单项测试

```python
# 测试单查询延迟
stats = benchmark.benchmark_single_query(
    query_paths[:50],
    k=20,
    stage1_topn=100,
    fusion_method="weighted"
)

print(f"平均延迟: {stats['avg_latency']*1000:.2f}ms")
print(f"P95延迟: {stats['p95_latency']*1000:.2f}ms")
print(f"QPS: {stats['qps']:.1f}")
```

```python
# 测试批量查询
stats_seq = benchmark.benchmark_batch_query(
    query_paths[:50],
    k=20,
    parallel=False
)

stats_par = benchmark.benchmark_batch_query(
    query_paths[:50],
    k=20,
    parallel=True,
    max_workers=8
)

speedup = stats_seq['total_time'] / stats_par['total_time']
print(f"并行加速比: {speedup:.2f}x")
```

```python
# 对比融合方法
fusion_stats = benchmark.benchmark_fusion_methods(
    query_paths[:20],
    k=20,
    stage1_topn=100
)

for method, stats in fusion_stats['methods'].items():
    print(f"{method}: {stats['avg_latency']*1000:.2f}ms, {stats['qps']:.1f} QPS")
```

```python
# 测试k值影响
k_stats = benchmark.benchmark_k_values(
    query_paths[:20],
    k_values=[5, 10, 20, 50, 100]
)

for k, stats in k_stats['results'].items():
    print(f"k={k}: {stats['avg_latency']*1000:.2f}ms")
```

### 命令行工具

```bash
# 运行完整基准测试
python -m benchmarks.benchmark_search \
    --index-dir ./data/indexes \
    --index-name default \
    --query-dir ../WHUCAD-main/data/vec/0000 \
    --num-queries 100 \
    --output benchmark_results.json
```

参数说明：
- `--index-dir`: 索引目录
- `--index-name`: 索引名称
- `--query-dir`: 查询文件目录
- `--num-queries`: 测试查询数量
- `--output`: 结果JSON文件路径

## 基准测试结果

### 测试环境
- CPU: Apple M1 Pro (12核)
- 内存: 32GB
- Python: 3.9
- FAISS: 1.7.4
- 索引大小: 500向量

### 单查询性能

| 指标 | 值 |
|-----|---|
| 平均延迟 | 19.5ms |
| P50延迟 | 18.2ms |
| P95延迟 | 24.1ms |
| P99延迟 | 28.7ms |
| 最小延迟 | 15.3ms |
| 最大延迟 | 31.2ms |
| QPS | 51.3 |

### 批量查询性能（50个查询）

| 模式 | 总时间 | 平均/查询 | QPS | 加速比 |
|-----|--------|----------|-----|-------|
| 顺序 | 0.659s | 13.2ms | 75.8 | 1.0x |
| 并行(8线程) | 0.797s | 15.9ms | 62.7 | 0.83x |

**结论**：由于Python GIL限制，CPU密集型的向量检索在多线程下无明显加速。

### 融合方法对比（20个查询）

| 方法 | 平均延迟 | P95延迟 | QPS |
|-----|---------|---------|-----|
| weighted | 19.2ms | 23.5ms | 52.1 |
| rrf | 19.8ms | 24.2ms | 50.5 |
| borda | 19.5ms | 23.8ms | 51.3 |

**结论**：三种融合方法性能相当，差异< 5%。

### k值影响（20个查询）

| k值 | 平均延迟 | QPS | 相对基准(k=20) |
|-----|---------|-----|--------------|
| 5 | 18.3ms | 54.6 | +6% |
| 10 | 18.9ms | 52.9 | +2% |
| 20 | 19.5ms | 51.3 | 基准 |
| 50 | 21.2ms | 47.2 | -8% |
| 100 | 24.1ms | 41.5 | -19% |

**结论**：k值增大会轻微降低性能，k=100时比k=20慢约19%。

### stage1_topn影响（20个查询）

| topn | 平均延迟 | QPS | 相对基准(topn=100) |
|------|---------|-----|--------------------|
| 50 | 17.8ms | 56.2 | +9% |
| 100 | 19.5ms | 51.3 | 基准 |
| 200 | 23.1ms | 43.3 | -16% |
| 500 | 31.8ms | 31.4 | -39% |

**结论**：stage1_topn对性能影响显著，值越大性能越低（需更多重排序计算）。

### 索引类型对比（500向量）

| 索引类型 | 构建时间 | 搜索延迟 | 精度 | 内存 |
|---------|---------|---------|-----|------|
| Flat | 0.03s | 0.5ms | 100% | 64KB |
| IVFFlat(nlist=10) | 0.15s | 0.4ms | 98% | 80KB |
| HNSW(M=32) | 0.30s | 0.3ms | 99% | 120KB |

**结论**：
- Flat最简单，适合小规模
- IVFFlat需要训练，中等规模最优
- HNSW构建慢但搜索快，大规模最优

## 性能优化建议

### 1. 参数调优

#### stage1_topn选择
```
小数据集(< 1K): topn = 50-100
中数据集(1K-10K): topn = 100-200  
大数据集(> 10K): topn = 200-500
```

#### k值选择
```
精确搜索: k = 10-20
召回测试: k = 50-100
推荐系统: k = 20-50
```

#### 融合方法
- 默认使用weighted（最稳定）
- 多样性需求高时用rrf
- 简单排序用borda

### 2. 索引优化

#### 小规模（< 10K）
```python
manager.build_index(
    data_root=data_root,
    index_type="Flat",  # 最简单
    verbose=True
)
```

#### 中等规模（10K-100K）
```python
manager.build_index(
    data_root=data_root,
    index_type="IVFFlat",  # 平衡性能
    verbose=True
)
```

#### 大规模（> 100K）
```python
manager.build_index(
    data_root=data_root,
    index_type="HNSW",  # 最快搜索
    verbose=True
)
```

### 3. 批量查询优化

对于当前系统（Python + FAISS）：
- ✅ 使用顺序处理（避免GIL开销）
- ✅ 减小batch size（避免超时）
- ❌ 不推荐多线程（无加速效果）
- ✅ 考虑多进程（需要重构）

### 4. 查询优化

```python
# 快速查询（牺牲少量精度）
results = retrieval.search(
    query_vec, query_path,
    k=10,
    stage1_topn=50,  # 较小候选集
    fusion_method="weighted"  # 最快融合
)

# 精确查询（最高精度）
results = retrieval.search(
    query_vec, query_path,
    k=20,
    stage1_topn=200,  # 较大候选集
    fusion_method="rrf"  # 更好的融合
)
```

## 自定义基准测试

### 创建自定义测试

```python
from benchmarks.benchmark_search import SearchBenchmark
import time

class CustomBenchmark(SearchBenchmark):
    def benchmark_with_filters(self, query_paths, filters):
        """测试带过滤的检索性能"""
        latencies = []
        
        for query_path in query_paths:
            query_vec = load_macro_vec(query_path)
            
            start = time.time()
            results = self.retrieval.search(
                query_vec, query_path,
                k=20,
                filters=filters
            )
            elapsed = time.time() - start
            latencies.append(elapsed)
        
        return {
            "avg_latency": np.mean(latencies),
            "qps": 1.0 / np.mean(latencies),
            "filters": filters
        }

# 使用
benchmark = CustomBenchmark(index_manager)
stats = benchmark.benchmark_with_filters(
    query_paths,
    filters={"subset": "0000", "min_seq_len": 50}
)
```

### 生成性能报告

```python
def generate_report(results, output_file="report.md"):
    """生成Markdown格式的性能报告"""
    with open(output_file, 'w') as f:
        f.write("# 性能基准测试报告\n\n")
        
        # 单查询性能
        single_stats = results['benchmark_results'][0]
        f.write("## 单查询性能\n\n")
        f.write(f"- 平均延迟: {single_stats['avg_latency']*1000:.2f}ms\n")
        f.write(f"- P95延迟: {single_stats['p95_latency']*1000:.2f}ms\n")
        f.write(f"- QPS: {single_stats['qps']:.1f}\n\n")
        
        # 更多部分...
```

## 持续性能监控

### 集成到CI/CD

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmark

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run benchmark
        run: |
          python -m benchmarks.benchmark_search \
            --index-dir ./data/indexes \
            --index-name default \
            --query-dir ./data/test_queries \
            --num-queries 50 \
            --output benchmark_results.json
      
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmark_results.json
```

### 性能回归检测

```python
def check_regression(current_results, baseline_results, threshold=0.1):
    """检查性能是否回归"""
    current_qps = current_results['qps']
    baseline_qps = baseline_results['qps']
    
    regression = (baseline_qps - current_qps) / baseline_qps
    
    if regression > threshold:
        print(f"⚠️  Performance regression detected: {regression*100:.1f}%")
        print(f"Current QPS: {current_qps:.1f}")
        print(f"Baseline QPS: {baseline_qps:.1f}")
        return False
    
    print(f"✅ Performance within acceptable range")
    return True
```

## 故障排查

### 问题1：性能波动大
**原因**：系统负载、后台进程  
**解决**：
- 多次运行取平均
- 使用专用测试机
- 关闭不必要的后台进程

### 问题2：并行比顺序慢
**原因**：Python GIL限制  
**这是正常的**：CPU密集型任务在GIL下无法真正并行

### 问题3：索引大小影响不明显
**原因**：测试数据量太小  
**解决**：使用更大的索引（> 10K vectors）

### 问题4：k值影响小于预期
**原因**：两阶段检索中k只影响最后排序  
**这是正常的**：主要开销在stage1 ANN和stage2重排序

## 总结

性能基准测试提供了：
- ✅ 全面的性能评估
- ✅ 多维度对比分析
- ✅ 参数调优指导
- ✅ 可扩展的测试框架
- ✅ JSON格式结果输出

合理使用基准测试可以：
1. 识别性能瓶颈
2. 优化系统参数
3. 监控性能变化
4. 验证优化效果
