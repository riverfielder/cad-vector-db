# 解释性检索 (Explainable Retrieval) 指南

## 概述

解释性检索为向量数据库检索结果提供详细的相似度分解和解释，帮助用户理解**为什么**某个结果被检索出来，以及**如何**计算相似度分数。

## 核心功能

### 1. 相似度分解

将最终的融合分数分解为三个层次：

- **Stage 1 相似度**: 基于特征向量的 ANN (近似最近邻) 搜索相似度
- **Stage 2 相似度**: 基于完整宏序列的精确相似度计算
- **最终融合分数**: 两个阶段按权重融合后的最终得分

### 2. 特征级分析

对 Stage 2 的序列级相似度进行详细分解：

- **命令匹配率**: CAD 命令序列的匹配程度
- **参数距离**: 命令参数的欧氏距离
- **逐步距离**: 每个序列步骤的距离分布
- **序列长度信息**: 查询和候选的序列长度

### 3. 融合贡献分析

分析不同阶段对最终得分的贡献：

- **Stage 1 贡献**: 特征级匹配的贡献值和百分比
- **Stage 2 贡献**: 序列级匹配的贡献值和百分比
- **融合方法**: 使用的融合策略 (weighted/linear/rrf/borda)

### 4. 可视化展示

生成 HTML 可视化报告，包含：

- 相似度进度条和百分比
- 彩色编码的匹配质量
- 详细的数值指标
- 易于理解的文字解释

## API 使用方法

### 方法 1: JSON 响应（带解释）

在搜索请求中设置 `explainable=true`：

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_file_path": "/path/to/query.h5",
    "k": 5,
    "stage1_topn": 30,
    "fusion_method": "weighted",
    "alpha": 0.6,
    "beta": 0.4,
    "explainable": true
  }'
```

**响应示例**:

```json
[
  {
    "id": "0000/00000000.h5",
    "score": 1.0,
    "sim_stage1": 1.0,
    "sim_stage2": 1.0,
    "metadata": {
      "subset": "0000",
      "seq_len": 7
    },
    "explanation": {
      "stage1_similarity": 1.0,
      "stage2_similarity": 1.0,
      "final_score": 1.0,
      "fusion_method": "weighted",
      "contributions": {
        "stage1_weight": 0.6,
        "stage2_weight": 0.4,
        "stage1_contribution": 0.6,
        "stage2_contribution": 0.4,
        "stage1_percentage": 60.0,
        "stage2_percentage": 40.0
      },
      "stage1_interpretation": "Excellent feature-level match",
      "stage2_interpretation": "Excellent sequence-level match"
    },
    "stage2_details": {
      "total_distance": 0.0,
      "cmd_penalty": 0.0,
      "param_l2": 0.0,
      "normalized_param_l2": 0.0,
      "sequence_length": 7,
      "query_seq_len": 7,
      "candidate_seq_len": 7,
      "cmd_matches": 7,
      "cmd_mismatches": 0,
      "cmd_match_rate": 1.0,
      "avg_param_distance_per_step": 0.0,
      "max_param_distance_per_step": 0.0,
      "step_distances": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
  }
]
```

### 方法 2: HTML 可视化报告

使用可视化端点直接生成 HTML 报告：

```bash
curl -X POST http://127.0.0.1:8000/search/visualize \
  -H "Content-Type: application/json" \
  -d '{
    "query_file_path": "/path/to/query.h5",
    "k": 5,
    "stage1_topn": 30,
    "fusion_method": "weighted",
    "alpha": 0.6,
    "beta": 0.4
  }'
```

**响应**:

```json
{
  "status": "success",
  "visualization_file": "explanation.html",
  "num_results": 5,
  "message": "Visualization saved to explanation.html"
}
```

生成的 HTML 文件 `explanation.html` 可以在浏览器中直接打开查看。

## Python 客户端示例

```python
import requests
import webbrowser

API_URL = "http://127.0.0.1:8000"

# 方法 1: 获取 JSON 解释
def search_with_explanation(query_path, k=5):
    response = requests.post(f"{API_URL}/search", json={
        "query_file_path": query_path,
        "k": k,
        "explainable": True
    })
    results = response.json()
    
    # 打印解释
    for i, result in enumerate(results, 1):
        print(f"\n=== Rank {i}: {result['id']} ===")
        print(f"Final Score: {result['score']:.4f}")
        
        if 'explanation' in result:
            exp = result['explanation']
            print(f"Stage 1: {exp['stage1_similarity']:.4f} - {exp['stage1_interpretation']}")
            print(f"Stage 2: {exp['stage2_similarity']:.4f} - {exp['stage2_interpretation']}")
            
            if 'contributions' in exp:
                contrib = exp['contributions']
                print(f"Stage 1 Contribution: {contrib['stage1_percentage']:.1f}%")
                print(f"Stage 2 Contribution: {contrib['stage2_percentage']:.1f}%")
    
    return results

# 方法 2: 生成可视化报告
def visualize_results(query_path, k=5):
    response = requests.post(f"{API_URL}/search/visualize", json={
        "query_file_path": query_path,
        "k": k,
        "fusion_method": "weighted",
        "alpha": 0.6,
        "beta": 0.4
    })
    result = response.json()
    
    if result['status'] == 'success':
        html_file = result['visualization_file']
        print(f"✅ Visualization saved: {html_file}")
        
        # 在浏览器中打开
        webbrowser.open(html_file)
    
    return result

# 使用示例
query_path = "/path/to/query.h5"
results = search_with_explanation(query_path, k=5)
visualize_results(query_path, k=5)
```

## 解释字段详解

### explanation 对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `stage1_similarity` | float | Stage 1 (特征级) 相似度 [0-1] |
| `stage2_similarity` | float | Stage 2 (序列级) 相似度 [0-1] |
| `final_score` | float | 最终融合分数 [0-1] |
| `fusion_method` | string | 融合方法 (weighted/linear/rrf/borda) |
| `contributions` | object | 各阶段贡献分解 |
| `stage1_interpretation` | string | Stage 1 质量文字解释 |
| `stage2_interpretation` | string | Stage 2 质量文字解释 |

### contributions 对象

**weighted 融合方法**:
- `stage1_weight`: Stage 1 权重 (alpha)
- `stage2_weight`: Stage 2 权重 (beta)
- `stage1_contribution`: Stage 1 贡献值
- `stage2_contribution`: Stage 2 贡献值
- `stage1_percentage`: Stage 1 贡献百分比
- `stage2_percentage`: Stage 2 贡献百分比

**linear 融合方法**:
- `stage1_contribution`: Stage 1 贡献值 (0.5 * sim1)
- `stage2_contribution`: Stage 2 贡献值 (0.5 * sim2)
- `stage1_percentage`: 固定 50%
- `stage2_percentage`: 固定 50%

### stage2_details 对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_distance` | float | 总距离（越小越相似）|
| `cmd_penalty` | float | 命令不匹配惩罚 |
| `param_l2` | float | 参数 L2 距离 |
| `normalized_param_l2` | float | 归一化参数距离 |
| `sequence_length` | int | 实际比较的序列长度 |
| `query_seq_len` | int | 查询序列长度 |
| `candidate_seq_len` | int | 候选序列长度 |
| `cmd_matches` | int | 命令匹配数量 |
| `cmd_mismatches` | int | 命令不匹配数量 |
| `cmd_match_rate` | float | 命令匹配率 [0-1] |
| `avg_param_distance_per_step` | float | 平均每步参数距离 |
| `max_param_distance_per_step` | float | 最大每步参数距离 |
| `step_distances` | array | 每步的参数距离列表 |

## 解释等级

### Stage 1 (特征级) 解释

- `> 0.9`: "Excellent feature-level match"
- `0.7 - 0.9`: "Good feature-level match"
- `0.5 - 0.7`: "Moderate feature-level match"
- `< 0.5`: "Weak feature-level match"

### Stage 2 (序列级) 解释

- `> 0.9`: "Excellent sequence-level match"
- `0.7 - 0.9`: "Good sequence-level match"
- `0.5 - 0.7`: "Moderate sequence-level match"
- `< 0.5`: "Weak sequence-level match"

## HTML 可视化特性

生成的 HTML 报告包含：

### 1. 查询信息卡片
- 查询文件路径
- 返回结果数量

### 2. 结果卡片（每个结果）
- **排名徽章**: 绿色显示排名
- **结果 ID**: 粗体显示文件标识
- **分数徽章**: 蓝色显示最终得分

### 3. 元数据展示
- 数据子集
- 序列长度

### 4. 相似度分解（彩色进度条）
- **Stage 1**: 红橙渐变进度条 + 数值 + 文字解释
- **Stage 2**: 青绿渐变进度条 + 数值 + 文字解释
- **Final**: 紫色渐变进度条 + 数值

### 5. 融合信息
- 融合方法显示
- 各阶段贡献卡片（权重、贡献值、百分比）

### 6. 序列级分析（网格布局）
- 总距离
- 命令匹配率
- 命令匹配数
- 平均参数距离
- 最大参数距离
- 序列长度对比

## 应用场景

### 1. 调试检索结果
当检索结果不符合预期时，通过解释可以发现：
- 是特征提取的问题（Stage 1 低）还是序列匹配的问题（Stage 2 低）
- 融合权重是否合理
- 命令序列是否真的相似

### 2. 优化融合参数
通过观察不同结果的贡献分解，可以调整 alpha 和 beta：
- 如果 Stage 1 贡献过高，可以降低 alpha
- 如果 Stage 2 贡献不足，可以提高 beta

### 3. 用户信任建立
向最终用户展示可视化报告，帮助其理解系统推荐逻辑，增强信任。

### 4. 研究和分析
在论文中展示检索结果时，使用可视化报告可以清晰地说明检索质量。

## 性能考虑

### explainable=false（默认）
- 只返回基本的相似度分数
- 更快的响应时间
- 适合生产环境大批量查询

### explainable=true
- 返回完整的解释和详细分解
- 额外计算开销约 10-20%
- 适合调试、分析和用户展示

### HTML 可视化
- 生成 HTML 文件约 20-50KB（取决于结果数量）
- 一次性生成，可多次查看
- 适合离线分析和报告

## 示例：分析一个实际结果

```json
{
  "id": "0000/00000498.h5",
  "score": 0.6433,
  "sim_stage1": 0.9848,
  "sim_stage2": 0.1312,
  "explanation": {
    "stage1_similarity": 0.9848,
    "stage2_similarity": 0.1312,
    "final_score": 0.6433,
    "fusion_method": "weighted",
    "contributions": {
      "stage1_weight": 0.6,
      "stage2_weight": 0.4,
      "stage1_contribution": 0.5909,
      "stage2_contribution": 0.0525,
      "stage1_percentage": 91.84,
      "stage2_percentage": 8.16
    },
    "stage1_interpretation": "Excellent feature-level match",
    "stage2_interpretation": "Weak sequence-level match"
  },
  "stage2_details": {
    "cmd_match_rate": 1.0,
    "cmd_matches": 7,
    "cmd_mismatches": 0,
    "avg_param_distance_per_step": 3.43,
    "max_param_distance_per_step": 13.0,
    "step_distances": [0.0, 0.0, 10.0, 0.0, 1.0, 13.0, 0.0]
  }
}
```

**分析**:

1. **整体评分**: 0.6433 (中等)
2. **Stage 1**: 0.9848 (优秀) - 特征级匹配非常好
3. **Stage 2**: 0.1312 (弱) - 序列级匹配较差

**深入原因**:
- 命令序列完全匹配（cmd_match_rate=1.0）
- 但参数差异较大（最大距离 13.0）
- 特别是第 3 步和第 6 步参数差异显著

**结论**: 这两个 CAD 模型使用了相同的命令序列，但参数设置差异较大，可能是相似的设计思路但不同的尺寸/比例。

## 总结

解释性检索功能为向量数据库提供了透明度和可解释性，使得：

- ✅ **用户理解**: 清楚知道为什么某个结果被推荐
- ✅ **系统调优**: 基于详细分析优化参数
- ✅ **信任建立**: 通过可视化增强用户信任
- ✅ **研究价值**: 为论文和报告提供详细的分析数据

**实现状态**: ✅ 已完成开发、测试和文档

**相关文档**:
- [API 测试结果](./API_TEST_RESULTS.md)
- [混合检索指南](./HYBRID_SEARCH_GUIDE.md)
