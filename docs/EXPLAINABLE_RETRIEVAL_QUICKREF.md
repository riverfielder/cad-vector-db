# 解释性检索快速参考

## API 请求

### JSON 响应（带解释）

```bash
POST /search
{
  "query_file_path": "/path/to/query.h5",
  "k": 5,
  "explainable": true  # 启用解释性检索
}
```

### HTML 可视化

```bash
POST /search/visualize
{
  "query_file_path": "/path/to/query.h5",
  "k": 5
}
```

## Python 示例

```python
import requests

# JSON 解释
response = requests.post("http://localhost:8000/search", json={
    "query_file_path": "/path/to/query.h5",
    "k": 5,
    "explainable": True
})
results = response.json()

# HTML 可视化
response = requests.post("http://localhost:8000/search/visualize", json={
    "query_file_path": "/path/to/query.h5",
    "k": 5
})
print(response.json()['message'])
```

## 响应字段

### explanation
- `stage1_similarity`: 特征级相似度 [0-1]
- `stage2_similarity`: 序列级相似度 [0-1]
- `final_score`: 最终分数 [0-1]
- `fusion_method`: 融合方法
- `contributions`: 各阶段贡献
- `stage1_interpretation`: 文字解释
- `stage2_interpretation`: 文字解释

### stage2_details
- `cmd_match_rate`: 命令匹配率
- `cmd_matches`: 匹配的命令数
- `avg_param_distance_per_step`: 平均参数距离
- `max_param_distance_per_step`: 最大参数距离
- `step_distances`: 每步距离数组

## 解释等级

- **> 0.9**: Excellent
- **0.7-0.9**: Good
- **0.5-0.7**: Moderate
- **< 0.5**: Weak

## 典型用例

### 1. 调试低分结果
```python
# 查看为什么分数低
result = results[0]
exp = result['explanation']
print(f"Stage 1: {exp['stage1_similarity']}")  # 特征级
print(f"Stage 2: {exp['stage2_similarity']}")  # 序列级
# 找出瓶颈
```

### 2. 优化融合权重
```python
# 观察各阶段贡献
contrib = exp['contributions']
print(f"S1: {contrib['stage1_percentage']:.1f}%")
print(f"S2: {contrib['stage2_percentage']:.1f}%")
# 调整 alpha/beta
```

### 3. 分析序列差异
```python
# 查看命令和参数匹配情况
details = result['stage2_details']
print(f"Cmd Match: {details['cmd_match_rate']:.1%}")
print(f"Avg Param Dist: {details['avg_param_distance_per_step']:.2f}")
```

## 性能

- **explainable=false**: 标准速度
- **explainable=true**: +10-20% 开销
- **HTML 生成**: ~50ms

## 更多信息

完整文档: [EXPLAINABLE_RETRIEVAL_GUIDE.md](./EXPLAINABLE_RETRIEVAL_GUIDE.md)
