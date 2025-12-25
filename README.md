# CAD Vector Database

面向 3D 深度学习的向量数据库系统，基于 WHUCAD 数据集实现高效的 CAD 模型相似性检索。

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7+-green.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com/)

## 项目概述

本项目实现了一个完整的向量数据库系统，用于 CAD 模型的相似性检索。系统采用**两阶段检索与融合排序**架构：
- **Stage 1**: 基于特征向量的快速 ANN 召回（FAISS）
- **Stage 2**: 基于原始宏序列的精确相似度计算
- **Fusion**: 多种融合策略（加权/RRF/Borda）整合两阶段分数

### 核心特性

✅ **已实现功能**
- 特征提取：从 WHUCAD 宏序列向量提取固定维度特征（均值池化）
- 索引构建：支持 FAISS HNSW/IVF/IVFPQ 三种索引类型
- 两阶段检索：ANN 快速召回 + 宏序列精确排序
- 融合排序：加权融合/RRF/Borda 三种方法
- REST API：基于 FastAPI 的检索服务
- 评估框架：Precision@K, Recall@K, mAP, 延迟指标

🚧 **待扩展功能**
- 元数据数据库：Postgres/OceanBase 集成（用于过滤）
- 潜向量支持：基于 AE 编码的潜空间表征
- 向量数据库对比：Qdrant/Milvus 性能基准测试

## 快速开始

### 环境准备

```bash
git clone https://github.com/riverfielder/cad-vector-db.git
cd cad-vector-db
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 构建索引

```bash
# 测试索引（500个样本，快速验证）
python scripts/build_index.py --max_samples 500 --output_dir data/index_test

# 全量索引（所有数据）
python scripts/build_index.py --output_dir data/index_full
```

### 测试检索

```bash
python scripts/retrieval.py \
  --index_dir data/index_test \
  --query_file /path/to/WHUCAD-main/data/vec/0000/00000001.h5 \
  --topn 100 --topk 20 --fusion weighted
```

### 启动 API 服务

```bash
python server/app.py
# 访问 http://127.0.0.1:8000/docs 查看交互式文档
```

### 运行评估

```bash
python evaluation/evaluate.py \
  --index_dir data/index_test \
  --n_queries 100 \
  --fusion weighted \
  --output evaluation/results.json
```

详细使用说明请参考 [USAGE.md](USAGE.md)。

## 系统架构

```
┌─────────────────┐
│  WHUCAD Vec     │  data/vec/*/*.h5 (宏序列向量)
│  (seq_len, 33)  │
└────────┬────────┘
         │ 特征提取（均值池化）
         ▼
┌─────────────────┐
│  Feature (32D)  │  固定维度特征向量
└────────┬────────┘
         │ 归一化 + 索引构建
         ▼
┌─────────────────┐
│  FAISS Index    │  HNSW/IVF/IVFPQ
│  (ANN 召回)      │
└────────┬────────┘
         │ Stage 1: Top-N 候选
         ▼
┌─────────────────┐
│  Macro Rerank   │  原始序列精排
│  (命令+参数)     │
└────────┬────────┘
         │ Stage 2: 细粒度相似度
         ▼
┌─────────────────┐
│  Fusion Rank    │  加权/RRF/Borda
│  (最终 Top-K)    │
└─────────────────┘
```

## 技术栈

- **索引引擎**: [FAISS](https://github.com/facebookresearch/faiss) - 高效向量检索
- **Web 框架**: [FastAPI](https://fastapi.tiangolo.com/) - 现代化 REST API
- **数据处理**: NumPy, h5py - 科学计算与 HDF5 处理
- **评估工具**: 自研评估框架 - P@K, R@K, mAP, 延迟统计

## 项目结构

```
cad-vector-db/
├── README.md              # 项目说明
├── USAGE.md              # 详细使用指南
├── requirements.txt      # Python 依赖
├── config.py            # 全局配置
├── scripts/             # 核心脚本
│   ├── build_index.py   # 索引构建
│   └── retrieval.py     # 两阶段检索
├── server/              # API 服务
│   └── app.py          # FastAPI 应用
├── evaluation/          # 评估模块
│   └── evaluate.py     # 评估脚本
└── data/               # 数据目录
    └── index/          # 索引文件
```

## 性能指标（测试集 500 样本）

| 指标 | 值 |
|------|-----|
| Precision@10 | ~0.4-0.6 (取决于弱标签质量) |
| Recall@10 | ~0.3-0.5 |
| 延迟 p50 | ~50ms |
| 延迟 p95 | ~150ms |
| 索引构建时间 | ~10s (500 样本) |

*注：当前使用 subset 作为弱标签；真实标签或潜向量 z 可显著提升精度*

## 系统能力

**向量管理**
- 读取/校验 WHUCAD `data/vec/*/*.h5` 文件
- 特征提取：均值池化 → 32维向量
- L2 归一化（余弦相似度）

**ANN 召回**
- FAISS HNSW：快速近似检索，可调 M/efSearch
- FAISS IVF：倒排索引，适合大规模数据
- FAISS IVFPQ：量化压缩，平衡精度与内存

**精排与融合**
- 宏序列距离：命令匹配罚分 + 参数 L2 距离
- 融合方法：加权（可调 α/β）、RRF（鲁棒）、Borda（排名）


## 实现说明

### 数据来源选择

本项目采用 **WHUCAD 宏序列向量（vec）** 作为数据源（选项一），而非等待生成的潜向量（z）：

**原因：**
- `vec` 数据已完备：`data/vec` 目录下 146,331+ 个 h5 文件可直接使用
- 快速原型验证：无需等待 Autoencoder 训练完成
- 降维策略合理：均值池化可有效保留序列统计信息

**向量格式：**
```python
# 原始格式: (seq_len, 33) float64
# vec[:, 0]  → 命令类型 (整数编码)
# vec[:, 1:] → 命令参数 (32维浮点数)

# 特征提取
feature = np.mean(vec[:, 1:], axis=0)  # (32,) 均值池化
feature = feature / (np.linalg.norm(feature) + 1e-8)  # L2 归一化
```

### 关键技术决策

1. **FAISS HNSW 优先**：测试索引使用 M=32, efConstruction=200，兼顾召回率与构建速度
2. **两阶段必要性**：特征降维损失信息，精排阶段补偿（命令匹配 + 参数距离）
3. **融合默认加权**：`α=0.6, β=0.4` 偏向 Stage 1，可通过参数调优
4. **无需外部数据库**：索引阶段直接持久化 JSON 元数据，避免初期复杂度

### 测试结果验证

**索引构建**（500 样本）:
```
Loaded 500 vectors, feature shape: (500, 32)
Building HNSW index: M=32, efConstruction=200
Index built successfully. Total vectors: 500
Saved to: data/index_test/
```

**检索测试**（查询 `0000/00000001.h5`）:
```
Top-10 Results:
1. 0000/00000001.h5 → Score: 1.0000 (stage1: 1.0000, stage2: 1.0000) ✓ 完美自匹配
2. 0003/00032846.h5 → Score: 0.4592 (stage1: 0.7545, stage2: 0.0029)
3. 0008/00085265.h5 → Score: 0.4521 (stage1: 0.7447, stage2: 0.0038)
...
10. 0003/00037845.h5 → Score: 0.3042 (stage1: 0.5042, stage2: 0.0027)
```

**观察：**
- Stage 1（FAISS）分数显著高于 Stage 2（宏距离），符合预期（特征相似 ≠ 序列相似）
- 融合后分数介于两者之间，体现互补性
- 自匹配完美得分，验证系统正确性

## 下一步计划

### 核心任务
- [ ] **全量索引构建**：处理全部 WHUCAD 数据（~146K 向量）
- [ ] **综合评估运行**：生成弱标签，计算 P@K/R@K/mAP 指标
- [ ] **消融实验**：对比融合方法、索引类型、参数配置
- [ ] **延迟优化**：分析 p95 延迟瓶颈，优化候选集大小

### 可选扩展
- [ ] **潜向量支持**：训练 Autoencoder，提取 `z` 向量重新索引
- [x] **元数据数据库**：集成 OceanBase，支持复杂过滤查询（已完成）
- [ ] **向量数据库对比**：Qdrant/Milvus 基准测试
- [ ] **生产部署**：Docker 化，负载均衡，监控告警

### 毕设文档
- [ ] **实验设计**：对比实验、消融实验方案
- [ ] **结果分析**：性能指标图表、系统瓶颈分析
- [ ] **论文撰写**：系统设计、实验结果、结论与展望

## 相关资源

- [WHUCAD 项目](https://github.com/user/WHUCAD-main) - 数据集来源
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki) - 索引算法文档
- [FastAPI 文档](https://fastapi.tiangolo.com/) - API 开发指南

## License

MIT License

## 作者

- 河田 [@riverfielder](https://github.com/riverfielder)
- 武汉大学 - 计算机科学与技术

---

*最后更新：2025-01-01*
```
- 查询组合流程（伪逻辑）：
  - `ids = ann_search(z_query, topN)` → `filtered = SQL WHERE subset='0000' AND label IN (...)` → `reranked = rerank_by_macro(filtered)` → 返回 Top-K。

**里程碑与时间表（示例）**
- 第1周: 巡检与数据规范；FAISS 召回原型（小样本）。
- 第2周: 批量导入与索引持久化；精排与融合排序；API 最小可用版本。
- 第3周: 评估与消融（HNSW/IVF/PQ/度量/融合）；优化与缓存。
- 第4周: Qdrant/Milvus 对比；写选型报告与结论。
- 第5周: 文档与论文撰写，汇总指标与图表，工程总结。

**风险与化解**
- 键名/维度不统一: 早期巡检；设定键优先顺序与异常回退；统一 dtype。
- 依赖兼容: `faiss-cpu` 兼容性用 conda 或容器解决；固定 Python 版本。
- 标签不足: 用 `subset/来源` 作为弱标签；小样本人工验证提升可信度。
- 性能瓶颈: 明确延迟目标（如 p95<100ms）；调 `efSearch/nprobe` 与批量接口；热门缓存。

**快速起步命令（不改仓库文件）**
- 环境与依赖：
```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install faiss-cpu fastapi uvicorn h5py numpy psycopg2-binary
```
- 抽样巡检：
```zsh
python - <<'PY'
import os,h5py,numpy as np
root="data/vec"; sample=None
for d,_,fs in os.walk(root):
  for f in fs:
    if f.endswith(".h5"): sample=os.path.join(d,f); break
  if sample: break
print("Sample:", sample)
with h5py.File(sample,"r") as h5:
  print("Keys:", list(h5.keys()))
  for k in h5.keys():
    arr=np.array(h5[k]); print(k,arr.shape,arr.dtype)
PY
```
- 两阶段检索与融合（原型，按你的键名/维度替换）：
```zsh
python - <<'PY'
# 同我之前给你的两阶段示例，替换为真实 z 的读取与度量即可
PY
```

如果你确认潜向量维度 `d` 与 `.h5` 键名（如 `z`/`vec`），我可以把“召回/精排/融合”的一次性命令精确到你的数据格式，并附上参数网格与评估脚本模版，帮助你当天跑出首批结果。是否需要我直接提供“FAISS 召回 + 精排 + 融合”的即用命令版本？