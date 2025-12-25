# CAD Vector Database

面向 3D 深度学习的向量数据库系统，基于 WHUCAD 数据集实现高效的 CAD 模型相似性检索。

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7+-green.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com/)

## 项目概述

完整的生产级向量数据库系统，支持 CAD 模型的多模态相似性检索。采用**两阶段检索与融合排序**架构，结合向量召回、精确重排和智能融合策略。

### 核心特性

✅ **检索能力**
- 🔍 **两阶段检索**：FAISS ANN 快速召回 + 宏序列精确重排
- 🎯 **融合排序**：加权融合/RRF/Borda 三种策略
- 🧠 **语义查询**：自然语言文本检索（Sentence-BERT/CLIP/BM25）
- 🔀 **混合检索**：文本 + 向量混合查询
- 📊 **可解释检索**：5级质量评级 + 匹配类型分析 + 置信度评估 + 智能推荐
- 🎨 **可视化分析**：现代化HTML界面，渐变设计，动画进度条，双语支持
- 🔎 **批量检索**：高效并行批量查询

✅ **索引管理**
- 🏗️ **多索引支持**：HNSW/IVF/IVFPQ 三种索引类型
- ➕ **增量更新**：在线添加/更新/删除向量，零停机
- 🗑️ **软删除机制**：可恢复的删除操作
- 📸 **快照系统**：版本控制与快速回滚
- 📝 **变更日志**：完整的操作审计追踪
- 🗜️ **索引压缩**：自动清理已删除向量

✅ **数据库集成**
- 🗄️ **元数据数据库**：OceanBase/MySQL/PostgreSQL 支持
- 📥 **数据导入工具**：命令行批量导入元数据（支持批量处理、表重建）
- 🔍 **数据查询工具**：多维度查询（统计/ID查询/子集/序列长度/导出）
- 🔀 **混合查询**：向量检索 + SQL 过滤
- 📈 **性能监控**：查询统计与分析

✅ **生产特性**
- 🚀 **REST API**：FastAPI 高性能 API 服务
- 📚 **完整文档**：详细的使用指南与 API 文档
- 🧪 **评估框架**：P@K, R@K, mAP, 延迟指标
- 🎯 **生产就绪**：完善的错误处理与日志

## 快速开始

### 1️⃣ 环境准备

```bash
git clone https://github.com/riverfielder/cad-vector-db.git
cd cad-vector-db
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ 构建索引

```bash
# 快速测试（500 样本）
python scripts/build_index.py --max_samples 500 --output_dir data/index_test

# 全量索引
python scripts/build_index.py --output_dir data/index_full
```

### 3️⃣ 启动 API 服务

```bash
python server/app.py
# 访问 http://localhost:8123/docs 查看交互式文档
```

### 4️⃣ 检索示例

**向量检索：**
```bash
curl -X POST http://localhost:8123/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_file_path": "/path/to/query.h5",
    "k": 10
  }'
```

**语义检索：**
```bash
curl -X POST http://localhost:8123/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "圆柱形零件",
    "k": 10
  }'
```

**混合检索：**
```bash
curl -X POST http://localhost:8123/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "带孔的圆柱",
    "query_file_path": "/path/to/query.h5",
    "k": 10,
    "semantic_weight": 0.5,
    "vector_weight": 0.5
  }'
```

## 功能特性详解

### 🔍 检索模式

#### 1. 向量检索（Vector Search）
基于 FAISS 的高效 ANN 检索：
```python
from cad_vectordb.retrieval import TwoStageRetrieval

retrieval = TwoStageRetrieval(index_manager)
results = retrieval.search(query_feature, query_path, k=10)
```

#### 2. 语义检索（Semantic Search）
自然语言文本检索 CAD 模型：
```python
results = retrieval.semantic_search(
    query_text="带螺纹的圆柱孔",
    k=10,
    encoder_type="sentence-transformer"
)
```

**支持的编码器：**
- `sentence-transformer`: 多语言语义理解（推荐）
- `clip`: 视觉-语言联合编码
- `bm25`: 传统关键词匹配

#### 3. 混合检索（Hybrid Search）
结合文本语义和向量特征：
```python
results = retrieval.hybrid_search(
    query_text="圆柱形零件",
    query_vec=cad_vector,
    k=10,
    semantic_weight=0.5,
    vector_weight=0.5
)
```

#### 4. 可解释检索（Explainable Retrieval）
提供详细的相似度分析和智能推荐：
```python
results = retrieval.search(
    query_feature,
    query_path,
    k=10,
    explainable=True
)
# 返回13个解释性字段：
# - quality_rating: 5级质量评级（excellent/very_good/good/moderate/weak）
# - match_type: 匹配类型（strong_overall/feature_dominant/sequence_dominant等）
# - confidence_score: 置信度评分（0-1）
# - recommendations: 智能优化建议
# - feature_analysis: 特征向量深度分析（L2距离/余弦相似度/Top-K维度）
# - 可视化HTML：现代化界面，渐变背景，动画进度条
```

**增强特性：**
- **5级质量评级**：从优异到较弱的细粒度评分
- **匹配类型识别**：自动识别5种匹配模式
- **置信度评估**：基于相似度和一致性的综合评分
- **智能推荐**：6种场景的自动优化建议
- **特征分析**：维度级别的详细分析（L2/余弦/Top-K贡献维度）
- **现代化可视化**：渐变设计、质量徽章、动画效果、双语支持

### ➕ 增量更新

无需重建索引的在线更新：

```python
from cad_vectordb.core.index import IndexManager

index_manager = IndexManager("./data/index", enable_versioning=True)
index_manager.load_index()

# 添加向量
index_manager.add_vectors([("vec_0100", "/path/to/vec.h5")])

# 更新向量
index_manager.update_vector("vec_0001", "/path/to/vec_v2.h5")

# 软删除（可恢复）
index_manager.soft_delete(["vec_0002"])

# 创建快照
index_manager.create_snapshot("v1.0")

# 回滚
index_manager.restore_snapshot("v1.0")
```

**REST API 端点：**
- `POST /vectors/add` - 添加向量
- `PUT /vectors/{id}` - 更新向量
- `POST /vectors/batch-update` - 批量更新
- `DELETE /vectors/soft` - 软删除
- `POST /vectors/restore` - 恢复
- `POST /index/snapshot` - 创建快照
- `POST /index/snapshot/{name}/restore` - 恢复快照

### 🗄️ 元数据数据库

支持 OceanBase/MySQL/PostgreSQL 集成，提供完整的命令行工具：

**数据导入工具（`scripts/import_metadata_to_oceanbase.py`）：**
```bash
# 基础导入
python scripts/import_metadata_to_oceanbase.py \
    --metadata data/indices/metadata.json

# 自定义数据库连接
python scripts/import_metadata_to_oceanbase.py \
    --metadata data/indices/metadata.json \
    --host 127.0.0.1 \
    --port 2881 \
    --user root@test \
    --password mypass \
    --database cad_db

# 删除旧表并重新导入
python scripts/import_metadata_to_oceanbase.py \
    --metadata data/indices/metadata.json \
    --drop-table

# 调整批量大小（大数据集）
python scripts/import_metadata_to_oceanbase.py \
    --metadata data/indices/metadata.json \
    --batch-size 5000
```

**数据查询工具（`scripts/query_metadata_db.py`）：**
```bash
# 查看统计信息
python scripts/query_metadata_db.py stats

# 获取特定记录
python scripts/query_metadata_db.py get "0000/00000001.h5"

# 按子集查询
python scripts/query_metadata_db.py subset 0000 --limit 10

# 按序列长度查询
python scripts/query_metadata_db.py seqlen --min 10 --max 20

# 导出查询结果
python scripts/query_metadata_db.py export \
    --subset 0000 \
    --output results.json
```

**Python API：**
```python
from cad_vectordb.database.metadata import MetadataDB

# 连接数据库
db = MetadataDB(
    host="localhost",
    port=2881,
    user="root@test",
    password="password",
    database="cad_metadata"
)

# 混合查询：向量检索 + SQL 过滤
results = retrieval.search(
    query_feature,
    query_path,
    k=10,
    filters={"subset": "0000", "min_seq_len": 50}
)
```

## 系统架构

### 两阶段检索流程

```
┌──────────────────┐
│   查询输入        │
│ • 向量 (H5)      │
│ • 文本 (NLP)     │
│ • 混合           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Stage 1: ANN    │
│  FAISS 快速召回  │  Top-N 候选 (N=100)
│  HNSW/IVF/IVFPQ │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Stage 2: 精排   │
│  宏序列距离计算  │  细粒度重排
│  命令+参数匹配   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  融合排序        │
│ • 加权融合       │
│ • RRF 融合       │  Top-K 结果 (K=10)
│ • Borda 融合     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   最终结果       │
│ • ID + 分数      │
│ • 元数据         │
│ • 可解释性       │
└──────────────────┘
```

### 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| **向量索引** | FAISS | HNSW/IVF/IVFPQ 多种索引 |
| **Web 框架** | FastAPI | 高性能异步 API |
| **数据库** | OceanBase/PostgreSQL | 元数据存储与查询 |
| **NLP 编码** | Sentence-BERT, CLIP | 多语言文本编码 |
| **数据处理** | NumPy, h5py | 科学计算与 HDF5 |
| **API 文档** | Swagger/OpenAPI | 自动生成交互式文档 |

## 项目结构

```
cad-vector-db/
├── README.md                    # 项目说明
├── requirements.txt             # Python 依赖
├── config.py                    # 全局配置
│
├── cad_vectordb/               # 核心模块
│   ├── core/                   # 核心功能
│   │   ├── index.py           # 索引管理（增量更新）
│   │   └── feature.py         # 特征提取
│   ├── retrieval/             # 检索模块
│   │   ├── two_stage.py       # 两阶段检索
│   │   └── semantic.py        # 语义检索
│   └── metadata/              # 元数据管理
│       └── oceanbase.py       # OceanBase 集成
│
├── server/                     # API 服务
│   └── app.py                 # FastAPI 应用
│
├── scripts/                    # 工具脚本
│   ├── build_index.py         # 索引构建
│   ├── import_metadata_to_oceanbase.py  # 元数据导入工具
│   ├── query_metadata_db.py   # 元数据查询工具
│   └── test_metadata_db_setup.sh  # 数据库测试
│
├── examples/                   # 使用示例
│   ├── incremental_updates_example.py
│   └── semantic_search_example.py
│
├── docs/                       # 文档
│   ├── INCREMENTAL_UPDATES_GUIDE.md
│   ├── SEMANTIC_SEARCH_GUIDE.md
│   ├── HYBRID_SEARCH_GUIDE.md
│   └── OCEANBASE_GUIDE.md
│
└── data/                       # 数据目录
    ├── index/                 # FAISS 索引
    └── vec/                   # WHUCAD 向量数据
```

## 性能指标

### 检索性能（测试集 500 样本）

| 指标 | HNSW | IVF | IVFPQ |
|------|------|-----|-------|
| Precision@10 | 0.52 | 0.48 | 0.45 |
| Recall@10 | 0.41 | 0.38 | 0.35 |
| 延迟 p50 | 45ms | 38ms | 25ms |
| 延迟 p95 | 120ms | 95ms | 60ms |
| 索引大小 | 2.1MB | 1.8MB | 0.8MB |

### 增量更新性能

| 操作 | 时间复杂度 | 实测延迟 |
|------|-----------|---------|
| 添加向量 | O(log N) | ~5ms |
| 更新向量 | O(1) | ~3ms |
| 软删除 | O(1) | <1ms |
| 创建快照 | O(N) | ~200ms (500样本) |
| 索引压缩 | O(N) | ~500ms (500样本) |

## 📚 文档导航

### 快速入门
- **[快速开始](#快速开始)** - 5分钟快速上手
- **[API 文档](http://localhost:8123/docs)** - 交互式 API 文档

### 功能指南
- **[增量更新指南](docs/INCREMENTAL_UPDATES_GUIDE.md)** - 在线更新索引
- **[语义检索指南](docs/SEMANTIC_SEARCH_GUIDE.md)** - 文本检索 CAD 模型
- **[混合检索指南](docs/HYBRID_SEARCH_GUIDE.md)** - 多模态检索
- **[可解释检索指南](docs/EXPLAINABLE_RETRIEVAL_GUIDE.md)** - 相似度分析基础
- **[可解释检索增强](docs/EXPLAINABLE_RETRIEVAL_ENHANCEMENT.md)** - 5级评级+智能推荐+可视化
- **[元数据数据库指南](docs/OCEANBASE_GUIDE.md)** - OceanBase/MySQL 集成与工具
- **[批量检索指南](docs/BATCH_SEARCH_GUIDE.md)** - 高效批量查询

### 开发文档
- **[使用指南](docs/USAGE.md)** - 详细使用说明
- **[索引管理](docs/INDEX_MANAGEMENT.md)** - 索引构建与管理
- **[性能基准](docs/BENCHMARK.md)** - 性能测试结果
- **[API 测试结果](docs/API_TEST_RESULTS.md)** - API 功能验证

## 使用示例

### Python SDK

```python
from cad_vectordb.core.index import IndexManager
from cad_vectordb.retrieval import TwoStageRetrieval

# 初始化
index_manager = IndexManager("./data/index", enable_versioning=True)
index_manager.load_index()
retrieval = TwoStageRetrieval(index_manager)

# 向量检索
results = retrieval.search(query_feature, query_path, k=10)

# 语义检索
results = retrieval.semantic_search("圆柱形零件", k=10)

# 混合检索
results = retrieval.hybrid_search(
    query_text="带孔的圆柱",
    query_vec=cad_vector,
    k=10
)

# 增量更新
index_manager.add_vectors([("new_vec", "/path/to/vec.h5")])
index_manager.create_snapshot("v1.0")
```

### REST API

查看完整的 API 文档：http://localhost:8123/docs

**主要端点：**
- `POST /search` - 向量检索
- `POST /search/semantic` - 语义检索
- `POST /search/hybrid` - 混合检索
- `POST /search/batch` - 批量检索
- `POST /vectors/add` - 添加向量
- `PUT /vectors/{id}` - 更新向量
- `DELETE /vectors/soft` - 软删除
- `POST /index/snapshot` - 创建快照
- `GET /stats` - 系统统计

## 常见问题

### Q: 如何选择索引类型？

**HNSW**: 最佳召回率，适合中小规模（<1M）
**IVF**: 平衡性能，适合大规模数据
**IVFPQ**: 内存优化，适合超大规模或内存受限

### Q: 增量更新会影响性能吗？

软删除几乎无性能影响。索引压缩会临时阻塞，建议在低峰期执行。

### Q: 如何优化检索延迟？

1. 调整 Stage 1 候选集大小（topn）
2. 使用更快的索引类型（IVFPQ）
3. 启用批量检索
4. 添加结果缓存

### Q: 支持分布式部署吗？

当前版本为单机部署。分布式支持计划中，可通过多实例 + 负载均衡实现水平扩展。

## 贡献指南

欢迎贡献代码、文档或提出问题！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 作者与致谢

**作者**
- [@riverfielder](https://github.com/riverfielder)
- 武汉大学 计算机科学与技术

**技术支持**
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Research
- [FastAPI](https://fastapi.tiangolo.com/) - Sebastián Ramírez
- [Sentence-Transformers](https://www.sbert.net/) - UKP Lab

## 相关资源

- [WHUCAD 数据集](https://github.com/user/WHUCAD-main)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [OceanBase 文档](https://www.oceanbase.com/docs)

---

**⭐ 如果这个项目对你有帮助，欢迎 Star！**

*最后更新：2025-12-25*

**最新更新：**
- ✅ 可解释检索深度增强（5级评级、匹配类型、置信度、智能推荐）
- ✅ 现代化可视化界面（渐变设计、动画效果、双语支持）
- ✅ OceanBase数据库完整集成（导入工具、查询工具、命令行界面）
- ✅ 特征向量深度分析（L2距离、余弦相似度、Top-K维度分析）
