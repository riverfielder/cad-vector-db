# CAD Vector Database

面向 3D 深度学习的向量数据库系统 - 毕业论文实现项目

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7+-green.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com/)

## 项目概述

本项目实现了一个功能完整的向量数据库系统，专门用于 CAD 模型的相似性检索。系统采用**两阶段检索与融合排序**架构，结合多种高级特性，为 3D 深度学习应用提供高效的检索服务。

### 核心架构

```
查询向量 → Stage 1 (FAISS ANN) → 候选集 → Stage 2 (精确重排序) → 融合排序 → Top-K 结果
            ↓                      ↓                              ↓
         特征级检索            序列级检索                      多策略融合
```

## 核心功能

### ✅ 已实现功能

#### 1. 向量检索
- **两阶段检索**：FAISS快速召回 + 宏序列精确排序
- **多种融合方法**：weighted（加权）/ rrf（倒数排名）/ borda（波达计数）
- **支持多种索引**：Flat / IVFFlat / HNSW

#### 2. 混合检索 (Hybrid Search)
- **元数据过滤**：基于OceanBase的元数据数据库
- **多维过滤**：子集、序列长度、标签等
- **高效集成**：向量检索 + 元数据筛选

#### 3. 可解释性检索 (Explainable Retrieval)
- **相似度分解**：Stage 1 / Stage 2 贡献度分析
- **特征级解释**：命令匹配率、参数距离等
- **HTML可视化**：直观的相似度解释界面

#### 4. 批量检索 (Batch Search)
- **批量处理**：一次请求处理多个查询
- **并行支持**：顺序/并行两种模式
- **统计信息**：QPS、成功率、耗时等指标

#### 5. 索引管理 (Index Management) 🆕
- **动态索引**：创建、加载、更新、删除
- **多索引支持**：管理多个不同用途的索引
- **完整性验证**：索引健康检查
- **统计信息**：详细的索引统计数据

#### 6. 性能基准测试 (Performance Benchmark) 🆕
- **多维度测试**：延迟、吞吐量、精度
- **参数分析**：k值、topn、融合方法影响
- **自动化报告**：JSON格式结果输出
- **持续监控**：性能回归检测

## 项目结构

```
cad-vector-db/
├── cad_vectordb/          # 核心库
│   ├── core/              # 核心功能
│   │   ├── index.py       # 索引管理
│   │   ├── retrieval.py   # 检索算法
│   │   └── feature.py     # 特征提取
│   ├── database/          # 数据库模块
│   │   └── metadata.py    # 元数据操作
│   ├── api/               # API服务
│   │   ├── app.py         # FastAPI应用
│   │   └── index_api.py   # 索引管理API
│   └── utils/             # 工具函数
│
├── benchmarks/            # 性能测试
│   └── benchmark_search.py
│
├── examples/              # 使用示例
│   ├── basic_search.py
│   ├── batch_search.py
│   └── index_management.py
│
├── docs/                  # 文档
│   ├── USAGE.md
│   ├── INDEX_MANAGEMENT.md
│   ├── BENCHMARK.md
│   ├── BATCH_SEARCH_GUIDE.md
│   └── EXPLAINABLE_RETRIEVAL_GUIDE.md
│
├── tests/                 # 单元测试
├── data/                  # 数据目录
│   └── indexes/           # 索引存储
│
├── config.py              # 配置文件
├── requirements.txt       # 依赖
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/riverfielder/cad-vector-db.git
cd cad-vector-db

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 构建索引

```python
from cad_vectordb.core.index import IndexManager

# 创建索引管理器
manager = IndexManager("./data/indexes")

# 构建索引
stats = manager.build_index(
    data_root="../WHUCAD-main/data/vec",
    index_type="Flat",  # 或 "IVFFlat", "HNSW"
    max_samples=500,    # 测试用，实际可设为 None
    verbose=True
)

# 保存索引
manager.save_index("default")
```

### 3. 执行检索

```python
from cad_vectordb.core.retrieval import TwoStageRetrieval
from cad_vectordb.core.feature import load_macro_vec

# 初始化检索系统
retrieval = TwoStageRetrieval(
    manager.index,
    manager.ids,
    manager.metadata
)

# 加载查询
query_path = "../WHUCAD-main/data/vec/0000/00000000.h5"
query_vec = load_macro_vec(query_path)

# 执行检索
results = retrieval.search(
    query_vec,
    query_path,
    k=10,
    stage1_topn=100,
    fusion_method="weighted"
)

# 查看结果
for i, result in enumerate(results, 1):
    print(f"{i}. {result['id']}: {result['score']:.4f}")
```

### 4. 启动API服务

```bash
# 启动服务器
cd server
python app.py

# 或使用uvicorn
uvicorn server.app:app --reload --port 8000
```

### 5. 使用REST API

```bash
# 检索
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_file_path": "/path/to/query.h5",
    "k": 10,
    "stage1_topn": 100,
    "fusion_method": "weighted"
  }'

# 批量检索
curl -X POST http://localhost:8000/search/batch \
  -H "Content-Type: application/json" \
  -d '{
    "query_file_paths": ["/path/to/query1.h5", "/path/to/query2.h5"],
    "k": 10,
    "parallel": true
  }'

# 索引管理
curl http://localhost:8000/index/list
curl http://localhost:8000/index/default/stats
```

## 性能基准

### 测试环境
- CPU: Apple M1 Pro (12核)
- 内存: 32GB
- 索引大小: 500向量
- Python: 3.9
- FAISS: 1.7.4

### 检索性能

| 指标 | 值 |
|------|---|
| 平均延迟 | 19.5ms |
| P95延迟 | 24.1ms |
| QPS | 51.3 |

### 批量检索（50个查询）

| 模式 | 总时间 | QPS |
|------|--------|-----|
| 顺序 | 0.659s | 75.8 |
| 并行 | 0.797s | 62.7 |

### 索引类型对比

| 类型 | 构建时间 | 搜索延迟 | 精度 |
|------|---------|---------|------|
| Flat | 0.03s | 0.5ms | 100% |
| IVFFlat | 0.15s | 0.4ms | 98% |
| HNSW | 0.30s | 0.3ms | 99% |

## 使用示例

### 示例1：基本检索

```python
# 见 examples/basic_search.py
python examples/basic_search.py
```

### 示例2：索引管理

```python
# 见 examples/index_management.py
python examples/index_management.py
```

### 示例3：性能测试

```bash
# 运行基准测试
python -m benchmarks.benchmark_search \
    --index-dir ./data/indexes \
    --index-name default \
    --query-dir ../WHUCAD-main/data/vec/0000 \
    --num-queries 100 \
    --output benchmark_results.json
```

## 文档

- [使用指南](docs/USAGE.md) - 基本使用方法
- [索引管理](docs/INDEX_MANAGEMENT.md) - 索引管理详解
- [性能基准](docs/BENCHMARK.md) - 性能测试指南
- [批量检索](docs/BATCH_SEARCH_GUIDE.md) - 批量检索使用
- [可解释性](docs/EXPLAINABLE_RETRIEVAL_GUIDE.md) - 可解释性检索

## 技术栈

- **向量检索**: FAISS (Facebook AI Similarity Search)
- **Web框架**: FastAPI
- **数据库**: OceanBase (元数据)
- **数据格式**: HDF5
- **数值计算**: NumPy
- **可视化**: Matplotlib, HTML/CSS

## 系统要求

- Python 3.7+
- 8GB+ RAM（推荐16GB）
- 支持的操作系统：Linux, macOS, Windows

## 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- faiss-cpu >= 1.7.4
- fastapi >= 0.100.0
- uvicorn >= 0.23.0
- numpy >= 1.21.0
- h5py >= 3.7.0
- pydantic >= 2.0.0

## 配置

编辑 `config.py` 进行配置：

```python
# 数据路径
DATA_ROOT = "../WHUCAD-main/data/vec"
INDEX_DIR = "./data/indexes"

# 索引参数
INDEX_TYPE = "Flat"  # "Flat", "IVFFlat", "HNSW"
FEATURE_DIM = 32

# 检索参数
STAGE1_TOPN = 100
STAGE2_TOPK = 20
FUSION_METHOD = "weighted"
FUSION_ALPHA = 0.6
FUSION_BETA = 0.4

# 元数据数据库
DB_CONFIG = {
    "host": "localhost",
    "port": 2881,
    "user": "root@test",
    "password": "",
    "database": "cad_metadata"
}
```

## 开发路线图

### 已完成 ✅
- [x] 基础向量检索
- [x] 两阶段检索与融合
- [x] REST API服务
- [x] 元数据数据库集成
- [x] 混合检索
- [x] 可解释性检索
- [x] 批量检索
- [x] 索引管理
- [x] 性能基准测试

### 计划中 🚧
- [ ] 查询缓存
- [ ] 分布式部署
- [ ] 向量数据库对比（Milvus, Qdrant）
- [ ] Web可视化界面
- [ ] 更多索引类型支持

## 贡献

欢迎贡献代码、报告问题或提出建议！

## 许可证

MIT License

## 致谢

- WHUCAD数据集提供方
- FAISS团队
- FastAPI社区

## 联系方式

- 作者：He Tian
- 项目：毕业论文 - 向量数据库的设计与实现面向3D深度学习
- GitHub: [riverfielder/cad-vector-db](https://github.com/riverfielder/cad-vector-db)

---

**论文相关**: 本项目是毕业论文"向量数据库的设计与实现面向3D深度学习"的实现部分，完整展示了向量数据库系统的设计、实现和优化过程。
