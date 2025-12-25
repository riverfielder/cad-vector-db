# CAD Vector Database - Quick Start Guide

## 完整实施流程（基于现有 WHUCAD vec 数据）

### 0. 环境准备

```bash
cd /Users/he.tian/bs/db
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. 构建索引（首次运行）

```bash
# 小样本测试（500个向量，快速验证）
python scripts/build_index.py --max_samples 500 --output_dir data/index_test

# 全量数据（所有向量，耗时较长）
python scripts/build_index.py --output_dir data/index_full
```

**输出**：
- `data/index/faiss_index.bin` - FAISS索引文件
- `data/index/id_map.json` - ID映射
- `data/index/metadata.json` - 元数据
- `data/index/config.json` - 索引配置

### 2. 测试检索（命令行）

```bash
# 单次查询测试
python scripts/retrieval.py \
  --index_dir data/index \
  --query_file /Users/he.tian/bs/WHUCAD-main/data/vec/0000/00000001.h5 \
  --topn 100 \
  --topk 20 \
  --fusion weighted
```

### 3. 启动API服务

```bash
# 启动服务器
python server/app.py
# 或者
cd server && uvicorn app:app --host 127.0.0.1 --port 8000
```

访问 http://127.0.0.1:8000/docs 查看交互式API文档

**API使用示例**：

```bash
# 检索
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_file_path": "/Users/he.tian/bs/WHUCAD-main/data/vec/0000/00000001.h5",
    "k": 20,
    "stage1_topn": 100,
    "fusion_method": "weighted",
    "alpha": 0.6,
    "beta": 0.4
  }'

# 获取统计信息
curl http://127.0.0.1:8000/stats

# 获取单个向量元数据
curl http://127.0.0.1:8000/vectors/0000/00000001.h5
```

### 4. 评估性能

```bash
# 评估（使用100个查询）
python evaluation/evaluate.py \
  --index_dir data/index \
  --n_queries 100 \
  --fusion weighted \
  --output evaluation/results_weighted.json

# 不同融合方法对比
python evaluation/evaluate.py --fusion rrf --output evaluation/results_rrf.json
python evaluation/evaluate.py --fusion borda --output evaluation/results_borda.json
```

**评估指标**：
- `precision@k`: 精确率
- `recall@k`: 召回率  
- `map`: 平均准确率均值
- `latency_p50/p95`: 延迟（毫秒）

### 5. 参数消融（论文实验）

#### a) 不同索引类型

```bash
# HNSW（默认）
python scripts/build_index.py --index_type HNSW --output_dir data/index_hnsw

# IVF
python scripts/build_index.py --index_type IVF --output_dir data/index_ivf

# IVFPQ（压缩）
python scripts/build_index.py --index_type IVFPQ --output_dir data/index_ivfpq
```

#### b) 修改 config.py 调参

编辑 `config.py`：
- `HNSW_M`: 16/32/64（邻居数，越大越精确但越慢）
- `HNSW_EF_SEARCH`: 64/128/256（搜索深度）
- `STAGE1_TOPN`: 50/100/200（召回候选数）
- `FUSION_ALPHA/BETA`: 调整融合权重

#### c) 不同融合方法

```bash
python evaluation/evaluate.py --fusion weighted
python evaluation/evaluate.py --fusion rrf
python evaluation/evaluate.py --fusion borda
```

### 6. 与 Qdrant/Milvus 对比（可选）

#### Qdrant

```bash
# 启动 Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 导入数据到 Qdrant（需另写脚本）
# 用相同查询集评估，对比延迟与精度
```

#### Milvus

```bash
# 启动 Milvus
docker run -p 19530:19530 -p 9091:9091 milvusdb/milvus-standalone

# 导入数据到 Milvus（需另写脚本）
# 用相同查询集评估，对比延迟与精度
```

## 目录结构

```
db/
├── README.md               # 项目说明（本文件）
├── USAGE.md               # 使用指南
├── requirements.txt       # Python依赖
├── config.py             # 配置文件
├── scripts/              # 脚本
│   ├── build_index.py    # 构建索引
│   └── retrieval.py      # 两阶段检索
├── server/               # API服务
│   └── app.py           # FastAPI应用
├── evaluation/           # 评估
│   ├── evaluate.py      # 评估脚本
│   └── results*.json    # 评估结果
└── data/                 # 数据目录
    └── index/           # 索引文件
```

## 关键技术点（论文写作参考）

### 1. 特征提取
- 从变长宏序列向量 `(seq_len, 33)` 提取固定维度特征
- 方法：均值池化参数列（忽略命令列）→ 32维向量
- 可扩展：最大池化、加权池化、Transformer编码

### 2. 两阶段检索
- **Stage 1（召回）**: FAISS ANN检索，快速召回Top-N候选
- **Stage 2（精排）**: 基于原始宏序列计算细粒度相似度
  - 命令匹配罚分 + 参数L2距离
  - 序列长度归一化

### 3. 融合排序
- **加权融合**: $S = \alpha \cdot sim_1' + \beta \cdot sim_2'$
- **RRF**: $S = \sum \frac{1}{k + rank}$（鲁棒，无需调参）
- **Borda计数**: 基于排名的简单融合

### 4. 评估体系
- 弱标签：同subset为相关（可人工标注小样本验证）
- 指标：P@K, R@K, mAP, 延迟p50/p95
- 消融：索引类型、参数、融合方法、Top-N

## 下一步工作

1. **生成潜向量z**（可选，提升精度）
   - 训练或使用已有AE checkpoint
   - `python test.py --mode enc --ckpt 1000`
   - 替换特征提取为加载z

2. **元数据数据库**（已实现，可选）
   - 安装 OceanBase：`docker run -d --name oceanbase-ce -p 2881:2881 -e MODE=mini oceanbase/oceanbase-ce`
   - 导入元数据：`python scripts/import_metadata_to_oceanbase.py --metadata data/index_test/metadata.json`
   - 查询统计：`python scripts/query_metadata_db.py stats`
   - 详细文档：参见 [docs/OCEANBASE_GUIDE.md](docs/OCEANBASE_GUIDE.md)

3. **部署与优化**
   - Docker化服务
   - 添加缓存层
   - 批量检索接口

4. **论文实验**
   - 完整消融实验矩阵
   - 与DeepCAD/其他方法对比
   - 可视化检索结果（调用WHUCAD可视化接口）

## 元数据数据库功能

### 快速开始

```bash
# 1. 启动 OceanBase（Docker）
docker run -d --name oceanbase-ce -p 2881:2881 -e MODE=mini oceanbase/oceanbase-ce

# 2. 安装依赖
pip install pymysql

# 3. 导入元数据
python scripts/import_metadata_to_oceanbase.py --metadata data/index_test/metadata.json

# 4. 查询统计
python scripts/query_metadata_db.py stats
```

### 主要功能

- ✅ 自动创建数据库和表
- ✅ 批量导入 JSON 元数据
- ✅ 支持按 subset、seq_len、label 过滤
- ✅ 提供 Python API 集成到检索流程
- ✅ 完整的统计和查询功能

### 使用场景

1. **过滤检索**：只在特定 subset 中检索
2. **序列长度过滤**：按序列长度范围过滤向量
3. **标签管理**：为向量添加标签（未来扩展）
4. **统计分析**：查看数据分布和统计信息

详细使用指南请参考 [docs/OCEANBASE_GUIDE.md](docs/OCEANBASE_GUIDE.md)

## 常见问题

**Q: 索引构建很慢？**
A: 先用 `--max_samples 500` 测试；全量数据视硬件需10分钟-1小时

**Q: 评估准确率很低？**
A: 当前用subset作为弱标签；建议人工标注小样本或使用更强的特征（潜向量z）

**Q: 如何提升检索速度？**
A: 调小 `HNSW_EF_SEARCH` 或使用 IVFPQ 压缩；trade-off精度与速度

**Q: 两阶段检索必要吗？**
A: 可以只用Stage1快速原型；Stage2+融合是论文创新点，能提升精度
