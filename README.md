**系统能力**
- 向量管理: 读取/校验 `data/vec/*/*.h5`，统一维度/键名/归一化，构建 ID 映射。
- ANN 召回: 基于潜向量 `z` 的近似最近邻检索（HNSW 或 IVF/PQ），支持 `cosine/L2`、Top-N。
- 精排与融合: 原始宏序列向量计算细粒度相似度；与召回分数融合排序（加权/RRF/Borda）。
- 过滤与查询: 按 `subset/label/source` 等元数据过滤；组合 Postgres/OceanBase 与向量索引。
- 持久化与热加载: 索引文件与映射持久化；服务重启快速加载。
- 评估与消融: P@K、Recall@K、mAP、延迟 p50/p95、吞吐；系统化参数消融。
- 对比与扩展: 与 Qdrant/Milvus 对比性能/工程复杂度；给出选型结论。

**数据与模式**
- 向量
  - `z`: 潜向量 `float32[d]`（建议 L2 归一化，`cosine` 更稳健）。
  - `vec`: 宏序列向量 `(seq_len, 1+N_ARGS)` 或一维编码（精排与展示）。
- 元数据（Postgres/OceanBase 表）
  - `items(id TEXT PK, file_path TEXT, subset TEXT, label TEXT, source TEXT, created_at TIMESTAMP, seq_len INT, args_dim INT)`
  - 可加索引：`(subset)`, `(label)`, `(source)`，提高过滤效率。
- 索引持久化
  - `faiss_index.bin`: 向量索引文件。
  - `id_map.json`: `faiss` 索引位置 → `id` 映射。
  - `config.json`: 索引/度量参数（`metric`, `M`, `efSearch`, `nlist`, `nprobe`, `pq_m`, `bits`）。

**架构与流程**
- Ingestion（离线批量）
  - 读取 HDF5 → 键名/维度校验 → 归一化/去重 → 写入 FAISS 索引；元数据入库（Postgres/OceanBase）。
- Index（基线）
  - FAISS-HNSW: `M=16/32`，`efConstruction=200`，`efSearch=64~256`。
  - 或 FAISS-IVF(+PQ): `nlist=4*sqrt(N)` 经验值，`nprobe=8~64`，PQ 压缩平衡精度-速度。
- API（FastAPI）
  - `POST /vectors` 批量插入（`id, z, metadata`）；`DELETE /vectors/{id}` 删除。
  - `POST /search` 召回与精排融合，支持 `k`, `metric`, `filters(subset/label)`, `stageN`。
  - `GET /vectors/{id}` 取向量/元数据；`GET /stats` 索引大小/延迟统计。
- 两阶段检索 + 融合
  - 阶段一: 用 `z` 做 ANN 取 Top-N。
  - 阶段二: 用 `vec` 计算精排分数（命令匹配罚分 + 参数 L2/DTW/加权编辑距离）。
  - 融合: Min-Max 归一化后加权和 $S=\\alpha s'_z+\\beta s'_{orig}$（起始 $\alpha=0.6,\\beta=0.4$）；或 RRF/Borda。
- 过滤策略
  - 先用 SQL 过滤候选（`subset/label/source`），再做精排；或在 Qdrant/Milvus 用 payload filter。

**评估与消融**
- 指标: Precision@K、Recall@K、mAP、延迟 p50/p95、QPS/吞吐。
- 数据划分: 用 train_val_test_split_data_split.json 构造查询/库；以 `label/subset` 为真值或弱标签。
- 消融变量
  - 索引类型: HNSW vs IVF(+PQ)。
  - 参数: `M/efSearch`、`nlist/nprobe`、`pq_m/bits`。
  - 度量: `cosine` vs `L2`（`cosine`需先单位化）。
  - N 候选: 50/100/200 的精度/延迟权衡。
  - 融合: 加权（网格调参）、RRF（`k0=60`）、Borda。
- 输出: 表格与曲线（P@K vs 延迟、mAP vs nprobe），结论与最佳参数范围。

**Qdrant/Milvus 对比路径（可选）**
- Qdrant
  - 部署: `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`
  - 客户端: `pip install qdrant-client`；集合创建（HNSW + cosine），payload 过滤。
- Milvus
  - 部署: `docker run -p 19530:19530 -p 9091:9091 milvusdb/milvus-standalone`
  - 客户端: `pip install pymilvus`；集合/索引（IVF/HNSW/PQ），批量导入，对比 nprobe/M。
- 复现实验: 用同一数据与评估协议，横向对比精度与延迟；记录工程复杂度与维护成本。

**实现要点与参数建议**
- 归一化: 召回前对 `z` 做 L2 归一化（`cosine`）；精排距离转相似度后做 Min-Max/Rank 归一化。
- HNSW: 起始 `M=32`、`efSearch=128`；增大 `efSearch` 提升精度但增大延迟。
- IVF: 起始 `nlist≈4*sqrt(N)`；`nprobe=16/32/64` 消融；大规模时配合 PQ。
- 融合权重: 从 `alpha=0.6,beta=0.4` 起，做网格搜索与交叉验证；或使用 RRF 借助排名稳健合并。
- 缓存: 对热门查询缓存候选与精排，降低延迟；批量接口合并请求。

**API 规范（示例）**
- `POST /vectors`
  - 请求: `{ "items": [ { "id": "...", "z": [..], "metadata": {"subset":"0000","label":"...","source":"z"} } ] }`
  - 响应: `{ "inserted": N, "duplicates": M }`
- `POST /search`
  - 请求: `{ "vector": [...], "k": 20, "metric":"cosine", "filters": {"subset":"0000","label":["..."]}, "stageN": 100, "fusion": {"method":"weighted","alpha":0.6} }`
  - 响应: `{ "results": [ { "id":"...", "score":0.92, "sim_z":0.88, "sim_orig":0.75, "metadata": {...} } ] }`

**Postgres/OceanBase**
- 表建表示例（Postgres）：
```sql
CREATE TABLE items (
  id TEXT PRIMARY KEY,
  file_path TEXT NOT NULL,
  subset TEXT,
  label TEXT,
  source TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  seq_len INT,
  args_dim INT
);
CREATE INDEX idx_items_subset ON items(subset);
CREATE INDEX idx_items_label ON items(label);
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