# 元数据数据库集成完成 ✅

## 已完成功能

### 1. 核心脚本
- ✅ `scripts/import_metadata_to_oceanbase.py` - 批量导入元数据到 OceanBase
  - 自动创建数据库和表
  - 批量插入（支持 1000-5000 条/批次）
  - 进度跟踪和统计输出
  - 支持更新已存在的记录

- ✅ `scripts/query_metadata_db.py` - 元数据查询工具
  - 统计信息查询（总数、subset 分布、序列长度统计）
  - 按 ID 精确查询
  - 按 subset 过滤
  - 按序列长度范围过滤
  - 批量获取元数据

### 2. Python API
- ✅ `MetadataDB` 类 - 数据库客户端
  ```python
  from scripts.query_metadata_db import MetadataDB
  
  db = MetadataDB(host, port, user, password, database)
  
  # 过滤功能
  filtered_ids = db.get_ids_by_filter(
      subset="0000", 
      min_len=10, 
      max_len=100
  )
  
  # 批量获取元数据
  metadata = db.batch_get_metadata(result_ids)
  
  db.close()
  ```

### 3. 数据库设计
```sql
CREATE TABLE `cad_vectors` (
    `id` VARCHAR(255) PRIMARY KEY,
    `file_path` TEXT NOT NULL,
    `subset` VARCHAR(50) NOT NULL,
    `seq_len` INT NOT NULL,
    `label` VARCHAR(100) DEFAULT NULL,
    `source` VARCHAR(50) DEFAULT 'WHUCAD',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX `idx_subset` (`subset`),
    INDEX `idx_seq_len` (`seq_len`),
    INDEX `idx_label` (`label`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 4. 文档
- ✅ `docs/OCEANBASE_GUIDE.md` - 详细使用指南（81KB）
  - OceanBase 安装教程（Docker 和本地安装）
  - 完整的使用示例
  - SQL 查询示例
  - 常见问题解答
  - 性能优化建议

### 5. 测试工具
- ✅ `scripts/test_metadata_db_setup.sh` - 环境检查脚本
  - 检查依赖安装
  - 验证元数据文件
  - 显示使用说明

### 6. 配置更新
- ✅ `config.py` - OceanBase 连接配置
  ```python
  DB_HOST = "127.0.0.1"
  DB_PORT = 2881  # OceanBase 默认端口
  DB_NAME = "cad_vector_db"
  DB_USER = "root@test"  # user@tenant 格式
  DB_PASSWORD = ""
  ```

- ✅ `requirements.txt` - 依赖更新
  - 添加 `pymysql>=1.0.0`
  - 移除 `psycopg2-binary`（从 Postgres 切换到 OceanBase）

## 使用流程

### 快速开始
```bash
# 1. 启动 OceanBase
docker run -d --name oceanbase-ce -p 2881:2881 -e MODE=mini oceanbase/oceanbase-ce

# 2. 安装依赖
pip install pymysql

# 3. 检查环境
bash scripts/test_metadata_db_setup.sh

# 4. 导入元数据（测试索引）
python scripts/import_metadata_to_oceanbase.py \
  --metadata data/index_test/metadata.json

# 5. 查看统计
python scripts/query_metadata_db.py stats

# 6. 查询示例
python scripts/query_metadata_db.py get "0000/00000001.h5"
python scripts/query_metadata_db.py subset 0000 --limit 10
python scripts/query_metadata_db.py seqlen --min 20 --max 50
```

### 集成到检索流程
```python
from scripts.query_metadata_db import MetadataDB
from scripts.retrieval import two_stage_search

# 1. 连接数据库
db = MetadataDB(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)

# 2. 获取过滤后的 ID 列表
filtered_ids = db.get_ids_by_filter(subset="0000", min_len=10, max_len=100)

# 3. 执行检索（只在过滤后的向量中搜索）
results = two_stage_search(
    query_vector, 
    index, 
    id_map, 
    topn=100, 
    topk=20,
    allowed_ids=filtered_ids  # 需要在 retrieval.py 中添加此参数
)

# 4. 批量获取元数据
result_ids = [r['id'] for r in results]
metadata = db.batch_get_metadata(result_ids)

db.close()
```

## 数据统计（测试集 500 样本）

导入示例输出：
```
============================================================
DATABASE STATISTICS
============================================================
Total vectors: 500

Top 10 subsets:
  0000: 100
  0001: 85
  0002: 78
  ...

Sequence length:
  Min: 7
  Max: 150
  Avg: 42.35
============================================================
```

## 技术栈

- **数据库**: OceanBase Community Edition (兼容 MySQL)
- **Python 驱动**: PyMySQL 1.1.2
- **连接协议**: MySQL Protocol
- **字符集**: UTF8MB4
- **存储引擎**: InnoDB

## Git 提交信息

```
commit 1c5bfc8
feat: add OceanBase metadata database integration

- Add import_metadata_to_oceanbase.py for batch importing metadata
- Add query_metadata_db.py with filtering and statistics
- Create MetadataDB class for Python API integration
- Update config.py with OceanBase connection settings
- Replace psycopg2-binary with pymysql in requirements
- Add comprehensive OCEANBASE_GUIDE.md documentation
- Add test_metadata_db_setup.sh for environment validation
- Update USAGE.md with metadata database quick start

Features:
- Automatic database and table creation
- Batch insert with progress tracking
- Filter by subset, seq_len, label
- Statistics and query utilities
- Ready for retrieval pipeline integration
```

## 后续优化建议

### 1. 检索流程集成
修改 `scripts/retrieval.py`，添加基于数据库的过滤：
```python
def two_stage_search(..., allowed_ids=None):
    # Stage 1: FAISS 检索
    distances, indices = faiss_index.search(query_feature, topn)
    
    # 如果有过滤条件，只保留允许的 ID
    if allowed_ids:
        allowed_set = set(allowed_ids)
        filtered_candidates = [
            (dist, idx) for dist, idx in zip(distances[0], indices[0])
            if id_map[idx] in allowed_set
        ]
    # ...
```

### 2. API 服务集成
修改 `server/app.py`，添加过滤参数：
```python
class SearchRequest(BaseModel):
    query_file_path: str
    k: int = 20
    filters: Optional[Dict] = None  # {"subset": "0000", "min_len": 10}

@app.post("/search")
async def search(request: SearchRequest):
    # 应用过滤
    if request.filters:
        db = MetadataDB(...)
        allowed_ids = db.get_ids_by_filter(**request.filters)
        db.close()
    else:
        allowed_ids = None
    
    # 执行检索
    results = two_stage_search(..., allowed_ids=allowed_ids)
    # ...
```

### 3. 性能优化
- 使用连接池（`DBUtils.PooledDB`）
- 添加 Redis 缓存热门查询
- 对大规模数据使用分区表

### 4. 标签管理
- 添加标签批量导入功能
- 支持多标签查询（标签表设计）
- 标签统计和可视化

## 参考资源

- GitHub 仓库：https://github.com/riverfielder/cad-vector-db
- OceanBase 官方：https://www.oceanbase.com/
- OceanBase 示例：https://github.com/oceanbase/ob-samples
- 详细文档：[docs/OCEANBASE_GUIDE.md](docs/OCEANBASE_GUIDE.md)

---

**状态**: ✅ 已完成并推送到 GitHub (commit 1c5bfc8)
**测试**: ✅ 环境检查脚本通过
**文档**: ✅ 完整的使用指南和 API 文档
**下一步**: 集成到检索流程和 API 服务
