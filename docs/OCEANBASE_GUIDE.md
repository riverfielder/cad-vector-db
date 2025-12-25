# OceanBase 元数据数据库使用指南

## 1. 安装 OceanBase

### 方式一：Docker 快速启动（推荐）

```bash
# 拉取并启动 OceanBase 容器
docker run -d --name oceanbase-ce \
  -p 2881:2881 \
  -e MODE=mini \
  -e OB_ROOT_PASSWORD='' \
  oceanbase/oceanbase-ce:latest

# 等待启动（约 2-3 分钟）
docker logs -f oceanbase-ce

# 看到 "boot success!" 表示启动成功
```

### 方式二：本地安装

参考官方文档：https://www.oceanbase.com/docs/

## 2. 安装 Python 依赖

```bash
cd .
source .venv/bin/activate
pip install pymysql
# 或者重新安装所有依赖
pip install -r requirements.txt
```

## 3. 配置数据库连接

编辑 `config.py`，修改数据库配置：

```python
# Database (OceanBase/MySQL)
DB_HOST = "127.0.0.1"
DB_PORT = 2881  # OceanBase 默认端口
DB_NAME = "cad_vector_db"
DB_USER = "root@test"  # OceanBase 格式: user@tenant
DB_PASSWORD = ""  # 设置您的密码
```

**注意：** OceanBase 用户名格式为 `user@tenant`，默认租户为 `test`。

## 4. 导入元数据

### 基本用法

```bash
# 导入测试索引的元数据
python scripts/import_metadata_to_oceanbase.py \
  --metadata data/index_test/metadata.json

# 导入全量索引的元数据
python scripts/import_metadata_to_oceanbase.py \
  --metadata data/index_full/metadata.json
```

### 高级选项

```bash
# 指定数据库连接参数
python scripts/import_metadata_to_oceanbase.py \
  --metadata data/index_test/metadata.json \
  --host 127.0.0.1 \
  --port 2881 \
  --user root@test \
  --password your_password \
  --database cad_vector_db

# 删除现有表重新导入
python scripts/import_metadata_to_oceanbase.py \
  --metadata data/index_test/metadata.json \
  --drop-table

# 调整批量插入大小
python scripts/import_metadata_to_oceanbase.py \
  --metadata data/index_test/metadata.json \
  --batch-size 5000
```

### 导入输出示例

```
============================================================
STEP 1: Connect to OceanBase
============================================================
✓ Successfully connected to OceanBase at 127.0.0.1:2881

============================================================
STEP 2: Create Database
============================================================
✓ Database `cad_vector_db` ready

============================================================
STEP 3: Create Table
============================================================
✓ Table `cad_vectors` created/verified

============================================================
STEP 4: Load Metadata
============================================================
Loading metadata from data/index_test/metadata.json...
✓ Loaded 500 records

============================================================
STEP 5: Import Data
============================================================
Progress: 500/500 (100%)

✓ Import completed!
  - Total records: 500
  - Inserted/Updated: 500

============================================================
STEP 6: Statistics
============================================================
DATABASE STATISTICS
============================================================
Total vectors: 500

Top 10 subsets:
  0000: 100
  0001: 80
  0002: 75
  ...

Sequence length:
  Min: 7
  Max: 150
  Avg: 42.35
============================================================

✓ All done! Connection closed.
```

## 5. 查询元数据

### 查看统计信息

```bash
python scripts/query_metadata_db.py stats
```

输出：
```json
{
  "total": 500,
  "subsets": {
    "0000": 100,
    "0001": 80,
    ...
  },
  "seq_len": {
    "min": 7,
    "max": 150,
    "avg": 42.35
  }
}
```

### 根据 ID 查询

```bash
python scripts/query_metadata_db.py get "0000/00000001.h5"
```

### 根据 subset 过滤

```bash
# 查询 subset 为 0000 的前 10 条记录
python scripts/query_metadata_db.py subset 0000 --limit 10
```

### 根据序列长度过滤

```bash
# 查询序列长度在 20-50 之间的记录
python scripts/query_metadata_db.py seqlen --min 20 --max 50 --limit 20
```

## 6. 在检索中使用元数据过滤

### Python API 示例

```python
from scripts.query_metadata_db import MetadataDB
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# 创建数据库客户端
db = MetadataDB(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)

# 获取满足条件的向量 ID 列表
filtered_ids = db.get_ids_by_filter(
    subset="0000",  # 只在 subset 0000 中检索
    min_len=10,     # 序列长度 >= 10
    max_len=100     # 序列长度 <= 100
)

print(f"Found {len(filtered_ids)} vectors matching filter")

# 在检索时使用这些 ID 进行过滤
# ... 执行向量检索 ...

# 批量获取检索结果的元数据
result_ids = ["0000/00000001.h5", "0000/00000002.h5"]
metadata_dict = db.batch_get_metadata(result_ids)

for vector_id, metadata in metadata_dict.items():
    print(f"{vector_id}: seq_len={metadata['seq_len']}, subset={metadata['subset']}")

db.close()
```

## 7. 数据库表结构

```sql
CREATE TABLE `cad_vectors` (
    `id` VARCHAR(255) PRIMARY KEY COMMENT 'Unique identifier (subset/filename)',
    `file_path` TEXT NOT NULL COMMENT 'Absolute path to h5 file',
    `subset` VARCHAR(50) NOT NULL COMMENT 'Subset directory (e.g., 0000, 0001)',
    `seq_len` INT NOT NULL COMMENT 'Sequence length of macro vector',
    `label` VARCHAR(100) DEFAULT NULL COMMENT 'Optional label (for future use)',
    `source` VARCHAR(50) DEFAULT 'WHUCAD' COMMENT 'Data source',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Import timestamp',
    INDEX `idx_subset` (`subset`),
    INDEX `idx_seq_len` (`seq_len`),
    INDEX `idx_label` (`label`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## 8. 直接使用 SQL 查询

```bash
# 连接到 OceanBase
docker exec -it oceanbase-ce obclient -h127.0.0.1 -P2881 -uroot@test -A

# 或者使用 mysql 客户端
mysql -h127.0.0.1 -P2881 -uroot@test
```

SQL 查询示例：

```sql
-- 使用数据库
USE cad_vector_db;

-- 查看表结构
DESC cad_vectors;

-- 统计总数
SELECT COUNT(*) FROM cad_vectors;

-- 按 subset 统计
SELECT subset, COUNT(*) as count 
FROM cad_vectors 
GROUP BY subset 
ORDER BY count DESC;

-- 查询特定条件
SELECT * FROM cad_vectors 
WHERE subset = '0000' 
  AND seq_len BETWEEN 20 AND 50 
LIMIT 10;

-- 序列长度分布
SELECT 
    FLOOR(seq_len/10)*10 as len_range,
    COUNT(*) as count
FROM cad_vectors
GROUP BY len_range
ORDER BY len_range;
```

## 9. 常见问题

**Q: OceanBase 启动失败？**
- 检查 Docker 是否运行：`docker ps`
- 查看日志：`docker logs oceanbase-ce`
- 确保端口 2881 未被占用：`lsof -i:2881`

**Q: 连接被拒绝？**
- 确认 OceanBase 已完全启动（需要 2-3 分钟）
- 检查配置中的用户名格式：`root@test`（包含租户名）
- 尝试使用 `obclient` 测试连接

**Q: 导入速度慢？**
- 增加 `--batch-size` 参数（如 5000）
- 检查网络连接和磁盘 I/O
- 考虑使用 `LOAD DATA` 语句（需要先导出 CSV）

**Q: 如何备份数据库？**
```bash
# 导出数据
docker exec oceanbase-ce mysqldump \
  -h127.0.0.1 -P2881 -uroot@test \
  cad_vector_db > backup.sql

# 恢复数据
docker exec -i oceanbase-ce mysql \
  -h127.0.0.1 -P2881 -uroot@test \
  cad_vector_db < backup.sql
```

## 10. 性能优化建议

1. **批量插入**：使用较大的 batch_size（1000-5000）
2. **索引优化**：根据查询模式添加合适的索引
3. **连接池**：生产环境使用连接池（如 `DBUtils`）
4. **分区表**：数据量大时考虑按 subset 分区

## 参考资源

- [OceanBase 官方文档](https://www.oceanbase.com/docs/)
- [PyMySQL 文档](https://pymysql.readthedocs.io/)
- [OceanBase 示例代码](https://github.com/oceanbase/ob-samples)
