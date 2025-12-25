# 数据目录说明

本目录包含CAD向量数据库所需的数据文件。

## 目录结构

```
data/
├── README.md                              # 本文件
├── train_val_test_split_data_split.json  # 数据集划分配置
├── vec/                                   # CAD向量数据目录
│   ├── 0000/                             # 子集0000 (1420个文件)
│   ├── 0001/                             # 子集0001 (1420个文件)
│   ├── 0002/                             # 子集0002 (1420个文件)
│   └── ...                               # 更多子集
├── index_test/                            # FAISS测试索引 (500样本)
├── index_full/                            # FAISS完整索引 (所有样本)
└── metadata.db                            # 元数据SQLite数据库
```

## 数据文件说明

### 1. 向量数据 (vec/)

- **格式**: HDF5 (.h5)
- **内容**: CAD宏序列向量
- **结构**: 每个.h5文件包含一个`vec`数组，形状为`(seq_len, 33)`
  - 第0列: 命令类型 (command)
  - 第1-32列: 命令参数 (parameters)
- **命名**: `{subset}/{id}.h5`，如`0000/00000001.h5`

### 2. 数据集划分 (train_val_test_split_data_split.json)

包含训练集、验证集和测试集的划分信息。

### 3. FAISS索引

- **index.faiss**: FAISS索引文件
- **ids.json**: 向量ID列表
- **config.json**: 索引配置参数

## 数据准备

### 方式1: 从WHUCAD复制（已完成）

当前项目已包含3个子集（0000-0002）的数据，共约4257个CAD模型。

如需更多数据，可以从WHUCAD项目复制：

```bash
# 复制更多子集
cp -r /path/to/WHUCAD-main/data/vec/0003 data/vec/
cp -r /path/to/WHUCAD-main/data/vec/0004 data/vec/
# ... 根据需要复制更多
```

### 方式2: 构建索引

使用项目提供的脚本构建FAISS索引：

```bash
# 快速测试索引（500样本）
python scripts/build_index.py --max_samples 500 --output_dir data/index_test

# 使用所有可用数据构建索引
python scripts/build_index.py --output_dir data/index_full

# 自定义索引类型
python scripts/build_index.py \
    --index_type HNSW \
    --hnsw_m 64 \
    --output_dir data/index_hnsw
```

## 数据统计

### 当前包含的数据

- **子集数量**: 3 (0000, 0001, 0002)
- **总文件数**: ~4257个.h5文件
- **磁盘占用**: ~40MB（H5文件）+ 2.4MB（JSON配置）

### 完整WHUCAD数据集

- **子集数量**: 104 (0000-0103)
- **总文件数**: ~146,331个.h5文件
- **磁盘占用**: ~1.2GB

## 配置说明

项目配置文件(`config.py`)中的数据路径已更新为：

```python
WHUCAD_DATA_ROOT = "data/vec"  # 项目内部路径
```

所有示例代码也已更新为使用项目内部数据路径。

## 注意事项

1. **大文件不提交**: `.gitignore`已配置排除`.h5`文件和索引文件
2. **保留配置**: JSON配置文件会被提交到git
3. **扩展数据**: 根据需要从WHUCAD复制更多子集
4. **索引重建**: 数据变更后需要重新构建FAISS索引

## 快速开始

1. 确认数据已复制：
   ```bash
   ls data/vec/
   # 应该看到: 0000  0001  0002
   ```

2. 构建测试索引：
   ```bash
   python scripts/build_index.py --max_samples 500 --output_dir data/index_test
   ```

3. 运行示例：
   ```bash
   python examples/basic_search.py
   ```

## 数据来源

数据来自WHUCAD-main项目，包含真实的CAD宏序列向量数据。

相关论文：
- WHUCAD: A Large-scale 3D Computer-Aided Design Dataset

## 许可证

数据使用需遵守WHUCAD项目的许可证要求。
