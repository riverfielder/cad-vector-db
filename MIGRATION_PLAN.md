# 目录结构优化方案

## 新目录结构

```
db/
├── README.md
├── requirements.txt
├── config.py
├── setup.py                    # 新增：包安装配置
│
├── cad_vectordb/              # 核心库目录
│   ├── __init__.py
│   ├── core/                  # 核心功能模块
│   │   ├── __init__.py
│   │   ├── index.py          # 索引构建和管理
│   │   ├── retrieval.py      # 检索算法
│   │   └── feature.py        # 特征提取
│   ├── database/             # 数据库模块
│   │   ├── __init__.py
│   │   ├── metadata.py       # 元数据数据库操作
│   │   └── models.py         # 数据模型
│   ├── api/                  # API服务
│   │   ├── __init__.py
│   │   ├── app.py           # FastAPI应用
│   │   └── models.py        # API数据模型
│   └── utils/               # 工具函数
│       ├── __init__.py
│       └── visualization.py  # 可视化工具
│
├── benchmarks/               # 性能基准测试
│   ├── __init__.py
│   ├── benchmark_search.py   # 检索性能测试
│   ├── benchmark_index.py    # 索引性能测试
│   └── report.py            # 测试报告生成
│
├── tests/                    # 单元测试
│   ├── __init__.py
│   ├── test_index.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── examples/                 # 使用示例
│   ├── basic_search.py
│   ├── batch_search.py
│   └── index_management.py
│
├── docs/                     # 文档
│   ├── USAGE.md
│   ├── BATCH_SEARCH_GUIDE.md
│   ├── EXPLAINABLE_RETRIEVAL_GUIDE.md
│   ├── INDEX_MANAGEMENT.md  # 新增
│   └── BENCHMARK.md         # 新增
│
└── data/                     # 数据目录
    └── indexes/              # 索引存储
        ├── default/
        └── test/
```

## 迁移步骤

1. 创建新目录结构
2. 移动和重构现有代码
3. 更新导入路径
4. 添加新功能（索引管理、性能测试）
5. 更新文档
