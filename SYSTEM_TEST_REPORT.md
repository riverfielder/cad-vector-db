# CAD向量数据库系统测试报告

**测试时间**: 2025-12-25 15:13:10

## 测试概况

- 总测试数: 37
- ✅ 通过: 25 (67.6%)
- ❌ 失败: 7 (18.9%)
- ⚠️ 警告: 5

## ❌ 失败的测试

- 导入 cad_vectordb.core.feature: module 'cad_vectordb.core.feature' has no attribute 'extract_feature_from_h5'
- 导入 cad_vectordb.utils.visualization: module 'cad_vectordb.utils.visualization' has no attribute 'generate_explanation_html'
- 索引管理器测试: 'IndexManager' object has no attribute 'list_available_indices'
- 特征提取测试: cannot import name 'extract_feature_from_h5' from 'cad_vectordb.core.feature' (/Users/he.tian/bs/db/cad_vectordb/core/feature.py)
- 检索系统测试: cannot import name 'extract_feature_from_h5' from 'cad_vectordb.core.feature' (/Users/he.tian/bs/db/cad_vectordb/core/feature.py)
- 可视化测试: cannot import name 'generate_explanation_html' from 'cad_vectordb.utils.visualization' (/Users/he.tian/bs/db/cad_vectordb/utils/visualization.py)
- 脚本 build_index.py: 文件不存在

## ⚠️ 警告

- 增量更新: 版本控制未启用
- 批量检索: 检索系统未初始化
- 数据库配置: 数据库密码未配置，跳过连接测试
- API健康检查: 状态码 404
- API统计接口: 状态码 500

## ⚡ 性能指标

- 导入 cad_vectordb.core.index: 0.125s
- 导入 cad_vectordb.database.metadata: 0.037s
- 导入 cad_vectordb.core.text_encoder: 0.002s
- 导入 config: 0.001s
- IndexManager初始化: 0.000s
- 导入 cad_vectordb.core.retrieval: 0.000s

