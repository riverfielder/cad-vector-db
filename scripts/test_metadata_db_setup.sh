#!/bin/bash
# 元数据数据库完整测试流程

set -e  # 遇到错误立即退出

echo "=========================================="
echo "OceanBase 元数据数据库测试"
echo "=========================================="
echo ""

# 检查依赖
echo "1. 检查依赖..."
python3 -c "import pymysql; print('✓ pymysql 已安装')"
python3 -c "import json; print('✓ json 已安装')"
echo ""

# 检查元数据文件
echo "2. 检查元数据文件..."
if [ -f "data/index_test/metadata.json" ]; then
    record_count=$(python3 -c "import json; print(len(json.load(open('data/index_test/metadata.json'))))")
    echo "✓ 找到测试元数据文件: $record_count 条记录"
else
    echo "✗ 未找到 data/index_test/metadata.json"
    echo "  请先运行: python scripts/build_index.py --max_samples 500 --output_dir data/index_test"
    exit 1
fi
echo ""

# 检查脚本文件
echo "3. 检查导入脚本..."
if [ -f "scripts/import_metadata_to_oceanbase.py" ]; then
    echo "✓ 导入脚本存在"
else
    echo "✗ 导入脚本不存在"
    exit 1
fi

if [ -f "scripts/query_metadata_db.py" ]; then
    echo "✓ 查询脚本存在"
else
    echo "✗ 查询脚本不存在"
    exit 1
fi
echo ""

# 显示使用说明
echo "=========================================="
echo "准备就绪！"
echo "=========================================="
echo ""
echo "下一步操作："
echo ""
echo "1. 启动 OceanBase（Docker）："
echo "   docker run -d --name oceanbase-ce -p 2881:2881 -e MODE=mini -e OB_ROOT_PASSWORD='' oceanbase/oceanbase-ce"
echo ""
echo "2. 等待启动完成（2-3分钟）："
echo "   docker logs -f oceanbase-ce"
echo "   # 看到 'boot success!' 表示成功"
echo ""
echo "3. 导入元数据："
echo "   python scripts/import_metadata_to_oceanbase.py --metadata data/index_test/metadata.json"
echo ""
echo "4. 查询统计："
echo "   python scripts/query_metadata_db.py stats"
echo ""
echo "5. 查看详细文档："
echo "   cat docs/OCEANBASE_GUIDE.md"
echo ""
echo "=========================================="
