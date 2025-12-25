# 生产级功能指南 (P0)

本指南介绍 CAD Vector Database v2.0 中新增的生产级功能，包括日志系统、配置管理和安全性增强。

## 📋 目录

1. [日志系统](#日志系统)
2. [配置管理](#配置管理)
3. [安全性](#安全性)
4. [快速开始](#快速开始)
5. [生产部署建议](#生产部署建议)

---

## 1️⃣ 日志系统

### 特性

✅ **结构化日志**
- JSON 格式便于解析和分析
- 支持控制台彩色输出
- 包含时间戳、日志级别、模块信息

✅ **日志轮转**
- 按文件大小自动轮转（默认 10MB）
- 保留多个备份文件（默认 5 个）
- 防止日志文件占用过多磁盘空间

✅ **上下文追踪**
- Request ID 追踪完整请求生命周期
- 性能计时（响应时间）
- 异常堆栈完整记录

### 使用方法

#### 基础用法

```python
from cad_vectordb.utils.logger import get_logger

# 创建 logger
logger = get_logger('my_module')

# 记录不同级别的日志
logger.debug("详细调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

#### 带上下文的日志

```python
# 添加请求ID和额外数据
logger.info(
    "用户执行搜索",
    extra={
        'request_id': '12345',
        'user_id': 'user001',
        'extra_data': {
            'query_type': 'semantic',
            'k': 20
        }
    }
)
```

#### 异常日志

```python
try:
    # 一些操作
    result = perform_search()
except Exception as e:
    logger.error("搜索失败", exc_info=True)  # 包含完整堆栈
```

### 配置选项

通过环境变量配置日志行为：

```bash
# 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# 日志格式 (json 或 text)
LOG_FORMAT=json

# 输出目标
LOG_TO_FILE=true
LOG_TO_CONSOLE=true

# 轮转配置
LOG_MAX_BYTES=10485760  # 10MB
LOG_BACKUP_COUNT=5

# 日志目录
LOG_DIR=logs
```

### 查看日志

```bash
# 实时查看日志
tail -f logs/cad_vectordb.api.log

# 使用 jq 解析 JSON 日志
tail -f logs/cad_vectordb.api.log | jq '.'

# 筛选错误日志
cat logs/cad_vectordb.api.log | jq 'select(.level=="ERROR")'

# 查找特定请求的所有日志
cat logs/cad_vectordb.api.log | jq 'select(.request_id=="abc123")'
```

---

## 2️⃣ 配置管理

### 特性

✅ **环境变量支持**
- 从 `.env` 文件加载配置
- 覆盖默认配置
- 环境隔离（开发/测试/生产）

✅ **配置验证**
- 自动验证配置项
- 类型安全
- 错误提示清晰

✅ **密钥管理**
- 敏感信息不出现在代码中
- 支持 `.env` 文件（不提交到 Git）

### 配置文件

#### 1. 创建 `.env` 文件

```bash
# 复制示例配置
cp .env.example .env

# 编辑配置
vim .env
```

#### 2. 配置示例

```bash
# .env
ENV=production
API_PORT=8000
ENABLE_AUTH=true
API_KEY=your-secret-key-here
DB_PASSWORD=your-db-password

# 日志级别
LOG_LEVEL=WARNING

# CORS
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### 使用配置

```python
from cad_vectordb.utils.config import get_config

# 获取配置
config = get_config()

# 访问配置项
print(f"Environment: {config.env}")
print(f"API Port: {config.server.port}")
print(f"Index Type: {config.index.index_type}")

# 完整配置字典（密钥已脱敏）
config_dict = config.to_dict()
```

### 环境变量说明

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ENV` | development | 环境类型 |
| `API_PORT` | 8000 | API 端口 |
| `ENABLE_AUTH` | false | 启用认证 |
| `API_KEY` | - | API 密钥 |
| `RATE_LIMIT_ENABLED` | false | 启用限流 |
| `LOG_LEVEL` | INFO | 日志级别 |

完整配置选项请参考 `.env.example`。

---

## 3️⃣ 安全性

### 特性

✅ **API 认证**
- API Key 认证
- 密钥哈希存储
- 支持密钥撤销

✅ **限流保护**
- Token Bucket 算法
- 按客户端 IP 限流
- 可配置速率

✅ **输入验证**
- 路径遍历防护
- 文件扩展名验证
- 参数范围检查

✅ **CORS 配置**
- 可配置允许的源
- 跨域请求控制

### API 认证

#### 1. 生成 API Key

```python
from cad_vectordb.utils.security import key_manager

# 生成新密钥
api_key = key_manager.generate_key("production")
print(f"Your API Key: {api_key}")

# 或使用命令行
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### 2. 配置认证

```bash
# .env
ENABLE_AUTH=true
API_KEY=your-generated-key-here
```

#### 3. 使用 API Key

```bash
# HTTP 请求中包含 API Key
curl -X POST http://localhost:8000/search \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query_file_path": "data/query.h5", "k": 10}'
```

### 限流配置

```bash
# .env
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100  # 每期间最多请求数
RATE_LIMIT_PERIOD=60     # 时间窗口（秒）
```

超过限流后，API 返回 429 状态码：

```json
{
  "detail": "Rate limit exceeded. Try again later. Remaining: 0"
}
```

### 输入验证

系统自动验证所有输入：

```python
from cad_vectordb.utils.security import InputValidator, PathValidator

# 验证文件路径
if PathValidator.is_safe_path("/data", user_path):
    # 安全路径
    process_file(user_path)

# 验证 k 参数
if InputValidator.validate_k_value(k, max_k=1000):
    # 有效的 k 值
    perform_search(k)

# 消毒文本输入
clean_text = InputValidator.sanitize_text(user_input)
```

### CORS 配置

```bash
# .env
# 允许所有源（开发环境）
ALLOWED_ORIGINS=*

# 限制特定源（生产环境）
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

---

## 4️⃣ 快速开始

### 安装依赖

```bash
# 更新依赖
pip install -r requirements.txt

# 或单独安装新依赖
pip install python-dotenv
```

### 配置环境

```bash
# 1. 复制配置模板
cp .env.example .env

# 2. 编辑配置
vim .env

# 3. 生成 API Key（如果启用认证）
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 4. 更新 .env 中的 API_KEY
```

### 启动服务

```bash
# 开发模式（自动重载）
python server/app.py

# 生产模式
ENV=production uvicorn server.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### 测试安全功能

```bash
# 1. 测试健康检查（无需认证）
curl http://localhost:8000/health

# 2. 测试认证（需要 API Key）
curl -X POST http://localhost:8000/search \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query_file_path": "data/query.h5", "k": 10}'

# 3. 测试限流（快速发送多个请求）
for i in {1..150}; do
  curl http://localhost:8000/health &
done
wait
```

---

## 5️⃣ 生产部署建议

### 环境配置

```bash
# .env (生产环境)
ENV=production

# 安全
ENABLE_AUTH=true
API_KEY=<strong-secret-key>
ALLOWED_ORIGINS=https://yourdomain.com

# 限流
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_PERIOD=60

# 日志
LOG_LEVEL=WARNING
LOG_TO_FILE=true
LOG_DIR=/var/log/cad-vectordb

# 数据库
DB_PASSWORD=<strong-db-password>
DB_POOL_SIZE=10

# 性能
API_WORKERS=4
```

### 系统服务

创建 systemd 服务文件：

```ini
# /etc/systemd/system/cad-vectordb.service
[Unit]
Description=CAD Vector Database API
After=network.target

[Service]
Type=simple
User=cadvectordb
WorkingDirectory=/opt/cad-vectordb
Environment="PATH=/opt/cad-vectordb/.venv/bin"
EnvironmentFile=/opt/cad-vectordb/.env
ExecStart=/opt/cad-vectordb/.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable cad-vectordb
sudo systemctl start cad-vectordb
sudo systemctl status cad-vectordb
```

### Nginx 反向代理

```nginx
# /etc/nginx/sites-available/cad-vectordb
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时配置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### 监控

```bash
# 日志监控
tail -f /var/log/cad-vectordb/cad_vectordb.api.log | jq '.'

# 错误日志
cat /var/log/cad-vectordb/cad_vectordb.api.log | jq 'select(.level=="ERROR")'

# 慢查询
cat /var/log/cad-vectordb/cad_vectordb.api.log | jq 'select(.duration_ms > 1000)'

# 系统资源
htop
df -h
```

### 安全检查清单

- [ ] ✅ 已设置强 API Key
- [ ] ✅ 已配置数据库密码
- [ ] ✅ CORS 限制到特定域名
- [ ] ✅ 启用限流保护
- [ ] ✅ 使用 HTTPS
- [ ] ✅ 日志级别设置为 WARNING 或 ERROR
- [ ] ✅ 定期轮换 API Key
- [ ] ✅ 定期审计日志
- [ ] ✅ 定期备份数据
- [ ] ✅ 监控系统资源使用

---

## 📚 相关文档

- [API 文档](API_DOCUMENTATION.md)
- [部署指南](DEPLOYMENT.md)
- [故障排查](TROUBLESHOOTING.md)
- [性能优化](PERFORMANCE_TUNING.md)

## 🐛 问题报告

如遇问题，请查看：
1. 日志文件：`logs/cad_vectordb.api.log`
2. 配置验证：确保 `.env` 文件配置正确
3. GitHub Issues: https://github.com/riverfielder/cad-vector-db/issues

## 📝 更新日志

### v2.0.0 (2025-12-25)
- ✅ 新增统一日志系统
- ✅ 新增配置管理（环境变量）
- ✅ 新增 API 认证和限流
- ✅ 新增输入验证和安全增强
- ✅ 新增自定义异常类
