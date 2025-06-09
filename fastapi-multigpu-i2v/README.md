# FastAPI Multi-GPU Video Generation API

基于 Wan2.1-I2V-14B-720P 模型的多卡分布式视频生成 API 服务，支持图像到视频（Image-to-Video）生成。采用模块化架构设计，支持华为昇腾 NPU 8卡分布式推理。

## 🚀 项目特色

- **🎯 多卡分布式**：支持 NV GPU, 华为昇腾 NPU 8卡并行推理
- **🔄 异步处理**：基于 FastAPI 的异步任务队列
- **🧩 模块化架构**：清晰的分层设计，易于维护和扩展
- **⚡ 性能优化**：注意力缓存、VAE并行等多种加速技术
- **📊 任务管理**：完整的任务状态跟踪和队列管理
- **🛡️ 容错机制**：健壮的错误处理和资源清理
- **🔒 企业级安全**：资源限制、并发控制、异常处理
- **📈 监控运维**：详细指标、健康检查、自动清理

## 📁 项目结构

```
src/
├── schemas/
│   ├── __init__.py
│   └── video.py              # 数据模型定义（请求/响应）
├── services/
│   ├── __init__.py
│   └── video_service.py      # 业务逻辑层（任务管理）
├── multigpu_pipeline.py      # 推理管道（分布式模型）
└── i2v_api.py               # API 接口层（FastAPI 应用）
generated_videos/             # 生成的视频文件存储
requirements.txt              # 项目依赖
README.md                     # 项目文档
```

## 🔧 环境要求

### 硬件要求
- **NPU**：华为昇腾 NPU × 8 卡
- **内存**：32GB+ 系统内存
- **存储**：100GB+ 可用空间

### 软件环境
- **Python**：3.11+
- **CANN**：华为昇腾 CANN 驱动
- **torch_npu**：PyTorch NPU 扩展

## 🛠️ 安装部署

### 1. 环境配置

```bash
# 设置核心环境变量
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

# 分布式通信配置
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 服务配置（可选）
export SERVER_HOST=0.0.0.0
export SERVER_PORT=8088
export MAX_CONCURRENT_TASKS=5
export TASK_TIMEOUT=1800
export CLEANUP_INTERVAL=300
export MAX_OUTPUT_DIR_SIZE=50
export ALLOWED_HOSTS="*"

# 模型配置（可选）
export MODEL_CKPT_DIR="/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"
```

### 2. 依赖安装

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 确保 torch_npu 正确安装
python -c "import torch_npu; print(torch_npu.__version__)"
```

### 3. 模型准备

确保模型文件位于指定路径：
```
/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P/
├── config.json
├── model.safetensors
└── ...
```

### 4. 启动服务

```bash
# 创建视频输出目录
mkdir -p generated_videos

# 启动 8 卡分布式服务
torchrun --nproc_per_node=8 src/i2v_api.py
```

服务启动后：
- **API 服务**：http://localhost:8088
- **API 文档**：http://localhost:8088/docs
- **健康检查**：http://localhost:8088/health
- **监控指标**：http://localhost:8088/metrics

## 📚 API 接口文档

### 接口概览

| 端点 | 方法 | 功能 | 状态码 |
|------|------|------|--------|
| `/video/submit` | POST | 提交视频生成任务 | 202 |
| `/video/status` | POST | 查询任务状态 | 200 |
| `/video/cancel` | POST | 取消任务 | 200 |
| `/health` | GET | 服务健康检查 | 200 |
| `/metrics` | GET | 获取详细指标 | 200 |
| `/docs` | GET | API 文档 | 200 |

### 1. 🎬 提交视频生成任务

**POST** `/video/submit`

#### 请求参数

```json
{
  "prompt": "A white cat wearing sunglasses sits on a surfboard at the beach",
  "image_url": "https://example.com/input.jpg",
  "image_size": "1280*720",
  "num_frames": 81,
  "guidance_scale": 3.0,
  "infer_steps": 30,
  "seed": 42,
  "negative_prompt": "blurry, low quality",
  
  // 分布式配置
  "vae_parallel": true,
  "ulysses_size": 8,
  "dit_fsdp": true,
  "t5_fsdp": true,
  
  // 性能优化
  "use_attentioncache": true,
  "start_step": 12,
  "attentioncache_interval": 4,
  "end_step": 37,
  "sample_solver": "unipc"
}
```

#### 响应

```json
{
  "requestId": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

#### 错误响应

```json
{
  "error": "VALIDATION_ERROR",
  "message": "参数验证失败的具体信息"
}
```

```json
{
  "detail": "服务器繁忙，请稍后重试"  // 429 状态码
}
```

### 2. 📊 查询任务状态

**POST** `/video/status`

#### 请求参数

```json
{
  "requestId": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

#### 响应

```json
{
  "status": "Succeed",
  "reason": null,
  "results": {
    "video_url": "/videos/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6.mp4",
    "video_path": "generated_videos/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6.mp4"
  },
  "queue_position": null,
  "progress": 1.0,
  "created_at": 1703847600.123,
  "updated_at": 1703847720.456
}
```

#### 任务状态说明

| 状态 | 描述 |
|------|------|
| `InQueue` | 任务已提交，等待处理 |
| `InProgress` | 正在生成视频 |
| `Succeed` | 生成成功 |
| `Failed` | 生成失败 |
| `Cancelled` | 任务已取消 |

### 3. ❌ 取消任务

**POST** `/video/cancel`

#### 请求参数

```json
{
  "requestId": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

#### 响应

```json
{
  "status": "Cancelled"
}
```

### 4. 🏥 健康检查

**GET** `/health`

#### 响应

```json
{
  "status": "healthy",
  "timestamp": 1703847600.123,
  "uptime": 3600.5,
  "rank": 0,
  "world_size": 8,
  "service": {
    "total_tasks": 15,
    "pipeline_world_size": 8,
    "pipeline_rank": 0
  },
  "resources": {
    "concurrent_tasks": 2,
    "max_concurrent": 5,
    "available_slots": 3
  }
}
```

### 5. 📈 监控指标

**GET** `/metrics`

#### 响应

```json
{
  "timestamp": 1703847600.123,
  "service": {
    "total_tasks": 15,
    "pipeline_world_size": 8,
    "pipeline_rank": 0
  },
  "resources": {
    "concurrent_tasks": 2,
    "max_concurrent": 5,
    "available_slots": 3
  },
  "system": {
    "rank": 0,
    "world_size": 8,
    "uptime": 3600.5
  }
}
```

## 🎛️ 配置参数

### 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SERVER_HOST` | 0.0.0.0 | 服务监听地址 |
| `SERVER_PORT` | 8088 | 服务端口 |
| `MAX_CONCURRENT_TASKS` | 5 | 最大并发任务数 |
| `TASK_TIMEOUT` | 1800 | 任务超时时间(秒) |
| `CLEANUP_INTERVAL` | 300 | 清理间隔(秒) |
| `MAX_OUTPUT_DIR_SIZE` | 50 | 最大输出目录大小(GB) |
| `ALLOWED_HOSTS` | * | 允许的主机列表 |
| `MODEL_CKPT_DIR` | /data/models/... | 模型文件路径 |

### 基础参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `prompt` | string | - | 10-500字符 | 视频描述提示词 |
| `image_url` | string | - | HTTP/HTTPS | 输入图像地址 |
| `image_size` | string | "1280*720" | 支持的分辨率 | 输出视频分辨率 |
| `num_frames` | int | 81 | 24-120 | 视频帧数 |
| `guidance_scale` | float | 3.0 | 1.0-20.0 | CFG 引导系数 |
| `infer_steps` | int | 30 | 20-100 | 去噪推理步数 |
| `seed` | int | null | 0-2147483647 | 随机数种子 |
| `negative_prompt` | string | null | 0-500字符 | 负面提示词 |

### 分布式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vae_parallel` | bool | false | VAE 并行编解码 |
| `ulysses_size` | int | 1 | Ulysses 序列并行组数 |
| `dit_fsdp` | bool | false | DiT 模型 FSDP 分片 |
| `t5_fsdp` | bool | false | T5 编码器 FSDP 分片 |
| `cfg_size` | int | 1 | CFG 并行组数 |

### 性能优化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_attentioncache` | bool | false | 启用注意力缓存 |
| `start_step` | int | 12 | 缓存起始步数 |
| `attentioncache_interval` | int | 4 | 缓存更新间隔 |
| `end_step` | int | 37 | 缓存结束步数 |
| `sample_solver` | string | "unipc" | 采样算法 |
| `sample_shift` | float | 5.0 | 采样偏移 |

## 🚀 使用示例

### 基础示例

```bash
# 提交视频生成任务
curl -X POST "http://localhost:8088/video/submit" \
-H "Content-Type: application/json" \
-d '{
  "prompt": "A serene lake with a swan gracefully swimming",
  "image_url": "https://picsum.photos/1280/720",
  "num_frames": 81,
  "guidance_scale": 3.0,
  "infer_steps": 30
}'

# 查询任务状态
curl -X POST "http://localhost:8088/video/status" \
-H "Content-Type: application/json" \
-d '{"requestId": "your-task-id-here"}'

# 检查服务健康状态
curl "http://localhost:8088/health"

# 获取监控指标
curl "http://localhost:8088/metrics"
```

### Python 客户端示例

```python
import requests
import time

# 提交任务
response = requests.post("http://localhost:8088/video/submit", json={
    "prompt": "A cat playing with a ball of yarn",
    "image_url": "https://example.com/cat.jpg",
    "num_frames": 81,
    "vae_parallel": True,
    "ulysses_size": 8,
    "use_attentioncache": True
})

if response.status_code == 429:
    print("服务器繁忙，请稍后重试")
    exit()

task_id = response.json()["requestId"]
print(f"Task submitted: {task_id}")

# 轮询任务状态
while True:
    status_response = requests.post("http://localhost:8088/video/status", 
                                   json={"requestId": task_id})
    status_data = status_response.json()
    
    print(f"Status: {status_data['status']}")
    
    if status_data["status"] == "Succeed":
        print(f"Video URL: {status_data['results']['video_url']}")
        break
    elif status_data["status"] == "Failed":
        print(f"Failed: {status_data['reason']}")
        break
    
    time.sleep(5)
```

### 错误处理示例

```python
import requests

try:
    response = requests.post("http://localhost:8088/video/submit", json={
        "prompt": "test",  # 太短，会触发验证错误
        "image_url": "invalid-url"
    })
    
    if response.status_code == 400:
        error_data = response.json()
        print(f"参数错误: {error_data['error']} - {error_data['message']}")
    elif response.status_code == 429:
        print("服务器繁忙，请稍后重试")
    elif response.status_code == 500:
        error_data = response.json()
        print(f"服务器错误: {error_data['error']} - {error_data['message']}")
    else:
        task_id = response.json()["requestId"]
        print(f"任务提交成功: {task_id}")

except requests.RequestException as e:
    print(f"网络错误: {str(e)}")
```

## 🏗️ 架构设计

### 分布式架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Rank 0 (主)   │    │   Rank 1-7      │    │   客户端请求     │
│  FastAPI 服务   │◄───┤   分布式推理     │◄───┤   HTTP API      │
│  任务管理       │    │   模型分片       │    │   WebSocket     │
│  资源控制       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │
         └────────┬─────────────────┘
                  ▼
    ┌─────────────────────────────────┐
    │       HCCL 通信后端             │
    │   (华为昇腾分布式通信框架)        │
    └─────────────────────────────────┘
```

### 数据流

```
HTTP 请求 → API 层 → 服务层 → 管道层 → 分布式模型 → 视频生成 → 文件保存 → HTTP 响应
   ↓           ↓        ↓        ↓          ↓          ↓          ↓          ↓
i2v_api.py → video_service.py → multigpu_pipeline.py → WanI2V → cache_video → 静态文件服务
   ↓
资源管理 → 并发控制 → 异常处理 → 监控记录
```

## ⚡ 性能特性

### 多卡并行加速

- **模型分片**：T5/DiT 模型 FSDP 分片，减少单卡内存占用
- **序列并行**：Ulysses 序列并行，处理长视频序列
- **VAE 并行**：视频编解码并行处理
- **CFG 并行**：分类器自由引导并行计算

### 推理优化

- **注意力缓存**：缓存中间注意力结果，减少重复计算
- **混合精度**：自动混合精度训练，提升计算效率
- **异步处理**：异步 I/O 和任务队列，提升并发能力

### 资源管理

- **并发控制**：限制同时处理的任务数量，防止资源耗尽
- **内存管理**：自动清理过期任务和临时文件
- **负载均衡**：智能任务调度，充分利用硬件资源

### 预期性能

| 配置 | 分辨率 | 帧数 | 生成时间 | 显存占用 | 并发数 |
|------|--------|------|----------|----------|--------|
| 8卡 NPU | 1280×720 | 81帧 | ~2-3分钟 | ~20GB | 1-5 |
| 8卡 NPU | 1280×720 | 121帧 | ~3-4分钟 | ~25GB | 1-3 |

## 🛠️ 故障排除

### 常见问题

#### 1. 服务启动失败

```bash
# 症状：ImportError 或模块不存在
# 解决方案：
export PYTHONPATH=/path/to/your/project:$PYTHONPATH
pip install -r requirements.txt

# 症状：端口被占用
# 解决方案：
export SERVER_PORT=8089  # 使用其他端口
```

#### 2. HCCL 初始化失败

```bash
# 症状：RuntimeError: HCCL init failed
# 解决方案：
ps aux | grep python | grep i2v_api | awk '{print $2}' | xargs kill -9
sleep 10
torchrun --nproc_per_node=8 src/i2v_api.py
```

#### 3. 任务提交被拒绝

```bash
# 症状：HTTP 429 "服务器繁忙"
# 原因：并发任务数超过限制
# 解决方案：
export MAX_CONCURRENT_TASKS=10  # 增加并发限制
# 或等待当前任务完成
```

#### 4. NPU 内存不足

```bash
# 检查 NPU 状态
npu-smi info
# 清理 NPU 缓存
python -c "import torch_npu; torch_npu.npu.empty_cache()"
# 调整并发数
export MAX_CONCURRENT_TASKS=2
```

#### 5. 任务失败常见原因

- **图像下载失败**：检查 `image_url` 是否可访问
- **参数验证失败**：检查帧数、分辨率等参数范围
- **模型文件缺失**：确认 `MODEL_CKPT_DIR` 路径正确

### 日志调试

```bash
# 启用详细日志
export PYTHONPATH=/path/to/your/project
export NCCL_DEBUG=INFO
torchrun --nproc_per_node=8 src/i2v_api.py

# 查看实时日志
tail -f /var/log/video_generation.log

# 查看错误日志
grep ERROR /var/log/video_generation.log
```

## 📊 监控和维护

### 健康检查

```bash
# 基础健康检查
curl http://localhost:8088/health

# 详细监控指标
curl http://localhost:8088/metrics

# 响应示例
{
  "status": "healthy",
  "timestamp": 1703847600.123,
  "uptime": 3600.5,
  "rank": 0,
  "world_size": 8,
  "service": {
    "total_tasks": 15,
    "pipeline_world_size": 8,
    "pipeline_rank": 0
  },
  "resources": {
    "concurrent_tasks": 2,
    "max_concurrent": 5,
    "available_slots": 3
  }
}
```

### 自动清理

服务会自动执行以下清理任务：

- **过期任务清理**：每5分钟清理超过30分钟的已完成任务
- **视频文件清理**：当存储超过50GB时自动删除最旧的文件
- **资源释放**：任务完成后自动释放并发槽位

### 手动维护

```bash
# 重启服务
ps aux | grep i2v_api | awk '{print $2}' | xargs kill -15
sleep 5
torchrun --nproc_per_node=8 src/i2v_api.py

# 清理生成的视频文件
find generated_videos -type f -mtime +7 -delete

# 检查磁盘使用
du -sh generated_videos/
```

## 🔒 安全注意事项

### 生产环境配置

```bash
# 限制允许的主机
export ALLOWED_HOSTS="api.example.com,*.example.com"

# 调整资源限制
export MAX_CONCURRENT_TASKS=3
export TASK_TIMEOUT=900  # 15分钟超时

# 设置安全的文件权限
chmod 750 generated_videos/
```

### 安全检查清单

- ✅ **输入验证**：严格验证图像 URL 和提示词
- ✅ **资源限制**：限制并发任务数量和视频长度
- ✅ **访问控制**：配置允许的主机列表
- ✅ **文件清理**：定期清理生成的视频文件
- ✅ **错误处理**：不暴露内部错误信息
- ⚠️ **认证机制**：生产环境建议添加 API Key 或 OAuth
- ⚠️ **HTTPS**：生产环境使用 HTTPS 加密传输
- ⚠️ **防火墙**：限制服务端口的网络访问

## 📋 依赖清单

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
aiohttp>=3.9.0
torch>=2.0.0
torch_npu
transformers>=4.30.0
diffusers>=0.20.0
numpy>=1.24.0
Pillow>=10.0.0
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **Wan AI Team** - 提供基础模型 Wan2.1-I2V-14B-720P
- **华为昇腾** - NPU 硬件和软件栈支持
- **FastAPI** - 高性能 Web 框架

---

**📞 技术支持**：如有问题，请提交 [Issue](https://github.com/BruceXcluding/Wan2.1/issues) 或联系维护团队。