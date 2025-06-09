# FastAPI Multi-GPU Video Generation API

基于 Wan2.1-I2V-14B-720P 模型的多卡分布式视频生成 API 服务，支持图像到视频（Image-to-Video）生成。采用模块化架构设计，支持华为昇腾 NPU 和 NVIDIA GPU 多卡分布式推理。

## 🚀 项目特色

- **🎯 多卡分布式**：支持 NPU/GPU 8卡并行推理，自动设备检测
- **🧠 T5 CPU 模式**：支持 T5 文本编码器在 CPU 上运行，节省显存
- **🔄 异步处理**：基于 FastAPI 的异步任务队列和状态管理
- **🧩 模块化架构**：清晰的分层设计，易于维护和扩展
- **⚡ 性能优化**：注意力缓存、VAE并行等多种加速技术
- **📊 任务管理**：完整的任务生命周期管理和队列控制
- **🛡️ 容错机制**：健壮的错误处理和资源清理
- **🔒 企业级安全**：资源限制、并发控制、异常处理
- **📈 监控运维**：健康检查、性能监控、调试工具
- **🎛️ 灵活配置**：配置生成器、环境预设、参数验证

## 📁 项目结构

```
fastapi-multigpu-i2v/
├── src/                              # 🎯 核心源码
│   ├── __init__.py
│   ├── i2v_api.py                    # FastAPI 主应用
│   ├── schemas/                      # 📋 数据模型
│   │   ├── __init__.py
│   │   └── video.py                  # 请求/响应模型定义
│   ├── services/                     # 🔧 业务逻辑层
│   │   ├── __init__.py
│   │   └── video_service.py          # 任务管理服务
│   ├── pipelines/                    # 🚀 推理管道
│   │   ├── __init__.py
│   │   ├── base_pipeline.py          # 管道基类
│   │   ├── npu_pipeline.py           # NPU 管道实现
│   │   ├── cuda_pipeline.py          # CUDA 管道实现
│   │   └── pipeline_factory.py       # 管道工厂
│   └── utils/                        # 🛠️ 工具类
│       ├── __init__.py
│       └── device_detector.py        # 设备自动检测
├── scripts/                          # 📜 启动脚本
│   ├── start_service_npu.sh          # NPU 专用启动脚本
│   ├── start_service_cuda.sh         # CUDA 专用启动脚本
│   ├── start_service_general.sh      # 通用启动脚本
│   └── debug/                        # 🔍 调试工具
│       ├── debug_t5_warmup.py        # T5 预热调试
│       ├── debug_pipeline.py         # 管道调试
│       ├── debug_device.py           # 设备检测调试
│       └── debug_memory.py           # 内存监控调试
├── tools/                            # 🛠️ 开发工具
│   ├── verify_structure.py           # 项目结构验证
│   ├── benchmark.py                  # 性能基准测试
│   ├── health_monitor.py             # 健康监控工具
│   └── config_generator.py           # 配置生成器
├── tests/                            # ✅ 测试用例
├── docs/                             # 📚 项目文档
├── generated_videos/                 # 📹 生成视频存储
├── logs/                             # 📝 日志文件
├── requirements.txt                  # 依赖清单
└── README.md                         # 项目文档
```

## 🔧 环境要求

### 硬件支持

#### NPU (华为昇腾)
- **设备型号**：910B1/910B2/910B4 等昇腾芯片
- **显存要求**：单卡 24GB+ (T5 CPU 模式) / 32GB+ (标准模式)
- **驱动版本**：CANN 8.0+

#### GPU (NVIDIA)
- **设备型号**：RTX 3090/4090, A100, H100 等
- **显存要求**：单卡 24GB+ (推荐 32GB+)
- **驱动版本**：CUDA 11.8+ / CUDA 12.0+

### 系统要求
- **CPU**：16+ 核心 (T5 CPU 模式建议 32+ 核心)
- **内存**：64GB+ 系统内存 (T5 CPU 模式需要更多)
- **存储**：200GB+ 可用空间 (模型 + 输出视频)
- **操作系统**：Linux (推荐 Ubuntu 20.04+)

### 软件环境
- **Python**：3.10+
- **PyTorch**：2.0+
- **设备扩展**：torch_npu (NPU) / torch (CUDA)

## 🛠️ 快速开始

### 1. 项目验证

```bash
# 克隆项目
git clone <repository-url>
cd fastapi-multigpu-i2v

# 验证项目结构
python3 tools/verify_structure.py

# 检测设备环境
python3 scripts/debug/debug_device.py
```

### 2. 环境配置

#### 自动配置生成
```bash
# 自动生成最优配置
python3 tools/config_generator.py --preset production

# 为开发环境生成配置
python3 tools/config_generator.py --preset development --output .env.dev

# 为内存受限环境生成配置
python3 tools/config_generator.py --preset memory_efficient
```

#### 手动环境配置
```bash
# NPU 环境变量
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export HCCL_TIMEOUT=3600

# CUDA 环境变量
export NCCL_TIMEOUT=3600
export CUDA_LAUNCH_BLOCKING=0

# 通用配置
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
```

### 3. 依赖安装

```bash
# 基础依赖
pip install -r requirements.txt

# NPU 环境验证
python3 -c "import torch_npu; print(f'NPU available: {torch_npu.npu.is_available()}')"

# CUDA 环境验证
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. 模型准备

```bash
# 设置模型路径
export MODEL_CKPT_DIR="/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"

# 验证模型文件
ls -la $MODEL_CKPT_DIR/
```

### 5. 启动服务

#### 🚀 智能启动 (推荐)
```bash
# 自动检测设备并启动最优配置
chmod +x scripts/start_service_general.sh
./scripts/start_service_general.sh
```

#### 🎯 设备专用启动
```bash
# NPU 专用启动 (华为昇腾)
chmod +x scripts/start_service_npu.sh
./scripts/start_service_npu.sh

# GPU 专用启动 (NVIDIA CUDA)
chmod +x scripts/start_service_cuda.sh  
./scripts/start_service_cuda.sh
```

#### 🎛️ 自定义配置启动
```bash
# T5 CPU 模式 (节省显存)
T5_CPU=true MAX_CONCURRENT_TASKS=2 ./scripts/start_service_npu.sh

# 高性能模式
T5_CPU=false MAX_CONCURRENT_TASKS=4 ./scripts/start_service_cuda.sh

# 调试模式
MAX_CONCURRENT_TASKS=1 TASK_TIMEOUT=3600 ./scripts/start_service_general.sh
```

### 6. 服务验证

```bash
# 健康检查
curl http://localhost:8088/health

# 设备信息
curl http://localhost:8088/device-info

# API 文档
open http://localhost:8088/docs
```

## 🔍 调试工具

### 设备检测调试
```bash
# 检测可用设备
python3 scripts/debug/debug_device.py

# 设备详细信息
python3 scripts/debug/debug_device.py --verbose
```

### T5 预热调试
```bash
# T5 CPU 模式预热测试
python3 scripts/debug/debug_t5_warmup.py --warmup-steps 3

# 指定模型路径测试
python3 scripts/debug/debug_t5_warmup.py --model-path /path/to/model
```

### 内存监控调试
```bash
# 查看当前内存状态
python3 scripts/debug/debug_memory.py --mode status

# 连续监控 60 秒
python3 scripts/debug/debug_memory.py --mode monitor --duration 60

# 模型加载内存测试
python3 scripts/debug/debug_memory.py --mode model-test

# 内存压力测试
python3 scripts/debug/debug_memory.py --mode stress-test
```

### 管道调试
```bash
# 测试管道创建
python3 scripts/debug/debug_pipeline.py

# 批量调试测试
bash scripts/debug/run_debug.sh
```

## 📊 监控和维护

### 健康监控工具
```bash
# 实时健康监控
python3 tools/health_monitor.py --url http://localhost:8088 --interval 30

# 连续监控 1 小时
python3 tools/health_monitor.py --duration 3600 --export health_report.json

# 告警监控
python3 tools/health_monitor.py --alert-on-error --alert-on-high-memory
```

### 性能基准测试
```bash
# 基础性能测试
python3 tools/benchmark.py --requests 10 --concurrent 2

# 压力测试
python3 tools/benchmark.py --requests 50 --concurrent 5 --duration 1800

# 导出测试报告
python3 tools/benchmark.py --export benchmark_report.json
```

### 配置管理
```bash
# 生成不同环境配置
python3 tools/config_generator.py --preset development --output configs/dev.env
python3 tools/config_generator.py --preset production --output configs/prod.env
python3 tools/config_generator.py --preset high_quality --output configs/hq.env

# 列出所有预设
python3 tools/config_generator.py --list-presets
```

## 📚 API 接口文档

### 核心接口

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/video/submit` | POST | 提交视频生成任务 | 支持分层参数配置 |
| `/video/status` | POST | 查询任务状态 | 实时进度跟踪 |
| `/video/cancel` | POST | 取消任务 | 支持优雅取消 |
| `/health` | GET | 服务健康检查 | 完整系统状态 |
| `/metrics` | GET | 监控指标 | 性能统计数据 |
| `/device-info` | GET | 设备信息 | 硬件配置详情 |
| `/docs` | GET | API 文档 | 交互式文档 |

### 请求参数分层

#### 基础参数 (普通用户)
```json
{
  "prompt": "A cat playing in the garden",
  "image_url": "https://example.com/input.jpg",
  "quality_preset": "balanced"
}
```

#### 进阶参数 (高级用户)
```json
{
  "prompt": "A cat playing in the garden",
  "image_url": "https://example.com/input.jpg", 
  "quality_preset": "custom",
  "guidance_scale": 4.0,
  "infer_steps": 40,
  "sample_solver": "dpmpp"
}
```

#### 专家参数 (系统管理员)
```json
{
  "prompt": "A cat playing in the garden",
  "image_url": "https://example.com/input.jpg",
  "t5_fsdp": true,
  "use_attentioncache": true,
  "cache_start_step": 12
}
```

### 响应示例

#### 健康检查响应
```json
{
  "status": "healthy",
  "timestamp": 1703847600.123,
  "uptime": 3600.5,
  "config": {
    "device_type": "npu",
    "device_count": 8,
    "t5_cpu": true,
    "max_concurrent": 2
  },
  "service": {
    "total_tasks": 15,
    "active_tasks": 1,
    "success_rate": 95.5
  },
  "resources": {
    "memory_usage": "68.5%",
    "available_slots": 1
  }
}
```

## ⚡ 性能优化

### T5 CPU 模式对比

| 模式 | T5 位置 | 显存占用 | 生成时间 | 并发能力 | 适用场景 |
|------|---------|----------|----------|----------|----------|
| 标准模式 | NPU/GPU | ~28GB | 2-3分钟 | 4-5任务 | 大显存环境 |
| T5 CPU | CPU | ~20GB | 2.5-3.5分钟 | 2-3任务 | 显存受限环境 |
| 混合模式 | 自适应 | ~24GB | 2.2-3分钟 | 3-4任务 | 平衡性能 |

### 设备特性对比

| 平台 | 优势 | 劣势 | 推荐配置 |
|------|------|------|----------|
| **NPU (昇腾)** | 高能效、大显存 | 生态相对较新 | T5_CPU=true, 保守并发 |
| **GPU (NVIDIA)** | 成熟生态、高性能 | 功耗较高 | T5_CPU=false, 激进并发 |

### 性能调优建议

#### 内存优化
```bash
# 显存受限环境
T5_CPU=true
DIT_FSDP=true
VAE_PARALLEL=false
MAX_CONCURRENT_TASKS=2
```

#### 速度优化  
```bash
# 追求最高性能
T5_CPU=false
VAE_PARALLEL=true
USE_ATTENTION_CACHE=true
ULYSSES_SIZE=8
```

#### 平衡配置
```bash
# 性能与稳定性平衡
T5_CPU=true
DIT_FSDP=true
VAE_PARALLEL=true
MAX_CONCURRENT_TASKS=3
```

## 🎛️ 配置参数

### 环境变量配置

#### 核心配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `T5_CPU` | false | T5 编码器是否使用 CPU |
| `DIT_FSDP` | true | DiT 模型是否使用 FSDP 分片 |
| `T5_FSDP` | false | T5 编码器是否使用 FSDP 分片 |
| `VAE_PARALLEL` | true | VAE 是否并行编解码 |
| `ULYSSES_SIZE` | 8 | Ulysses 序列并行组数 |

#### 服务配置

| 变量名 | 默认值 | T5 CPU 模式默认值 | 说明 |
|--------|--------|-------------------|------|
| `SERVER_HOST` | 0.0.0.0 | 0.0.0.0 | 服务监听地址 |
| `SERVER_PORT` | 8088 | 8088 | 服务端口 |
| `MAX_CONCURRENT_TASKS` | 5 | 2 | 最大并发任务数 |
| `TASK_TIMEOUT` | 1800 | 2400 | 任务超时时间(秒) |
| `CLEANUP_INTERVAL` | 300 | 300 | 清理间隔(秒) |
| `MAX_OUTPUT_DIR_SIZE` | 50 | 50 | 最大输出目录大小(GB) |
| `ALLOWED_HOSTS` | * | * | 允许的主机列表 |
| `MODEL_CKPT_DIR` | /data/models/... | /data/models/... | 模型文件路径 |

#### 通信配置

| 变量名 | 默认值 | T5 CPU 模式调整 | 说明 |
|--------|--------|-----------------|------|
| `HCCL_TIMEOUT` | 1800 | 2400 | HCCL 通信超时(秒) |
| `HCCL_CONNECT_TIMEOUT` | 600 | 900 | HCCL 连接超时(秒) |
| `HCCL_BUFFSIZE` | 512 | 256 | HCCL 缓冲区大小 |

#### NPU 专用环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ALGO` | 0 | NPU 算法选择 |
| `PYTORCH_NPU_ALLOC_CONF` | expandable_segments:True | NPU 内存分配策略 |
| `TASK_QUEUE_ENABLE` | 2 | NPU 任务队列启用级别 |
| `CPU_AFFINITY_CONF` | 1 | CPU 亲和性配置 |
| `ASCEND_LAUNCH_BLOCKING` | 0 | NPU 启动阻塞模式 |
| `ASCEND_GLOBAL_LOG_LEVEL` | 1 | NPU 全局日志级别 |
| `NPU_VISIBLE_DEVICES` | 0,1,2,3,4,5,6,7 | 可见 NPU 设备 |
| `ASCEND_RT_VISIBLE_DEVICES` | 0,1,2,3,4,5,6,7 | NPU 运行时可见设备 |

#### CUDA 专用环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `NCCL_TIMEOUT` | 3600 | NCCL 通信超时(秒) |
| `CUDA_LAUNCH_BLOCKING` | 0 | CUDA 启动阻塞模式 |
| `NCCL_DEBUG` | INFO | NCCL 调试级别 |
| `NCCL_IB_DISABLE` | 1 | 禁用 InfiniBand |
| `NCCL_P2P_DISABLE` | 0 | 禁用点对点通信 |
| `CUDA_VISIBLE_DEVICES` | 0,1,2,3,4,5,6,7 | 可见 CUDA 设备 |

#### CPU 优化配置

| 变量名 | 默认值 | T5 CPU 模式建议 | 说明 |
|--------|--------|-----------------|------|
| `OMP_NUM_THREADS` | 8 | 16 | OpenMP 线程数 |
| `MKL_NUM_THREADS` | 8 | 16 | MKL 线程数 |
| `OPENBLAS_NUM_THREADS` | 8 | 16 | OpenBLAS 线程数 |
| `TOKENIZERS_PARALLELISM` | false | false | 分词器并行处理 |

### API 请求参数

#### 基础参数

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

#### 分布式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vae_parallel` | bool | false | VAE 并行编解码 |
| `ulysses_size` | int | 1 | Ulysses 序列并行组数 |
| `dit_fsdp` | bool | false | DiT 模型 FSDP 分片 |
| `t5_fsdp` | bool | false | T5 编码器 FSDP 分片 |
| `cfg_size` | int | 1 | CFG 并行组数 |

#### 性能优化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_attentioncache` | bool | false | 启用注意力缓存 |
| `start_step` | int | 12 | 缓存起始步数 |
| `attentioncache_interval` | int | 4 | 缓存更新间隔 |
| `end_step` | int | 37 | 缓存结束步数 |
| `sample_solver` | string | "unipc" | 采样算法 |
| `sample_shift` | float | 5.0 | 采样偏移 |

### 质量预设配置

#### 预设模式对比

| 预设模式 | 推理步数 | 引导系数 | 采样算法 | 适用场景 |
|----------|----------|----------|----------|----------|
| `fast` | 20 | 2.5 | unipc | 快速测试 |
| `balanced` | 30 | 3.0 | unipc | 日常使用 |
| `high` | 50 | 4.0 | dpmpp | 高质量输出 |
| `custom` | 用户定义 | 用户定义 | 用户定义 | 自定义配置 |

#### 支持的分辨率

| 分辨率 | 比例 | 显存需求 | 生成时间 | 推荐场景 |
|--------|------|----------|----------|----------|
| `1280*720` | 16:9 | 标准 | 标准 | 横屏视频 |
| `720*1280` | 9:16 | 标准 | 标准 | 竖屏视频 |
| `1024*576` | 16:9 | 较低 | 较快 | 快速预览 |
| `1920*1080` | 16:9 | 较高 | 较慢 | 高清输出 |

#### 支持的帧数

| 帧数 | 时长 (24fps) | 显存需求 | 生成时间 | 推荐场景 |
|------|--------------|----------|----------|----------|
| 41 | ~1.7秒 | 较低 | 较快 | 短视频片段 |
| 61 | ~2.5秒 | 标准 | 标准 | 常规视频 |
| 81 | ~3.4秒 | 标准 | 标准 | 默认长度 |
| 121 | ~5.0秒 | 较高 | 较慢 | 长视频 |

### 配置模板示例

#### 开发环境配置

```bash
# 开发测试环境
export T5_CPU=true
export MAX_CONCURRENT_TASKS=1
export TASK_TIMEOUT=3600
export DIT_FSDP=false
export VAE_PARALLEL=false
export ULYSSES_SIZE=1
export ASCEND_GLOBAL_LOG_LEVEL=0  # 详细日志
```

#### 生产环境配置

```bash
# 生产环境 - NPU 高性能
export T5_CPU=false
export MAX_CONCURRENT_TASKS=5
export TASK_TIMEOUT=1800
export DIT_FSDP=true
export VAE_PARALLEL=true
export ULYSSES_SIZE=8
export USE_ATTENTION_CACHE=true
```

#### 内存优化配置

```bash
# 内存受限环境
export T5_CPU=true
export MAX_CONCURRENT_TASKS=2
export TASK_TIMEOUT=2400
export DIT_FSDP=true
export VAE_PARALLEL=false
export ULYSSES_SIZE=4
export HCCL_TIMEOUT=3600
```

#### 调试配置

```bash
# 调试模式
export T5_CPU=true
export MAX_CONCURRENT_TASKS=1
export TASK_TIMEOUT=7200
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_LAUNCH_BLOCKING=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

## 🚀 使用示例

### API 调用示例

#### 基础示例

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

# 检查服务配置
health_response = requests.get("http://localhost:8088/health")
health_data = health_response.json()
print(f"T5 CPU mode: {health_data['config']['t5_cpu']}")
print(f"Max concurrent: {health_data['config']['max_concurrent']}")
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

## 🛠️ 故障排除

### 常见问题及解决方案

#### 1. 启动失败
```bash
# 检查环境
python3 tools/verify_structure.py

# 检查设备
python3 scripts/debug/debug_device.py

# 检查依赖
pip install -r requirements.txt
```

#### 2. 内存不足
```bash
# 使用 T5 CPU 模式
T5_CPU=true MAX_CONCURRENT_TASKS=1 ./scripts/start_service_npu.sh

# 内存监控
python3 scripts/debug/debug_memory.py --mode monitor
```

#### 3. T5 预热失败
```bash
# T5 预热调试
python3 scripts/debug/debug_t5_warmup.py --warmup-steps 1

# 增加系统内存
echo "vm.swappiness=10" >> /etc/sysctl.conf
```

#### 4. 设备通信错误
```bash
# NPU 环境重置
export HCCL_TIMEOUT=7200
pkill -f i2v_api && sleep 10

# GPU 环境重置  
export NCCL_TIMEOUT=7200
nvidia-smi --gpu-reset
```

### 调试流程

```bash
# 1. 验证项目结构
python3 tools/verify_structure.py

# 2. 检测硬件环境
python3 scripts/debug/debug_device.py

# 3. 测试 T5 预热
python3 scripts/debug/debug_t5_warmup.py

# 4. 监控内存使用
python3 scripts/debug/debug_memory.py --mode status

# 5. 启动服务
./scripts/start_service_general.sh

# 6. 健康检查
curl http://localhost:8088/health
```

## 🔒 生产部署

### 安全配置

```bash
# 访问控制
export ALLOWED_HOSTS="api.company.com,*.company.com"

# 资源限制
export MAX_CONCURRENT_TASKS=3
export TASK_TIMEOUT=1800
export MAX_OUTPUT_DIR_SIZE=100

# 文件权限
chmod 750 generated_videos/
chown www-data:www-data generated_videos/
```

### 反向代理 (Nginx)

```nginx
upstream i2v_backend {
    server 127.0.0.1:8088;
}

server {
    listen 80;
    server_name api.company.com;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://i2v_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 1800s;
    }
    
    location /videos/ {
        alias /path/to/generated_videos/;
        expires 1d;
    }
}
```

### 系统服务 (systemd)

```ini
[Unit]
Description=FastAPI Multi-GPU I2V Service
After=network.target

[Service]
Type=simple
User=i2v-service
WorkingDirectory=/opt/fastapi-multigpu-i2v
Environment=T5_CPU=true
Environment=MAX_CONCURRENT_TASKS=2
ExecStart=/opt/fastapi-multigpu-i2v/scripts/start_service_npu.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 📋 依赖清单

### 核心依赖
```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
aiohttp>=3.9.0
torch>=2.0.0
transformers>=4.30.0
diffusers>=0.20.0
numpy>=1.24.0
Pillow>=10.0.0
```

### 设备特定
```txt
# NPU 环境
torch_npu

# GPU 环境  
torch[cuda]

# 可选依赖
psutil>=5.9.0        # 系统监控
opencv-python>=4.8.0 # 视频处理
```

## 🤝 贡献指南

### 开发流程

1. **Fork 项目**
2. **创建功能分支**: `git checkout -b feature/amazing-feature`
3. **验证代码**: `python3 tools/verify_structure.py`
4. **运行测试**: `python3 -m pytest tests/`
5. **提交代码**: `git commit -m 'Add amazing feature'`
6. **推送分支**: `git push origin feature/amazing-feature`
7. **创建 PR**

### 代码规范

```bash
# 代码格式化
black src/ tools/ scripts/
isort src/ tools/ scripts/

# 类型检查
mypy src/

# 测试覆盖
pytest --cov=src tests/
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **Wan AI Team** - 提供 Wan2.1-I2V-14B-720P 模型
- **华为昇腾** - NPU 硬件和 CANN 软件栈
- **NVIDIA** - GPU 硬件和 CUDA 软件栈  
- **FastAPI** - 高性能异步 Web 框架
- **PyTorch** - 深度学习框架

---

## 📞 技术支持

- **问题反馈**: [GitHub Issues](https://github.com/your-repo/issues)
- **讨论交流**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **技术文档**: [项目 Wiki](https://github.com/your-repo/wiki)

## 🎯 快速验证

```bash
# 一键验证和启动
git clone <repository-url>
cd fastapi-multigpu-i2v

# 验证环境
python3 tools/verify_structure.py
python3 scripts/debug/debug_device.py

# 启动服务 (自动检测最佳配置)
./scripts/start_service_general.sh

# 测试 API
curl http://localhost:8088/health
curl -X POST http://localhost:8088/video/submit \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test video", "image_url": "https://picsum.photos/1280/720"}'
```

**🚀 开始你的 AI 视频生成之旅！**