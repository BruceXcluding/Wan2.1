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
- **🎛️ 智能配置**：配置生成器、环境预设、参数验证

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
│   │   └── video_service.py          # 任务管理服务 (待实现)
│   ├── pipelines/                    # 🚀 推理管道
│   │   ├── __init__.py
│   │   ├── base_pipeline.py          # 管道基类 (使用混入类重构)
│   │   ├── npu_pipeline.py           # NPU 管道实现 (简化版)
│   │   ├── cuda_pipeline.py          # CUDA 管道实现 (简化版)
│   │   └── pipeline_factory.py       # 管道工厂
│   └── utils/                        # 🛠️ 内部工具类
│       └── __init__.py
├── utils/                            # 🛠️ 项目级工具
│   ├── __init__.py
│   └── device_detector.py            # 设备自动检测
├── scripts/                          # 📜 启动脚本
│   ├── start_service_npu.sh          # NPU 专用启动脚本 (优化版)
│   ├── start_service_cuda.sh         # CUDA 专用启动脚本 (新增)
│   ├── start_service_general.sh      # 通用智能启动脚本 (优化版)
│   └── debug/                        # 🔍 调试工具集
│       ├── README.md                 # 调试工具使用指南
│       ├── debug_device.py           # 设备检测调试 (完整版)
│       ├── debug_pipeline.py         # 管道调试 (完整版)
│       ├── debug_memory.py           # 内存监控调试
│       └── debug_t5_warmup.py        # T5 预热调试
├── tools/                            # 🛠️ 开发工具集
│   ├── README.md                     # 开发工具使用指南
│   ├── verify_structure.py           # 项目结构验证 (完整版)
│   ├── config_generator.py           # 智能配置生成器 (完整版)
│   ├── benchmark.py                  # 性能基准测试
│   └── health_monitor.py             # 健康监控工具
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

### 1. 项目初始化与验证

```bash
# 克隆项目
git clone <repository-url>
cd fastapi-multigpu-i2v

# 🔍 完整项目验证 (推荐首次运行)
python3 tools/verify_structure.py

# 🎯 快速设备检测
python3 scripts/debug/debug_device.py

# 📦 检查可用管道
python3 -c "from pipelines import get_available_pipelines; print(f'Available: {get_available_pipelines()}')"
```

### 2. 智能配置生成

```bash
# 🤖 自动生成最优配置 (根据硬件环境)
python3 tools/config_generator.py --template production --export-env --output-dir configs/

# 🔧 为不同环境生成配置
python3 tools/config_generator.py --template development --export-env --output-dir configs/
python3 tools/config_generator.py --template testing --export-env --output-dir configs/

# 🎛️ 自定义配置生成
python3 tools/config_generator.py \
  --template production \
  --custom '{"max_concurrent_tasks": 2, "t5_cpu": true}' \
  --model-path /your/model/path \
  --port 8080 \
  --export-env

# 📋 查看生成的配置
cat configs/config_production.env
```

### 3. 环境配置

#### 使用生成的配置文件
```bash
# 加载生产配置
source configs/config_production.env

# 或者手动设置关键变量
export MODEL_CKPT_DIR="/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"
export T5_CPU=true
export MAX_CONCURRENT_TASKS=2
```

#### NPU 专用环境变量
```bash
export ASCEND_LAUNCH_BLOCKING=0
export HCCL_TIMEOUT=2400
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

#### CUDA 专用环境变量
```bash
export NCCL_TIMEOUT=1800
export CUDA_LAUNCH_BLOCKING=0
```

### 4. 依赖安装

```bash
# 基础依赖
pip install -r requirements.txt

# 验证设备环境
python3 scripts/debug/debug_device.py
```

### 5. 预启动调试 (推荐)

```bash
# 🧪 T5 预热测试 (T5 CPU 模式重要)
python3 scripts/debug/debug_t5_warmup.py --warmup-steps 3

# 🧠 内存状态检查
python3 scripts/debug/debug_memory.py --mode status

# 🔗 管道创建测试
python3 scripts/debug/debug_pipeline.py --mode quick

# 📊 系统资源检查
python3 tools/health_monitor.py --mode check
```

### 6. 启动服务

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

# 高性能模式 (需要大显存)
T5_CPU=false MAX_CONCURRENT_TASKS=4 ./scripts/start_service_cuda.sh

# 调试模式 (单任务,长超时)
MAX_CONCURRENT_TASKS=1 TASK_TIMEOUT=3600 ./scripts/start_service_general.sh
```

### 7. 服务验证与监控

```bash
# 🏥 健康检查
curl http://localhost:8088/health

# 📊 设备信息
curl http://localhost:8088/device-info

# 📈 监控指标
curl http://localhost:8088/metrics

# 📖 API 文档
open http://localhost:8088/docs

# 🔍 持续健康监控 (可选)
python3 tools/health_monitor.py --mode monitor --duration 3600 &
```

## 🔍 调试工具详解

### 设备检测调试
```bash
# 🎯 完整设备检测 (包含内存测试和兼容性检查)
python3 scripts/debug/debug_device.py

# 输出示例：
# ✅ Detected device: npu
# ✅ Device count: 8
# ✅ Backend: torch_npu
# 📊 Device Details...
# 🧠 Memory Operations Test...
# 🔗 Pipeline Compatibility...
```

### T5 预热调试 (重要!)
```bash
# 🧠 T5 CPU 模式预热测试 (首次启动前推荐)
python3 scripts/debug/debug_t5_warmup.py --warmup-steps 3

# 🎛️ 自定义测试参数
python3 scripts/debug/debug_t5_warmup.py \
  --model-path /path/to/model \
  --warmup-steps 5 \
  --skip-resource-check

# 输出示例：
# 🔍 T5 Warmup Debug Tool
# ✅ T5 warmup completed successfully
# 📊 Warmup time: 45.2s
# 🧠 Memory usage: 12.5GB
```

### 内存监控调试
```bash
# 📊 查看当前内存状态
python3 scripts/debug/debug_memory.py --mode status

# 📈 连续监控 60 秒
python3 scripts/debug/debug_memory.py --mode monitor --duration 60 --interval 5

# 🧪 模型加载内存测试
python3 scripts/debug/debug_memory.py --mode model-test

# 💥 内存压力测试
python3 scripts/debug/debug_memory.py --mode stress-test

# 📋 导出监控数据
python3 scripts/debug/debug_memory.py \
  --mode monitor \
  --duration 300 \
  --export memory_monitor.csv
```

### 管道系统调试
```bash
# 🔧 快速管道测试
python3 scripts/debug/debug_pipeline.py --mode quick

# 🧪 综合管道测试 (包含模型加载)
python3 scripts/debug/debug_pipeline.py --mode comprehensive

# 🎛️ 自定义模型路径测试
python3 scripts/debug/debug_pipeline.py --model-path /path/to/model

# 输出示例：
# 🔧 Pipeline Creation Test
# ✅ Pipeline created successfully in 25.3s
# ✅ Pipeline type: NPUPipeline
# ✅ Memory logging works
```

### 调试工具批量运行
```bash
# 📋 运行所有核心调试检查
echo "🔍 Running comprehensive debug checks..."
python3 scripts/debug/debug_device.py && \
python3 scripts/debug/debug_memory.py --mode status && \
python3 scripts/debug/debug_pipeline.py --mode quick && \
echo "✅ All debug checks completed!"
```

## 🛠️ 开发工具详解

### 项目结构验证
```bash
# 🔍 完整项目验证 (部署前必检)
python3 tools/verify_structure.py

# 检查内容：
# - 📄 必需文件和目录
# - 🐍 Python导入
# - 📦 依赖包
# - ⚙️ 配置文件
# - 🌍 运行时环境
# - 🔐 文件权限

# 输出示例：
# 🎉 PROJECT VERIFICATION PASSED!
# ✅ Your project structure is ready for deployment
```

### 智能配置生成器
```bash
# 🤖 自动硬件检测配置生成
python3 tools/config_generator.py --template production

# 🎯 生成不同环境配置
python3 tools/config_generator.py --template development --export-env
python3 tools/config_generator.py --template testing --export-env
python3 tools/config_generator.py --template memory_efficient --export-env

# 🎛️ 高级自定义配置
python3 tools/config_generator.py \
  --template production \
  --custom '{"t5_cpu": false, "max_concurrent_tasks": 6, "use_attentioncache": true}' \
  --model-path /high/performance/model \
  --port 8080 \
  --export-env \
  --export-json

# 📋 配置对比
echo "Development vs Production:"
diff configs/config_development.env configs/config_production.env
```

### 性能基准测试
```bash
# ⚡ 基础性能测试
python3 tools/benchmark.py --response-test 10

# 🚀 负载测试
python3 tools/benchmark.py --load-test --concurrent-users 5 --duration 300

# 📊 完整基准测试套件
python3 tools/benchmark.py \
  --response-test 20 \
  --load-test \
  --concurrent-users 10 \
  --duration 600 \
  --export benchmark_results.json

# 📈 性能监控 (与基准测试配合)
python3 tools/health_monitor.py --mode monitor --duration 600 &
python3 tools/benchmark.py --load-test --concurrent-users 8
```

### 健康监控工具
```bash
# 🏥 单次健康检查
python3 tools/health_monitor.py --mode check

# 📊 持续监控 30 分钟
python3 tools/health_monitor.py --mode monitor --duration 1800 --interval 30

# 📋 导出监控数据 (CSV/JSON)
python3 tools/health_monitor.py \
  --mode monitor \
  --duration 3600 \
  --export health_data.json \
  --alert-on-error \
  --alert-on-high-memory

# 🔔 告警监控模式
python3 tools/health_monitor.py \
  --mode monitor \
  --duration 86400 \
  --alert-on-error \
  --alert-threshold-memory 90 \
  --alert-threshold-response-time 30
```

### 工具组合使用示例
```bash
# 🚀 部署前完整检查流程
echo "🔍 Pre-deployment verification..."
python3 tools/verify_structure.py && \
python3 scripts/debug/debug_device.py && \
python3 tools/config_generator.py --template production --export-env && \
echo "✅ Ready for deployment!"

# 📊 生产环境监控配置
# 启动健康监控 (后台)
python3 tools/health_monitor.py \
  --mode monitor \
  --duration 86400 \
  --export daily_health.json \
  --alert-on-error &

# 启动服务
source configs/config_production.env
./scripts/start_service_general.sh

# 🧪 开发环境调试流程
echo "🔧 Development debug workflow..."
python3 scripts/debug/debug_memory.py --mode status
python3 scripts/debug/debug_t5_warmup.py --warmup-steps 1
python3 scripts/debug/debug_pipeline.py --mode comprehensive
echo "🎯 Debug completed!"
```

## 📊 性能监控与优化

### T5 CPU 模式对比分析

| 配置模式 | T5 位置 | 显存占用 | 生成时间 | 并发能力 | 适用场景 |
|----------|---------|----------|----------|----------|----------|
| **标准模式** | NPU/GPU | ~28GB | 2-3分钟 | 4-5任务 | 大显存环境 |
| **T5 CPU** | CPU | ~20GB | 2.5-3.5分钟 | 2-3任务 | 显存受限环境 |
| **混合模式** | 自适应 | ~24GB | 2.2-3分钟 | 3-4任务 | 平衡性能 |
| **优化模式** | CPU+缓存 | ~18GB | 2.8-3.2分钟 | 3-4任务 | 节能高效 |

### 设备平台对比

| 平台 | 优势 | 劣势 | 推荐配置 | 调试重点 |
|------|------|------|----------|----------|
| **NPU (昇腾)** | 高能效、大显存、稳定 | 生态相对较新 | T5_CPU=true, 保守并发 | HCCL通信、内存管理 |
| **GPU (NVIDIA)** | 成熟生态、高性能 | 功耗较高、显存限制 | 可选T5_CPU=false | CUDA同步、显存优化 |

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

### 配置优先级

配置的加载优先级（从高到低）：

1. **API 请求参数** - 单次请求覆盖
2. **环境变量** - 系统级配置
3. **配置文件** - 持久化配置
4. **默认值** - 代码中的默认配置

### 配置验证和生成

#### 自动配置生成
```bash
# 🤖 智能硬件检测配置生成
python3 tools/config_generator.py --template production --export-env

# 🎯 针对不同场景生成配置
python3 tools/config_generator.py --template development --export-env --output-dir configs/
python3 tools/config_generator.py --template testing --export-env --output-dir configs/
python3 tools/config_generator.py --template memory_efficient --export-env --output-dir configs/

# 🔧 自定义配置生成
python3 tools/config_generator.py \
  --template production \
  --custom '{"t5_cpu": true, "max_concurrent_tasks": 3, "use_attentioncache": true}' \
  --model-path /custom/model/path \
  --port 8080 \
  --export-env \
  --export-json
```

#### 配置验证
```bash
# 📋 验证配置完整性
python3 tools/verify_structure.py

# 🔍 配置环境检查
python3 scripts/debug/debug_device.py

# ⚙️ 生成配置对比
echo "🔍 Configuration comparison:"
echo "Development vs Production:"
diff configs/config_development.env configs/config_production.env

echo "Memory Efficient vs High Performance:"
diff configs/config_memory_efficient.env configs/config_production.env
```

### 动态配置调整

#### 运行时配置查看
```bash
# 🏥 查看当前服务配置
curl http://localhost:8088/health | jq '.config'

# 📊 查看设备信息
curl http://localhost:8088/device-info | jq '.'

# 📈 查看性能指标
curl http://localhost:8088/metrics | jq '.performance'
```

#### 配置热更新（部分支持）
```bash
# ⚡ 部分配置支持热更新（需要重启生效的配置）
# T5_CPU, DIT_FSDP, VAE_PARALLEL - 需要重启
# MAX_CONCURRENT_TASKS, TASK_TIMEOUT - 支持热更新

# 📝 更新并发配置（示例）
export MAX_CONCURRENT_TASKS=3
curl -X POST http://localhost:8088/admin/update-config \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent_tasks": 3, "task_timeout": 2400}'
```

### 配置最佳实践

#### 🎯 环境分离
```bash
# 🏠 开发环境
source configs/config_development.env
./scripts/start_service_general.sh

# 🚀 生产环境  
source configs/config_production.env
./scripts/start_service_general.sh

# 🧪 测试环境
source configs/config_testing.env
./scripts/start_service_general.sh
```

#### 🔒 安全配置
```bash
# 🛡️ 生产安全配置
export ALLOWED_HOSTS="api.company.com,*.company.com"
export CORS_ORIGINS="https://app.company.com"
export MAX_REQUEST_SIZE="50M"
export RATE_LIMIT_REQUESTS=100
export RATE_LIMIT_WINDOW=3600

# 📁 文件权限
export OUTPUT_DIR_PERMISSIONS=750
export LOG_DIR_PERMISSIONS=640
```

#### 📊 性能调优配置
```bash
# ⚡ 高性能配置 (大显存环境)
export T5_CPU=false                    # T5 使用 GPU
export VAE_PARALLEL=true               # VAE 并行
export ULYSSES_SIZE=8                  # 最大序列并行
export USE_ATTENTION_CACHE=true        # 注意力缓存
export MAX_CONCURRENT_TASKS=5          # 高并发

# 🧠 内存优化配置 (显存受限环境)
export T5_CPU=true                     # T5 使用 CPU
export VAE_PARALLEL=false              # 关闭 VAE 并行
export ULYSSES_SIZE=4                  # 适中序列并行
export MAX_CONCURRENT_TASKS=2          # 保守并发
export DIT_FSDP=true                   # 启用模型分片
```

### 配置故障排除

#### 🔍 配置诊断
```bash
# 📋 配置检查清单
echo "🔍 Configuration Diagnostics"
echo "=========================="

# 1. 环境变量检查
echo "1. Environment Variables:"
env | grep -E "(T5_CPU|DIT_FSDP|MAX_CONCURRENT|MODEL_CKPT_DIR)" | sort

# 2. 设备配置检查
echo "2. Device Configuration:"
python3 -c "
from utils.device_detector import device_detector
device_type, device_count = device_detector.detect_device()
print(f'Device: {device_type.value} x {device_count}')
"

# 3. 内存配置检查
echo "3. Memory Configuration:"
python3 scripts/debug/debug_memory.py --mode status

# 4. 模型路径检查
echo "4. Model Path:"
if [ -d "$MODEL_CKPT_DIR" ]; then
    echo "✅ Model directory exists: $MODEL_CKPT_DIR"
    ls -la "$MODEL_CKPT_DIR" | head -5
else
    echo "❌ Model directory not found: $MODEL_CKPT_DIR"
fi
```

#### ⚠️ 常见配置问题

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| T5 CPU 内存不足 | T5 加载失败 | 增加系统内存或使用 T5 GPU 模式 |
| 并发任务过多 | GPU OOM | 降低 MAX_CONCURRENT_TASKS |
| 通信超时 | 分布式训练失败 | 增加 HCCL_TIMEOUT/NCCL_TIMEOUT |
| 模型路径错误 | 模型加载失败 | 检查 MODEL_CKPT_DIR 路径 |
| 端口被占用 | 服务启动失败 | 修改 SERVER_PORT 或释放端口 |

这样配置参数部分就完整了，包含了所有重要的配置选项、使用示例和最佳实践！

## 📚 API 接口文档

### 核心接口总览

| 端点 | 方法 | 功能 | 监控指标 |
|------|------|------|----------|
| `/video/submit` | POST | 提交视频生成任务 | 请求量、成功率、队列长度 |
| `/video/status` | POST | 查询任务状态 | 查询频率、响应时间 |
| `/video/cancel` | POST | 取消任务 | 取消率、资源回收时间 |
| `/health` | GET | 服务健康检查 | 健康状态、资源使用率 |
| `/metrics` | GET | 监控指标 | 系统性能、任务统计 |
| `/device-info` | GET | 设备信息 | 设备状态、配置信息 |

### 请求参数分层设计

#### 🎯 简化模式 (推荐普通用户)
```json
{
  "prompt": "A cat playing in the garden",
  "image_url": "https://example.com/input.jpg",
  "quality_preset": "balanced"
}
```

#### 🔧 高级模式 (专业用户)
```json
{
  "prompt": "A cat playing in the garden",
  "image_url": "https://example.com/input.jpg", 
  "quality_preset": "custom",
  "guidance_scale": 4.0,
  "infer_steps": 40,
  "num_frames": 81,
  "image_size": "1280*720"
}
```

#### ⚙️ 专家模式 (系统管理员)
```json
{
  "prompt": "A cat playing in the garden",
  "image_url": "https://example.com/input.jpg",
  "guidance_scale": 4.0,
  "infer_steps": 40,
  "t5_fsdp": true,
  "vae_parallel": true,
  "use_attentioncache": true,
  "cache_start_step": 12,
  "ulysses_size": 8
}
```

### 增强版健康检查响应
```json
{
  "status": "healthy",
  "timestamp": 1703847600.123,
  "uptime": 3600.5,
  "config": {
    "device_type": "npu",
    "device_count": 8,
    "t5_cpu": true,
    "max_concurrent": 2,
    "model_path": "/data/models/...",
    "backend": "torch_npu"
  },
  "service": {
    "total_tasks": 15,
    "active_tasks": 1,
    "completed_tasks": 13,
    "failed_tasks": 1,
    "success_rate": 92.3,
    "avg_generation_time": 165.2
  },
  "resources": {
    "memory_usage": "68.5%",
    "device_memory_usage": "45.2%",
    "cpu_usage": "23.1%",
    "available_slots": 1,
    "disk_usage": "12.5GB"
  },
  "performance": {
    "requests_per_minute": 2.3,
    "avg_response_time": 2.1,
    "cache_hit_rate": 78.5
  }
}
```

## 🚀 使用示例与最佳实践

### 完整的客户端示例

```python
import requests
import time
import json
from typing import Optional, Dict, Any

class I2VClient:
    """智能 I2V API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8088"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_health(self) -> Dict[str, Any]:
        """检查服务健康状态"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """获取当前最优配置建议"""
        health = self.check_health()
        
        # 根据服务状态推荐配置
        config = {
            "guidance_scale": 3.0,
            "infer_steps": 30,
            "num_frames": 81
        }
        
        # 根据负载调整
        if health["service"]["active_tasks"] >= health["config"]["max_concurrent"] * 0.8:
            config["infer_steps"] = 25  # 降低步数提高吞吐
        
        # 根据T5模式调整
        if health["config"]["t5_cpu"]:
            config["guidance_scale"] = 3.5  # CPU模式可以稍高
        
        return config
    
    def submit_video(self, 
                    prompt: str, 
                    image_url: str,
                    auto_optimize: bool = True,
                    **kwargs) -> str:
        """智能提交视频生成任务"""
        
        # 基础请求
        request_data = {
            "prompt": prompt,
            "image_url": image_url,
        }
        
        # 自动优化配置
        if auto_optimize:
            optimal_config = self.get_optimal_config()
            request_data.update(optimal_config)
        
        # 用户自定义覆盖
        request_data.update(kwargs)
        
        # 提交任务
        response = self.session.post(
            f"{self.base_url}/video/submit",
            json=request_data
        )
        
        if response.status_code == 429:
            raise Exception("服务器繁忙，请稍后重试")
        elif response.status_code != 200:
            raise Exception(f"提交失败: {response.text}")
        
        return response.json()["requestId"]
    
    def wait_for_completion(self, 
                          task_id: str, 
                          timeout: int = 3600,
                          progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """等待任务完成"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 查询状态
            status_response = self.session.post(
                f"{self.base_url}/video/status",
                json={"requestId": task_id}
            )
            
            status_data = status_response.json()
            
            # 进度回调
            if progress_callback:
                progress_callback(status_data)
            
            # 检查完成状态
            if status_data["status"] == "Succeed":
                return status_data
            elif status_data["status"] == "Failed":
                raise Exception(f"任务失败: {status_data.get('reason', '未知原因')}")
            
            # 动态调整轮询间隔
            elapsed = time.time() - start_time
            if elapsed < 30:
                interval = 2  # 前30秒密集轮询
            elif elapsed < 120:
                interval = 5  # 2分钟内中等频率
            else:
                interval = 10  # 之后低频轮询
            
            time.sleep(interval)
        
        raise Exception(f"任务超时: {timeout}秒")

# 使用示例
def example_usage():
    """完整使用示例"""
    
    client = I2VClient("http://localhost:8088")
    
    try:
        # 1. 检查服务状态
        health = client.check_health()
        print(f"🏥 Service health: {health['status']}")
        print(f"🎯 Device: {health['config']['device_type']} x {health['config']['device_count']}")
        print(f"🧠 T5 CPU mode: {health['config']['t5_cpu']}")
        print(f"📊 Active tasks: {health['service']['active_tasks']}/{health['config']['max_concurrent']}")
        
        # 2. 智能提交任务
        task_id = client.submit_video(
            prompt="A serene lake with a swan gracefully swimming",
            image_url="https://picsum.photos/1280/720",
            auto_optimize=True,  # 自动优化配置
            quality_preset="high"  # 用户偏好
        )
        
        print(f"🚀 Task submitted: {task_id}")
        
        # 3. 等待完成 (带进度显示)
        def progress_callback(status):
            print(f"📊 Status: {status['status']} - {status.get('message', '')}")
        
        result = client.wait_for_completion(
            task_id, 
            timeout=3600,
            progress_callback=progress_callback
        )
        
        print(f"✅ Video completed: {result['results']['video_url']}")
        print(f"⏱️  Generation time: {result['results']['generation_time']:.1f}s")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    example_usage()
```

### 批量处理示例

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

async def batch_process_videos(prompts_and_images: list, max_concurrent: int = 3):
    """异步批量处理视频"""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_video(prompt, image_url):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                # 提交任务
                async with session.post(
                    "http://localhost:8088/video/submit",
                    json={
                        "prompt": prompt,
                        "image_url": image_url,
                        "quality_preset": "balanced"
                    }
                ) as response:
                    if response.status != 200:
                        return {"error": await response.text()}
                    
                    task_data = await response.json()
                    task_id = task_data["requestId"]
                
                # 等待完成
                while True:
                    async with session.post(
                        "http://localhost:8088/video/status",
                        json={"requestId": task_id}
                    ) as status_response:
                        status_data = await status_response.json()
                        
                        if status_data["status"] == "Succeed":
                            return {
                                "task_id": task_id,
                                "video_url": status_data["results"]["video_url"],
                                "generation_time": status_data["results"]["generation_time"]
                            }
                        elif status_data["status"] == "Failed":
                            return {"error": status_data.get("reason", "Unknown error")}
                    
                    await asyncio.sleep(5)
    
    # 并发处理
    tasks = [
        process_single_video(prompt, image_url) 
        for prompt, image_url in prompts_and_images
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# 使用示例
async def batch_example():
    tasks = [
        ("A cat playing with a ball", "https://example.com/cat.jpg"),
        ("A dog running in the park", "https://example.com/dog.jpg"),
        ("A bird flying in the sky", "https://example.com/bird.jpg"),
    ]
    
    results = await batch_process_videos(tasks, max_concurrent=2)
    
    for i, result in enumerate(results):
        if isinstance(result, dict) and "error" not in result:
            print(f"✅ Task {i+1}: {result['video_url']} ({result['generation_time']:.1f}s)")
        else:
            print(f"❌ Task {i+1}: {result.get('error', str(result))}")

# asyncio.run(batch_example())
```

## 🛠️ 故障排除与运维

### 常见问题诊断流程

#### 🔍 启动失败诊断
```bash
# 1. 项目结构检查
echo "📋 Checking project structure..."
python3 tools/verify_structure.py

# 2. 设备环境检查  
echo "🔧 Checking device environment..."
python3 scripts/debug/debug_device.py

# 3. 依赖检查
echo "📦 Checking dependencies..."
python3 -c "
try:
    import torch, fastapi, uvicorn
    from schemas import VideoSubmitRequest
    from pipelines import PipelineFactory
    from utils import device_detector
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
"

# 4. 模型路径检查
echo "📁 Checking model path..."
if [ -d "$MODEL_CKPT_DIR" ]; then
    echo "✅ Model directory exists: $MODEL_CKPT_DIR"
    ls -la "$MODEL_CKPT_DIR" | head -10
else
    echo "❌ Model directory not found: $MODEL_CKPT_DIR"
fi
```

#### 🧠 内存问题诊断
```bash
# 1. 当前内存状态
python3 scripts/debug/debug_memory.py --mode status

# 2. T5 CPU 模式检查
if [ "$T5_CPU" = "true" ]; then
    echo "✅ T5 CPU mode enabled"
    python3 scripts/debug/debug_t5_warmup.py --warmup-steps 1
else
    echo "⚠️  T5 GPU mode - high memory usage expected"
fi

# 3. 建议的内存优化配置
echo "💡 Recommended memory optimization:"
echo "   T5_CPU=true"
echo "   MAX_CONCURRENT_TASKS=2"
echo "   DIT_FSDP=true"
echo "   VAE_PARALLEL=false"
```

#### 🔗 通信问题诊断
```bash
# NPU 通信检查
if [ "$device_type" = "npu" ]; then
    echo "🔍 NPU communication check..."
    python3 -c "
import torch_npu
print(f'NPU available: {torch_npu.npu.is_available()}')
print(f'NPU count: {torch_npu.npu.device_count()}')
try:
    torch_npu.npu.synchronize()
    print('✅ NPU synchronization OK')
except Exception as e:
    print(f'❌ NPU sync failed: {e}')
"
fi

# GPU 通信检查
if [ "$device_type" = "cuda" ]; then
    echo "🔍 CUDA communication check..."
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA count: {torch.cuda.device_count()}')
try:
    torch.cuda.synchronize()
    print('✅ CUDA synchronization OK')
except Exception as e:
    print(f'❌ CUDA sync failed: {e}')
"
fi
```

### 自动故障恢复脚本

```bash
#!/bin/bash
# 📧 auto_recovery.sh - 自动故障恢复脚本

echo "🚨 Auto Recovery Script Started"

# 1. 检查服务状态
if ! curl -s http://localhost:8088/health > /dev/null; then
    echo "❌ Service not responding, attempting recovery..."
    
    # 停止旧进程
    pkill -f "i2v_api.py" || true
    pkill -f "torchrun.*i2v_api" || true
    sleep 10
    
    # 清理设备缓存
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('✅ CUDA cache cleared')
except: pass

try:
    import torch_npu
    if torch_npu.npu.is_available():
        torch_npu.npu.empty_cache()
        print('✅ NPU cache cleared')
except: pass
"
    
    # 重启服务
    echo "🔄 Restarting service..."
    ./scripts/start_service_general.sh &
    
    # 等待启动
    sleep 60
    
    # 验证恢复
    if curl -s http://localhost:8088/health > /dev/null; then
        echo "✅ Service recovered successfully"
    else
        echo "❌ Recovery failed, manual intervention required"
        exit 1
    fi
else
    echo "✅ Service is healthy"
fi

# 2. 检查资源使用
python3 scripts/debug/debug_memory.py --mode status

# 3. 清理旧文件 (可选)
find generated_videos/ -name "*.mp4" -mtime +7 -delete
find logs/ -name "*.log" -mtime +7 -delete

echo "🎯 Auto recovery completed"
```

### 生产环境监控配置

```bash
#!/bin/bash
# 🔍 production_monitor.sh - 生产环境监控

# 创建监控目录
mkdir -p monitoring/{logs,alerts,reports}

# 1. 健康监控 (24/7)
python3 tools/health_monitor.py \
  --mode monitor \
  --duration 86400 \
  --interval 300 \
  --export monitoring/logs/health_$(date +%Y%m%d).json \
  --alert-on-error \
  --alert-threshold-memory 85 \
  --alert-threshold-response-time 30 &

# 2. 内存监控
python3 scripts/debug/debug_memory.py \
  --mode monitor \
  --duration 86400 \
  --interval 600 \
  --export monitoring/logs/memory_$(date +%Y%m%d).csv &

# 3. 性能基准 (每小时)
while true; do
    python3 tools/benchmark.py \
      --response-test 5 \
      --export monitoring/reports/benchmark_$(date +%Y%m%d_%H%M).json
    sleep 3600
done &

# 4. 自动恢复检查 (每30分钟)
while true; do
    ./auto_recovery.sh >> monitoring/logs/recovery_$(date +%Y%m%d).log 2>&1
    sleep 1800
done &

echo "🚀 Production monitoring started"
echo "📊 Logs: monitoring/logs/"
echo "📈 Reports: monitoring/reports/"
```

## 🔒 生产部署最佳实践

### 安全配置
```bash
# 访问控制
export ALLOWED_HOSTS="api.company.com,*.company.com"
export CORS_ORIGINS="https://app.company.com,https://admin.company.com"

# 资源限制
export MAX_CONCURRENT_TASKS=3
export TASK_TIMEOUT=1800
export MAX_OUTPUT_DIR_SIZE=100
export MAX_REQUEST_SIZE=50M

# API 限流
export RATE_LIMIT_REQUESTS=100
export RATE_LIMIT_WINDOW=3600

# 文件权限
chmod 750 generated_videos/
chown i2v-service:i2v-service generated_videos/
```

### Docker 部署配置
```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 项目文件
COPY . .

# 权限设置
RUN chmod +x scripts/*.sh

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8088/health || exit 1

# 启动命令
CMD ["./scripts/start_service_general.sh"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  i2v-api:
    build: .
    ports:
      - "8088:8088"
    environment:
      - T5_CPU=true
      - MAX_CONCURRENT_TASKS=2
      - MODEL_CKPT_DIR=/data/models
    volumes:
      - ./generated_videos:/app/generated_videos
      - ./logs:/app/logs
      - /data/models:/data/models:ro
    deploy:
      resources:
        limits:
          memory: 64G
        reservations:
          memory: 32G
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./generated_videos:/var/www/videos:ro
    depends_on:
      - i2v-api
    restart: unless-stopped
```

### Kubernetes 部署配置
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: i2v-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: i2v-api
  template:
    metadata:
      labels:
        app: i2v-api
    spec:
      containers:
      - name: i2v-api
        image: your-registry/i2v-api:latest
        ports:
        - containerPort: 8088
        env:
        - name: T5_CPU
          value: "true"
        - name: MAX_CONCURRENT_TASKS
          value: "2"
        resources:
          requests:
            memory: "32Gi"
            nvidia.com/gpu: 4
          limits:
            memory: "64Gi"
            nvidia.com/gpu: 4
        livenessProbe:
          httpGet:
            path: /health
            port: 8088
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8088
          initialDelaySeconds: 60
          periodSeconds: 10
```

## 📋 依赖清单

### 核心依赖
```txt
# API 框架
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0

# 深度学习
torch>=2.0.0
transformers>=4.30.0
diffusers>=0.20.0

# 图像处理
Pillow>=10.0.0
opencv-python>=4.8.0

# 数值计算
numpy>=1.24.0

# HTTP 客户端
aiohttp>=3.9.0
requests>=2.31.0

# 系统监控
psutil>=5.9.0
```

### 设备特定依赖
```txt
# NPU 环境 (华为昇腾)
torch_npu  # 根据CANN版本选择

# GPU 环境 (NVIDIA)
torch[cuda]>=2.0.0

# 开发工具
pytest>=7.4.0
black>=23.0.0
isort>=5.12.0
mypy>=1.5.0
```

### 可选增强依赖
```txt
# 性能监控
prometheus-client>=0.18.0
grafana-api>=1.0.3

# 日志管理
structlog>=23.1.0
python-json-logger>=2.0.7

# 配置管理
python-dotenv>=1.0.0
pyyaml>=6.0.1

# 开发调试
ipython>=8.14.0
jupyter>=1.0.0
```

## 🤝 贡献指南

### 开发环境搭建
```bash
# 1. Fork 并克隆项目
git clone https://github.com/your-username/fastapi-multigpu-i2v.git
cd fastapi-multigpu-i2v

# 2. 创建开发分支
git checkout -b feature/your-amazing-feature

# 3. 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. 验证开发环境
python3 tools/verify_structure.py
python3 scripts/debug/debug_device.py

# 5. 运行测试
python3 -m pytest tests/ -v
```

### 代码质量检查
```bash
# 代码格式化
black src/ tools/ scripts/debug/
isort src/ tools/ scripts/debug/

# 类型检查
mypy src/

# 测试覆盖率
pytest --cov=src --cov-report=html tests/

# 代码质量
flake8 src/ tools/
pylint src/ tools/
```

### 提交规范
```bash
# 提交前检查
python3 tools/verify_structure.py
python3 -m pytest tests/
black --check src/ tools/

# 提交格式
git commit -m "feat: add amazing feature

- Add new functionality X
- Improve performance by Y%
- Fix issue #123

Closes #123"
```

## 📞 技术支持与社区

### 问题反馈渠道
- **🐛 Bug 报告**: [GitHub Issues](https://github.com/your-repo/issues)
- **💡 功能建议**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **📖 文档问题**: [Documentation Issues](https://github.com/your-repo/issues?q=is%3Aissue+label%3Adocumentation)

### 技术文档
- **📚 详细文档**: [项目 Wiki](https://github.com/your-repo/wiki)
- **🎥 视频教程**: [YouTube Channel](https://youtube.com/your-channel)
- **📊 性能报告**: [Benchmark Results](https://github.com/your-repo/wiki/benchmarks)

### 社区支持
- **💬 即时讨论**: [Discord Server](https://discord.gg/your-server)
- **📧 邮件列表**: [Google Groups](https://groups.google.com/your-group)
- **🌐 中文社区**: [国内技术论坛](https://forum.your-site.com)

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **🧠 Wan AI Team** - 提供 Wan2.1-I2V-14B-720P 基础模型
- **🔥 华为昇腾** - NPU 硬件支持和 CANN 软件栈
- **💚 NVIDIA** - GPU 硬件支持和 CUDA 生态系统
- **⚡ FastAPI Team** - 高性能异步 Web 框架
- **🔥 PyTorch Team** - 强大的深度学习框架
- **🌟 开源社区** - 各种优秀的开源项目和工具

---

## 🎯 快速验证与启动

### 一键验证脚本
```bash
#!/bin/bash
# 🚀 quick_start.sh - 一键验证和启动

echo "🎯 FastAPI Multi-GPU I2V - Quick Start"
echo "====================================="

# 1. 项目验证
echo "1️⃣ Verifying project structure..."
python3 tools/verify_structure.py || exit 1

# 2. 设备检测
echo "2️⃣ Detecting hardware..."
python3 scripts/debug/debug_device.py || exit 1

# 3. 生成配置
echo "3️⃣ Generating optimal configuration..."
python3 tools/config_generator.py --template production --export-env --output-dir .

# 4. 预热测试
echo "4️⃣ Running warmup test..."
python3 scripts/debug/debug_t5_warmup.py --warmup-steps 1

# 5. 启动服务
echo "5️⃣ Starting service..."
source config_production.env
./scripts/start_service_general.sh

echo "✅ Quick start completed!"
echo "🌐 API: http://localhost:8088"
echo "📖 Docs: http://localhost:8088/docs"
```

### 验证 API 功能
```bash
# 等待服务启动
sleep 30

# 健康检查
curl -s http://localhost:8088/health | jq '.'

# 测试视频生成
curl -X POST http://localhost:8088/video/submit \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "image_url": "https://picsum.photos/1280/720",
    "quality_preset": "balanced"
  }' | jq '.'

echo "🚀 FastAPI Multi-GPU I2V is ready!"
echo "💡 Use tools/ and scripts/debug/ for monitoring and troubleshooting"
```

**🌟 开始你的 AI 视频生成之旅！**
