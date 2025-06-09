#!/bin/bash

echo "🚀 Starting CUDA (NVIDIA) Multi-GPU Video Generation API..."

# 设置 CUDA 专用环境变量
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-"3600"}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-"0"}
export NCCL_DEBUG=${NCCL_DEBUG:-"INFO"}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-"1"}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-"0"}

# GPU 设备配置
export DEVICE_COUNT=${DEVICE_COUNT:-$(nvidia-smi -L | wc -l || echo "8")}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

# 通用分布式配置
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}

# 模型配置 (CUDA 优化)
export T5_CPU=${T5_CPU:-"false"}          # CUDA 可以 T5 用 GPU
export DIT_FSDP=${DIT_FSDP:-"true"}
export T5_FSDP=${T5_FSDP:-"false"}
export VAE_PARALLEL=${VAE_PARALLEL:-"true"}
export CFG_SIZE=${CFG_SIZE:-"1"}
export ULYSSES_SIZE=${ULYSSES_SIZE:-"8"}

# 性能优化配置
export USE_ATTENTION_CACHE=${USE_ATTENTION_CACHE:-"true"}
export CACHE_START_STEP=${CACHE_START_STEP:-"12"}
export CACHE_INTERVAL=${CACHE_INTERVAL:-"4"}
export CACHE_END_STEP=${CACHE_END_STEP:-"37"}

# 业务配置 (CUDA 更高性能)
export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-"4"}
export TASK_TIMEOUT=${TASK_TIMEOUT:-"1800"}        # 30分钟，CUDA 较快
export CLEANUP_INTERVAL=${CLEANUP_INTERVAL:-"300"}
export MAX_OUTPUT_DIR_SIZE=${MAX_OUTPUT_DIR_SIZE:-"50"}
export SERVER_PORT=${SERVER_PORT:-"8088"}
export ALLOWED_HOSTS=${ALLOWED_HOSTS:-"*"}

# 模型路径
export MODEL_CKPT_DIR=${MODEL_CKPT_DIR:-"/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"}
export MODEL_TASK=${MODEL_TASK:-"i2v-14B"}
export DEFAULT_SIZE=${DEFAULT_SIZE:-"1280*720"}
export DEFAULT_FRAME_NUM=${DEFAULT_FRAME_NUM:-"81"}
export DEFAULT_SAMPLE_STEPS=${DEFAULT_SAMPLE_STEPS:-"40"}

# CPU 线程优化 (CUDA 可以用更少 CPU)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"8"}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"8"}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-"8"}

# Python 路径
export PYTHONPATH=/workspace/Wan2.1:$PYTHONPATH

echo "📋 CUDA Configuration:"
echo "  - Device Count: $DEVICE_COUNT"
echo "  - CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "  - T5 CPU Mode: $T5_CPU"
echo "  - DIT FSDP: $DIT_FSDP"
echo "  - VAE Parallel: $VAE_PARALLEL"
echo "  - Max Concurrent: $MAX_CONCURRENT_TASKS"
echo "  - Timeout: ${TASK_TIMEOUT}s"
echo "  - Server Port: $SERVER_PORT"
echo "  - Model Path: $MODEL_CKPT_DIR"

# 检查 CUDA 环境
echo "🔍 Checking CUDA environment..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
else
    echo "⚠️  nvidia-smi not found, continuing anyway..."
fi

# 检查 Python 环境
echo "🐍 Checking Python environment..."
python3 -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f'  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)')
except ImportError as e:
    print(f'❌ torch import failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Python environment check failed!"
    exit 1
fi

# 清理旧进程
echo "🧹 Cleaning up old processes..."
pkill -f "i2v_api.py" || true
pkill -f "torchrun.*i2v_api" || true
sleep 3

# 创建必要目录
mkdir -p generated_videos
mkdir -p logs

# 清理 CUDA 缓存
echo "🗑️  Clearing CUDA cache..."
python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('✅ CUDA cache cleared')
    else:
        print('⚠️  CUDA not available')
except Exception as e:
    print(f'⚠️  Cache clear warning: {e}')
"

# 设置信号处理
trap 'echo "🛑 Stopping CUDA service..."; pkill -f "torchrun.*i2v_api"; exit 0' INT TERM

# 启动 CUDA 分布式服务
echo "🚀 Starting $DEVICE_COUNT-GPU distributed service..."
echo "📡 Master: $MASTER_ADDR:$MASTER_PORT"
echo "🌐 Server will start on port $SERVER_PORT"
echo ""

# 使用更高性能的 NCCL 启动参数
torchrun \
    --nproc_per_node=$DEVICE_COUNT \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=1 \
    --node_rank=0 \
    src/i2v_api.py 2>&1 | tee logs/cuda_service.log

echo "🏁 CUDA service stopped."