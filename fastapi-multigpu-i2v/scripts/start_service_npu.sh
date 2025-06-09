#!/bin/bash

echo "🚀 Starting NPU (Ascend) Multi-GPU Video Generation API..."

# 设置 NPU 专用环境变量
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export HCCL_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_BUFFSIZE=256
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_GLOBAL_LOG_LEVEL=1

# NPU 设备配置
export DEVICE_COUNT=${DEVICE_COUNT:-$(npu-smi info | grep -c "NPU.*:" || echo "8")}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

# 通用分布式配置
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}

# 模型配置 (NPU 优化)
export T5_CPU=${T5_CPU:-"true"}           # NPU 建议 T5 使用 CPU
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

# 业务配置 (NPU 保守配置)
export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-"2"}
export TASK_TIMEOUT=${TASK_TIMEOUT:-"2400"}        # 40分钟，NPU 较慢
export CLEANUP_INTERVAL=${CLEANUP_INTERVAL:-"300"}
export MAX_OUTPUT_DIR_SIZE=${MAX_OUTPUT_DIR_SIZE:-"50"}
export SERVER_PORT=${SERVER_PORT:-"8088"}
export ALLOWED_HOSTS=${ALLOWED_HOSTS:-"*"}

# 模型路径
export MODEL_CKPT_DIR=${MODEL_CKPT_DIR:-"/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"}
export MODEL_TASK=${MODEL_TASK:-"i2v-14B"}
export DEFAULT_SIZE=${DEFAULT_SIZE:-"1280*720"}
export DEFAULT_FRAME_NUM=${DEFAULT_FRAME_NUM:-"81"}
export DEFAULT_SAMPLE_STEPS=${DEFAULT_SAMPLE_STEPS:-"30"}

# CPU 线程优化 (NPU 需要更多 CPU 资源)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"16"}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"16"}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-"16"}

# Python 路径
export PYTHONPATH=/workspace/Wan2.1:$PYTHONPATH

echo "📋 NPU Configuration:"
echo "  - Device Count: $DEVICE_COUNT"
echo "  - NPU Devices: $NPU_VISIBLE_DEVICES"
echo "  - T5 CPU Mode: $T5_CPU"
echo "  - DIT FSDP: $DIT_FSDP"
echo "  - VAE Parallel: $VAE_PARALLEL"
echo "  - Max Concurrent: $MAX_CONCURRENT_TASKS"
echo "  - Timeout: ${TASK_TIMEOUT}s"
echo "  - Server Port: $SERVER_PORT"
echo "  - Model Path: $MODEL_CKPT_DIR"

# 检查 NPU 环境
echo "🔍 Checking NPU environment..."
if command -v npu-smi &> /dev/null; then
    echo "NPU Status:"
    npu-smi info | head -20
else
    echo "⚠️  npu-smi not found, continuing anyway..."
fi

# 检查 Python 环境
echo "🐍 Checking Python environment..."
python3 -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch_npu
    print(f'torch_npu available: {torch_npu.npu.is_available()}')
    print(f'NPU device count: {torch_npu.npu.device_count()}')
except ImportError as e:
    print(f'❌ torch_npu import failed: {e}')
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
sleep 5

# 创建必要目录
mkdir -p generated_videos
mkdir -p logs

# 清理 NPU 缓存
echo "🗑️  Clearing NPU cache..."
python3 -c "
try:
    import torch_npu
    torch_npu.npu.empty_cache()
    print('✅ NPU cache cleared')
except Exception as e:
    print(f'⚠️  Cache clear warning: {e}')
"

# 设置信号处理
trap 'echo "🛑 Stopping NPU service..."; pkill -f "torchrun.*i2v_api"; exit 0' INT TERM

# 启动 NPU 分布式服务
echo "🚀 Starting $DEVICE_COUNT-NPU distributed service..."
echo "📡 Master: $MASTER_ADDR:$MASTER_PORT"
echo "🌐 Server will start on port $SERVER_PORT"
echo ""

torchrun \
    --nproc_per_node=$DEVICE_COUNT \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=1 \
    --node_rank=0 \
    src/i2v_api.py 2>&1 | tee logs/npu_service.log

echo "🏁 NPU service stopped."