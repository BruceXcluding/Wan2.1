#!/bin/bash

echo "Starting Universal Multi-GPU Video Generation API..."

# 自动检测设备类型
python3 -c "
import sys
import os
sys.path.append('src')
from utils.device_detector import device_detector
device_type, device_count = device_detector.detect_device()
print(f'DETECTED_DEVICE={device_type.value}')
print(f'DEVICE_COUNT={device_count}')
" > device_info.tmp

source device_info.tmp
rm device_info.tmp

echo "Detected device: $DETECTED_DEVICE with $DEVICE_COUNT devices"

# 设置通用环境变量
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 根据设备类型设置特定环境变量
if [ "$DETECTED_DEVICE" = "npu" ]; then
    echo "Configuring for NPU (Ascend)..."
    export ALGO=0
    export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
    export TASK_QUEUE_ENABLE=2
    export CPU_AFFINITY_CONF=1
    export HCCL_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_BUFFSIZE=256
    export ASCEND_LAUNCH_BLOCKING=0
    export ASCEND_GLOBAL_LOG_LEVEL=1
    export NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [ "$DETECTED_DEVICE" = "cuda" ]; then
    echo "Configuring for CUDA (NVIDIA)..."
    export NCCL_TIMEOUT=3600
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    echo "Unsupported device: $DETECTED_DEVICE"
    exit 1
fi

# 可配置选项
export T5_CPU=${T5_CPU:-"false"}
export DIT_FSDP=${DIT_FSDP:-"true"}
export T5_FSDP=${T5_FSDP:-"false"}
export VAE_PARALLEL=${VAE_PARALLEL:-"true"}
export ULYSSES_SIZE=${ULYSSES_SIZE:-"8"}

# 业务配置（根据设备自动调整）
if [ "$T5_CPU" = "true" ] || [ "$DETECTED_DEVICE" = "npu" ]; then
    export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-"2"}
    export TASK_TIMEOUT=${TASK_TIMEOUT:-"2400"}
else
    export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-"5"}
    export TASK_TIMEOUT=${TASK_TIMEOUT:-"1800"}
fi

export SERVER_PORT=${SERVER_PORT:-"8088"}
export MODEL_CKPT_DIR=${MODEL_CKPT_DIR:-"/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"}
export PYTHONPATH=/workspace/Wan2.1:$PYTHONPATH

# CPU 优化
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

echo "Configuration:"
echo "  - Device: $DETECTED_DEVICE ($DEVICE_COUNT cards)"
echo "  - T5 CPU: $T5_CPU"
echo "  - DIT FSDP: $DIT_FSDP"
echo "  - VAE Parallel: $VAE_PARALLEL"
echo "  - Max Concurrent: $MAX_CONCURRENT_TASKS"
echo "  - Server Port: $SERVER_PORT"

# 清理旧进程
echo "Cleaning up old processes..."
pkill -f "i2v_api.py" || true
pkill -f "torchrun.*i2v_api" || true
sleep 10

# 创建输出目录
mkdir -p generated_videos

# 检查设备状态
if [ "$DETECTED_DEVICE" = "npu" ]; then
    echo "Checking NPU status..."
    npu-smi info || echo "NPU-SMI not available"
elif [ "$DETECTED_DEVICE" = "cuda" ]; then
    echo "Checking CUDA status..."
    nvidia-smi || echo "nvidia-smi not available"
fi

# 清理缓存
echo "Clearing device cache..."
python3 -c "
import torch
try:
    if '$DETECTED_DEVICE' == 'npu':
        import torch_npu
        torch_npu.npu.empty_cache()
        print('NPU cache cleared')
    elif '$DETECTED_DEVICE' == 'cuda':
        torch.cuda.empty_cache()
        print('CUDA cache cleared')
except Exception as e:
    print(f'Cache clear warning: {e}')
"

# 启动服务
echo "Starting $DEVICE_COUNT-card distributed service on $DETECTED_DEVICE..."
torchrun \
    --nproc_per_node=$DEVICE_COUNT \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=1 \
    --node_rank=0 \
    src/i2v_api.py

echo "Service stopped."