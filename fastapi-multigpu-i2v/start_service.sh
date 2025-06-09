#!/bin/bash

echo "Starting Wan2.1 I2V API with configurable options..."

# 设置核心环境变量
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

# 分布式通信配置
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL 基础配置
export HCCL_TIMEOUT=1800           # 30分钟超时
export HCCL_CONNECT_TIMEOUT=600    # 10分钟连接超时
export HCCL_BUFFSIZE=512          # 缓冲区大小
export ASCEND_LAUNCH_BLOCKING=0    # 异步模式
export ASCEND_GLOBAL_LOG_LEVEL=1

# NPU 配置
export NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ==================== 可配置选项 ====================

# T5 CPU 模式 (true/false) - 关键配置
export T5_CPU=${T5_CPU:-"false"}

# 分布式配置
export DIT_FSDP=${DIT_FSDP:-"true"}
export T5_FSDP=${T5_FSDP:-"false"}
export VAE_PARALLEL=${VAE_PARALLEL:-"true"}
export ULYSSES_SIZE=${ULYSSES_SIZE:-"8"}

# 业务配置
if [ "$T5_CPU" = "true" ]; then
    # T5 CPU 模式的优化配置
    export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-"2"}
    export TASK_TIMEOUT=${TASK_TIMEOUT:-"2400"}  # 40分钟
    export HCCL_TIMEOUT=2400  # 延长 HCCL 超时
    echo "T5 CPU mode enabled - optimized for 32GB NPU memory"
else
    # 标准模式配置
    export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-"5"}
    export TASK_TIMEOUT=${TASK_TIMEOUT:-"1800"}  # 30分钟
    echo "Standard mode - T5 on NPU"
fi

# 服务配置
export SERVER_PORT=${SERVER_PORT:-"8088"}
export MAX_OUTPUT_DIR_SIZE=${MAX_OUTPUT_DIR_SIZE:-"50"}

# 模型路径
export MODEL_CKPT_DIR=${MODEL_CKPT_DIR:-"/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"}

# Python路径
export PYTHONPATH=/workspace/Wan2.1:$PYTHONPATH

# ==================== 显示配置 ====================

echo "Configuration:"
echo "  - T5 CPU mode: $T5_CPU"
echo "  - DIT FSDP: $DIT_FSDP"
echo "  - VAE Parallel: $VAE_PARALLEL"
echo "  - Ulysses Size: $ULYSSES_SIZE"
echo "  - Max Concurrent: $MAX_CONCURRENT_TASKS"
echo "  - Task Timeout: $TASK_TIMEOUT seconds"
echo "  - Server Port: $SERVER_PORT"
echo "  - Model Path: $MODEL_CKPT_DIR"

# ==================== 启动服务 ====================

# 创建输出目录
mkdir -p generated_videos

# 清理旧进程
echo "Cleaning up old processes..."
pkill -f "i2v_api.py" || true
pkill -f "torchrun.*i2v_api" || true
sleep 5

# 检查 NPU 状态
echo "Checking NPU status..."
npu-smi info

# 清理NPU缓存
echo "Clearing NPU cache..."
python3 -c "import torch_npu; torch_npu.npu.empty_cache(); print('NPU cache cleared')" || true

# 启动服务
echo "Starting 8-card distributed service..."
torchrun \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/i2v_api.py

echo "Service stopped."