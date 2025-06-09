#!/bin/bash

echo "Starting Wan2.1 I2V API with warmup..."

# 设置核心环境变量
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

# 分布式通信配置
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL 配置 - 针对 T5 CPU 模式优化
export HCCL_TIMEOUT=3600           # 延长超时到1小时
export HCCL_CONNECT_TIMEOUT=1200   # 20分钟连接超时
export HCCL_BUFFSIZE=256          # 减小缓冲区
export ASCEND_LAUNCH_BLOCKING=0    # 异步模式
export ASCEND_GLOBAL_LOG_LEVEL=1

# NPU 配置
export NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# T5 CPU 模式配置
export T5_CPU=${T5_CPU:-"true"}
export DIT_FSDP=${DIT_FSDP:-"true"}
export T5_FSDP=${T5_FSDP:-"false"}
export VAE_PARALLEL=${VAE_PARALLEL:-"false"}  # T5 CPU 模式建议关闭
export ULYSSES_SIZE=${ULYSSES_SIZE:-"4"}       # 减少序列并行度

# 业务配置
export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-"1"}  # T5 CPU 模式建议单任务
export TASK_TIMEOUT=${TASK_TIMEOUT:-"3600"}  # 1小时超时
export SERVER_PORT=${SERVER_PORT:-"8088"}
export MAX_OUTPUT_DIR_SIZE=${MAX_OUTPUT_DIR_SIZE:-"30"}

# 模型路径
export MODEL_CKPT_DIR=${MODEL_CKPT_DIR:-"/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"}

# Python路径
export PYTHONPATH=/workspace/Wan2.1:$PYTHONPATH

# CPU 优化 - T5 CPU 模式需要
export OMP_NUM_THREADS=16          # 设置CPU线程数
export MKL_NUM_THREADS=16          # Intel MKL 线程数
export OPENBLAS_NUM_THREADS=16     # OpenBLAS 线程数

echo "T5 CPU Mode Configuration:"
echo "  - T5 CPU: $T5_CPU"
echo "  - Max Concurrent: $MAX_CONCURRENT_TASKS"
echo "  - VAE Parallel: $VAE_PARALLEL"
echo "  - Ulysses Size: $ULYSSES_SIZE"
echo "  - CPU Threads: $OMP_NUM_THREADS"
echo "  - HCCL Timeout: $HCCL_TIMEOUT"

# 创建输出目录
mkdir -p generated_videos

# 彻底清理旧进程
echo "Cleaning up old processes..."
pkill -f "i2v_api.py" || true
pkill -f "torchrun.*i2v_api" || true
pkill -f "python.*i2v_api" || true
sleep 10

# 检查 NPU 状态
echo "Checking NPU status..."
npu-smi info

# 清理NPU缓存
echo "Clearing NPU cache..."
python3 -c "
import torch
try:
    import torch_npu
    torch_npu.npu.empty_cache()
    print('NPU cache cleared')
except Exception as e:
    print(f'Cache clear warning: {e}')
"

# 检查可用内存
echo "System memory status:"
free -h

# 启动服务
echo "Starting service with T5 CPU mode optimizations..."
echo "This may take 5-10 minutes for T5 warmup..."

torchrun \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=1 \
    --node_rank=0 \
    fastapi-multigpu-i2v/src/i2v_api.py

echo "Service stopped."