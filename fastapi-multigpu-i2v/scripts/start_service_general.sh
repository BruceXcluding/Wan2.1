#!/bin/bash
"""
通用智能启动脚本 - 优化版
自动检测硬件环境并启动最优配置
"""

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}🚀 FastAPI Multi-GPU I2V Service - General Launcher${NC}"
echo "=================================================="

# 默认配置
export MODEL_CKPT_DIR="${MODEL_CKPT_DIR:-/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P}"
export T5_CPU="${T5_CPU:-true}"
export DIT_FSDP="${DIT_FSDP:-true}"
export T5_FSDP="${T5_FSDP:-false}"
export VAE_PARALLEL="${VAE_PARALLEL:-true}"
export CFG_SIZE="${CFG_SIZE:-1}"
export ULYSSES_SIZE="${ULYSSES_SIZE:-1}"
export USE_ATTENTION_CACHE="${USE_ATTENTION_CACHE:-false}"
export CACHE_START_STEP="${CACHE_START_STEP:-12}"
export CACHE_INTERVAL="${CACHE_INTERVAL:-4}"
export CACHE_END_STEP="${CACHE_END_STEP:-37}"
export MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-2}"
export TASK_TIMEOUT="${TASK_TIMEOUT:-1800}"
export SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
export SERVER_PORT="${SERVER_PORT:-8088}"

# 分布式配置
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# 系统优化
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

# Python 路径 - 确保项目路径在 Python path 中
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo -e "${BLUE}📋 General Configuration:${NC}"
echo "  - Project Root: $PROJECT_ROOT"
echo "  - Model Path: $MODEL_CKPT_DIR"
echo "  - T5 CPU Mode: $T5_CPU"
echo "  - DIT FSDP: $DIT_FSDP"
echo "  - VAE Parallel: $VAE_PARALLEL"
echo "  - Max Concurrent: $MAX_CONCURRENT_TASKS"
echo "  - Timeout: ${TASK_TIMEOUT}s"
echo "  - Server: $SERVER_HOST:$SERVER_PORT"

# 检查模型路径
if [ ! -d "$MODEL_CKPT_DIR" ]; then
    echo -e "${YELLOW}⚠️  Model directory not found: $MODEL_CKPT_DIR${NC}"
    echo -e "${YELLOW}   Continuing anyway (model will be downloaded if needed)${NC}"
fi

# 自动设备检测
echo -e "${BLUE}🔍 Auto-detecting hardware environment...${NC}"
DETECTED_DEVICE=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from utils.device_detector import device_detector
    device_type, device_count = device_detector.detect_device()
    print(f'{device_type.value}:{device_count}')
except Exception as e:
    print(f'cpu:1')  # 默认回退
" 2>/dev/null)

# 解析检测结果
IFS=':' read -r DEVICE_TYPE DEVICE_COUNT <<< "$DETECTED_DEVICE"

echo -e "${GREEN}✅ Detected: $DEVICE_TYPE with $DEVICE_COUNT device(s)${NC}"

# 设置设备相关环境变量
if [ "$DEVICE_TYPE" = "npu" ]; then
    export NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-$(seq -s, 0 $((DEVICE_COUNT-1)))}"
    export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
    export HCCL_TIMEOUT="${HCCL_TIMEOUT:-1800}"
    export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
    export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
    
    echo -e "${BLUE}📱 NPU Configuration:${NC}"
    echo "  - NPU Devices: $NPU_VISIBLE_DEVICES"
    echo "  - HCCL Timeout: $HCCL_TIMEOUT"
elif [ "$DEVICE_TYPE" = "cuda" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((DEVICE_COUNT-1)))}"
    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
    
    echo -e "${BLUE}🎮 CUDA Configuration:${NC}"
    echo "  - CUDA Devices: $CUDA_VISIBLE_DEVICES"
    echo "  - NCCL Timeout: $NCCL_TIMEOUT"
fi

# 验证 Python 环境
echo -e "${BLUE}🐍 Checking Python environment...${NC}"
python3 -c "
import sys
print(f'Python: {sys.version}')
sys.path.insert(0, '$PROJECT_ROOT')

# 检查基础导入
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    
    if '$DEVICE_TYPE' == 'npu':
        import torch_npu
        print(f'torch_npu available: {torch_npu.npu.is_available()}')
        print(f'NPU device count: {torch_npu.npu.device_count()}')
    elif '$DEVICE_TYPE' == 'cuda':
        print(f'CUDA available: {torch.cuda.is_available()}')
        print(f'CUDA device count: {torch.cuda.device_count()}')
    
    # 检查项目模块
    from schemas import VideoSubmitRequest
    from pipelines import PipelineFactory, get_available_pipelines
    from utils import device_detector
    print('✅ All project modules imported successfully')
    print(f'Available pipelines: {get_available_pipelines()}')
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'⚠️  Environment check warning: {e}')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Python environment check failed!${NC}"
    echo -e "${YELLOW}💡 Please check your PyTorch installation and project dependencies${NC}"
    exit 1
fi

# 清理设备缓存
echo -e "${BLUE}🗑️  Clearing device cache...${NC}"
python3 -c "
try:
    if '$DEVICE_TYPE' == 'npu':
        import torch_npu
        torch_npu.npu.empty_cache()
        print('✅ NPU cache cleared')
    elif '$DEVICE_TYPE' == 'cuda':
        import torch
        torch.cuda.empty_cache()
        print('✅ CUDA cache cleared')
    else:
        print('✅ No device cache to clear')
except Exception as e:
    print(f'⚠️  Cache clear warning: {e}')
"

# 清理旧进程
echo -e "${BLUE}🧹 Cleaning up old processes...${NC}"
pkill -f "i2v_api.py" || true
pkill -f "torchrun.*i2v_api" || true
sleep 3

# 创建必要目录
mkdir -p generated_videos
mkdir -p logs

# 设置信号处理
trap 'echo -e "${YELLOW}🛑 Stopping service...${NC}"; pkill -f "torchrun.*i2v_api"; pkill -f "python.*i2v_api"; exit 0' INT TERM

# 启动服务
echo ""
if [ "$DEVICE_COUNT" -gt 1 ]; then
    echo -e "${GREEN}🚀 Starting $DEVICE_COUNT-device distributed service on $DEVICE_TYPE...${NC}"
    echo -e "${BLUE}📡 Master: $MASTER_ADDR:$MASTER_PORT${NC}"
    echo -e "${BLUE}🌐 Server will start on http://$SERVER_HOST:$SERVER_PORT${NC}"
    echo -e "${BLUE}📖 API docs: http://$SERVER_HOST:$SERVER_PORT/docs${NC}"
    echo ""
    
    # 启动分布式服务
    LOG_FILE="logs/${DEVICE_TYPE}_distributed_$(date +%Y%m%d_%H%M%S).log"
    
    torchrun \
        --nproc_per_node=$DEVICE_COUNT \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --nnodes=1 \
        --node_rank=0 \
        src/i2v_api.py 2>&1 | tee "$LOG_FILE"
else
    echo -e "${GREEN}🚀 Starting single-device service on $DEVICE_TYPE...${NC}"
    echo -e "${BLUE}🌐 Server will start on http://$SERVER_HOST:$SERVER_PORT${NC}"
    echo -e "${BLUE}📖 API docs: http://$SERVER_HOST:$SERVER_PORT/docs${NC}"
    echo ""
    
    # 启动单设备服务
    LOG_FILE="logs/${DEVICE_TYPE}_single_$(date +%Y%m%d_%H%M%S).log"
    
    python3 src/i2v_api.py 2>&1 | tee "$LOG_FILE"
fi

echo -e "${YELLOW}Service stopped.${NC}"