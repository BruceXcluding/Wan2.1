#!/bin/bash
"""
NPU 专用启动脚本 - 优化版
针对华为昇腾 NPU 优化的启动配置
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

echo -e "${BLUE}🚀 FastAPI Multi-GPU I2V Service - NPU Launcher${NC}"
echo "=============================================="

# NPU 专用配置
export MODEL_CKPT_DIR="${MODEL_CKPT_DIR:-/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P}"
export T5_CPU="${T5_CPU:-true}"          # NPU 推荐 T5 CPU 模式
export DIT_FSDP="${DIT_FSDP:-true}"       # NPU 推荐 FSDP
export T5_FSDP="${T5_FSDP:-false}"
export VAE_PARALLEL="${VAE_PARALLEL:-true}"
export CFG_SIZE="${CFG_SIZE:-1}"
export ULYSSES_SIZE="${ULYSSES_SIZE:-1}"
export MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-2}"  # NPU 保守并发
export TASK_TIMEOUT="${TASK_TIMEOUT:-2400}"              # NPU 延长超时
export SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
export SERVER_PORT="${SERVER_PORT:-8088}"

# NPU 特定环境变量
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export HCCL_TIMEOUT="${HCCL_TIMEOUT:-1800}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"

# 分布式配置
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# 系统优化
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

# Python 路径
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# 自动检测 NPU 设备
echo -e "${BLUE}🔍 Detecting NPU devices...${NC}"
DEVICE_COUNT=$(python3 -c "
try:
    import torch_npu
    if torch_npu.npu.is_available():
        print(torch_npu.npu.device_count())
    else:
        print('0')
except ImportError:
    print('0')
" 2>/dev/null || echo "0")

if [ "$DEVICE_COUNT" = "0" ]; then
    echo -e "${RED}❌ No NPU devices detected!${NC}"
    echo -e "${YELLOW}💡 Please check:${NC}"
    echo "   - NPU driver installation"
    echo "   - torch_npu installation"
    echo "   - Device availability"
    exit 1
fi

export NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-$(seq -s, 0 $((DEVICE_COUNT-1)))}"

echo -e "${GREEN}✅ Detected $DEVICE_COUNT NPU device(s)${NC}"
echo -e "${BLUE}📋 NPU Configuration:${NC}"
echo "  - Device Count: $DEVICE_COUNT"
echo "  - NPU Devices: $NPU_VISIBLE_DEVICES"
echo "  - T5 CPU Mode: $T5_CPU"
echo "  - DIT FSDP: $DIT_FSDP"
echo "  - VAE Parallel: $VAE_PARALLEL"
echo "  - Max Concurrent: $MAX_CONCURRENT_TASKS"
echo "  - Timeout: ${TASK_TIMEOUT}s"
echo "  - Server: $SERVER_HOST:$SERVER_PORT"
echo "  - Model Path: $MODEL_CKPT_DIR"

# 检查 NPU 状态
if command -v npu-smi &> /dev/null; then
    echo -e "${BLUE}📊 NPU Status:${NC}"
    npu-smi info | head -20
else
    echo -e "${YELLOW}⚠️  npu-smi not found, continuing anyway...${NC}"
fi

# 验证 Python 环境
echo -e "${BLUE}🐍 Checking NPU Python environment...${NC}"
python3 -c "
import sys
print(f'Python: {sys.version}')
sys.path.insert(0, '$PROJECT_ROOT')
try:
    import torch_npu
    print(f'torch_npu available: {torch_npu.npu.is_available()}')
    print(f'NPU device count: {torch_npu.npu.device_count()}')
    
    # 检查项目模块
    from schemas import VideoSubmitRequest
    from pipelines import PipelineFactory, get_available_pipelines
    from utils import device_detector
    print('✅ All project modules imported successfully')
    print(f'Available pipelines: {get_available_pipelines()}')
    
except ImportError as e:
    print(f'❌ torch_npu import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'⚠️  Environment check warning: {e}')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ NPU Python environment check failed!${NC}"
    exit 1
fi

# 清理旧进程和缓存
echo -e "${BLUE}🧹 Cleaning up...${NC}"
pkill -f "i2v_api.py" || true
pkill -f "torchrun.*i2v_api" || true
sleep 5

# 清理 NPU 缓存
echo -e "${BLUE}🗑️  Clearing NPU cache...${NC}"
python3 -c "
try:
    import torch_npu
    torch_npu.npu.empty_cache()
    print('✅ NPU cache cleared')
except Exception as e:
    print(f'⚠️  Cache clear warning: {e}')
"

# 创建必要目录
mkdir -p generated_videos
mkdir -p logs

# 设置信号处理
trap 'echo -e "${YELLOW}🛑 Stopping NPU service...${NC}"; pkill -f "torchrun.*i2v_api"; exit 0' INT TERM

# 启动 NPU 服务
echo ""
if [ "$DEVICE_COUNT" -gt 1 ]; then
    echo -e "${GREEN}🚀 Starting $DEVICE_COUNT-NPU distributed service...${NC}"
    echo -e "${BLUE}📡 Master: $MASTER_ADDR:$MASTER_PORT${NC}"
    echo -e "${BLUE}🌐 Server will start on http://$SERVER_HOST:$SERVER_PORT${NC}"
    echo -e "${BLUE}📖 API docs: http://$SERVER_HOST:$SERVER_PORT/docs${NC}"
    echo ""
    
    LOG_FILE="logs/npu_distributed_$(date +%Y%m%d_%H%M%S).log"
    
    torchrun \
        --nproc_per_node=$DEVICE_COUNT \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --nnodes=1 \
        --node_rank=0 \
        src/i2v_api.py 2>&1 | tee "$LOG_FILE"
else
    echo -e "${GREEN}🚀 Starting single-NPU service...${NC}"
    echo -e "${BLUE}🌐 Server will start on http://$SERVER_HOST:$SERVER_PORT${NC}"
    echo -e "${BLUE}📖 API docs: http://$SERVER_HOST:$SERVER_PORT/docs${NC}"
    echo ""
    
    LOG_FILE="logs/npu_single_$(date +%Y%m%d_%H%M%S).log"
    
    python3 src/i2v_api.py 2>&1 | tee "$LOG_FILE"
fi

echo -e "${YELLOW}NPU service stopped.${NC}"