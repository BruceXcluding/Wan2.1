#!/bin/bash

echo "🚀 Starting Universal Multi-GPU Video Generation API..."

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "📂 Project root: $PROJECT_ROOT"
echo "📂 Script directory: $SCRIPT_DIR"

# 设置 Python 路径（在设备检测之前）
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH}"
echo "🐍 PYTHONPATH: $PYTHONPATH"

# 自动检测设备类型
echo "🔍 Detecting hardware devices..."
python3 -c "
import sys
import os

# 确保路径正确
project_root = '$PROJECT_ROOT'
src_path = os.path.join(project_root, 'src')

# 检查路径是否存在
if not os.path.exists(src_path):
    print(f'Error: src path does not exist: {src_path}', file=sys.stderr)
    print('DETECTED_DEVICE=cuda')
    print('DEVICE_COUNT=1')
    exit()

# 添加到 Python 路径
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f'DEBUG: Python paths added: {src_path}', file=sys.stderr)
print(f'DEBUG: Current sys.path: {sys.path[:3]}...', file=sys.stderr)

try:
    # 尝试导入设备检测器
    from utils.device_detector import device_detector
    device_type, device_count = device_detector.detect_device()
    print(f'DETECTED_DEVICE={device_type.value}')
    print(f'DEVICE_COUNT={device_count}')
    print(f'DEBUG: Successfully detected {device_type.value} with {device_count} devices', file=sys.stderr)
except ImportError as e:
    print(f'Import error: {e}', file=sys.stderr)
    # 尝试简单的设备检测
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f'DETECTED_DEVICE=cuda')
            print(f'DEVICE_COUNT={device_count}')
            print(f'DEBUG: Fallback CUDA detection: {device_count} devices', file=sys.stderr)
        else:
            print('DETECTED_DEVICE=cuda')
            print('DEVICE_COUNT=1')
            print('DEBUG: No CUDA available, using default', file=sys.stderr)
    except:
        print('DETECTED_DEVICE=cuda')
        print('DEVICE_COUNT=1')
        print('DEBUG: Fallback to default values', file=sys.stderr)
except Exception as e:
    print(f'Detection error: {e}', file=sys.stderr)
    print('DETECTED_DEVICE=cuda')
    print('DEVICE_COUNT=1')
" > device_info.tmp 2> device_debug.log

# 检查设备检测是否成功
if [ ! -f device_info.tmp ] || [ ! -s device_info.tmp ]; then
    echo "⚠️  Device detection failed, using defaults..."
    echo "DETECTED_DEVICE=cuda" > device_info.tmp
    echo "DEVICE_COUNT=1" >> device_info.tmp
fi

# 显示调试信息
if [ -f device_debug.log ]; then
    echo "🔍 Device detection debug info:"
    cat device_debug.log
    rm device_debug.log
fi

source device_info.tmp
rm device_info.tmp

echo "✅ Detected device: $DETECTED_DEVICE with $DEVICE_COUNT devices"

# 设置通用环境变量
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}

# 根据设备类型设置特定环境变量
if [ "$DETECTED_DEVICE" = "npu" ]; then
    echo "⚙️  Configuring for NPU (Ascend)..."
    export ALGO=0
    export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
    export TASK_QUEUE_ENABLE=2
    export CPU_AFFINITY_CONF=1
    export HCCL_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_BUFFSIZE=256
    export ASCEND_LAUNCH_BLOCKING=0
    export ASCEND_GLOBAL_LOG_LEVEL=1
    
    # 动态设置可见设备
    if [ $DEVICE_COUNT -gt 0 ]; then
        DEVICE_LIST=$(seq -s, 0 $((DEVICE_COUNT-1)))
        export NPU_VISIBLE_DEVICES=$DEVICE_LIST
        export ASCEND_RT_VISIBLE_DEVICES=$DEVICE_LIST
        echo "🎯 NPU devices: $DEVICE_LIST"
    fi
    
elif [ "$DETECTED_DEVICE" = "cuda" ]; then
    echo "⚙️  Configuring for CUDA (NVIDIA)..."
    export NCCL_TIMEOUT=3600
    export CUDA_LAUNCH_BLOCKING=0
    export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=0
    
    # 动态设置可见设备
    if [ $DEVICE_COUNT -gt 0 ]; then
        DEVICE_LIST=$(seq -s, 0 $((DEVICE_COUNT-1)))
        export CUDA_VISIBLE_DEVICES=$DEVICE_LIST
        echo "🎯 CUDA devices: $DEVICE_LIST"
    fi
    
else
    echo "❌ Unsupported device: $DETECTED_DEVICE"
    echo "💡 Supported devices: npu, cuda"
    exit 1
fi

# 模型和服务配置
export MODEL_CKPT_DIR=${MODEL_CKPT_DIR:-"/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"}
export MODEL_TASK=${MODEL_TASK:-"i2v-14B"}
export DEFAULT_SIZE=${DEFAULT_SIZE:-"1280*720"}
export DEFAULT_FRAME_NUM=${DEFAULT_FRAME_NUM:-"81"}
export DEFAULT_SAMPLE_STEPS=${DEFAULT_SAMPLE_STEPS:-"30"}

# 可配置的模型选项
export T5_CPU=${T5_CPU:-"false"}
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

# 业务配置（根据设备和 T5 模式自动调整）
if [ "$T5_CPU" = "true" ] || [ "$DETECTED_DEVICE" = "npu" ]; then
    export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-"2"}
    export TASK_TIMEOUT=${TASK_TIMEOUT:-"2400"}  # 40分钟
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"16"}  # T5 CPU 需要更多线程
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"16"}
    export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-"16"}
else
    export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-"4"}
    export TASK_TIMEOUT=${TASK_TIMEOUT:-"1800"}  # 30分钟
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"8"}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"8"}
    export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-"8"}
fi

# 服务配置
export SERVER_HOST=${SERVER_HOST:-"0.0.0.0"}
export SERVER_PORT=${SERVER_PORT:-"8088"}
export CLEANUP_INTERVAL=${CLEANUP_INTERVAL:-"300"}
export MAX_OUTPUT_DIR_SIZE=${MAX_OUTPUT_DIR_SIZE:-"50"}
export ALLOWED_HOSTS=${ALLOWED_HOSTS:-"*"}

echo "📋 Configuration Summary:"
echo "  - Device: $DETECTED_DEVICE ($DEVICE_COUNT cards)"
echo "  - Model Path: $MODEL_CKPT_DIR"
echo "  - T5 CPU Mode: $T5_CPU"
echo "  - DIT FSDP: $DIT_FSDP"
echo "  - VAE Parallel: $VAE_PARALLEL"
echo "  - Ulysses Size: $ULYSSES_SIZE"
echo "  - Attention Cache: $USE_ATTENTION_CACHE"
echo "  - Max Concurrent: $MAX_CONCURRENT_TASKS"
echo "  - Task Timeout: ${TASK_TIMEOUT}s"
echo "  - Server: $SERVER_HOST:$SERVER_PORT"
echo "  - CPU Threads: $OMP_NUM_THREADS"

# 验证项目结构
echo "🔍 Verifying project structure..."
if [ ! -f "$PROJECT_ROOT/src/utils/device_detector.py" ]; then
    echo "⚠️  device_detector.py not found at: $PROJECT_ROOT/src/utils/device_detector.py"
    echo "💡 Creating minimal device detector..."
    
    mkdir -p "$PROJECT_ROOT/src/utils"
    cat > "$PROJECT_ROOT/src/utils/__init__.py" << 'EOF'
# Utils package
EOF

    cat > "$PROJECT_ROOT/src/utils/device_detector.py" << 'EOF'
"""
设备自动检测工具
"""
from enum import Enum
from typing import Tuple

class DeviceType(Enum):
    NPU = "npu"
    CUDA = "cuda"

class DeviceDetector:
    """设备检测器"""
    
    def detect_device(self) -> Tuple[DeviceType, int]:
        """检测可用设备"""
        # 优先检测 NPU
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                device_count = torch_npu.npu.device_count()
                return DeviceType.NPU, device_count
        except ImportError:
            pass
        
        # 检测 CUDA
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                return DeviceType.CUDA, device_count
        except ImportError:
            pass
        
        # 默认返回单卡 CUDA
        return DeviceType.CUDA, 1

# 全局实例
device_detector = DeviceDetector()
EOF
    echo "✅ Created minimal device detector"
fi

# 验证模型路径
if [ ! -d "$MODEL_CKPT_DIR" ]; then
    echo "⚠️  Model directory not found: $MODEL_CKPT_DIR"
    echo "💡 Please set MODEL_CKPT_DIR environment variable"
    echo "💡 Example: export MODEL_CKPT_DIR=/path/to/your/model"
fi

# 清理旧进程
echo "🧹 Cleaning up old processes..."
pkill -f "i2v_api.py" 2>/dev/null || true
pkill -f "torchrun.*i2v_api" 2>/dev/null || true
sleep 5

# 创建必要目录
echo "📁 Creating directories..."
mkdir -p "$PROJECT_ROOT/generated_videos"
mkdir -p "$PROJECT_ROOT/logs"

# 设置工作目录
cd "$PROJECT_ROOT"
echo "📍 Working directory: $(pwd)"

# 检查设备状态
echo "🔍 Checking device status..."
if [ "$DETECTED_DEVICE" = "npu" ]; then
    if command -v npu-smi &> /dev/null; then
        echo "NPU Status:"
        npu-smi info | head -10
    else
        echo "⚠️  npu-smi not available"
    fi
elif [ "$DETECTED_DEVICE" = "cuda" ]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
    else
        echo "⚠️  nvidia-smi not available"
    fi
fi

# 验证 Python 环境
echo "🐍 Checking Python environment..."
python3 -c "
import sys
import os
print(f'Python: {sys.version}')
print(f'Python executable: {sys.executable}')
print(f'Current working directory: {os.getcwd()}')
print(f'PYTHONPATH: {os.environ.get(\"PYTHONPATH\", \"Not set\")}')

# 检查 src 目录
src_path = '$PROJECT_ROOT/src'
print(f'Checking src path: {src_path}')
print(f'Src path exists: {os.path.exists(src_path)}')

if os.path.exists(src_path):
    utils_path = os.path.join(src_path, 'utils')
    print(f'Utils path exists: {os.path.exists(utils_path)}')
    if os.path.exists(utils_path):
        detector_path = os.path.join(utils_path, 'device_detector.py')
        print(f'Device detector exists: {os.path.exists(detector_path)}')

try:
    if '$DETECTED_DEVICE' == 'npu':
        import torch_npu
        print(f'torch_npu available: {torch_npu.npu.is_available()}')
        if torch_npu.npu.is_available():
            print(f'NPU device count: {torch_npu.npu.device_count()}')
    elif '$DETECTED_DEVICE' == 'cuda':
        import torch
        print(f'PyTorch: {torch.__version__}')
        print(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'CUDA version: {torch.version.cuda}')
            print(f'GPU device count: {torch.cuda.device_count()}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'⚠️  Environment check warning: {e}')
"

if [ $? -ne 0 ]; then
    echo "❌ Python environment check failed!"
    echo "💡 Please check your PyTorch installation"
    exit 1
fi

# 清理设备缓存
echo "🗑️  Clearing device cache..."
python3 -c "
try:
    if '$DETECTED_DEVICE' == 'npu':
        import torch_npu
        torch_npu.npu.empty_cache()
        print('✅ NPU cache cleared')
    elif '$DETECTED_DEVICE' == 'cuda':
        import torch
        torch.cuda.empty_cache()
        print('✅ CUDA cache cleared')
except Exception as e:
    print(f'⚠️  Cache clear warning: {e}')
"

# 设置信号处理
trap 'echo "🛑 Stopping service..."; pkill -f "torchrun.*i2v_api"; exit 0' INT TERM

# 启动分布式服务
echo ""
echo "🚀 Starting $DEVICE_COUNT-card distributed service on $DETECTED_DEVICE..."
echo "📡 Master: $MASTER_ADDR:$MASTER_PORT"
echo "🌐 Server will start on http://$SERVER_HOST:$SERVER_PORT"
echo "📖 API docs: http://$SERVER_HOST:$SERVER_PORT/docs"
echo ""

# 启动服务并记录日志
LOG_FILE="$PROJECT_ROOT/logs/${DETECTED_DEVICE}_service_$(date +%Y%m%d_%H%M%S).log"

torchrun \
    --nproc_per_node=$DEVICE_COUNT \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=1 \
    --node_rank=0 \
    src/i2v_api.py 2>&1 | tee "$LOG_FILE"

echo "🏁 Service stopped. Logs saved to: $LOG_FILE"