#!/usr/bin/env python3
"""
T5 CPU 模式调试工具
用于单独测试 T5 预热过程，诊断 T5 CPU 模式的问题
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch_npu

# 设置环境变量
os.environ["T5_CPU"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONPATH"] = f"/workspace/Wan2.1:{os.environ.get('PYTHONPATH', '')}"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "logs" / "t5_debug.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """设置测试环境"""
    logger.info("Setting up test environment...")
    
    # 创建日志目录
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # NPU 环境设置
    os.environ.setdefault("ALGO", "0")
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HCCL_TIMEOUT", "1800")
    
    # 设置单卡模式
    torch.npu.set_device(0)
    
    logger.info("Environment setup completed")

def check_system_resources():
    """检查系统资源"""
    try:
        import psutil
        
        logger.info("=" * 60)
        logger.info("SYSTEM RESOURCE CHECK")
        logger.info("=" * 60)
        
        # CPU 信息
        logger.info(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        logger.info(f"CPU usage: {psutil.cpu_percent(interval=1):.1f}%")
        
        # 内存信息
        memory = psutil.virtual_memory()
        logger.info(f"Memory: {memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB ({memory.percent:.1f}%)")
        logger.info(f"Available memory: {memory.available / 1024**3:.1f}GB")
        
        # NPU 信息
        logger.info("\nNPU Status:")
        try:
            npu_count = torch.npu.device_count()
            logger.info(f"NPU count: {npu_count}")
            
            for i in range(min(npu_count, 8)):
                allocated = torch.npu.memory_allocated(i) / 1024**3
                reserved = torch.npu.memory_reserved(i) / 1024**3
                logger.info(f"NPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                
        except Exception as e:
            logger.warning(f"NPU check failed: {str(e)}")
        
        logger.info("=" * 60)
        
    except ImportError:
        logger.warning("psutil not available, skipping system resource check")
    except Exception as e:
        logger.error(f"System resource check failed: {str(e)}")

def test_imports():
    """测试必要的导入"""
    logger.info("Testing imports...")
    
    try:
        # 测试设备检测
        from utils.device_detector import device_detector
        device_type, device_count = device_detector.detect_device()
        logger.info(f"Device detection: {device_type.value} x {device_count}")
        
        # 测试 wan 模块
        import wan
        logger.info(f"WAN module imported successfully")
        logger.info(f"Available configs: {list(wan.configs.WAN_CONFIGS.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Import test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

def test_t5_cpu_warmup(model_path: str = None, max_warmup_steps: int = 3):
    """测试 T5 CPU 预热"""
    logger.info("=" * 60)
    logger.info("T5 CPU WARMUP TEST")
    logger.info("=" * 60)
    
    try:
        # 导入 wan 模块
        import wan
        
        # 获取配置
        cfg = wan.configs.WAN_CONFIGS["i2v-14B"]
        logger.info(f"Using config: {cfg}")
        
        # 设置模型路径
        if not model_path:
            model_path = "/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"
        
        logger.info(f"Model path: {model_path}")
        
        # 检查模型路径
        if not Path(model_path).exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False
        
        # 加载模型
        logger.info("Loading WanI2V model...")
        load_start = time.time()
        
        model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=model_path,
            device_id=0,
            rank=0,
            t5_cpu=True,
            dit_fsdp=False,
            t5_fsdp=False,
            use_vae_parallel=False,
            use_usp=False
        )
        
        load_time = time.time() - load_start
        logger.info(f"Model loaded in {load_time:.2f}s")
        
        # 记录加载后的内存使用
        try:
            allocated = torch.npu.memory_allocated(0) / 1024**3
            reserved = torch.npu.memory_reserved(0) / 1024**3
            logger.info(f"NPU memory after model load: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
        except:
            pass
        
        # T5 预热测试
        logger.info(f"Starting T5 warmup with {max_warmup_steps} steps...")
        
        dummy_prompts = [
            "warm up text for testing",
            "a beautiful landscape with mountains",
            "testing T5 encoder performance",
            "video generation prompt",
            "final warmup step test"
        ][:max_warmup_steps]
        
        warmup_start = time.time()
        successful_steps = 0
        
        with torch.no_grad():
            for i, prompt in enumerate(dummy_prompts):
                logger.info(f"Warmup step {i+1}/{len(dummy_prompts)}: '{prompt}'")
                step_start = time.time()
                
                try:
                    # 调用 T5 编码器
                    result = model.text_encoder([prompt], torch.device('cpu'))
                    step_time = time.time() - step_start
                    
                    # 记录结果信息
                    if hasattr(result, 'shape'):
                        logger.info(f"Step {i+1} SUCCESS in {step_time:.2f}s, result shape: {result.shape}")
                    elif hasattr(result, '__len__'):
                        logger.info(f"Step {i+1} SUCCESS in {step_time:.2f}s, result length: {len(result)}")
                    else:
                        logger.info(f"Step {i+1} SUCCESS in {step_time:.2f}s, result type: {type(result)}")
                    
                    successful_steps += 1
                    
                    # 记录内存使用
                    try:
                        allocated = torch.npu.memory_allocated(0) / 1024**3
                        reserved = torch.npu.memory_reserved(0) / 1024**3
                        logger.info(f"  NPU memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                    except:
                        pass
                    
                except Exception as e:
                    logger.error(f"Step {i+1} FAILED: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    break
        
        total_warmup_time = time.time() - warmup_start
        
        # 输出总结
        logger.info("=" * 60)
        logger.info("T5 WARMUP SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total warmup time: {total_warmup_time:.2f}s")
        logger.info(f"Successful steps: {successful_steps}/{len(dummy_prompts)}")
        logger.info(f"Average time per step: {total_warmup_time/len(dummy_prompts):.2f}s")
        
        if successful_steps == len(dummy_prompts):
            logger.info("✅ T5 CPU warmup test PASSED")
            return True
        else:
            logger.error("❌ T5 CPU warmup test FAILED")
            return False
        
    except Exception as e:
        logger.error(f"T5 CPU warmup test failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="T5 CPU Warmup Debug Tool")
    parser.add_argument("--model-path", type=str, 
                       default="/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P",
                       help="Path to model checkpoint directory")
    parser.add_argument("--warmup-steps", type=int, default=3,
                       help="Number of warmup steps to test")
    parser.add_argument("--skip-resource-check", action="store_true",
                       help="Skip system resource check")
    parser.add_argument("--skip-import-test", action="store_true",
                       help="Skip import test")
    
    args = parser.parse_args()
    
    logger.info("🚀 T5 CPU Warmup Debug Tool")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Warmup steps: {args.warmup_steps}")
    logger.info("=" * 60)
    
    try:
        # 设置环境
        setup_environment()
        
        # 检查系统资源
        if not args.skip_resource_check:
            check_system_resources()
        
        # 测试导入
        if not args.skip_import_test:
            if not test_imports():
                logger.error("Import test failed, aborting")
                return False
        
        # 测试 T5 预热
        success = test_t5_cpu_warmup(args.model_path, args.warmup_steps)
        
        if success:
            logger.info("🎉 All tests PASSED!")
            return True
        else:
            logger.error("💥 Tests FAILED!")
            return False
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)