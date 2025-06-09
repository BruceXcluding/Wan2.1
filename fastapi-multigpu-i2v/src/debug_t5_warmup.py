#!/usr/bin/env python3
"""
T5 CPU 模式调试工具
用于单独测试 T5 预热过程
"""

import os
import sys
import time
import logging
import torch
import torch_npu

# 设置环境变量
os.environ["T5_CPU"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_t5_cpu_warmup():
    """测试 T5 CPU 预热"""
    try:
        logger.info("Starting T5 CPU warmup test...")
        
        # 导入 wan 模块
        import wan
        
        # 获取配置
        cfg = wan.configs.WAN_CONFIGS["i2v-14B"]
        
        # 只测试 T5 部分
        logger.info("Loading T5 model...")
        
        # 模拟单卡加载
        model = wan.WanI2V(
            config=cfg,
            checkpoint_dir="/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P",
            device_id=0,
            rank=0,
            t5_cpu=True,
            dit_fsdp=False,
            use_vae_parallel=False,
        )
        
        logger.info("T5 model loaded, starting warmup...")
        
        # 预热测试
        dummy_prompts = [
            "warm up text",
            "test prompt for warmup"
        ]
        
        start_time = time.time()
        
        with torch.no_grad():
            for i, prompt in enumerate(dummy_prompts):
                logger.info(f"Warmup step {i+1}/{len(dummy_prompts)}: {prompt}")
                step_start = time.time()
                
                try:
                    result = model.text_encoder([prompt], torch.device('cpu'))
                    step_time = time.time() - step_start
                    logger.info(f"Step {i+1} completed in {step_time:.2f}s, result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
                except Exception as e:
                    logger.error(f"Step {i+1} failed: {str(e)}")
                    raise
        
        total_time = time.time() - start_time
        logger.info(f"T5 CPU warmup completed successfully in {total_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"T5 CPU warmup test failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return False

def check_system_resources():
    """检查系统资源"""
    import psutil
    
    logger.info("System Resource Check:")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    logger.info(f"Total memory: {psutil.virtual_memory().total / 1024**3:.2f} GB")
    
    # 检查 NPU
    try:
        logger.info("NPU Status:")
        for i in range(8):
            allocated = torch.npu.memory_allocated(i) / 1024**3
            reserved = torch.npu.memory_reserved(i) / 1024**3
            logger.info(f"NPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    except Exception as e:
        logger.warning(f"NPU check failed: {str(e)}")

if __name__ == "__main__":
    logger.info("T5 CPU Warmup Debug Tool")
    logger.info("=" * 50)
    
    # 检查系统资源
    check_system_resources()
    
    logger.info("=" * 50)
    
    # 测试 T5 预热
    success = test_t5_cpu_warmup()
    
    if success:
        logger.info("✅ T5 CPU warmup test PASSED")
        sys.exit(0)
    else:
        logger.error("❌ T5 CPU warmup test FAILED")
        sys.exit(1)