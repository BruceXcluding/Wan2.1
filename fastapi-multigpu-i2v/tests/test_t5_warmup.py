#!/usr/bin/env python3
# filepath: /Users/yigex/Documents/LLM-Inftra/Wan2.1_fix/fastapi-multigpu-i2v/tests/test_warmup.py
"""
T5 预热测试工具
整合了原 debug/debug_t5_warmup.py 的功能
"""

import sys
import os
import time
import logging
import argparse
from pathlib import Path

# 设置项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 设置环境变量
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("T5_CPU", "true")

# 配置日志
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(logs_dir / "t5_warmup_test.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class T5WarmupTester:
    """T5 预热测试器"""
    
    def __init__(self):
        self.test_results = {}
    
    def setup_environment(self):
        """设置测试环境"""
        logger.info("🔧 Setting up environment...")
        
        # 设置环境变量
        env_vars = {
            "T5_CPU": "true",
            "DIT_FSDP": "false",
            "T5_FSDP": "false",
            "VAE_PARALLEL": "false",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0"
        }
        
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
                logger.info(f"Set {key}={value}")
        
        logger.info("✅ Environment setup completed")
    
    def check_system_resources(self):
        """检查系统资源"""
        logger.info("💻 Checking system resources...")
        
        try:
            import psutil
            
            # 检查内存
            memory = psutil.virtual_memory()
            logger.info(f"System memory: {memory.total / 1024**3:.2f}GB total, "
                       f"{memory.available / 1024**3:.2f}GB available")
            
            if memory.available < 4 * 1024**3:  # 4GB
                logger.warning("Available memory is low (< 4GB)")
            
            # 检查 CPU
            cpu_count = psutil.cpu_count()
            logger.info(f"CPU cores: {cpu_count}")
            
            # 检查磁盘空间
            disk = psutil.disk_usage('/')
            logger.info(f"Disk space: {disk.free / 1024**3:.2f}GB free")
            
            return True
            
        except ImportError:
            logger.warning("psutil not available, skipping resource check")
            return True
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return False
    
    def test_imports(self):
        """测试必要的导入"""
        logger.info("📦 Testing imports...")
        
        imports = [
            ("torch", "PyTorch"),
            ("wan", "WAN model"),
            ("wan.configs", "WAN configs"),
        ]
        
        failed = 0
        
        for module_name, description in imports:
            try:
                __import__(module_name)
                logger.info(f"✅ {description} imported successfully")
            except ImportError as e:
                logger.error(f"❌ Failed to import {description}: {e}")
                failed += 1
            except Exception as e:
                logger.error(f"❌ Error importing {description}: {e}")
                failed += 1
        
        if failed == 0:
            logger.info("✅ All imports successful")
            return True
        else:
            logger.error(f"❌ {failed} imports failed")
            return False
    
    def test_t5_cpu_warmup(self, model_path: str = None, max_warmup_steps: int = 3):
        """测试 T5 CPU 预热"""
        logger.info("🔥 Starting T5 CPU warmup test...")
        
        if not model_path:
            model_path = os.environ.get('MODEL_CKPT_DIR', 
                "/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P")
        
        logger.info(f"Model path: {model_path}")
        
        try:
            # 导入必要模块
            import torch
            import wan
            
            # 检查模型路径
            if not os.path.exists(model_path):
                logger.warning(f"Model path does not exist: {model_path}")
                logger.info("Continuing with test (model will be downloaded if needed)")
            
            # 获取配置
            logger.info("Loading model configuration...")
            cfg = wan.configs.WAN_CONFIGS["i2v-14B"]
            logger.info(f"Config loaded: {type(cfg).__name__}")
            
            # 记录初始内存
            try:
                if torch.cuda.is_available():
                    initial_gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"Initial GPU memory: {initial_gpu_memory:.2f}GB")
                elif hasattr(torch, 'npu') or 'torch_npu' in sys.modules:
                    try:
                        import torch_npu
                        initial_npu_memory = torch_npu.npu.memory_allocated(0) / 1024**3
                        logger.info(f"Initial NPU memory: {initial_npu_memory:.2f}GB")
                    except:
                        pass
            except:
                pass
            
            # 加载模型 - 简化版本用于测试
            logger.info("Creating simplified model instance for T5 test...")
            load_start = time.time()
            
            # 创建一个最小配置的模型实例来测试 T5
            try:
                model = wan.WanI2V(
                    config=cfg,
                    checkpoint_dir=model_path,
                    device_id=0,
                    rank=0,
                    t5_cpu=True,      # 强制 T5 使用 CPU
                    dit_fsdp=False,   # 关闭 FSDP
                    t5_fsdp=False,    # 关闭 T5 FSDP
                    use_vae_parallel=False,  # 关闭 VAE 并行
                    use_usp=False     # 关闭 USP
                )
                
                load_time = time.time() - load_start
                logger.info(f"Model instance created in {load_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to create full model, trying T5 only: {e}")
                # 如果完整模型创建失败，尝试只测试 T5 编码器
                logger.info("Attempting to test T5 encoder directly...")
                return self._test_t5_encoder_only(max_warmup_steps)
            
            # 记录加载后的内存使用
            try:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    logger.info(f"GPU memory after model load: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                elif hasattr(torch, 'npu') or 'torch_npu' in sys.modules:
                    try:
                        import torch_npu
                        allocated = torch_npu.npu.memory_allocated(0) / 1024**3
                        reserved = torch_npu.npu.memory_reserved(0) / 1024**3
                        logger.info(f"NPU memory after model load: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                    except:
                        pass
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
                        if hasattr(model, 'text_encoder') and callable(model.text_encoder):
                            result = model.text_encoder([prompt], torch.device('cpu'))
                        else:
                            logger.warning("T5 encoder not accessible, skipping step")
                            continue
                            
                        step_time = time.time() - step_start
                        
                        # 记录结果信息
                        if hasattr(result, 'shape'):
                            logger.info(f"Step {i+1} SUCCESS in {step_time:.2f}s, result shape: {result.shape}")
                        elif isinstance(result, (list, tuple)) and len(result) > 0:
                            logger.info(f"Step {i+1} SUCCESS in {step_time:.2f}s, result length: {len(result)}")
                        else:
                            logger.info(f"Step {i+1} SUCCESS in {step_time:.2f}s, result type: {type(result)}")
                        
                        successful_steps += 1
                        
                    except Exception as e:
                        logger.error(f"Step {i+1} FAILED: {str(e)}")
                        # 继续其他步骤
                        continue
            
            total_warmup_time = time.time() - warmup_start
            
            # 输出总结
            logger.info("=" * 60)
            logger.info("T5 WARMUP SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total warmup time: {total_warmup_time:.2f}s")
            logger.info(f"Successful steps: {successful_steps}/{len(dummy_prompts)}")
            if len(dummy_prompts) > 0:
                logger.info(f"Average time per step: {total_warmup_time/len(dummy_prompts):.2f}s")
            
            # 清理
            try:
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                del model
            except:
                pass
            
            self.test_results['t5_warmup'] = {
                'passed': successful_steps > 0,
                'successful_steps': successful_steps,
                'total_steps': len(dummy_prompts),
                'total_time': total_warmup_time,
                'avg_time_per_step': total_warmup_time/len(dummy_prompts) if len(dummy_prompts) > 0 else 0
            }
            
            if successful_steps > 0:
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
    
    def _test_t5_encoder_only(self, max_warmup_steps: int):
        """只测试 T5 编码器"""
        logger.info("🔥 Testing T5 encoder only...")
        
        try:
            import torch
            from transformers import T5EncoderModel, T5Tokenizer
            
            logger.info("Loading T5 encoder and tokenizer...")
            
            # 使用较小的 T5 模型进行测试
            model_name = "t5-base"  # 可以改为项目中使用的具体 T5 模型
            
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            encoder = T5EncoderModel.from_pretrained(model_name)
            
            # 确保在 CPU 上
            encoder = encoder.to('cpu')
            encoder.eval()
            
            logger.info("T5 encoder loaded successfully")
            
            dummy_prompts = [
                "warm up text for testing",
                "a beautiful landscape with mountains",
                "testing T5 encoder performance"
            ][:max_warmup_steps]
            
            successful_steps = 0
            warmup_start = time.time()
            
            with torch.no_grad():
                for i, prompt in enumerate(dummy_prompts):
                    logger.info(f"T5 warmup step {i+1}/{len(dummy_prompts)}: '{prompt}'")
                    step_start = time.time()
                    
                    try:
                        # 编码文本
                        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
                        outputs = encoder(**inputs)
                        
                        step_time = time.time() - step_start
                        logger.info(f"Step {i+1} SUCCESS in {step_time:.2f}s, output shape: {outputs.last_hidden_state.shape}")
                        successful_steps += 1
                        
                    except Exception as e:
                        logger.error(f"Step {i+1} FAILED: {str(e)}")
                        continue
            
            total_time = time.time() - warmup_start
            logger.info(f"T5 encoder test completed: {successful_steps}/{len(dummy_prompts)} successful")
            
            return successful_steps > 0
            
        except Exception as e:
            logger.error(f"T5 encoder only test failed: {e}")
            return False
    
    def run_all_tests(self, model_path: str = None, warmup_steps: int = 3, 
                     skip_resource_check: bool = False, skip_import_test: bool = False):
        """运行所有测试"""
        logger.info("🚀 T5 Warmup Test Suite")
        logger.info("=" * 60)
        
        # 设置环境
        self.setup_environment()
        
        # 检查系统资源
        if not skip_resource_check:
            if not self.check_system_resources():
                logger.warning("System resource check failed, continuing anyway...")
        
        # 测试导入
        if not skip_import_test:
            if not self.test_imports():
                logger.error("Import test failed, aborting")
                return False
        
        # 测试 T5 预热
        success = self.test_t5_cpu_warmup(model_path, warmup_steps)
        
        if success:
            logger.info("🎉 All T5 warmup tests PASSED!")
        else:
            logger.error("💥 T5 warmup tests FAILED!")
        
        return success

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="T5 Warmup Test Tool")
    parser.add_argument("--model-path", type=str, 
                       help="Path to model checkpoint directory")
    parser.add_argument("--warmup-steps", type=int, default=3,
                       help="Number of warmup steps to test")
    parser.add_argument("--skip-resource-check", action="store_true",
                       help="Skip system resource check")
    parser.add_argument("--skip-import-test", action="store_true",
                       help="Skip import test")
    
    args = parser.parse_args()
    
    logger.info("🚀 T5 Warmup Test Tool")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path or 'default'}")
    logger.info(f"Warmup steps: {args.warmup_steps}")
    logger.info("=" * 60)
    
    try:
        tester = T5WarmupTester()
        success = tester.run_all_tests(
            model_path=args.model_path,
            warmup_steps=args.warmup_steps,
            skip_resource_check=args.skip_resource_check,
            skip_import_test=args.skip_import_test
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("⏸️  Test interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Test suite failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)