# 修改 NPUPipeline 类，添加环境检查方法

"""
简化的NPU管道实现
"""
import os
import torch
from typing import Optional
from datetime import timedelta
from .base_pipeline import BasePipeline

# NPU 特定导入
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)

class NPUPipeline(BasePipeline):
    """华为昇腾 NPU 视频生成管道 - 简化版"""
    
    def __init__(self, ckpt_dir: str, **model_args):
        if not NPU_AVAILABLE:
            raise RuntimeError("torch_npu not available, cannot use NPU pipeline")
        
        self.device_type = "npu"
        
        # 🔧 添加：环境检查和修复
        self._check_and_fix_npu_environment()
        
        super().__init__(ckpt_dir, **model_args)
        
        # 初始化
        self.initialize()
    
    def _check_and_fix_npu_environment(self):
        """检查和修复 NPU 环境"""
        logger.info("🔍 Checking NPU environment...")
        
        try:
            # 1. 检查 NPU 可用性
            if not torch_npu.npu.is_available():
                raise RuntimeError("NPU is not available")
            
            npu_count = torch_npu.npu.device_count()
            logger.info(f"✅ NPU available: {npu_count} devices")
            
            # 2. 检查当前进程的 NPU 设备设置
            current_device = torch_npu.npu.current_device()
            logger.info(f"Current NPU device: {current_device}")
            
            # 3. 检查和设置 HCCL 环境变量
            self._check_hccl_environment()
            
            # 4. 检查分布式环境
            self._check_distributed_environment()
            
            # 5. 设置 NPU 设备
            if hasattr(self, 'local_rank'):
                torch_npu.npu.set_device(self.local_rank)
                logger.info(f"✅ Set NPU device to: {self.local_rank}")
            
            # 6. 验证 NPU 功能
            self._test_npu_functionality()
            
        except Exception as e:
            logger.error(f"❌ NPU environment check failed: {e}")
            raise RuntimeError(f"NPU environment check failed: {e}")
    
    def _check_hccl_environment(self):
        """检查 HCCL 环境变量"""
        logger.info("🔧 Checking HCCL environment...")
        
        # 必需的 HCCL 环境变量
        required_vars = {
            'HCCL_TIMEOUT': '3600',
            'HCCL_BUFFSIZE': '1024', 
            'HCCL_CONNECT_TIMEOUT': '1200',
            'HCCL_EXEC_TIMEOUT': '0',
            'HCCL_HEARTBEAT_TIMEOUT': '0',
            'ASCEND_GLOBAL_LOG_LEVEL': '3',
            'ASCEND_SLOG_PRINT_TO_STDOUT': '1',
            'ASCEND_GLOBAL_EVENT_ENABLE': '0'
        }
        
        # 检查和设置环境变量
        for var, default_value in required_vars.items():
            if var not in os.environ:
                os.environ[var] = default_value
                logger.info(f"  🔧 Set {var} = {default_value}")
            else:
                logger.info(f"  ✅ {var} = {os.environ[var]}")
        
        # 检查 NPU_VISIBLE_DEVICES
        if 'NPU_VISIBLE_DEVICES' in os.environ:
            logger.info(f"  ✅ NPU_VISIBLE_DEVICES = {os.environ['NPU_VISIBLE_DEVICES']}")
        else:
            logger.warning("  ⚠️  NPU_VISIBLE_DEVICES not set")
        
        # 检查 ASCEND_DEVICE_ID
        if 'ASCEND_DEVICE_ID' in os.environ:
            logger.info(f"  ✅ ASCEND_DEVICE_ID = {os.environ['ASCEND_DEVICE_ID']}")
        else:
            # 设置默认值
            os.environ['ASCEND_DEVICE_ID'] = '0'
            logger.info(f"  🔧 Set ASCEND_DEVICE_ID = 0")
    
    def _check_distributed_environment(self):
        """检查分布式环境"""
        logger.info("🌐 Checking distributed environment...")
        
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        logger.info(f"  - WORLD_SIZE: {world_size}")
        logger.info(f"  - RANK: {rank}")
        logger.info(f"  - LOCAL_RANK: {local_rank}")
        
        if world_size > 1:
            # 检查分布式相关环境变量
            required_dist_vars = ['MASTER_ADDR', 'MASTER_PORT']
            for var in required_dist_vars:
                if var not in os.environ:
                    logger.warning(f"  ⚠️  {var} not set")
                else:
                    logger.info(f"  ✅ {var} = {os.environ[var]}")
            
            # 检查 NPU 设备数量与进程数量匹配
            npu_count = torch_npu.npu.device_count()
            if local_rank >= npu_count:
                logger.warning(f"  ⚠️  LOCAL_RANK({local_rank}) >= NPU_COUNT({npu_count})")
        else:
            logger.info("  ✅ Single device mode")
    
    def _test_npu_functionality(self):
        """测试 NPU 基本功能"""
        logger.info("🧪 Testing NPU functionality...")
        
        try:
            # 创建测试张量
            test_tensor = torch.randn(2, 2)
            npu_tensor = test_tensor.npu()
            
            # 测试基本运算
            result = npu_tensor + npu_tensor
            cpu_result = result.cpu()
            
            logger.info("  ✅ NPU tensor operations working")
            
            # 测试内存管理
            memory_allocated = torch_npu.npu.memory_allocated() / 1024**2  # MB
            logger.info(f"  ✅ NPU memory allocated: {memory_allocated:.2f}MB")
            
            # 清理测试张量
            del test_tensor, npu_tensor, result, cpu_result
            torch_npu.npu.empty_cache()
            
        except Exception as e:
            logger.error(f"  ❌ NPU functionality test failed: {e}")
            raise RuntimeError(f"NPU functionality test failed: {e}")
    
    def _setup_environment(self):
        """设置NPU环境变量"""
        # 这个方法现在主要在 _check_and_fix_npu_environment 中处理
        # 保留一些运行时设置
        os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "0")
        
        # NPU 特定配置
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        
        logger.info("✅ NPU environment setup completed")
    
    def _set_device(self):
        """设置NPU设备"""
        try:
            torch_npu.npu.set_device(self.local_rank)  # ✅ 正确的API
            logger.info(f"✅ Set NPU device to: {self.local_rank}")
        except Exception as e:
            logger.error(f"❌ Failed to set NPU device: {e}")
            raise
        
    def _move_to_device(self, tensor):
        """移动张量到NPU"""
        return tensor.npu()
    
    def _configure_model(self, model):
        """配置NPU模型"""
        logger.info("🔧 Configuring model for NPU...")
        
        # NPU特定的模型配置
        try:
            # 如果模型有特定的 NPU 优化方法，可以在这里调用
            if hasattr(model, 'npu'):
                model = model.npu()
                logger.info("  ✅ Model moved to NPU")
            
            # 设置模型为评估模式
            if hasattr(model, 'eval'):
                model.eval()
                logger.info("  ✅ Model set to eval mode")
                
        except Exception as e:
            logger.warning(f"  ⚠️  Model NPU configuration warning: {e}")
    
    def _generate_video_device_specific(self, request, img, size: str, frame_num: int):
        """NPU特定的视频生成逻辑"""
        try:
            logger.info(f"🎬 NPU generating video: size={size}, frames={frame_num}")
            
            # 🔧 参数映射 - 严格按照 generate.py:428-437
            generation_params = {
                'max_area': self._calculate_max_area(size),      # 根据尺寸计算
                'frame_num': frame_num,                          # 帧数
                'shift': 3.0,                                    # i2v 默认 shift
                'sample_solver': 'unipc',                        # 默认采样器
                'sampling_steps': request.infer_steps or 40,     # 推理步数
                'guide_scale': request.guidance_scale or 5.0,    # 引导系数
                'seed': request.seed or self._generate_seed(),   # 随机种子
                'offload_model': False                           # NPU 通常不需要卸载
            }
            
            logger.info(f"NPU generation params: {generation_params}")
            
            # 调用模型生成 - 与 generate.py:428-437 完全一致
            video_tensor = self.model.generate(
                request.prompt,  # 第一个参数：提示词
                img,            # 第二个参数：PIL.Image 对象
                **generation_params  # 其余参数
            )
            
            if self.rank == 0:
                return video_tensor
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ NPU video generation failed: {str(e)}")
            raise
    
    def _calculate_max_area(self, size: str) -> int:
        """根据尺寸字符串计算 max_area"""
        try:
            if '*' in size:
                width, height = map(int, size.split('*'))
                return width * height
            else:
                # 默认值
                return 921600  # 1280*720
        except:
            return 921600
            
    def _generate_seed(self) -> int:
        """生成随机种子"""
        import random
        return random.randint(0, 2**32 - 1)
    
    def _log_memory_usage(self):
        """记录NPU内存使用"""
        try:
            memory_allocated = torch.npu.memory_allocated(self.local_rank) / 1024**3
            memory_reserved = torch.npu.memory_reserved(self.local_rank) / 1024**3
            logger.info(f"Rank {self.rank} NPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        except Exception as e:
            logger.warning(f"Failed to get NPU memory info: {str(e)}")
    
    def _empty_cache(self):
        """清理NPU缓存"""
        torch.npu.empty_cache()

    def _init_distributed_common(self, backend: str, timeout_seconds: int = 3600):
        """NPU 分布式初始化"""
        try:
            import torch.distributed as dist
            from datetime import timedelta
            
            logger.info(f"Initializing NPU distributed with backend: {backend}")
            
            # 设置当前 NPU 设备
            torch_npu.npu.set_device(self.local_rank)
            
            # 初始化分布式进程组
            dist.init_process_group(
                backend=backend,  # "hccl"
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank,
                timeout=timedelta(seconds=timeout_seconds)
            )
            
            logger.info(f"✅ NPU distributed initialized: rank={self.rank}/{self.world_size}")
            
        except Exception as e:
            logger.error(f"❌ NPU distributed initialization failed: {e}")
            raise
    
    def _get_backend(self) -> str:
        """获取NPU分布式后端"""
        return "hccl"
    
    def _get_timeout_seconds(self) -> int:
        """获取超时时间"""
        return 3600  # NPU 需要更长的超时时间