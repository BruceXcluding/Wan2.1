"""
简化的NPU管道实现
"""
import os
import torch
from typing import Optional
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
        super().__init__(ckpt_dir, **model_args)
        
        # 初始化
        self.initialize()
    
    def _setup_environment(self):
        """设置NPU环境变量"""
        os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "0")
        os.environ.setdefault("HCCL_TIMEOUT", "1800")
        os.environ.setdefault("HCCL_BUFFSIZE", "512")
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "600")
        
        # NPU 特定配置
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
    
    def _set_device(self):
        """设置NPU设备"""
        torch.npu.set_device(self.local_rank)
    
    def _move_to_device(self, tensor):
        """移动张量到NPU"""
        return tensor.npu()
    
    def _configure_model(self, model):
        """配置NPU模型"""
        # NPU特定的模型配置可以在这里添加
        pass
    

    def _generate_video_device_specific(self, request, img, size: str, frame_num: int):
        """NPU特定的视频生成逻辑"""
        try:
            logger.info(f"NPU generating video: size={size}, frames={frame_num}")
            
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
            logger.error(f"NPU video generation failed: {str(e)}")
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
    
    def _get_backend(self) -> str:
        """获取NPU分布式后端"""
        return "hccl"