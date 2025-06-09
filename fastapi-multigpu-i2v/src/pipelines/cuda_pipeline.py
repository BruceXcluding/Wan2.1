"""
简化的CUDA管道实现
"""
import os
import torch
from typing import Optional
from .base_pipeline import BasePipeline

import logging
logger = logging.getLogger(__name__)

class CUDAPipeline(BasePipeline):
    """NVIDIA CUDA GPU 视频生成管道 - 简化版"""
    
    def __init__(self, ckpt_dir: str, **model_args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, cannot use CUDA pipeline")
        
        self.device_type = "cuda"
        super().__init__(ckpt_dir, **model_args)
        
        # 初始化
        self.initialize()
    
    def _setup_environment(self):
        """设置CUDA环境变量"""
        os.environ.setdefault("NCCL_TIMEOUT", "1800")
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    
    def _set_device(self):
        """设置CUDA设备"""
        torch.cuda.set_device(self.local_rank)
    
    def _move_to_device(self, tensor):
        """移动张量到CUDA"""
        return tensor.cuda()
    
    def _configure_model(self, model):
        """配置CUDA模型"""
        # CUDA特定的模型配置可以在这里添加
        pass
    
    def _generate_video_device_specific(self, request, img, size: str, frame_num: int):
        """CUDA特定的视频生成"""
        try:
            from wan.configs import MAX_AREA_CONFIGS
            
            # 调用模型生成
            video_tensor = self.model.generate(
                request.prompt,
                img,
                max_area=MAX_AREA_CONFIGS[size],
                frame_num=frame_num,
                shift=getattr(request, 'sample_shift', 5.0),
                sample_solver=getattr(request, 'sample_solver', 'unipc'),
                sampling_steps=request.infer_steps or 30,
                guide_scale=request.guidance_scale or 3.0,
                n_prompt=getattr(request, 'negative_prompt', ""),
                seed=getattr(request, 'seed', None)
            )
            
            return video_tensor
            
        except Exception as e:
            if "out of memory" in str(e).lower():
                logger.error(f"CUDA out of memory on rank {self.rank}: {str(e)}")
                torch.cuda.empty_cache()
                raise Exception(f"GPU 显存不足，请降低并发数或帧数: {str(e)}")
            else:
                raise
    
    def _log_memory_usage(self):
        """记录CUDA内存使用"""
        try:
            memory_allocated = torch.cuda.memory_allocated(self.local_rank) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.local_rank) / 1024**3
            logger.info(f"Rank {self.rank} CUDA Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        except Exception as e:
            logger.warning(f"Failed to get CUDA memory info: {str(e)}")
    
    def _empty_cache(self):
        """清理CUDA缓存"""
        torch.cuda.empty_cache()
    
    def _get_backend(self) -> str:
        """获取CUDA分布式后端"""
        return "nccl"