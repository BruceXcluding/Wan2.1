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
        """NPU特定的视频生成"""
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
            if "timeout" in str(e).lower():
                logger.error(f"NPU generation timeout on rank {self.rank}: {str(e)}")
                raise Exception(f"NPU 生成超时，请检查网络和设备状态: {str(e)}")
            elif "out of memory" in str(e).lower():
                logger.error(f"NPU out of memory on rank {self.rank}: {str(e)}")
                torch.npu.empty_cache()
                raise Exception(f"NPU 显存不足，请降低并发数或帧数: {str(e)}")
            else:
                raise
    
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