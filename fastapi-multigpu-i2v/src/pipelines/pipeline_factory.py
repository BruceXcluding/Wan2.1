import logging
from typing import Type
from .base_pipeline import BasePipeline
from .npu_pipeline import NPUPipeline
from .cuda_pipeline import CUDAPipeline
from ..utils.device_detector import device_detector, DeviceType

logger = logging.getLogger(__name__)

class PipelineFactory:
    """管道工厂类"""
    
    @staticmethod
    def create_pipeline(ckpt_dir: str, **model_args) -> BasePipeline:
        """根据可用设备创建合适的管道"""
        
        # 检测设备
        device_type, device_count = device_detector.detect_device()
        
        logger.info(f"Detected device: {device_type.value}, count: {device_count}")
        
        # 根据设备类型创建对应的管道
        if device_type == DeviceType.NPU:
            logger.info("Creating NPU pipeline")
            return NPUPipeline(ckpt_dir, **model_args)
        
        elif device_type == DeviceType.CUDA:
            logger.info("Creating CUDA pipeline")
            return CUDAPipeline(ckpt_dir, **model_args)
        
        else:
            raise RuntimeError(f"Unsupported device type: {device_type}")
    
    @staticmethod
    def get_available_devices():
        """获取可用设备信息"""
        device_type, device_count = device_detector.detect_device()
        return {
            "device_type": device_type.value,
            "device_count": device_count,
            "backend": device_detector.get_backend_name()
        }