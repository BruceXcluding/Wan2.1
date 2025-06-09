"""
简化的管道工厂
"""
import os
import logging
from typing import Dict, Any
from utils.device_detector import device_detector, DeviceType
from pipelines.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)

class PipelineFactory:
    """管道工厂类 - 简化版"""
    
    @staticmethod
    def get_available_devices() -> Dict[str, Any]:
        """获取可用设备信息"""
        try:
            device_type, device_count = device_detector.detect_device()
            
            device_info = {
                "device_type": device_type.value,
                "device_count": device_count,
                "backend": "torch_npu" if device_type == DeviceType.NPU else "torch"
            }
            
            logger.info(f"Detected devices: {device_info}")
            return device_info
            
        except Exception as e:
            logger.error(f"Failed to detect devices: {str(e)}")
            return {
                "device_type": "cuda",  # 默认回退到CUDA
                "device_count": 1,
                "backend": "torch"
            }
    
    @staticmethod
    def create_pipeline(**config) -> BasePipeline:
        """创建推理管道"""
        device_info = PipelineFactory.get_available_devices()
        device_type = device_info["device_type"]
        
        logger.info(f"Creating pipeline for device type: {device_type}")
        
        # 确保必需的参数
        if 'ckpt_dir' not in config:
            config['ckpt_dir'] = os.environ.get('MODEL_CKPT_DIR', '/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P')
        
        try:
            if device_type == "npu":
                from pipelines.npu_pipeline import NPUPipeline
                return NPUPipeline(**config)
            else:  # 默认使用CUDA或CPU
                from pipelines.cuda_pipeline import CUDAPipeline
                return CUDAPipeline(**config)
                
        except Exception as e:
            logger.error(f"Failed to create {device_type} pipeline: {e}")
            # 如果NPU失败，尝试CUDA作为后备
            if device_type == "npu":
                logger.warning("NPU pipeline failed, trying CUDA as fallback")
                try:
                    from pipelines.cuda_pipeline import CUDAPipeline
                    return CUDAPipeline(**config)
                except Exception as cuda_e:
                    logger.error(f"CUDA fallback also failed: {cuda_e}")
            raise
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """简化的配置验证"""
        # 检查模型路径
        ckpt_dir = config.get('ckpt_dir')
        if ckpt_dir and not os.path.exists(ckpt_dir):
            logger.warning(f"Model checkpoint directory not found: {ckpt_dir}")
            # 不要直接失败，让模型加载时处理
        
        return True