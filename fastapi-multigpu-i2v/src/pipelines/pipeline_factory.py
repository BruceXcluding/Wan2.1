"""
管道工厂
负责创建和管理不同类型的推理管道
"""
import os
import logging
from typing import Dict, Any, Tuple

# 修复导入 - 使用绝对导入
from utils.device_detector import device_detector, DeviceType
from pipelines.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)

class PipelineFactory:
    """管道工厂类"""
    
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
                "device_type": "unknown",
                "device_count": 0,
                "backend": "unknown"
            }
    
    @staticmethod
    def create_pipeline(**config) -> BasePipeline:
        """创建推理管道"""
        device_info = PipelineFactory.get_available_devices()
        device_type = device_info["device_type"]
        
        logger.info(f"Creating pipeline for device type: {device_type}")
        
        if device_type == "npu":
            from pipelines.npu_pipeline import NPUPipeline
            return NPUPipeline(**config)
        elif device_type == "cuda":
            from pipelines.cuda_pipeline import CUDAPipeline
            return CUDAPipeline(**config)
        else:
            # 默认使用 CUDA 管道作为后备
            logger.warning(f"Unknown device type {device_type}, falling back to CUDA pipeline")
            from pipelines.cuda_pipeline import CUDAPipeline
            return CUDAPipeline(**config)
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """验证配置"""
        required_keys = [
            "ckpt_dir", "task", "size", "frame_num", "sample_steps"
        ]
        
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required config key: {key}")
                return False
        
        # 验证模型路径
        if not os.path.exists(config["ckpt_dir"]):
            logger.error(f"Model checkpoint directory not found: {config['ckpt_dir']}")
            return False
        
        # 验证参数范围
        if config["frame_num"] < 24 or config["frame_num"] > 121:
            logger.error(f"Invalid frame_num: {config['frame_num']} (must be 24-121)")
            return False
        
        if config["sample_steps"] < 20 or config["sample_steps"] > 100:
            logger.error(f"Invalid sample_steps: {config['sample_steps']} (must be 20-100)")
            return False
        
        return True