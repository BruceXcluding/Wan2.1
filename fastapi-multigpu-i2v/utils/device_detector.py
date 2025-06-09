import os
import sys
import logging
from typing import Tuple, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """设备类型枚举"""
    CUDA = "cuda"
    NPU = "npu"
    CPU = "cpu"

class DeviceDetector:
    """设备检测器"""
    
    def __init__(self):
        self.device_type = None
        self.device_count = 0
        self.is_distributed = False
        
    def detect_device(self) -> Tuple[DeviceType, int]:
        """检测可用设备类型和数量"""
        
        # 1. 优先检测 NPU
        if self._check_npu_available():
            self.device_type = DeviceType.NPU
            self.device_count = self._get_npu_count()
            logger.info(f"Detected NPU devices: {self.device_count}")
            return DeviceType.NPU, self.device_count
            
        # 2. 检测 CUDA
        elif self._check_cuda_available():
            self.device_type = DeviceType.CUDA
            self.device_count = self._get_cuda_count()
            logger.info(f"Detected CUDA devices: {self.device_count}")
            return DeviceType.CUDA, self.device_count
            
        # 3. 回退到 CPU
        else:
            self.device_type = DeviceType.CPU
            self.device_count = 1
            logger.warning("No GPU devices detected, using CPU")
            return DeviceType.CPU, 1
    
    def _check_npu_available(self) -> bool:
        """检查 NPU 是否可用"""
        try:
            import torch_npu
            import torch
            return torch.npu.is_available() and torch.npu.device_count() > 0
        except ImportError:
            logger.debug("torch_npu not available")
            return False
        except Exception as e:
            logger.debug(f"NPU check failed: {str(e)}")
            return False
    
    def _check_cuda_available(self) -> bool:
        """检查 CUDA 是否可用"""
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except Exception as e:
            logger.debug(f"CUDA check failed: {str(e)}")
            return False
    
    def _get_npu_count(self) -> int:
        """获取 NPU 数量"""
        try:
            import torch
            return torch.npu.device_count()
        except:
            return 0
    
    def _get_cuda_count(self) -> int:
        """获取 CUDA 数量"""
        try:
            import torch
            return torch.cuda.device_count()
        except:
            return 0
    
    def get_backend_name(self) -> str:
        """获取分布式后端名称"""
        if self.device_type == DeviceType.NPU:
            return "hccl"
        elif self.device_type == DeviceType.CUDA:
            return "nccl"
        else:
            return "gloo"
    
    def get_device_name(self, device_id: int = 0) -> str:
        """获取设备名称"""
        if self.device_type == DeviceType.NPU:
            return f"npu:{device_id}"
        elif self.device_type == DeviceType.CUDA:
            return f"cuda:{device_id}"
        else:
            return "cpu"
    
    def set_device(self, device_id: int):
        """设置当前设备"""
        if self.device_type == DeviceType.NPU:
            import torch
            torch.npu.set_device(device_id)
        elif self.device_type == DeviceType.CUDA:
            import torch
            torch.cuda.set_device(device_id)
    
    def empty_cache(self):
        """清理设备缓存"""
        try:
            if self.device_type == DeviceType.NPU:
                import torch
                torch.npu.empty_cache()
            elif self.device_type == DeviceType.CUDA:
                import torch
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Failed to empty cache: {str(e)}")
    
    def get_memory_info(self, device_id: int = 0) -> Tuple[float, float]:
        """获取设备内存信息 (allocated_gb, reserved_gb)"""
        try:
            if self.device_type == DeviceType.NPU:
                import torch
                allocated = torch.npu.memory_allocated(device_id) / 1024**3
                reserved = torch.npu.memory_reserved(device_id) / 1024**3
                return allocated, reserved
            elif self.device_type == DeviceType.CUDA:
                import torch
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                reserved = torch.cuda.memory_reserved(device_id) / 1024**3
                return allocated, reserved
            else:
                return 0.0, 0.0
        except Exception as e:
            logger.warning(f"Failed to get memory info: {str(e)}")
            return 0.0, 0.0

# 全局设备检测器实例
device_detector = DeviceDetector()