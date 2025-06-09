"""
推理管道包
支持多硬件后端的视频生成管道
"""
import logging

logger = logging.getLogger(__name__)

# 基础组件 - 始终可用
from .base_pipeline import BasePipeline
from .pipeline_factory import PipelineFactory

# 导出的基础组件
__all__ = ["BasePipeline", "PipelineFactory"]

# 设备特定的管道 - 条件导入
_device_pipelines = {}

# 尝试导入 NPU 管道
try:
    from .npu_pipeline import NPUPipeline
    _device_pipelines["npu"] = NPUPipeline
    __all__.append("NPUPipeline")
    logger.info("NPU pipeline available")
except ImportError as e:
    logger.info(f"NPU pipeline not available: {e}")
    NPUPipeline = None

# 尝试导入 CUDA 管道
try:
    from .cuda_pipeline import CUDAPipeline
    _device_pipelines["cuda"] = CUDAPipeline
    __all__.append("CUDAPipeline")
    logger.info("CUDA pipeline available")
except ImportError as e:
    logger.info(f"CUDA pipeline not available: {e}")
    CUDAPipeline = None

# 导出设备信息
def get_available_pipelines():
    """获取可用的管道类型"""
    return list(_device_pipelines.keys())

def get_pipeline_class(device_type: str):
    """根据设备类型获取管道类"""
    return _device_pipelines.get(device_type.lower())

# 添加到导出列表
__all__.extend(["get_available_pipelines", "get_pipeline_class"])

# 版本信息
__version__ = "1.0.0"
__author__ = "BruceXcluding"
__description__ = "Multi-hardware backend pipelines for video generation"

logger.info(f"Pipelines package initialized. Available: {get_available_pipelines()}")