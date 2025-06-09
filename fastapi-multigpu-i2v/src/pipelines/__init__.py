"""
管道包
包含视频生成管道的基类和具体实现
"""

from .pipeline_factory import PipelineFactory
from .base_pipeline import BasePipeline

# 条件导入 - 避免在不支持的环境中导入失败
__all__ = ["PipelineFactory", "BasePipeline"]

# 尝试导入 NPU 管道
try:
    from .npu_pipeline import NPUPipeline
    __all__.append("NPUPipeline")
except ImportError:
    # NPU 环境不可用
    pass

# 尝试导入 CUDA 管道
try:
    from .cuda_pipeline import CUDAPipeline
    __all__.append("CUDAPipeline")
except ImportError:
    # CUDA 环境不可用
    pass

# 提供便捷的创建函数
def create_pipeline(**kwargs):
    """便捷的管道创建函数"""
    return PipelineFactory.create_pipeline(**kwargs)

def get_available_devices():
    """获取可用设备信息"""
    return PipelineFactory.get_available_devices()