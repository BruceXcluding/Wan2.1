"""
外层工具包 (utils/)
项目级通用工具函数
"""

from .device_detector import (
    device_detector,    # 全局实例对象 - 主要使用这个
    DeviceDetector,     # 类 - 可以创建新实例
    DeviceType         # 枚举 - 设备类型
)

__all__ = [
    "device_detector",  # 全局实例，可以直接调用 device_detector.detect_device()
    "DeviceDetector",   # 类，可以创建新的检测器实例
    "DeviceType"        # 枚举，可以用于类型检查和比较
]

__version__ = "1.0.0"