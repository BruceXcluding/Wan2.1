"""
外层工具包 (utils/)
项目级通用工具函数
"""

from .device_detector import device_detector, DeviceType

__all__ = ["device_detector", "DeviceType"]
__version__ = "1.0.0"