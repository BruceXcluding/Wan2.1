#!/usr/bin/env python3
"""
设备检测调试工具
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.device_detector import device_detector, DeviceType

def main():
    print("🔍 Device Detection Debug Tool")
    print("=" * 50)
    
    # 检测设备
    device_type, device_count = device_detector.detect_device()
    
    print(f"Detected device: {device_type.value}")
    print(f"Device count: {device_count}")
    print(f"Backend: {device_detector.get_backend_name()}")
    
    # 详细信息
    for i in range(device_count):
        device_name = device_detector.get_device_name(i)
        print(f"Device {i}: {device_name}")
        
        allocated, reserved = device_detector.get_memory_info(i)
        print(f"  Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

if __name__ == "__main__":
    main()