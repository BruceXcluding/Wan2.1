#!/usr/bin/env python3
"""
设备检测调试工具 - 完整版
"""

import sys
import os
from pathlib import Path

# 添加项目路径 - 修复路径问题
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))  # 外层utils
sys.path.insert(0, str(project_root / "src"))  # src模块

try:
    from utils.device_detector import device_detector, DeviceType
    from pipelines import get_available_pipelines
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please check project structure and dependencies")
    sys.exit(1)

def test_basic_detection():
    """测试基础设备检测"""
    print("🔍 Basic Device Detection")
    print("-" * 40)
    
    try:
        # 检测设备
        device_type, device_count = device_detector.detect_device()
        
        print(f"✅ Detected device: {device_type.value}")
        print(f"✅ Device count: {device_count}")
        print(f"✅ Backend: {device_detector.get_backend_name()}")
        
        return True
    except Exception as e:
        print(f"❌ Basic detection failed: {e}")
        return False

def test_device_details():
    """测试设备详细信息"""
    print("\n📊 Device Details")
    print("-" * 40)
    
    try:
        device_type, device_count = device_detector.detect_device()
        
        for i in range(device_count):
            print(f"\nDevice {i}:")
            
            # 设备名称
            try:
                device_name = device_detector.get_device_name(i)
                print(f"  Name: {device_name}")
            except Exception as e:
                print(f"  Name: Failed to get ({e})")
            
            # 内存信息
            try:
                allocated, reserved = device_detector.get_memory_info(i)
                print(f"  Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            except Exception as e:
                print(f"  Memory: Failed to get ({e})")
            
            # 设备属性
            try:
                if device_type == DeviceType.CUDA:
                    import torch
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(i)
                        print(f"  Compute capability: {props.major}.{props.minor}")
                        print(f"  Total memory: {props.total_memory / 1024**3:.2f}GB")
                        print(f"  Multiprocessors: {props.multi_processor_count}")
                elif device_type == DeviceType.NPU:
                    import torch_npu
                    print(f"  NPU device available: {torch_npu.npu.is_available()}")
            except Exception as e:
                print(f"  Properties: Failed to get ({e})")
        
        return True
    except Exception as e:
        print(f"❌ Device details failed: {e}")
        return False

def test_memory_operations():
    """测试内存操作"""
    print("\n🧠 Memory Operations Test")
    print("-" * 40)
    
    try:
        device_type, device_count = device_detector.detect_device()
        
        if device_count == 0:
            print("⚠️  No devices available for memory test")
            return True
        
        device_id = 0
        print(f"Testing on device {device_id} ({device_type.value})")
        
        # 获取初始内存
        initial_allocated, initial_reserved = device_detector.get_memory_info(device_id)
        print(f"Initial: {initial_allocated:.2f}GB allocated, {initial_reserved:.2f}GB reserved")
        
        # 分配一些内存
        if device_type == DeviceType.CUDA:
            import torch
            test_tensor = torch.randn(1000, 1000, device=f'cuda:{device_id}')
        elif device_type == DeviceType.NPU:
            import torch_npu
            test_tensor = torch.randn(1000, 1000, device=f'npu:{device_id}')
        else:
            print("⚠️  CPU device, skipping memory allocation test")
            return True
        
        # 检查内存变化
        after_allocated, after_reserved = device_detector.get_memory_info(device_id)
        print(f"After allocation: {after_allocated:.2f}GB allocated, {after_reserved:.2f}GB reserved")
        
        memory_increase = after_allocated - initial_allocated
        print(f"Memory increase: {memory_increase:.3f}GB")
        
        # 清理
        del test_tensor
        device_detector.empty_cache()
        
        final_allocated, final_reserved = device_detector.get_memory_info(device_id)
        print(f"After cleanup: {final_allocated:.2f}GB allocated, {final_reserved:.2f}GB reserved")
        
        return True
    except Exception as e:
        print(f"❌ Memory operations failed: {e}")
        return False

def test_pipeline_compatibility():
    """测试与管道的兼容性"""
    print("\n🔗 Pipeline Compatibility")
    print("-" * 40)
    
    try:
        available_pipelines = get_available_pipelines()
        print(f"Available pipelines: {available_pipelines}")
        
        device_type, device_count = device_detector.detect_device()
        expected_pipeline = device_type.value.lower()
        
        if expected_pipeline in available_pipelines:
            print(f"✅ Expected pipeline '{expected_pipeline}' is available")
        else:
            print(f"⚠️  Expected pipeline '{expected_pipeline}' not available")
            print(f"   This might indicate missing dependencies")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline compatibility check failed: {e}")
        return False

def test_environment_variables():
    """测试环境变量"""
    print("\n🌍 Environment Variables")
    print("-" * 40)
    
    env_vars = {
        'CUDA_VISIBLE_DEVICES': 'CUDA device visibility',
        'NPU_VISIBLE_DEVICES': 'NPU device visibility', 
        'RANK': 'Distributed training rank',
        'WORLD_SIZE': 'Distributed training world size',
        'LOCAL_RANK': 'Local device rank',
        'MASTER_ADDR': 'Distributed master address',
        'MASTER_PORT': 'Distributed master port'
    }
    
    for var, description in env_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}={value} ({description})")
        else:
            print(f"⚪ {var}=<not set> ({description})")

def main():
    print("🔍 Device Detection Debug Tool - Complete Version")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(("Basic Detection", test_basic_detection()))
    test_results.append(("Device Details", test_device_details()))
    test_results.append(("Memory Operations", test_memory_operations()))
    test_results.append(("Pipeline Compatibility", test_pipeline_compatibility()))
    
    # 环境变量检查（不算作测试结果）
    test_environment_variables()
    
    # 总结
    print("\n📋 Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All device detection tests PASSED!")
        return True
    else:
        print("💥 Some device detection tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)