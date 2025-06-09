#!/usr/bin/env python3
"""
管道调试工具 - 完整版
测试管道创建和基本功能
"""

import sys
import os
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from pipelines import PipelineFactory, get_available_pipelines, get_pipeline_class
    from utils import device_detector
    from schemas import VideoSubmitRequest
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please check project structure and dependencies")
    sys.exit(1)

def test_device_detection():
    """测试设备检测"""
    print("🔍 Device Detection Test")
    print("-" * 40)
    
    try:
        device_type, device_count = device_detector.detect_device()
        print(f"✅ Detected: {device_type.value} x {device_count}")
        return device_type.value, device_count
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return None, 0

def test_pipeline_availability():
    """测试管道可用性"""
    print("\n📦 Pipeline Availability")
    print("-" * 40)
    
    try:
        available_pipelines = get_available_pipelines()
        print(f"Available pipelines: {available_pipelines}")
        
        for pipeline_type in available_pipelines:
            pipeline_class = get_pipeline_class(pipeline_type)
            print(f"✅ {pipeline_type}: {pipeline_class.__name__}")
        
        return available_pipelines
    except Exception as e:
        print(f"❌ Pipeline availability check failed: {e}")
        return []

def test_factory_device_info():
    """测试工厂设备信息"""
    print("\n🏭 Factory Device Info")
    print("-" * 40)
    
    try:
        device_info = PipelineFactory.get_available_devices()
        print(f"Factory device info: {device_info}")
        return device_info
    except Exception as e:
        print(f"❌ Factory device info failed: {e}")
        return {}

def test_pipeline_creation():
    """测试管道创建"""
    print("\n🔧 Pipeline Creation Test")
    print("-" * 40)
    
    # 获取默认模型路径
    default_model_path = "/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"
    model_path = os.environ.get('MODEL_CKPT_DIR', default_model_path)
    
    print(f"Using model path: {model_path}")
    
    try:
        print("Creating pipeline...")
        start_time = time.time()
        
        pipeline = PipelineFactory.create_pipeline(
            ckpt_dir=model_path,
            t5_cpu=True,  # 安全的配置
            dit_fsdp=False,  # 简化配置
            vae_parallel=False
        )
        
        creation_time = time.time() - start_time
        print(f"✅ Pipeline created successfully in {creation_time:.2f}s")
        print(f"Pipeline type: {type(pipeline).__name__}")
        print(f"Pipeline device: {getattr(pipeline, 'device_type', 'unknown')}")
        
        return pipeline
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return None

def test_pipeline_basic_methods(pipeline):
    """测试管道基本方法"""
    print("\n🧪 Pipeline Basic Methods Test")
    print("-" * 40)
    
    if not pipeline:
        print("⚠️  No pipeline to test")
        return False
    
    try:
        # 测试基本属性
        print(f"Pipeline rank: {getattr(pipeline, 'rank', 'N/A')}")
        print(f"Pipeline world_size: {getattr(pipeline, 'world_size', 'N/A')}")
        print(f"Pipeline local_rank: {getattr(pipeline, 'local_rank', 'N/A')}")
        
        # 测试内存方法
        try:
            pipeline._log_memory_usage()
            print("✅ Memory logging works")
        except Exception as e:
            print(f"⚠️  Memory logging failed: {e}")
        
        # 测试缓存清理
        try:
            pipeline._empty_cache()
            print("✅ Cache clearing works")
        except Exception as e:
            print(f"⚠️  Cache clearing failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline methods test failed: {e}")
        return False

def test_request_creation():
    """测试请求对象创建"""
    print("\n📝 Request Creation Test")
    print("-" * 40)
    
    try:
        # 创建测试请求
        test_request = VideoSubmitRequest(
            prompt="A cat walking in the garden",
            image_url="https://example.com/test.jpg",
            image_size="1280*720",
            num_frames=41,
            guidance_scale=3.0,
            infer_steps=30
        )
        
        print("✅ Test request created successfully")
        print(f"Request prompt: {test_request.prompt}")
        print(f"Request params: image_size={test_request.image_size}, frames={test_request.num_frames}")
        
        return test_request
    except Exception as e:
        print(f"❌ Request creation failed: {e}")
        return None

def test_pipeline_cleanup(pipeline):
    """测试管道清理"""
    print("\n🧹 Pipeline Cleanup Test")
    print("-" * 40)
    
    if not pipeline:
        print("⚠️  No pipeline to cleanup")
        return True
    
    try:
        print("Cleaning up pipeline...")
        pipeline.cleanup()
        print("✅ Pipeline cleanup successful")
        return True
    except Exception as e:
        print(f"❌ Pipeline cleanup failed: {e}")
        return False

def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 Comprehensive Pipeline Test")
    print("=" * 60)
    
    test_results = []
    pipeline = None
    
    try:
        # 1. 设备检测
        device_type, device_count = test_device_detection()
        test_results.append(("Device Detection", device_type is not None))
        
        # 2. 管道可用性
        available_pipelines = test_pipeline_availability()
        test_results.append(("Pipeline Availability", len(available_pipelines) > 0))
        
        # 3. 工厂设备信息
        device_info = test_factory_device_info()
        test_results.append(("Factory Device Info", len(device_info) > 0))
        
        # 4. 管道创建
        pipeline = test_pipeline_creation()
        test_results.append(("Pipeline Creation", pipeline is not None))
        
        # 5. 管道基本方法
        basic_methods_ok = test_pipeline_basic_methods(pipeline)
        test_results.append(("Pipeline Basic Methods", basic_methods_ok))
        
        # 6. 请求创建
        test_request = test_request_creation()
        test_results.append(("Request Creation", test_request is not None))
        
        # 7. 管道清理
        cleanup_ok = test_pipeline_cleanup(pipeline)
        test_results.append(("Pipeline Cleanup", cleanup_ok))
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        test_results.append(("Comprehensive Test", False))
    
    # 打印结果
    print("\n📋 Test Results Summary")
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
    
    success = passed == total
    if success:
        print("🎉 All pipeline tests PASSED!")
    else:
        print("💥 Some pipeline tests FAILED!")
    
    return success

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Debug Tool")
    parser.add_argument("--mode", choices=["quick", "comprehensive"], 
                       default="comprehensive", help="Test mode")
    parser.add_argument("--model-path", type=str,
                       help="Model checkpoint path")
    
    args = parser.parse_args()
    
    # 设置模型路径
    if args.model_path:
        os.environ['MODEL_CKPT_DIR'] = args.model_path
    
    print("🔧 Pipeline Debug Tool - Complete Version")
    print("=" * 60)
    
    if args.mode == "quick":
        # 快速测试
        device_type, device_count = test_device_detection()
        available_pipelines = test_pipeline_availability()
        success = device_type is not None and len(available_pipelines) > 0
    else:
        # 综合测试
        success = run_comprehensive_test()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)