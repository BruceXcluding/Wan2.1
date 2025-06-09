#!/usr/bin/env python3
# filepath: /Users/yigex/Documents/LLM-Inftra/Wan2.1_fix/fastapi-multigpu-i2v/tests/test_pipeline.py
"""
管道测试工具
整合了原 debug/debug_pipeline.py 的功能
"""

import sys
import os
import time
import argparse
from pathlib import Path

# 设置项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from pipelines import PipelineFactory, get_available_pipelines
    from utils.device_detector import device_detector
    from schemas import VideoSubmitRequest
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please check project structure and dependencies")
    sys.exit(1)

class PipelineTester:
    """管道测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.pipeline = None
    
    def test_device_detection(self):
        """测试设备检测"""
        print("🔍 Device Detection Test")
        print("-" * 40)
        
        try:
            device_type, device_count = device_detector.detect_device()
            print(f"✅ Detected: {device_type.value} x {device_count}")
            
            self.test_results['device_detection'] = {
                'passed': True,
                'device_type': device_type.value,
                'device_count': device_count
            }
            return device_type.value, device_count
            
        except Exception as e:
            print(f"❌ Device detection failed: {e}")
            self.test_results['device_detection'] = {'passed': False, 'error': str(e)}
            return None, 0
    
    def test_pipeline_availability(self):
        """测试管道可用性"""
        print("\n📦 Pipeline Availability")
        print("-" * 40)
        
        try:
            available_pipelines = get_available_pipelines()
            print(f"Available pipelines: {available_pipelines}")
            
            if len(available_pipelines) > 0:
                print("✅ Pipeline system is available")
                self.test_results['pipeline_availability'] = {
                    'passed': True,
                    'pipelines': available_pipelines
                }
            else:
                print("⚠️  No pipelines available")
                self.test_results['pipeline_availability'] = {
                    'passed': False,
                    'pipelines': []
                }
            
            return available_pipelines
            
        except Exception as e:
            print(f"❌ Pipeline availability check failed: {e}")
            self.test_results['pipeline_availability'] = {'passed': False, 'error': str(e)}
            return []
    
    def test_factory_device_info(self):
        """测试工厂设备信息"""
        print("\n🏭 Factory Device Info")
        print("-" * 40)
        
        try:
            device_info = PipelineFactory.get_available_devices()
            print(f"Factory device info: {device_info}")
            
            self.test_results['factory_device_info'] = {
                'passed': True,
                'device_info': device_info
            }
            return device_info
            
        except Exception as e:
            print(f"❌ Factory device info failed: {e}")
            self.test_results['factory_device_info'] = {'passed': False, 'error': str(e)}
            return {}
    
    def test_pipeline_creation(self, model_path: str = None):
        """测试管道创建"""
        print("\n🔧 Pipeline Creation Test")
        print("-" * 40)
        
        # 获取默认模型路径
        if not model_path:
            model_path = os.environ.get('MODEL_CKPT_DIR', 
                "/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P")
        
        print(f"Using model path: {model_path}")
        
        try:
            print("Creating pipeline...")
            start_time = time.time()
            
            # 使用安全的配置进行测试
            self.pipeline = PipelineFactory.create_pipeline(
                ckpt_dir=model_path,
                t5_cpu=True,  # 安全的配置
                dit_fsdp=False,  # 简化配置
                vae_parallel=False
            )
            
            creation_time = time.time() - start_time
            print(f"✅ Pipeline created successfully in {creation_time:.2f}s")
            print(f"Pipeline type: {type(self.pipeline).__name__}")
            
            if hasattr(self.pipeline, 'device_type'):
                print(f"Pipeline device: {self.pipeline.device_type}")
            
            self.test_results['pipeline_creation'] = {
                'passed': True,
                'creation_time': creation_time,
                'pipeline_type': type(self.pipeline).__name__
            }
            return self.pipeline
            
        except Exception as e:
            print(f"❌ Pipeline creation failed: {e}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            self.test_results['pipeline_creation'] = {'passed': False, 'error': str(e)}
            return None
    
    def test_pipeline_basic_methods(self):
        """测试管道基本方法"""
        print("\n🧪 Pipeline Basic Methods Test")
        print("-" * 40)
        
        if not self.pipeline:
            print("⚠️  No pipeline to test")
            return False
        
        try:
            # 测试基本属性
            print(f"Pipeline rank: {getattr(self.pipeline, 'rank', 'N/A')}")
            print(f"Pipeline world_size: {getattr(self.pipeline, 'world_size', 'N/A')}")
            print(f"Pipeline local_rank: {getattr(self.pipeline, 'local_rank', 'N/A')}")
            
            # 测试内存方法
            try:
                if hasattr(self.pipeline, '_log_memory_usage'):
                    self.pipeline._log_memory_usage()
                    print("✅ Memory logging works")
                else:
                    print("⚪ Memory logging method not available")
            except Exception as e:
                print(f"⚠️  Memory logging failed: {e}")
            
            # 测试缓存清理
            try:
                if hasattr(self.pipeline, '_empty_cache'):
                    self.pipeline._empty_cache()
                    print("✅ Cache clearing works")
                else:
                    print("⚪ Cache clearing method not available")
            except Exception as e:
                print(f"⚠️  Cache clearing failed: {e}")
            
            self.test_results['pipeline_basic_methods'] = {'passed': True}
            return True
            
        except Exception as e:
            print(f"❌ Pipeline methods test failed: {e}")
            self.test_results['pipeline_basic_methods'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_request_creation(self):
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
            
            self.test_results['request_creation'] = {
                'passed': True,
                'request_data': {
                    'prompt': test_request.prompt,
                    'image_size': test_request.image_size,
                    'num_frames': test_request.num_frames
                }
            }
            return test_request
            
        except Exception as e:
            print(f"❌ Request creation failed: {e}")
            self.test_results['request_creation'] = {'passed': False, 'error': str(e)}
            return None
    
    def test_pipeline_cleanup(self):
        """测试管道清理"""
        print("\n🧹 Pipeline Cleanup Test")
        print("-" * 40)
        
        if not self.pipeline:
            print("⚪ No pipeline to cleanup")
            return True
        
        try:
            print("Cleaning up pipeline...")
            if hasattr(self.pipeline, 'cleanup'):
                self.pipeline.cleanup()
                print("✅ Pipeline cleanup successful")
            else:
                print("⚪ Pipeline cleanup method not available")
            
            self.test_results['pipeline_cleanup'] = {'passed': True}
            return True
            
        except Exception as e:
            print(f"❌ Pipeline cleanup failed: {e}")
            self.test_results['pipeline_cleanup'] = {'passed': False, 'error': str(e)}
            return False
    
    def run_quick_test(self):
        """快速测试"""
        print("🚀 Quick Pipeline Test")
        print("=" * 60)
        
        device_type, device_count = self.test_device_detection()
        available_pipelines = self.test_pipeline_availability()
        
        success = device_type is not None and len(available_pipelines) > 0
        
        if success:
            print("\n🎉 Quick test PASSED! Pipeline system is ready.")
        else:
            print("\n💥 Quick test FAILED! Check device or pipeline setup.")
        
        return success
    
    def run_comprehensive_test(self, model_path: str = None):
        """运行综合测试"""
        print("🚀 Comprehensive Pipeline Test")
        print("=" * 60)
        
        test_functions = [
            ("Device Detection", self.test_device_detection),
            ("Pipeline Availability", self.test_pipeline_availability),
            ("Factory Device Info", self.test_factory_device_info),
            ("Pipeline Creation", lambda: self.test_pipeline_creation(model_path)),
            ("Pipeline Basic Methods", self.test_pipeline_basic_methods),
            ("Request Creation", self.test_request_creation),
            ("Pipeline Cleanup", self.test_pipeline_cleanup),
        ]
        
        passed = 0
        total = len(test_functions)
        
        for test_name, test_func in test_functions:
            try:
                result = test_func()
                if result:
                    passed += 1
            except Exception as e:
                print(f"💥 {test_name} crashed: {e}")
        
        # 打印结果
        print("\n📋 Test Results Summary")
        print("=" * 60)
        
        for test_name, _ in test_functions:
            result_key = test_name.lower().replace(' ', '_')
            result = self.test_results.get(result_key, {'passed': False})
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            print(f"{test_name:<25} {status}")
        
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
    parser = argparse.ArgumentParser(description="Pipeline Test Tool")
    parser.add_argument("--mode", choices=["quick", "comprehensive"], 
                       default="comprehensive", help="Test mode")
    parser.add_argument("--model-path", type=str,
                       help="Model checkpoint path")
    
    args = parser.parse_args()
    
    # 设置模型路径
    if args.model_path:
        os.environ['MODEL_CKPT_DIR'] = args.model_path
    
    print("🔧 Pipeline Test Tool")
    print("=" * 60)
    
    try:
        tester = PipelineTester()
        
        if args.mode == "quick":
            success = tester.run_quick_test()
        else:
            success = tester.run_comprehensive_test(args.model_path)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)