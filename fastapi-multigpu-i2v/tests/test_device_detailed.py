#!/usr/bin/env python3
# filepath: /Users/yigex/Documents/LLM-Inftra/Wan2.1_fix/fastapi-multigpu-i2v/tests/test_device_detailed.py
"""
设备检测详细测试工具
整合了原 debug/debug_device.py 的功能
"""

import sys
import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List

# 设置项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from utils.device_detector import device_detector, DeviceType
    from pipelines import get_available_pipelines
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please check project structure and dependencies")
    sys.exit(1)

class DeviceDetailedTester:
    """设备详细测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.device_type = None
        self.device_count = 0
    
    def test_basic_detection(self) -> bool:
        """测试基础设备检测"""
        print("🔍 Basic Device Detection")
        print("-" * 40)
        
        try:
            # 检测设备
            self.device_type, self.device_count = device_detector.detect_device()
            
            print(f"✅ Detected device: {self.device_type.value}")
            print(f"✅ Device count: {self.device_count}")
            
            # 获取后端名称
            try:
                backend = device_detector.get_backend_name()
                print(f"✅ Backend: {backend}")
            except:
                print(f"⚠️  Backend: Unable to determine")
            
            self.test_results['basic_detection'] = {
                'passed': True,
                'device_type': self.device_type.value,
                'device_count': self.device_count
            }
            return True
            
        except Exception as e:
            print(f"❌ Basic detection failed: {e}")
            self.test_results['basic_detection'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_device_details(self) -> bool:
        """测试设备详细信息"""
        print("\n📊 Device Details")
        print("-" * 40)
        
        try:
            if self.device_count == 0:
                print("⚠️  No devices available for detailed testing")
                return True
            
            device_details = []
            
            for i in range(min(self.device_count, 4)):  # 最多测试4个设备
                print(f"\nDevice {i}:")
                device_info = {'id': i}
                
                # 设备名称
                try:
                    device_name = device_detector.get_device_name(i)
                    print(f"  Name: {device_name}")
                    device_info['name'] = device_name
                except Exception as e:
                    print(f"  Name: Failed to get ({e})")
                    device_info['name'] = f"Error: {e}"
                
                # 内存信息
                try:
                    allocated, reserved = device_detector.get_memory_info(i)
                    print(f"  Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    device_info['memory'] = {
                        'allocated_gb': allocated,
                        'reserved_gb': reserved
                    }
                except Exception as e:
                    print(f"  Memory: Failed to get ({e})")
                    device_info['memory'] = f"Error: {e}"
                
                # 设备属性
                try:
                    if self.device_type == DeviceType.CUDA:
                        import torch
                        if torch.cuda.is_available():
                            props = torch.cuda.get_device_properties(i)
                            print(f"  Compute capability: {props.major}.{props.minor}")
                            print(f"  Total memory: {props.total_memory / 1024**3:.2f}GB")
                            print(f"  Multiprocessors: {props.multi_processor_count}")
                            device_info['properties'] = {
                                'compute_capability': f"{props.major}.{props.minor}",
                                'total_memory_gb': props.total_memory / 1024**3,
                                'multiprocessors': props.multi_processor_count
                            }
                    elif self.device_type == DeviceType.NPU:
                        import torch_npu
                        available = torch_npu.npu.is_available()
                        print(f"  NPU device available: {available}")
                        device_info['properties'] = {
                            'npu_available': available
                        }
                except Exception as e:
                    print(f"  Properties: Failed to get ({e})")
                    device_info['properties'] = f"Error: {e}"
                
                device_details.append(device_info)
            
            self.test_results['device_details'] = {
                'passed': True,
                'devices': device_details
            }
            return True
            
        except Exception as e:
            print(f"❌ Device details failed: {e}")
            self.test_results['device_details'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_memory_operations(self) -> bool:
        """测试内存操作"""
        print("\n🧠 Memory Operations Test")
        print("-" * 40)
        
        try:
            if self.device_count == 0:
                print("⚠️  No devices available for memory test")
                return True
            
            if self.device_type == DeviceType.CPU:
                print("⚠️  CPU device, skipping memory allocation test")
                return True
            
            device_id = 0
            print(f"Testing on device {device_id} ({self.device_type.value})")
            
            # 获取初始内存
            initial_allocated, initial_reserved = device_detector.get_memory_info(device_id)
            print(f"Initial: {initial_allocated:.2f}GB allocated, {initial_reserved:.2f}GB reserved")
            
            # 分配一些内存
            test_tensor = None
            if self.device_type == DeviceType.CUDA:
                import torch
                test_tensor = torch.randn(1000, 1000, device=f'cuda:{device_id}')
            elif self.device_type == DeviceType.NPU:
                import torch_npu
                test_tensor = torch.randn(1000, 1000, device=f'npu:{device_id}')
            
            # 检查内存变化
            after_allocated, after_reserved = device_detector.get_memory_info(device_id)
            print(f"After allocation: {after_allocated:.2f}GB allocated, {after_reserved:.2f}GB reserved")
            
            memory_increase = after_allocated - initial_allocated
            print(f"Memory increase: {memory_increase:.3f}GB")
            
            # 清理
            if test_tensor is not None:
                del test_tensor
            device_detector.empty_cache()
            
            final_allocated, final_reserved = device_detector.get_memory_info(device_id)
            print(f"After cleanup: {final_allocated:.2f}GB allocated, {final_reserved:.2f}GB reserved")
            
            self.test_results['memory_operations'] = {
                'passed': True,
                'initial_allocated': initial_allocated,
                'peak_allocated': after_allocated,
                'final_allocated': final_allocated,
                'memory_increase': memory_increase
            }
            return True
            
        except Exception as e:
            print(f"❌ Memory operations failed: {e}")
            self.test_results['memory_operations'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_pipeline_compatibility(self) -> bool:
        """测试与管道的兼容性"""
        print("\n🔗 Pipeline Compatibility")
        print("-" * 40)
        
        try:
            available_pipelines = get_available_pipelines()
            print(f"Available pipelines: {available_pipelines}")
            
            expected_pipeline = self.device_type.value.lower()
            
            if expected_pipeline in available_pipelines:
                print(f"✅ Expected pipeline '{expected_pipeline}' is available")
                compatible = True
            else:
                print(f"⚠️  Expected pipeline '{expected_pipeline}' not available")
                print(f"   This might indicate missing dependencies")
                compatible = False
            
            self.test_results['pipeline_compatibility'] = {
                'passed': True,
                'available_pipelines': available_pipelines,
                'expected_pipeline': expected_pipeline,
                'compatible': compatible
            }
            return True
            
        except Exception as e:
            print(f"❌ Pipeline compatibility check failed: {e}")
            self.test_results['pipeline_compatibility'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_performance_benchmark(self) -> bool:
        """简单的性能基准测试"""
        print("\n⚡ Performance Benchmark")
        print("-" * 40)
        
        try:
            if self.device_count == 0 or self.device_type == DeviceType.CPU:
                print("⚠️  Skipping performance test (no GPU/NPU available)")
                return True
            
            device_id = 0
            iterations = 10
            
            print(f"Running {iterations} iterations of matrix multiplication...")
            
            times = []
            
            for i in range(iterations):
                start_time = time.time()
                
                if self.device_type == DeviceType.CUDA:
                    import torch
                    a = torch.randn(1000, 1000, device=f'cuda:{device_id}')
                    b = torch.randn(1000, 1000, device=f'cuda:{device_id}')
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()  # 确保计算完成
                elif self.device_type == DeviceType.NPU:
                    import torch_npu
                    a = torch.randn(1000, 1000, device=f'npu:{device_id}')
                    b = torch.randn(1000, 1000, device=f'npu:{device_id}')
                    c = torch.matmul(a, b)
                    torch_npu.npu.synchronize()  # 确保计算完成
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"Average time: {avg_time*1000:.2f}ms")
            print(f"Min time: {min_time*1000:.2f}ms")
            print(f"Max time: {max_time*1000:.2f}ms")
            
            # 估算 FLOPS (1000x1000 矩阵乘法大约是 2 * 1000^3 = 2G FLOPS)
            flops = 2 * 1000**3
            avg_gflops = (flops / avg_time) / 1e9
            print(f"Estimated performance: {avg_gflops:.2f} GFLOPS")
            
            self.test_results['performance_benchmark'] = {
                'passed': True,
                'avg_time_ms': avg_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'estimated_gflops': avg_gflops
            }
            return True
            
        except Exception as e:
            print(f"❌ Performance benchmark failed: {e}")
            self.test_results['performance_benchmark'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_environment_variables(self) -> Dict[str, str]:
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
            'MASTER_PORT': 'Distributed master port',
            'MODEL_CKPT_DIR': 'Model checkpoint directory',
            'T5_CPU': 'T5 CPU mode',
            'DIT_FSDP': 'DIT FSDP mode',
            'VAE_PARALLEL': 'VAE parallel mode',
            'ULYSSES_SIZE': 'Ulysses parallelism size'
        }
        
        env_status = {}
        
        for var, description in env_vars.items():
            value = os.environ.get(var)
            if value:
                print(f"✅ {var}={value}")
                status = "✅"
            else:
                print(f"⚪ {var}=<not set>")
                status = "⚪"
                value = "<not set>"
            
            env_status[var] = {'value': value, 'status': status, 'description': description}
        
        self.test_results['environment_variables'] = env_status
        return env_status
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("🔍 Device Detection Detailed Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # 运行所有测试
        tests = [
            ("Basic Detection", self.test_basic_detection),
            ("Device Details", self.test_device_details),
            ("Memory Operations", self.test_memory_operations),
            ("Pipeline Compatibility", self.test_pipeline_compatibility),
            ("Performance Benchmark", self.test_performance_benchmark),
        ]
        
        test_results = []
        passed = 0
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                test_results.append((test_name, result))
                if result:
                    passed += 1
            except Exception as e:
                print(f"💥 {test_name} crashed: {e}")
                test_results.append((test_name, False))
        
        # 环境变量检查（不算作测试结果）
        self.test_environment_variables()
        
        # 计算测试时间
        test_duration = time.time() - start_time
        
        # 总结
        print("\n📋 Test Summary")
        print("=" * 60)
        
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<25} {status}")
        
        print("-" * 60)
        print(f"Total: {passed}/{total} tests passed")
        print(f"Test duration: {test_duration:.2f}s")
        
        # 生成建议
        self._generate_recommendations(passed, total)
        
        if passed == total:
            print("🎉 All device detection tests PASSED!")
        else:
            print("💥 Some device detection tests FAILED!")
        
        # 返回完整的测试结果
        return {
            'summary': {
                'passed': passed,
                'total': total,
                'success_rate': (passed / total) * 100,
                'duration': test_duration
            },
            'tests': test_results,
            'details': self.test_results
        }
    
    def _generate_recommendations(self, passed: int, total: int) -> None:
        """生成建议"""
        print(f"\n💡 Recommendations:")
        
        success_rate = (passed / total) * 100
        
        if success_rate < 100:
            print(f"  🔧 Some tests failed. Check the error messages above.")
        
        if self.device_type and self.device_count > 0:
            if self.device_type == DeviceType.NPU:
                print(f"  🚀 NPU detected! Recommended for production use.")
                if self.device_count >= 8:
                    print(f"     export ULYSSES_SIZE=8")
            elif self.device_type == DeviceType.CUDA:
                print(f"  🎮 CUDA detected! Good for development and production.")
                if self.device_count >= 4:
                    print(f"     export ULYSSES_SIZE=4")
            else:
                print(f"  💻 CPU only. Consider upgrading hardware for better performance.")
        
        # 基于测试结果的具体建议
        if 'memory_operations' in self.test_results:
            if self.test_results['memory_operations'].get('passed'):
                print(f"  ✅ Memory operations working correctly.")
            else:
                print(f"  ⚠️  Memory operations failed. Check device drivers.")
        
        if 'pipeline_compatibility' in self.test_results:
            if not self.test_results['pipeline_compatibility'].get('compatible', True):
                print(f"  ⚠️  Pipeline compatibility issues detected.")
                print(f"     Install missing dependencies for {self.device_type.value} support.")

def main():
    """主函数"""
    try:
        tester = DeviceDetailedTester()
        results = tester.run_all_tests()
        
        # 返回是否所有测试都通过
        success = results['summary']['passed'] == results['summary']['total']
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