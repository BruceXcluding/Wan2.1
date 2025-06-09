#!/usr/bin/env python3
# filepath: /Users/yigex/Documents/LLM-Inftra/Wan2.1_fix/fastapi-multigpu-i2v/tests/test_env.py
"""
FastAPI Multi-GPU I2V - 环境测试工具
====================================

这个脚本执行全面的环境检查，包括：
- 📁 项目结构验证
- 🧪 模块导入测试  
- 🖥️  硬件环境检测
- 🌍 环境变量检查
- 💡 智能建议和故障排除

使用方法:
    python3 tests/test_env.py
    
或者从项目根目录:
    cd /path/to/fastapi-multigpu-i2v
    python3 tests/test_env.py

作者: Multi-GPU I2V Team
版本: 1.0.0
"""
import sys
import os
import traceback
from pathlib import Path
import importlib.util
from typing import Dict, List, Tuple, Any, Optional
import time

def setup_project_paths() -> Tuple[Path, Path]:
    """设置项目路径并返回关键路径"""
    # 获取当前脚本路径
    current_script = Path(__file__).resolve()
    
    # 计算项目根目录（测试脚本在 tests/ 目录下）
    project_root = current_script.parent.parent
    src_root = project_root / "src"
    
    # 添加到 Python 路径
    paths_to_add = [str(project_root), str(src_root)]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return project_root, src_root

def check_project_structure(project_root: Path, src_root: Path) -> Dict[str, bool]:
    """检查项目结构"""
    print(f"📂 Project structure check:")
    
    directories_to_check = [
        ("src", src_root),
        ("src/schemas", src_root / "schemas"),
        ("src/pipelines", src_root / "pipelines"),
        ("src/services", src_root / "services"),
        ("src/utils", src_root / "utils"),
        ("utils", project_root / "utils"),
        ("scripts", project_root / "scripts"),
        ("configs", project_root / "configs"),
        ("logs", project_root / "logs"),
        ("tests", project_root / "tests"),
        ("generated_videos", project_root / "generated_videos"),
    ]
    
    structure_status = {}
    
    for name, path in directories_to_check:
        exists = path.exists()
        has_init = (path / "__init__.py").exists() if exists else False
        status = "✅" if exists else "❌"
        init_info = " (has __init__.py)" if has_init else " (no __init__.py)" if exists else ""
        print(f"  {status} {name:<25} {init_info}")
        structure_status[name] = exists
    
    return structure_status

def check_key_files(project_root: Path, src_root: Path) -> Dict[str, bool]:
    """检查关键文件"""
    print(f"\n📄 Key files check:")
    
    files_to_check = [
        ("src/schemas/video.py", src_root / "schemas" / "video.py"),
        ("src/schemas/__init__.py", src_root / "schemas" / "__init__.py"),
        ("src/pipelines/pipeline_factory.py", src_root / "pipelines" / "pipeline_factory.py"),
        ("src/pipelines/__init__.py", src_root / "pipelines" / "__init__.py"),
        ("src/pipelines/base_pipeline.py", src_root / "pipelines" / "base_pipeline.py"),
        ("utils/device_detector.py", project_root / "utils" / "device_detector.py"),
        ("utils/__init__.py", project_root / "utils" / "__init__.py"),
        ("src/i2v_api.py", src_root / "i2v_api.py"),
        ("requirements.txt", project_root / "requirements.txt"),
        ("README.md", project_root / "README.md"),
        ("scripts/start_service_general.sh", project_root / "scripts" / "start_service_general.sh"),
    ]
    
    file_status = {}
    
    for name, path in files_to_check:
        exists = path.exists()
        status = "✅" if exists else "❌"
        size = f" ({path.stat().st_size} bytes)" if exists else ""
        print(f"  {status} {name:<40} {size}")
        file_status[name] = exists
    
    return file_status

def test_module_imports() -> Tuple[int, int, List[str]]:
    """测试模块导入"""
    print(f"\n🧪 Module import testing:")
    
    import_tests = [
        # 基础 Python 模块
        ("sys", None, "Python built-in"),
        ("os", None, "Python built-in"),
        ("pathlib", None, "Python built-in"),
        ("logging", None, "Python built-in"),
        ("asyncio", None, "Python built-in"),
        ("json", None, "Python built-in"),
        ("time", None, "Python built-in"),
        ("uuid", None, "Python built-in"),
        ("datetime", None, "Python built-in"),
        
        # 第三方核心依赖
        ("fastapi", None, "FastAPI framework"),
        ("uvicorn", None, "ASGI server"),
        ("pydantic", None, "Data validation"),
        ("torch", "version", "PyTorch"),
        
        # 第三方可选依赖
        ("PIL", None, "Pillow image library"),
        ("requests", None, "HTTP library"),
        ("aiofiles", None, "Async file operations"),
        ("numpy", None, "NumPy"),
        
        # 项目核心模块 - schemas
        ("schemas", None, "Project schemas package"),
        ("schemas.video", "VideoSubmitRequest", "Video request models"),
        ("schemas.video", "VideoSubmitResponse", "Video response models"),
        ("schemas.video", "VideoStatusRequest", "Video status models"),
        ("schemas.video", "VideoStatusResponse", "Video status response"),
        ("schemas.video", "VideoCancelRequest", "Video cancel models"),
        ("schemas.video", "VideoCancelResponse", "Video cancel response"),
        ("schemas.video", "TaskStatus", "Task status enum"),
        ("schemas.video", "VideoResults", "Video results model"),
        ("schemas.video", "HealthResponse", "Health response model"),
        ("schemas.video", "MetricsResponse", "Metrics response model"),
        
        # 项目核心模块 - pipelines
        ("pipelines", None, "Project pipelines package"),
        ("pipelines.pipeline_factory", "PipelineFactory", "Pipeline factory class"),
        ("pipelines.pipeline_factory", "get_available_pipelines", "Available pipelines function"),
        ("pipelines.base_pipeline", "BasePipeline", "Base pipeline class"),
        
        # 项目核心模块 - utils
        ("utils", None, "Project utils package"),
        ("utils.device_detector", "device_detector", "Device detector instance"),
        ("utils.device_detector", "DeviceDetector", "Device detector class"),
        ("utils.device_detector", "DeviceType", "Device type enum"),
        
        # 项目核心模块 - services (如果存在)
        ("services", None, "Project services package"),
    ]
    
    success_count = 0
    total_count = len(import_tests)
    failed_imports = []
    
    for module_name, item_name, description in import_tests:
        try:
            if item_name == "version":
                # 特殊处理版本信息
                module = __import__(module_name)
                version = getattr(module, "__version__", "unknown")
                print(f"  ✅ {module_name:<35} v{version} - {description}")
            elif item_name:
                # 导入特定的类或函数
                module = __import__(module_name, fromlist=[item_name])
                obj = getattr(module, item_name)
                obj_type = type(obj).__name__
                print(f"  ✅ {module_name}.{item_name:<25} ({obj_type}) - {description}")
            else:
                # 只导入模块
                module = __import__(module_name)
                module_file = getattr(module, "__file__", "built-in")
                if module_file != "built-in":
                    module_file = Path(module_file).name
                print(f"  ✅ {module_name:<35} ({module_file}) - {description}")
            success_count += 1
            
        except ImportError as e:
            error_msg = f"{module_name}" + (f".{item_name}" if item_name else "") + f" - Import Error: {e}"
            print(f"  ❌ {error_msg}")
            failed_imports.append(error_msg)
        except AttributeError as e:
            error_msg = f"{module_name}" + (f".{item_name}" if item_name else "") + f" - Attribute Error: {e}"
            print(f"  ❌ {error_msg}")
            failed_imports.append(error_msg)
        except Exception as e:
            error_msg = f"{module_name}" + (f".{item_name}" if item_name else "") + f" - Other Error: {e}"
            print(f"  ⚠️  {error_msg}")
            failed_imports.append(error_msg)
    
    return success_count, total_count, failed_imports

def test_special_functionality() -> Dict[str, Any]:
    """测试特殊功能"""
    print(f"\n🎯 Special functionality tests:")
    
    test_results = {
        "device_detection": False,
        "pipeline_detection": False,
        "schema_completeness": False,
        "device_info": None,
        "available_pipelines": None,
        "schema_classes_count": 0
    }
    
    # 测试设备检测
    try:
        from utils.device_detector import device_detector
        device_type, device_count = device_detector.detect_device()
        device_info = device_detector.get_device_info()
        
        print(f"  ✅ Device detection: {device_type.value} x {device_count}")
        print(f"     Backend: {device_info.get('backend', 'unknown')}")
        
        test_results["device_detection"] = True
        test_results["device_info"] = {
            "type": device_type.value,
            "count": device_count,
            "backend": device_info.get('backend', 'unknown')
        }
        
        # 测试设备相关方法
        is_distributed = device_detector.is_distributed_available()
        print(f"     Distributed available: {is_distributed}")
        
    except Exception as e:
        print(f"  ❌ Device detection failed: {e}")
        print(f"     Debug info: {traceback.format_exc()}")
    
    # 测试可用管道
    try:
        from pipelines.pipeline_factory import get_available_pipelines, PipelineFactory
        pipelines = get_available_pipelines()
        device_info = PipelineFactory.get_available_devices()
        
        print(f"  ✅ Available pipelines: {pipelines}")
        print(f"     Pipeline device info: {device_info}")
        
        test_results["pipeline_detection"] = True
        test_results["available_pipelines"] = pipelines
        
    except Exception as e:
        print(f"  ❌ Pipeline detection failed: {e}")
        print(f"     Debug info: {traceback.format_exc()}")
    
    # 测试 schemas 完整性
    try:
        from schemas import (
            VideoSubmitRequest, VideoSubmitResponse,
            VideoStatusRequest, VideoStatusResponse,
            VideoCancelRequest, VideoCancelResponse,
            TaskStatus, VideoResults, HealthResponse, MetricsResponse
        )
        
        schema_classes = [
            VideoSubmitRequest, VideoSubmitResponse,
            VideoStatusRequest, VideoStatusResponse,
            VideoCancelRequest, VideoCancelResponse,
            TaskStatus, VideoResults, HealthResponse, MetricsResponse
        ]
        
        print(f"  ✅ All schema classes loaded: {len(schema_classes)} classes")
        
        # 测试TaskStatus枚举
        task_statuses = [status.value for status in TaskStatus]
        print(f"     TaskStatus values: {task_statuses}")
        
        test_results["schema_completeness"] = True
        test_results["schema_classes_count"] = len(schema_classes)
        
    except Exception as e:
        print(f"  ❌ Schema completeness test failed: {e}")
        print(f"     Debug info: {traceback.format_exc()}")
    
    return test_results

def check_environment_variables() -> Dict[str, str]:
    """检查环境变量"""
    print(f"\n🌍 Environment variables check:")
    
    env_vars = [
        "MODEL_CKPT_DIR",
        "PYTHONPATH", 
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "T5_CPU",
        "DIT_FSDP",
        "VAE_PARALLEL",
        "ULYSSES_SIZE",
        "MAX_CONCURRENT_TASKS",
        "SERVER_HOST",
        "SERVER_PORT",
        "MASTER_ADDR",
        "MASTER_PORT",
    ]
    
    env_status = {}
    
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        status = "✅" if value != "Not set" else "⚪"
        print(f"  {status} {var:<25} = {value}")
        env_status[var] = value
    
    return env_status

def check_hardware() -> Dict[str, Any]:
    """检查硬件环境"""
    print(f"\n🖥️  Hardware-specific checks:")
    
    hardware_info = {
        "npu_available": False,
        "cuda_available": False,
        "npu_count": 0,
        "cuda_count": 0,
        "npu_devices": [],
        "cuda_devices": []
    }
    
    # NPU 检查
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            npu_count = torch_npu.npu.device_count()
            print(f"  ✅ NPU available: {npu_count} devices")
            hardware_info["npu_available"] = True
            hardware_info["npu_count"] = npu_count
            
            for i in range(min(npu_count, 4)):  # 只显示前4个设备
                try:
                    # 尝试获取内存信息
                    memory_info = torch_npu.npu.memory_stats(i) if hasattr(torch_npu.npu, 'memory_stats') else {}
                    allocated = memory_info.get('allocated_bytes.all.current', 'N/A')
                    total = memory_info.get('reserved_bytes.all.current', 'N/A')
                    
                    device_info = {
                        "id": i,
                        "allocated": allocated,
                        "total": total
                    }
                    hardware_info["npu_devices"].append(device_info)
                    print(f"    📱 NPU {i}: {allocated} allocated / {total} total")
                except Exception as e:
                    print(f"    📱 NPU {i}: Info unavailable ({e})")
        else:
            print(f"  ⚪ NPU not available")
    except ImportError:
        print(f"  ⚪ torch_npu not installed")
    except Exception as e:
        print(f"  ❌ NPU check failed: {e}")
    
    # CUDA 检查
    try:
        import torch
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            print(f"  ✅ CUDA available: {cuda_count} devices")
            hardware_info["cuda_available"] = True
            hardware_info["cuda_count"] = cuda_count
            
            for i in range(min(cuda_count, 4)):  # 只显示前4个设备
                try:
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1024**3
                    
                    device_info = {
                        "id": i,
                        "name": props.name,
                        "memory_gb": memory_gb,
                        "compute_capability": f"{props.major}.{props.minor}"
                    }
                    hardware_info["cuda_devices"].append(device_info)
                    print(f"    🎮 GPU {i}: {props.name} ({memory_gb:.1f}GB, CC {props.major}.{props.minor})")
                except Exception as e:
                    print(f"    🎮 GPU {i}: Info unavailable ({e})")
        else:
            print(f"  ⚪ CUDA not available")
    except Exception as e:
        print(f"  ❌ CUDA check failed: {e}")
    
    return hardware_info

def generate_recommendations(
    success_rate: float, 
    failed_imports: List[str],
    env_status: Dict[str, str],
    hardware_info: Dict[str, Any],
    test_results: Dict[str, Any]
) -> None:
    """生成建议和下一步操作"""
    print(f"\n💡 Recommendations:")
    
    # 基于成功率的建议
    if success_rate < 100:
        print(f"  📦 Install missing dependencies:")
        print(f"     pip install -r requirements.txt")
        
        if any("torch" in fail for fail in failed_imports):
            print(f"     🔥 PyTorch installation required")
        
        if any("fastapi" in fail for fail in failed_imports):
            print(f"     🚀 FastAPI installation required")
    
    # 目录创建建议
    project_root = Path(__file__).parent.parent
    if not (project_root / "logs").exists():
        print(f"  📁 Create logs directory: mkdir -p logs")
    
    if not (project_root / "generated_videos").exists():
        print(f"  📁 Create output directory: mkdir -p generated_videos")
    
    # 环境变量建议
    if env_status.get("MODEL_CKPT_DIR") == "Not set":
        print(f"  🔧 Set model path:")
        print(f"     export MODEL_CKPT_DIR=/path/to/models")
    
    # 硬件特定建议
    if test_results.get("device_detection") and test_results.get("device_info"):
        device_info = test_results["device_info"]
        device_type = device_info["type"]
        device_count = device_info["count"]
        
        if device_type == "npu" and device_count > 0:
            print(f"  🚀 NPU detected! Recommended configuration:")
            print(f"     ./scripts/start_service_general.sh")
            if device_count >= 8:
                print(f"     Consider: export ULYSSES_SIZE=8")
                print(f"     Consider: export MAX_CONCURRENT_TASKS=2")
        elif device_type == "cuda" and device_count > 0:
            print(f"  🚀 CUDA detected! Recommended configuration:")
            print(f"     ./scripts/start_service_general.sh")
            if device_count >= 4:
                print(f"     Consider: export ULYSSES_SIZE=4")
                print(f"     Consider: export MAX_CONCURRENT_TASKS=3")
        else:
            print(f"  💻 CPU only detected:")
            print(f"     export T5_CPU=true")
            print(f"     export MAX_CONCURRENT_TASKS=1")
    
    # 故障排除建议
    if success_rate < 70:
        print(f"  🔧 Troubleshooting:")
        print(f"     1. Check Python version: python3 --version (>= 3.8)")
        print(f"     2. Check virtual environment: which python3")
        print(f"     3. Update pip: pip install --upgrade pip")
        print(f"     4. Reinstall dependencies: pip install -r requirements.txt --force-reinstall")
    
    # 性能优化建议
    if success_rate >= 90:
        print(f"  ⚡ Performance optimization:")
        print(f"     export OMP_NUM_THREADS=16")
        print(f"     export MKL_NUM_THREADS=16")
        if hardware_info.get("npu_available"):
            print(f"     export HCCL_TIMEOUT=1800")
        elif hardware_info.get("cuda_available"):
            print(f"     export NCCL_TIMEOUT=1800")

def test_imports():
    """主测试函数"""
    print("🔍 FastAPI Multi-GPU I2V Environment Test")
    print("=" * 55)
    
    start_time = time.time()
    
    # 设置项目路径
    project_root, src_root = setup_project_paths()
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Source root: {src_root}")
    print(f"📁 Current working directory: {Path.cwd()}")
    print(f"📁 Test script location: {Path(__file__)}")
    
    # 验证关键路径
    print(f"\n🔍 Path validation:")
    print(f"  - Project root exists: {project_root.exists()}")
    print(f"  - Source root exists: {src_root.exists()}")
    print(f"  - Utils directory: {(project_root / 'utils').exists()}")
    print(f"  - Device detector file: {(project_root / 'utils' / 'device_detector.py').exists()}")
    
    print(f"\n🔍 Python path (first 8 entries):")
    for i, path in enumerate(sys.path[:8]):
        print(f"  {i+1:2d}. {path}")
    
    # 项目结构检查
    print(f"\n" + "="*55)
    structure_status = check_project_structure(project_root, src_root)
    
    # 关键文件检查
    file_status = check_key_files(project_root, src_root)
    
    # 模块导入测试
    print(f"\n" + "="*55)
    success_count, total_count, failed_imports = test_module_imports()
    
    # 特殊功能测试
    print(f"\n" + "="*55)
    test_results = test_special_functionality()
    
    # 环境变量检查
    print(f"\n" + "="*55)
    env_status = check_environment_variables()
    
    # 硬件检查
    print(f"\n" + "="*55)
    hardware_info = check_hardware()
    
    # 计算测试时间
    test_duration = time.time() - start_time
    
    # 总结报告
    print(f"\n" + "="*55)
    print(f"📊 Test Summary:")
    print(f"=" * 55)
    
    success_rate = (success_count / total_count) * 100
    print(f"Import success rate: {success_count}/{total_count} ({success_rate:.1f}%)")
    print(f"Test duration: {test_duration:.2f}s")
    
    # 状态评估
    if success_rate >= 95:
        status_emoji = "🎉"
        status_text = "Excellent! Project is ready to run."
    elif success_rate >= 85:
        status_emoji = "✅"
        status_text = "Very Good! Minor issues detected."
    elif success_rate >= 70:
        status_emoji = "🟡"
        status_text = "Good! Some optional dependencies missing."
    elif success_rate >= 50:
        status_emoji = "⚠️"
        status_text = "Warning! Core dependencies missing."
    else:
        status_emoji = "❌"
        status_text = "Critical! Many dependencies missing."
    
    print(f"{status_emoji} {status_text}")
    
    # 关键组件状态
    print(f"\nKey components status:")
    print(f"  Device detection: {'✅' if test_results.get('device_detection') else '❌'}")
    print(f"  Pipeline system: {'✅' if test_results.get('pipeline_detection') else '❌'}")
    print(f"  Schema system: {'✅' if test_results.get('schema_completeness') else '❌'}")
    print(f"  Hardware support: {'✅' if hardware_info.get('npu_available') or hardware_info.get('cuda_available') else '⚪ CPU only'}")
    
    # 失败导入详情
    if failed_imports:
        print(f"\n❌ Failed imports ({len(failed_imports)}):")
        for fail in failed_imports[:5]:  # 只显示前5个
            print(f"  - {fail}")
        if len(failed_imports) > 5:
            print(f"  ... and {len(failed_imports) - 5} more")
    
    # 生成建议
    print(f"\n" + "="*55)
    generate_recommendations(success_rate, failed_imports, env_status, hardware_info, test_results)
    
    print(f"\n🎯 Test completed! Check the results above.")
    print(f"💾 For detailed logs, check the output above.")
    
    # 返回测试是否通过（70%以上成功率且关键组件可用）
    critical_components_ok = (
        test_results.get('device_detection', False) and
        test_results.get('schema_completeness', False)
    )
    
    return success_rate >= 70 and critical_components_ok

# 在文件末尾添加快速测试选项

def quick_test():
    """快速测试 - 只检查关键功能"""
    print("🚀 Quick Environment Test")
    print("=" * 40)
    
    project_root, src_root = setup_project_paths()
    
    # 只测试关键导入
    critical_imports = [
        ("utils.device_detector", "device_detector"),
        ("schemas", "VideoSubmitRequest"), 
        ("pipelines", "get_available_pipelines")
    ]
    
    failed = 0
    for module_name, item_name in critical_imports:
        try:
            module = __import__(module_name, fromlist=[item_name])
            getattr(module, item_name)
            print(f"✅ {module_name}.{item_name}")
        except Exception as e:
            print(f"❌ {module_name}.{item_name}: {e}")
            failed += 1
    
    # 快速设备检测
    try:
        from utils.device_detector import device_detector
        device_type, device_count = device_detector.detect_device()
        print(f"✅ Device: {device_type.value} x {device_count}")
    except Exception as e:
        print(f"❌ Device detection: {e}")
        failed += 1
    
    if failed == 0:
        print("🎉 Quick test PASSED! System ready.")
        return True
    else:
        print(f"💥 Quick test FAILED! {failed} issues found.")
        print("   Run 'python3 tests/test_env.py' for detailed diagnosis.")
        return False

# 修改 main 函数支持快速模式
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Environment Test Tool')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    args = parser.parse_args()
    
    try:
        print(f"FastAPI Multi-GPU I2V Environment Tester v1.0.0")
        print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if args.quick:
            success = quick_test()
        else:
            success = test_imports()
        
        print(f"\nTest completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)