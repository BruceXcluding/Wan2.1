#!/usr/bin/env python3
"""
项目结构验证工具
验证项目文件结构完整性、依赖关系和配置正确性
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import importlib.util

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class StructureVerifier:
    """项目结构验证器"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
    
    def log_error(self, message: str):
        """记录错误"""
        self.errors.append(message)
        logger.error(message)
    
    def log_warning(self, message: str):
        """记录警告"""
        self.warnings.append(message)
        logger.warning(message)
    
    def log_success(self, message: str):
        """记录成功"""
        self.success_count += 1
        logger.info(f"✅ {message}")
    
    def check_required_files(self) -> bool:
        """检查必需文件"""
        print("🔍 Checking required files...")
        
        required_files = [
            # 核心源码
            "src/__init__.py",
            "src/i2v_api.py",
            "src/schemas/__init__.py",
            "src/schemas/video.py",
            "src/services/__init__.py", 
            "src/services/video_service.py",
            "src/pipelines/__init__.py",
            "src/pipelines/base_pipeline.py",
            "src/pipelines/pipeline_factory.py",
            "src/utils/__init__.py",
            "src/utils/device_detector.py",
            
            # 脚本和工具
            "scripts/start_service_general.sh",
            "scripts/debug/debug_t5_warmup.py",
            "scripts/debug/debug_memory.py",
            "scripts/debug/debug_device.py",
            "scripts/debug/debug_pipeline.py",
            
            # 配置文件
            "requirements.txt",
            "README.md",
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in required_files:
            self.total_checks += 1
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
                self.log_success(f"Found {file_path}")
            else:
                missing_files.append(file_path)
                self.log_error(f"Missing required file: {file_path}")
        
        if missing_files:
            print(f"\n❌ Missing {len(missing_files)} required files:")
            for f in missing_files:
                print(f"   - {f}")
        
        return len(missing_files) == 0
    
    def check_directory_structure(self) -> bool:
        """检查目录结构"""
        print("\n🗂️  Checking directory structure...")
        
        required_dirs = [
            "src",
            "src/schemas", 
            "src/services",
            "src/pipelines",
            "src/utils",
            "scripts",
            "scripts/debug",
            "tools",
            "generated_videos",
        ]
        
        missing_dirs = []
        
        for dir_path in required_dirs:
            self.total_checks += 1
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.log_success(f"Directory exists: {dir_path}")
            else:
                missing_dirs.append(dir_path)
                self.log_error(f"Missing directory: {dir_path}")
        
        # 检查可选目录
        optional_dirs = ["tests", "docs", "logs", "requirements"]
        for dir_path in optional_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.log_success(f"Optional directory exists: {dir_path}")
            else:
                self.log_warning(f"Optional directory missing: {dir_path}")
        
        return len(missing_dirs) == 0
    
    def check_python_imports(self) -> bool:
        """检查 Python 导入"""
        print("\n🐍 Checking Python imports...")
        
        import_tests = [
            ("schemas.video", ["VideoSubmitRequest", "VideoStatusResponse"]),
            ("utils.device_detector", ["device_detector", "DeviceType"]),
            ("services.video_service", ["VideoService"]),
            ("pipelines.pipeline_factory", ["PipelineFactory"]),
        ]
        
        failed_imports = []
        
        for module_name, expected_attrs in import_tests:
            self.total_checks += 1
            try:
                module = importlib.import_module(module_name)
                
                # 检查预期属性
                missing_attrs = []
                for attr in expected_attrs:
                    if not hasattr(module, attr):
                        missing_attrs.append(attr)
                
                if missing_attrs:
                    self.log_error(f"Module {module_name} missing attributes: {missing_attrs}")
                    failed_imports.append(module_name)
                else:
                    self.log_success(f"Module {module_name} imports correctly")
                    
            except ImportError as e:
                self.log_error(f"Failed to import {module_name}: {str(e)}")
                failed_imports.append(module_name)
            except Exception as e:
                self.log_error(f"Error testing {module_name}: {str(e)}")
                failed_imports.append(module_name)
        
        return len(failed_imports) == 0
    
    def check_dependencies(self) -> bool:
        """检查依赖包"""
        print("\n📦 Checking dependencies...")
        
        # 基础依赖
        required_packages = [
            "fastapi",
            "uvicorn", 
            "pydantic",
            "torch",
            "asyncio",
            "pathlib",
        ]
        
        # 设备特定依赖（可选）
        optional_packages = [
            "torch_npu",
            "psutil",
            "opencv-python",
        ]
        
        missing_required = []
        missing_optional = []
        
        for package in required_packages:
            self.total_checks += 1
            try:
                importlib.import_module(package)
                self.log_success(f"Required package available: {package}")
            except ImportError:
                missing_required.append(package)
                self.log_error(f"Missing required package: {package}")
        
        for package in optional_packages:
            try:
                importlib.import_module(package)
                self.log_success(f"Optional package available: {package}")
            except ImportError:
                missing_optional.append(package)
                self.log_warning(f"Optional package missing: {package}")
        
        if missing_optional:
            print(f"\n⚠️  Missing {len(missing_optional)} optional packages (may affect functionality):")
            for p in missing_optional:
                print(f"   - {p}")
        
        return len(missing_required) == 0
    
    def check_configuration_files(self) -> bool:
        """检查配置文件"""
        print("\n⚙️  Checking configuration files...")
        
        config_checks = []
        
        # 检查 requirements.txt
        self.total_checks += 1
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    requirements = f.read().strip()
                    if requirements:
                        self.log_success("requirements.txt exists and not empty")
                    else:
                        self.log_warning("requirements.txt exists but is empty")
            except Exception as e:
                self.log_error(f"Error reading requirements.txt: {str(e)}")
                config_checks.append("requirements.txt")
        else:
            self.log_error("requirements.txt not found")
            config_checks.append("requirements.txt")
        
        # 检查启动脚本
        self.total_checks += 1
        start_script = self.project_root / "scripts" / "start_service_general.sh"
        if start_script.exists():
            if os.access(start_script, os.X_OK):
                self.log_success("start_service_general.sh exists and is executable")
            else:
                self.log_warning("start_service_general.sh exists but not executable")
        else:
            self.log_error("start_service_general.sh not found")
            config_checks.append("start_service_general.sh")
        
        return len(config_checks) == 0
    
    def check_runtime_environment(self) -> bool:
        """检查运行时环境"""
        print("\n🔧 Checking runtime environment...")
        
        env_issues = []
        
        # 检查 Python 版本
        self.total_checks += 1
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.log_success(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            self.log_error(f"Python version too old: {python_version.major}.{python_version.minor}.{python_version.micro} (requires >= 3.8)")
            env_issues.append("python_version")
        
        # 检查设备可用性
        self.total_checks += 1
        try:
            from utils.device_detector import device_detector
            device_type, device_count = device_detector.detect_device()
            self.log_success(f"Device detection: {device_type.value} x {device_count}")
        except Exception as e:
            self.log_error(f"Device detection failed: {str(e)}")
            env_issues.append("device_detection")
        
        # 检查端口可用性
        self.total_checks += 1
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 8088))
            sock.close()
            
            if result == 0:
                self.log_warning("Port 8088 is already in use")
            else:
                self.log_success("Port 8088 is available")
        except Exception as e:
            self.log_warning(f"Could not check port availability: {str(e)}")
        
        return len(env_issues) == 0
    
    def check_permissions(self) -> bool:
        """检查文件权限"""
        print("\n🔐 Checking file permissions...")
        
        permission_issues = []
        
        # 检查脚本执行权限
        scripts_to_check = [
            "scripts/start_service_general.sh",
            "scripts/debug/debug_t5_warmup.py",
            "scripts/debug/debug_memory.py",
        ]
        
        for script_path in scripts_to_check:
            self.total_checks += 1
            full_path = self.project_root / script_path
            if full_path.exists():
                if os.access(full_path, os.X_OK):
                    self.log_success(f"Script {script_path} is executable")
                else:
                    self.log_warning(f"Script {script_path} is not executable")
            else:
                self.log_error(f"Script {script_path} not found")
                permission_issues.append(script_path)
        
        # 检查写入权限
        write_dirs = ["generated_videos", "logs"]
        for dir_name in write_dirs:
            self.total_checks += 1
            dir_path = self.project_root / dir_name
            try:
                dir_path.mkdir(exist_ok=True)
                test_file = dir_path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                self.log_success(f"Directory {dir_name} is writable")
            except Exception as e:
                self.log_error(f"Directory {dir_name} is not writable: {str(e)}")
                permission_issues.append(dir_name)
        
        return len(permission_issues) == 0
    
    def run_all_checks(self) -> Dict[str, bool]:
        """运行所有检查"""
        print("🚀 FastAPI Multi-GPU I2V Project Structure Verification")
        print("=" * 80)
        
        results = {}
        
        # 运行各项检查
        results["files"] = self.check_required_files()
        results["directories"] = self.check_directory_structure()
        results["imports"] = self.check_python_imports()
        results["dependencies"] = self.check_dependencies()
        results["configuration"] = self.check_configuration_files()
        results["environment"] = self.check_runtime_environment()
        results["permissions"] = self.check_permissions()
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """打印检查总结"""
        print("\n" + "=" * 80)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 80)
        
        passed_checks = sum(results.values())
        total_categories = len(results)
        
        print(f"Categories passed: {passed_checks}/{total_categories}")
        print(f"Individual checks: {self.success_count}/{self.total_checks}")
        
        print("\nCategory Results:")
        for category, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {category:15} {status}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   - {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        print("\n" + "=" * 80)
        
        if all(results.values()) and len(self.errors) == 0:
            print("🎉 PROJECT VERIFICATION PASSED!")
            print("✅ Your project structure is ready for deployment")
        else:
            print("❌ PROJECT VERIFICATION FAILED!")
            print("⚠️  Please fix the issues above before proceeding")
        
        return all(results.values()) and len(self.errors) == 0

def main():
    """主函数"""
    verifier = StructureVerifier()
    
    try:
        results = verifier.run_all_checks()
        success = verifier.print_summary(results)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nVerification interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Verification failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())