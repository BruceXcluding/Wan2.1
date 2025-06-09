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
sys.path.insert(0, str(PROJECT_ROOT))  # 外层utils
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # src模块

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class StructureVerifier:
    """项目结构验证器"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.errors = []
        self.warnings = []
        self.successes = []
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
        self.successes.append(message)
        logger.debug(message)
    
    def check_required_files(self) -> bool:
        """检查必需文件"""
        print("\n📄 Checking required files...")
        
        required_files = [
            # 核心 API
            "src/i2v_api.py",
            
            # 数据模型
            "src/schemas/__init__.py",
            "src/schemas/video.py",
            
            # 管道系统
            "src/pipelines/__init__.py",
            "src/pipelines/base_pipeline.py",
            "src/pipelines/npu_pipeline.py", 
            "src/pipelines/cuda_pipeline.py",
            "src/pipelines/pipeline_factory.py",
            
            # 工具模块
            "utils/__init__.py",
            "utils/device_detector.py",
            
            # 启动脚本
            "scripts/start_service_general.sh",
            "scripts/start_service_npu.sh",
            
            # 调试工具
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
        
        print(f"✅ Found {len(existing_files)} required files")
        return len(missing_files) == 0
    
    def check_directory_structure(self) -> bool:
        """检查目录结构"""
        print("\n🗂️  Checking directory structure...")
        
        required_dirs = [
            "src",
            "src/schemas", 
            "src/services",
            "src/pipelines",
            "scripts",
            "scripts/debug",
            "tools",
            "utils",
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
        
        # 检查并创建可选目录
        optional_dirs = ["generated_videos", "logs"]
        for dir_path in optional_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                try:
                    full_path.mkdir(exist_ok=True)
                    self.log_success(f"Created directory: {dir_path}")
                except Exception as e:
                    self.log_warning(f"Could not create directory {dir_path}: {e}")
        
        return len(missing_dirs) == 0
    
    def check_python_imports(self) -> bool:
        """检查 Python 导入"""
        print("\n🐍 Checking Python imports...")
        
        import_tests = [
            # 基础导入
            ("torch", "PyTorch"),
            ("fastapi", "FastAPI"),
            ("uvicorn", "Uvicorn"),
            ("pydantic", "Pydantic"),
            
            # 项目模块导入
            ("schemas", "Project schemas"),
            ("pipelines", "Project pipelines"),
            ("utils", "Project utils"),
        ]
        
        failed_imports = []
        
        for module_name, description in import_tests:
            self.total_checks += 1
            try:
                if module_name in ["schemas", "pipelines", "utils"]:
                    # 项目模块特殊处理
                    spec = importlib.util.spec_from_file_location(
                        module_name, 
                        self.project_root / "src" / module_name / "__init__.py"
                        if module_name in ["schemas", "pipelines"] 
                        else self.project_root / module_name / "__init__.py"
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                else:
                    # 外部库
                    __import__(module_name)
                
                self.log_success(f"Import successful: {description}")
            except ImportError as e:
                failed_imports.append((module_name, str(e)))
                self.log_error(f"Import failed: {description} - {str(e)}")
            except Exception as e:
                failed_imports.append((module_name, str(e)))
                self.log_warning(f"Import warning: {description} - {str(e)}")
        
        if failed_imports:
            print(f"\n❌ {len(failed_imports)} import failures:")
            for module, error in failed_imports:
                print(f"   - {module}: {error}")
        
        return len(failed_imports) == 0
    
    def check_dependencies(self) -> bool:
        """检查依赖包"""
        print("\n📦 Checking dependencies...")
        
        # 检查 requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            self.log_error("requirements.txt not found")
            return False
        
        try:
            # 读取要求
            with open(requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            print(f"Found {len(requirements)} requirements")
            
            # 检查关键依赖
            critical_deps = ["torch", "fastapi", "uvicorn", "pydantic"]
            missing_critical = []
            
            for dep in critical_deps:
                self.total_checks += 1
                if not any(dep in req for req in requirements):
                    missing_critical.append(dep)
                    self.log_error(f"Critical dependency missing: {dep}")
                else:
                    self.log_success(f"Critical dependency found: {dep}")
            
            return len(missing_critical) == 0
            
        except Exception as e:
            self.log_error(f"Error reading requirements.txt: {str(e)}")
            return False
    
    def check_configuration_files(self) -> bool:
        """检查配置文件"""
        print("\n⚙️  Checking configuration...")
        
        config_checks = []
        
        # 检查配置文件
        config_files = [
            ("src/config.py", False),  # 可选
            ("src/device_manager.py", False),  # 可选
        ]
        
        for config_file, required in config_files:
            self.total_checks += 1
            full_path = self.project_root / config_file
            if full_path.exists():
                # 检查文件是否为空
                if full_path.stat().st_size == 0:
                    self.log_warning(f"Configuration file is empty: {config_file}")
                else:
                    self.log_success(f"Configuration file exists: {config_file}")
                config_checks.append(True)
            else:
                if required:
                    self.log_error(f"Required configuration file missing: {config_file}")
                    config_checks.append(False)
                else:
                    self.log_warning(f"Optional configuration file missing: {config_file}")
                    config_checks.append(True)
        
        # 检查环境变量示例
        env_examples = [".env.example", "config/production.env.example"]
        for env_file in env_examples:
            full_path = self.project_root / env_file
            if full_path.exists():
                self.log_success(f"Environment example found: {env_file}")
            else:
                self.log_warning(f"Environment example missing: {env_file}")
        
        return all(config_checks)
    
    def check_runtime_environment(self) -> bool:
        """检查运行时环境"""
        print("\n🌍 Checking runtime environment...")
        
        env_checks = []
        
        # Python 版本检查
        self.total_checks += 1
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.log_success(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            env_checks.append(True)
        else:
            self.log_error(f"Python version too old: {python_version.major}.{python_version.minor}, requires >= 3.8")
            env_checks.append(False)
        
        # 设备检测
        self.total_checks += 1
        try:
            from utils.device_detector import device_detector
            device_type, device_count = device_detector.detect_device()
            self.log_success(f"Device detection: {device_type.value} x {device_count}")
            env_checks.append(True)
        except Exception as e:
            self.log_error(f"Device detection failed: {str(e)}")
            env_checks.append(False)
        
        # 关键环境变量检查
        important_env_vars = {
            "MODEL_CKPT_DIR": "Model checkpoint directory",
            "PYTHONPATH": "Python path"
        }
        
        for var, description in important_env_vars.items():
            value = os.environ.get(var)
            if value:
                self.log_success(f"Environment variable {var}: {value}")
            else:
                self.log_warning(f"Environment variable {var} not set ({description})")
        
        return all(env_checks)
    
    def check_permissions(self) -> bool:
        """检查文件权限"""
        print("\n🔐 Checking file permissions...")
        
        permission_issues = []
        
        # 检查脚本执行权限
        scripts_to_check = [
            "scripts/start_service_general.sh",
            "scripts/start_service_npu.sh",
        ]
        
        for script_path in scripts_to_check:
            self.total_checks += 1
            full_path = self.project_root / script_path
            if full_path.exists():
                if os.access(full_path, os.X_OK):
                    self.log_success(f"Script {script_path} is executable")
                else:
                    self.log_warning(f"Script {script_path} is not executable")
                    # 尝试修复权限
                    try:
                        full_path.chmod(0o755)
                        self.log_success(f"Fixed permissions for {script_path}")
                    except Exception as e:
                        self.log_error(f"Could not fix permissions for {script_path}: {e}")
                        permission_issues.append(script_path)
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
        print("\n📋 Verification Summary")
        print("=" * 80)
        
        # 打印结果
        passed = 0
        total = len(results)
        
        for check_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{check_name.title():<20} {status}")
            if result:
                passed += 1
        
        print("-" * 80)
        print(f"Total Checks: {self.total_checks}")
        print(f"Categories: {passed}/{total} passed")
        print(f"Successes: {len(self.successes)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Errors: {len(self.errors)}")
        
        # 显示错误和警告
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
    print("🔍 Project Structure Verifier")
    print("Checking project integrity and configuration...")
    
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