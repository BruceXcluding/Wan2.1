#!/usr/bin/env python3
"""
配置生成器工具
根据硬件环境和需求自动生成最优配置
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HardwareInfo:
    """硬件信息"""
    device_type: str
    device_count: int
    total_memory: float
    system_memory: float
    cpu_count: int
    backend: str

@dataclass 
class ServiceConfig:
    """服务配置"""
    # 模型配置
    model_path: str
    t5_cpu: bool
    dit_fsdp: bool
    t5_fsdp: bool
    vae_parallel: bool
    cfg_size: int
    ulysses_size: int
    
    # 性能优化
    use_attentioncache: bool
    cache_start_step: int
    cache_interval: int
    cache_end_step: int
    
    # 业务配置
    max_concurrent_tasks: int
    task_timeout: int
    server_port: int
    
    # 环境变量
    environment_vars: Dict[str, str]

class ConfigGenerator:
    """配置生成器"""
    
    def __init__(self):
        self.templates = {
            "development": {
                "t5_cpu": True,
                "dit_fsdp": False,
                "max_concurrent_tasks": 1,
                "task_timeout": 1800,
                "use_attentioncache": False
            },
            "production": {
                "t5_cpu": True,
                "dit_fsdp": True,
                "max_concurrent_tasks": 3,
                "task_timeout": 2400,
                "use_attentioncache": True
            },
            "testing": {
                "t5_cpu": True,
                "dit_fsdp": False,
                "max_concurrent_tasks": 1,
                "task_timeout": 600,
                "use_attentioncache": False
            }
        }
    
    def detect_hardware(self) -> HardwareInfo:
        """检测硬件信息"""
        try:
            from utils.device_detector import device_detector
            device_type, device_count = device_detector.detect_device()
            
            return HardwareInfo(
                device_type=device_type.value,
                device_count=device_count,
                total_memory=self._get_device_memory(device_type.value),
                system_memory=self._get_system_memory(),
                cpu_count=os.cpu_count() or 1,
                backend="torch_npu" if device_type.value == "npu" else "torch"
            )
        except Exception as e:
            logger.warning(f"Hardware detection failed: {e}")
            return HardwareInfo(
                device_type="cuda",
                device_count=1,
                total_memory=8.0,
                system_memory=16.0,
                cpu_count=8,
                backend="torch"
            )
    
    def _get_device_memory(self, device_type: str) -> float:
        """获取设备内存大小（GB）"""
        try:
            if device_type == "cuda":
                import torch
                if torch.cuda.is_available():
                    return torch.cuda.get_device_properties(0).total_memory / 1024**3
            elif device_type == "npu":
                import torch_npu
                if torch_npu.npu.is_available():
                    return 32.0  # 假设NPU内存
            return 8.0  # 默认值
        except:
            return 8.0
    
    def _get_system_memory(self) -> float:
        """获取系统内存大小（GB）"""
        try:
            import psutil
            return psutil.virtual_memory().total / 1024**3
        except:
            return 16.0  # 默认值
    
    def generate_config(self, 
                       template: str = "production",
                       model_path: Optional[str] = None,
                       server_port: int = 8088,
                       custom_options: Optional[Dict[str, Any]] = None) -> ServiceConfig:
        """生成服务配置"""
        
        # 检测硬件
        hw = self.detect_hardware()
        print(f"🔍 Detected: {hw.device_type} x {hw.device_count} ({hw.total_memory:.1f}GB)")
        
        # 获取模板配置
        base_config = self.templates.get(template, self.templates["production"]).copy()
        
        # 根据硬件优化配置
        if hw.device_count > 1:
            base_config["dit_fsdp"] = True
            base_config["vae_parallel"] = True
            base_config["max_concurrent_tasks"] = min(hw.device_count, 4)
        
        # 内存优化
        if hw.total_memory < 16:
            base_config["t5_cpu"] = True
            base_config["max_concurrent_tasks"] = 1
        
        # 应用自定义选项
        if custom_options:
            base_config.update(custom_options)
        
        # 生成环境变量
        env_vars = self._generate_environment_vars(hw, base_config.get("t5_cpu", True))
        
        return ServiceConfig(
            model_path=model_path or "/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P",
            t5_cpu=base_config.get("t5_cpu", True),
            dit_fsdp=base_config.get("dit_fsdp", True),
            t5_fsdp=base_config.get("t5_fsdp", False),
            vae_parallel=base_config.get("vae_parallel", True),
            cfg_size=base_config.get("cfg_size", 1),
            ulysses_size=base_config.get("ulysses_size", 1),
            use_attentioncache=base_config.get("use_attentioncache", False),
            cache_start_step=base_config.get("cache_start_step", 12),
            cache_interval=base_config.get("cache_interval", 4),
            cache_end_step=base_config.get("cache_end_step", 37),
            max_concurrent_tasks=base_config.get("max_concurrent_tasks", 2),
            task_timeout=base_config.get("task_timeout", 1800),
            server_port=server_port,
            environment_vars=env_vars
        )
    
    def _generate_environment_vars(self, hw: HardwareInfo, t5_cpu: bool) -> Dict[str, str]:
        """生成环境变量"""
        env_vars = {
            "T5_CPU": str(t5_cpu).lower(),
            "DIT_FSDP": "true",
            "VAE_PARALLEL": "true" if hw.device_count > 1 else "false",
            "OMP_NUM_THREADS": str(min(hw.cpu_count, 16)),
        }
        
        if hw.device_type == "npu":
            env_vars.update({
                "ASCEND_LAUNCH_BLOCKING": "0",
                "HCCL_TIMEOUT": "2400" if t5_cpu else "1800",
            })
        elif hw.device_type == "cuda":
            env_vars.update({
                "CUDA_LAUNCH_BLOCKING": "0",
                "NCCL_TIMEOUT": "1800",
            })
        
        return env_vars
    
    def print_config(self, config: ServiceConfig):
        """打印配置信息"""
        print(f"\n📋 Generated Configuration")
        print("=" * 60)
        
        print("Model Configuration:")
        print(f"  Model Path:        {config.model_path}")
        print(f"  T5 CPU Mode:       {config.t5_cpu}")
        print(f"  DiT FSDP:          {config.dit_fsdp}")
        print(f"  T5 FSDP:           {config.t5_fsdp}")
        print(f"  VAE Parallel:      {config.vae_parallel}")
        print(f"  CFG Size:          {config.cfg_size}")
        print(f"  Ulysses Size:      {config.ulysses_size}")
        
        print("\nPerformance Optimization:")
        print(f"  Attention Cache:   {config.use_attentioncache}")
        if config.use_attentioncache:
            print(f"  Cache Start Step:  {config.cache_start_step}")
            print(f"  Cache Interval:    {config.cache_interval}")
            print(f"  Cache End Step:    {config.cache_end_step}")
        
        print("\nBusiness Configuration:")
        print(f"  Max Concurrent:    {config.max_concurrent_tasks}")
        print(f"  Task Timeout:      {config.task_timeout}s")
        print(f"  Server Port:       {config.server_port}")
        
        print("\nEnvironment Variables:")
        for key, value in config.environment_vars.items():
            print(f"  {key}={value}")
    
    def export_env_file(self, config: ServiceConfig, filename: str):
        """导出环境变量文件"""
        with open(filename, 'w') as f:
            f.write(f"# Generated configuration\n")
            f.write(f"MODEL_CKPT_DIR={config.model_path}\n")
            f.write(f"SERVER_PORT={config.server_port}\n")
            f.write(f"MAX_CONCURRENT_TASKS={config.max_concurrent_tasks}\n")
            f.write(f"TASK_TIMEOUT={config.task_timeout}\n")
            
            for key, value in config.environment_vars.items():
                f.write(f"{key}={value}\n")
        
        print(f"✅ Environment file exported: {filename}")
    
    def export_json_config(self, config: ServiceConfig, filename: str):
        """导出JSON配置"""
        config_dict = asdict(config)
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ JSON config exported: {filename}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Configuration Generator for Multi-GPU I2V Service")
    parser.add_argument("--template", choices=["development", "production", "testing"],
                       default="production", help="Configuration template")
    parser.add_argument("--model-path", type=str,
                       help="Model checkpoint directory path")
    parser.add_argument("--port", type=int, default=8088,
                       help="Server port")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Output directory for generated files")
    parser.add_argument("--export-env", action="store_true",
                       help="Export environment file (.env)")
    parser.add_argument("--export-json", action="store_true",
                       help="Export JSON configuration")
    parser.add_argument("--custom", type=str,
                       help="Custom options as JSON string")
    
    args = parser.parse_args()
    
    try:
        generator = ConfigGenerator()
        
        # 解析自定义选项
        custom_options = None
        if args.custom:
            try:
                custom_options = json.loads(args.custom)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in custom options: {str(e)}")
                return 1
        
        # 生成配置
        config = generator.generate_config(
            template=args.template,
            model_path=args.model_path,
            server_port=args.port,
            custom_options=custom_options
        )
        
        # 打印配置
        generator.print_config(config)
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 导出文件
        if args.export_env:
            env_file = output_dir / f"config_{args.template}.env"
            generator.export_env_file(config, str(env_file))
        
        if args.export_json:
            json_file = output_dir / f"config_{args.template}.json"
            generator.export_json_config(config, str(json_file))
        
        print("\n🎉 Configuration generation completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Configuration generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())