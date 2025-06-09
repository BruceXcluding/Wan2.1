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
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HardwareInfo:
    """硬件信息"""
    device_type: str
    device_count: int
    memory_per_device: float  # GB
    total_memory: float      # GB
    cpu_cores: int
    system_memory: float     # GB

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
        self.hardware_info: Optional[HardwareInfo] = None
        self.config_templates = {
            "development": {
                "description": "开发环境 - 快速启动，较低资源占用",
                "max_concurrent_factor": 0.5,
                "use_optimization": False,
                "timeout_factor": 1.0
            },
            "production": {
                "description": "生产环境 - 最优性能和稳定性",
                "max_concurrent_factor": 1.0,
                "use_optimization": True,
                "timeout_factor": 1.5
            },
            "testing": {
                "description": "测试环境 - 快速响应，适合自动化测试",
                "max_concurrent_factor": 0.3,
                "use_optimization": False,
                "timeout_factor": 0.8
            }
        }
    
    def detect_hardware(self) -> HardwareInfo:
        """检测硬件信息"""
        print("🔍 Detecting hardware information...")
        
        try:
            # 检测设备类型和数量
            from utils.device_detector import device_detector
            device_type, device_count = device_detector.detect_device()
            
            # 检测设备内存
            memory_per_device = self._get_device_memory(device_type.value)
            total_memory = memory_per_device * device_count
            
            # 检测CPU信息
            cpu_cores = os.cpu_count() or 1
            
            # 检测系统内存
            system_memory = self._get_system_memory()
            
            hardware_info = HardwareInfo(
                device_type=device_type.value,
                device_count=device_count,
                memory_per_device=memory_per_device,
                total_memory=total_memory,
                cpu_cores=cpu_cores,
                system_memory=system_memory
            )
            
            self.hardware_info = hardware_info
            
            print(f"✅ Hardware detected:")
            print(f"   Device: {device_type.value} x {device_count}")
            print(f"   Memory: {memory_per_device:.1f}GB per device ({total_memory:.1f}GB total)")
            print(f"   CPU: {cpu_cores} cores")
            print(f"   System Memory: {system_memory:.1f}GB")
            
            return hardware_info
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {str(e)}")
            # 返回默认值
            return HardwareInfo(
                device_type="unknown",
                device_count=1,
                memory_per_device=32.0,
                total_memory=32.0,
                cpu_cores=16,
                system_memory=64.0
            )
    
    def _get_device_memory(self, device_type: str) -> float:
        """获取设备内存大小（GB）"""
        try:
            if device_type == "npu":
                # NPU 内存检测
                import torch_npu
                if torch_npu.npu.is_available():
                    # 尝试获取 NPU 内存信息（可能需要特定API）
                    return 32.0  # 默认 32GB，实际应通过 NPU API 获取
            elif device_type == "cuda":
                # CUDA 内存检测
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    return props.total_memory / (1024**3)  # 转换为GB
        except Exception as e:
            logger.warning(f"Failed to get device memory: {str(e)}")
        
        return 32.0  # 默认值
    
    def _get_system_memory(self) -> float:
        """获取系统内存大小（GB）"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # 尝试通过其他方式获取
            try:
                # Linux
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            # 提取数值（KB）并转换为GB
                            memory_kb = int(line.split()[1])
                            return memory_kb / (1024**2)
            except:
                pass
        except Exception as e:
            logger.warning(f"Failed to get system memory: {str(e)}")
        
        return 64.0  # 默认值
    
    def generate_config(self, 
                       template: str = "production",
                       model_path: Optional[str] = None,
                       server_port: int = 8088,
                       custom_options: Optional[Dict[str, Any]] = None) -> ServiceConfig:
        """生成服务配置"""
        
        if not self.hardware_info:
            self.detect_hardware()
        
        hw = self.hardware_info
        template_config = self.config_templates.get(template, self.config_templates["production"])
        
        print(f"🛠️  Generating {template} configuration...")
        print(f"   Template: {template_config['description']}")
        
        # 基础配置
        model_path = model_path or "/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"
        
        # 根据硬件选择 T5 CPU 模式
        t5_cpu = hw.device_type == "npu" or hw.memory_per_device < 24.0
        
        # 分布式配置
        dit_fsdp = hw.device_count > 1
        t5_fsdp = hw.device_count > 4 and not t5_cpu
        vae_parallel = hw.device_count > 1
        
        # 并行配置
        cfg_size = min(2, hw.device_count)
        ulysses_size = min(8, hw.device_count)
        
        # 性能优化
        use_optimization = template_config["use_optimization"]
        use_attentioncache = use_optimization and hw.memory_per_device >= 32.0
        
        # 业务配置
        base_concurrent = 2 if t5_cpu else min(5, int(hw.total_memory / 16))
        max_concurrent_tasks = int(base_concurrent * template_config["max_concurrent_factor"])
        max_concurrent_tasks = max(1, max_concurrent_tasks)
        
        base_timeout = 1800 if not t5_cpu else 2400
        task_timeout = int(base_timeout * template_config["timeout_factor"])
        
        # 环境变量
        environment_vars = self._generate_environment_vars(hw, t5_cpu)
        
        config = ServiceConfig(
            model_path=model_path,
            t5_cpu=t5_cpu,
            dit_fsdp=dit_fsdp,
            t5_fsdp=t5_fsdp,
            vae_parallel=vae_parallel,
            cfg_size=cfg_size,
            ulysses_size=ulysses_size,
            use_attentioncache=use_attentioncache,
            cache_start_step=12 if use_attentioncache else 0,
            cache_interval=4 if use_attentioncache else 1,
            cache_end_step=37 if use_attentioncache else 40,
            max_concurrent_tasks=max_concurrent_tasks,
            task_timeout=task_timeout,
            server_port=server_port,
            environment_vars=environment_vars
        )
        
        # 应用自定义选项
        if custom_options:
            for key, value in custom_options.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"   Custom: {key} = {value}")
        
        return config
    
    def _generate_environment_vars(self, hw: HardwareInfo, t5_cpu: bool) -> Dict[str, str]:
        """生成环境变量"""
        env_vars = {
            # 通用环境变量
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": "/workspace/Wan2.1:${PYTHONPATH}",
            "OMP_NUM_THREADS": str(min(16, hw.cpu_cores)),
            "MKL_NUM_THREADS": str(min(16, hw.cpu_cores)),
            "OPENBLAS_NUM_THREADS": str(min(16, hw.cpu_cores)),
        }
        
        # 设备特定环境变量
        if hw.device_type == "npu":
            env_vars.update({
                "ALGO": "0",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "TASK_QUEUE_ENABLE": "2",
                "CPU_AFFINITY_CONF": "1",
                "HCCL_TIMEOUT": "3600",
                "HCCL_CONNECT_TIMEOUT": "1200",
                "HCCL_BUFFSIZE": "256",
                "ASCEND_LAUNCH_BLOCKING": "0",
                "ASCEND_GLOBAL_LOG_LEVEL": "1",
            })
            
            # 设置可见设备
            if hw.device_count > 1:
                device_list = ",".join(str(i) for i in range(hw.device_count))
                env_vars["NPU_VISIBLE_DEVICES"] = device_list
                env_vars["ASCEND_RT_VISIBLE_DEVICES"] = device_list
        
        elif hw.device_type == "cuda":
            env_vars.update({
                "NCCL_TIMEOUT": "3600",
            })
            
            # 设置可见设备
            if hw.device_count > 1:
                device_list = ",".join(str(i) for i in range(hw.device_count))
                env_vars["CUDA_VISIBLE_DEVICES"] = device_list
        
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
    
    def export_env_file(self, config: ServiceConfig, filepath: str):
        """导出环境变量文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# Auto-generated environment configuration\n")
                f.write(f"# Generated by ConfigGenerator\n\n")
                
                # 模型配置
                f.write("# Model Configuration\n")
                f.write(f'MODEL_CKPT_DIR="{config.model_path}"\n')
                f.write(f'T5_CPU="{str(config.t5_cpu).lower()}"\n')
                f.write(f'DIT_FSDP="{str(config.dit_fsdp).lower()}"\n')
                f.write(f'T5_FSDP="{str(config.t5_fsdp).lower()}"\n')
                f.write(f'VAE_PARALLEL="{str(config.vae_parallel).lower()}"\n')
                f.write(f'CFG_SIZE="{config.cfg_size}"\n')
                f.write(f'ULYSSES_SIZE="{config.ulysses_size}"\n\n')
                
                # 性能优化
                f.write("# Performance Optimization\n")
                f.write(f'USE_ATTENTION_CACHE="{str(config.use_attentioncache).lower()}"\n')
                f.write(f'CACHE_START_STEP="{config.cache_start_step}"\n')
                f.write(f'CACHE_INTERVAL="{config.cache_interval}"\n')
                f.write(f'CACHE_END_STEP="{config.cache_end_step}"\n\n')
                
                # 业务配置
                f.write("# Business Configuration\n")
                f.write(f'MAX_CONCURRENT_TASKS="{config.max_concurrent_tasks}"\n')
                f.write(f'TASK_TIMEOUT="{config.task_timeout}"\n')
                f.write(f'SERVER_PORT="{config.server_port}"\n\n')
                
                # 环境变量
                f.write("# Environment Variables\n")
                for key, value in config.environment_vars.items():
                    f.write(f'{key}="{value}"\n')
            
            print(f"✅ Environment file exported to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export environment file: {str(e)}")
    
    def export_json(self, config: ServiceConfig, filepath: str):
        """导出 JSON 配置文件"""
        try:
            config_dict = asdict(config)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✅ JSON config exported to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export JSON config: {str(e)}")
    
    def generate_start_script(self, config: ServiceConfig, filepath: str):
        """生成启动脚本"""
        try:
            script_content = f"""#!/bin/bash
# Auto-generated service start script
# Generated by ConfigGenerator

set -e

echo "🚀 Starting FastAPI Multi-GPU I2V Service"
echo "=========================================="

# Check if model directory exists
if [ ! -d "{config.model_path}" ]; then
    echo "❌ Model directory not found: {config.model_path}"
    exit 1
fi

# Set environment variables
export MODEL_CKPT_DIR="{config.model_path}"
export T5_CPU="{str(config.t5_cpu).lower()}"
export DIT_FSDP="{str(config.dit_fsdp).lower()}"
export T5_FSDP="{str(config.t5_fsdp).lower()}"
export VAE_PARALLEL="{str(config.vae_parallel).lower()}"
export CFG_SIZE="{config.cfg_size}"
export ULYSSES_SIZE="{config.ulysses_size}"
export USE_ATTENTION_CACHE="{str(config.use_attentioncache).lower()}"
export CACHE_START_STEP="{config.cache_start_step}"
export CACHE_INTERVAL="{config.cache_interval}"
export CACHE_END_STEP="{config.cache_end_step}"
export MAX_CONCURRENT_TASKS="{config.max_concurrent_tasks}"
export TASK_TIMEOUT="{config.task_timeout}"
export SERVER_PORT="{config.server_port}"

# Set system environment variables
"""
            
            for key, value in config.environment_vars.items():
                script_content += f'export {key}="{value}"\n'
            
            hw_info = self.hardware_info
            if hw_info and hw_info.device_count > 1:
                # 分布式启动
                script_content += f"""
# Start distributed service
echo "Starting {hw_info.device_count}-device distributed service..."
torchrun \\
    --nproc_per_node={hw_info.device_count} \\
    --master_addr=127.0.0.1 \\
    --master_port=29500 \\
    --nnodes=1 \\
    --node_rank=0 \\
    src/i2v_api.py
"""
            else:
                # 单卡启动
                script_content += """
# Start single device service
echo "Starting single-device service..."
python3 src/i2v_api.py
"""
            
            script_content += """
echo "Service stopped."
"""
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # 设置执行权限
            os.chmod(filepath, 0o755)
            
            print(f"✅ Start script generated: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to generate start script: {str(e)}")

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
    parser.add_argument("--generate-script", action="store_true",
                       help="Generate start script")
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
            generator.export_json(config, str(json_file))
        
        if args.generate_script:
            script_file = output_dir / f"start_{args.template}.sh"
            generator.generate_start_script(config, str(script_file))
        
        print(f"\n🎉 Configuration generation completed!")
        print(f"Template: {args.template}")
        print(f"Output directory: {output_dir.absolute()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Configuration generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())