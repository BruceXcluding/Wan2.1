"""
简化的管道工厂
"""
import os
import sys
import logging
from typing import Dict, Any
from pathlib import Path

# 确保能正确找到项目根目录和 utils
def setup_project_paths():
    """设置项目路径，确保能找到所有模块"""
    current_file = Path(__file__).resolve()
    
    # 计算路径：src/pipelines/pipeline_factory.py -> 项目根目录
    project_root = current_file.parent.parent.parent
    src_root = current_file.parent.parent
    utils_root = project_root / "utils"
    
    # 要添加的路径列表
    paths_to_add = [
        str(project_root),      # 项目根目录
        str(src_root),          # src 目录  
        str(utils_root)         # utils 目录（直接添加）
    ]
    
    # 添加到 sys.path，避免重复
    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    return project_root, src_root, utils_root

# 设置路径
project_root, src_root, utils_root = setup_project_paths()

# 导入设备检测器 - 多种导入方式，确保成功
device_detector = None
DeviceType = None

try:
    # 方法1：标准导入
    from utils.device_detector import device_detector, DeviceType
except ImportError:
    try:
        # 方法2：直接导入
        import device_detector as dd
        device_detector = dd.device_detector
        DeviceType = dd.DeviceType
    except ImportError:
        try:
            # 方法3：从 utils 包导入
            from utils import device_detector as dd
            device_detector = dd.device_detector  
            DeviceType = dd.DeviceType
        except ImportError as e:
            # 如果都失败了，打印调试信息
            print(f"❌ Failed to import device_detector: {e}")
            print(f"Project root: {project_root}")
            print(f"Utils root: {utils_root}")
            print(f"Utils exists: {utils_root.exists()}")
            print(f"device_detector.py exists: {(utils_root / 'device_detector.py').exists()}")
            print(f"Current sys.path: {sys.path[:5]}")
            raise ImportError(f"Cannot import device_detector from any method: {e}")

# 导入基础管道类
from pipelines.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)

class PipelineFactory:
    """管道工厂类 - 简化版"""
    
    @staticmethod
    def get_available_devices() -> Dict[str, Any]:
        """获取可用设备信息"""
        try:
            if device_detector is None:
                raise RuntimeError("device_detector not properly imported")
                
            device_type, device_count = device_detector.detect_device()
            
            device_info = {
                "device_type": device_type.value,
                "device_count": device_count,
                "backend": "torch_npu" if device_type.value == "npu" else "torch"
            }
            
            logger.info(f"Detected devices: {device_info}")
            return device_info
            
        except Exception as e:
            logger.error(f"Failed to detect devices: {str(e)}")
            # 返回默认的 CPU 配置
            return {
                "device_type": "cpu",
                "device_count": 1,
                "backend": "torch"
            }
    
    @staticmethod
    def create_pipeline(**config) -> BasePipeline:
        """创建推理管道"""
        device_info = PipelineFactory.get_available_devices()
        device_type = device_info["device_type"]
        
        logger.info(f"Creating pipeline for device type: {device_type}")
        
        # 设置默认模型路径
        if 'ckpt_dir' not in config:
            config['ckpt_dir'] = os.environ.get(
                'MODEL_CKPT_DIR', 
                '/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P'
            )
        
        try:
            # 根据设备类型创建相应的管道
            if device_type == "npu":
                from pipelines.npu_pipeline import NPUPipeline
                return NPUPipeline(**config)
            elif device_type == "cuda":
                from pipelines.cuda_pipeline import CUDAPipeline
                return CUDAPipeline(**config)
            else:
                # CPU 或其他设备，使用 CUDA 管道作为默认
                from pipelines.cuda_pipeline import CUDAPipeline
                return CUDAPipeline(**config)
                
        except Exception as e:
            logger.error(f"Failed to create {device_type} pipeline: {e}")
            
            # 如果 NPU 管道创建失败，尝试 CUDA 作为备用
            if device_type == "npu":
                logger.warning("NPU pipeline failed, trying CUDA as fallback")
                try:
                    from pipelines.cuda_pipeline import CUDAPipeline
                    return CUDAPipeline(**config)
                except Exception as cuda_e:
                    logger.error(f"CUDA fallback also failed: {cuda_e}")
                    raise
            else:
                raise
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """简化的配置验证"""
        ckpt_dir = config.get('ckpt_dir')
        if ckpt_dir and not os.path.exists(ckpt_dir):
            logger.warning(f"Model checkpoint directory not found: {ckpt_dir}")
            # 不要因为模型路径不存在就返回 False，可能是需要下载
        
        return True
    
    # 修正分布式参数计算逻辑
    
    @staticmethod
    def get_optimal_config(device_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """根据设备信息生成最优配置"""
        if device_info is None:
            device_info = PipelineFactory.get_available_devices()
        
        device_type = device_info["device_type"]
        device_count = device_info["device_count"]
        
        # 基础配置
        config = {
            "device_type": device_type,
            "device_count": device_count,
            "task": "i2v-14B",
        }
        
        # 计算分布式推理参数 - 参照 generate.py 的逻辑
        if device_count > 1:
            ulysses_size, ring_size = PipelineFactory._calculate_optimal_parallel_config(device_count)
            
            # 验证配置 - 参照 generate.py:334
            assert ulysses_size * ring_size == device_count, \
                f"ulysses_size({ulysses_size}) * ring_size({ring_size}) != device_count({device_count})"
        else:
            ulysses_size = 1
            ring_size = 1
        
        # 根据设备类型调整配置
        if device_type == "npu":
            config.update({
                "t5_cpu": True,
                "dit_fsdp": True,
                "t5_fsdp": False,
                "init_on_cpu": True,
                "ulysses_size": ulysses_size,
                "ring_size": ring_size,
            })
        elif device_type == "cuda":
            config.update({
                "t5_cpu": device_count > 4,
                "dit_fsdp": True,
                "t5_fsdp": False,
                "init_on_cpu": True,
                "ulysses_size": ulysses_size,
                "ring_size": ring_size,
            })
        else:
            # CPU 配置
            config.update({
                "t5_cpu": True,
                "dit_fsdp": False,
                "t5_fsdp": False,
                "init_on_cpu": True,
                "ulysses_size": 1,
                "ring_size": 1,
            })
        
        return config
    
    @staticmethod
    def _calculate_optimal_parallel_config(device_count: int) -> tuple:
        """计算最优的并行配置"""
        if device_count == 1:
            return 1, 1
        
        # 对于 NPU (昇腾)，通常优先使用序列并行
        # 对于小规模设备，全部用 ulysses 并行
        if device_count <= 8:
            return device_count, 1
        
        # 对于大规模设备，寻找平衡的因子分解
        import math
        sqrt_count = int(math.sqrt(device_count))
        
        # 寻找最接近平方根的因子，优先较大的 ulysses_size
        for ulysses_size in range(sqrt_count, 0, -1):
            if device_count % ulysses_size == 0:
                ring_size = device_count // ulysses_size
                return ulysses_size, ring_size
        
        # 如果没有找到平衡的分解，回退到全 ulysses
        return device_count, 1
        
def get_available_pipelines() -> list:
    """获取可用的管道类型"""
    try:
        device_info = PipelineFactory.get_available_devices()
        device_type = device_info["device_type"]
        
        available = ["base"]  # 基础管道总是可用的
        
        if device_type == "npu":
            available.append("npu")
        elif device_type == "cuda": 
            available.append("cuda")
        
        return available
        
    except Exception as e:
        logger.error(f"Failed to get available pipelines: {e}")
        return ["base"]  # 返回最基本的选项

def create_default_pipeline(**kwargs) -> BasePipeline:
    """创建默认管道"""
    try:
        return PipelineFactory.create_pipeline(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create default pipeline: {e}")
        raise

def get_device_info() -> Dict[str, Any]:
    """获取设备信息的便捷函数"""
    return PipelineFactory.get_available_devices()

def get_optimal_config() -> Dict[str, Any]:
    """获取最优配置的便捷函数"""
    return PipelineFactory.get_optimal_config()

# 模块级别的便捷变量
try:
    _device_info = PipelineFactory.get_available_devices()
    DEVICE_TYPE = _device_info["device_type"]
    DEVICE_COUNT = _device_info["device_count"]
    BACKEND = _device_info["backend"]
except Exception:
    # 如果检测失败，使用默认值
    DEVICE_TYPE = "cpu"
    DEVICE_COUNT = 1
    BACKEND = "torch"

# 导出的公共接口
__all__ = [
    'PipelineFactory',
    'get_available_pipelines', 
    'create_default_pipeline',
    'get_device_info',
    'get_optimal_config',
    'DEVICE_TYPE',
    'DEVICE_COUNT', 
    'BACKEND'
]