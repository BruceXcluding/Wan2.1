#!/usr/bin/env python3
"""
内存监控调试工具
监控 NPU/GPU 内存使用情况，诊断内存泄漏和峰值使用
"""

import os
import sys
import time
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: float
    device_id: int
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization: float
    description: str = ""

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, device_type: str = "auto"):
        self.device_type = self._detect_device_type() if device_type == "auto" else device_type
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # 秒
        
        logger.info(f"Memory monitor initialized for device type: {self.device_type}")
    
    def _detect_device_type(self) -> str:
        """自动检测设备类型"""
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                return "npu"
        except ImportError:
            pass
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        
        return "cpu"
    
    def get_device_count(self) -> int:
        """获取设备数量"""
        try:
            if self.device_type == "npu":
                import torch_npu
                return torch_npu.npu.device_count()
            elif self.device_type == "cuda":
                import torch
                return torch.cuda.device_count()
            else:
                return 0
        except Exception as e:
            logger.warning(f"Failed to get device count: {e}")
            return 0
    
    def get_memory_info(self, device_id: int = 0) -> Tuple[float, float, float, float]:
        """获取设备内存信息 (allocated_mb, reserved_mb, free_mb, total_mb)"""
        try:
            if self.device_type == "npu":
                import torch_npu
                allocated = torch_npu.npu.memory_allocated(device_id) / 1024**2
                reserved = torch_npu.npu.memory_reserved(device_id) / 1024**2
                # NPU 总内存获取（可能需要特定 API）
                total = 32 * 1024  # 默认 32GB，实际应该通过 NPU API 获取
                free = total - reserved
                return allocated, reserved, free, total
                
            elif self.device_type == "cuda":
                import torch
                allocated = torch.cuda.memory_allocated(device_id) / 1024**2
                reserved = torch.cuda.memory_reserved(device_id) / 1024**2
                total = torch.cuda.get_device_properties(device_id).total_memory / 1024**2
                free = total - reserved
                return allocated, reserved, free, total
                
            else:
                return 0.0, 0.0, 0.0, 0.0
                
        except Exception as e:
            logger.warning(f"Failed to get memory info for device {device_id}: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def get_system_memory(self) -> Dict[str, float]:
        """获取系统内存信息"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / 1024**3,
                "available_gb": memory.available / 1024**3,
                "used_gb": memory.used / 1024**3,
                "percent": memory.percent
            }
        except ImportError:
            logger.warning("psutil not available, cannot get system memory info")
            return {}
        except Exception as e:
            logger.warning(f"Failed to get system memory info: {e}")
            return {}
    
    def take_snapshot(self, device_id: int = 0, description: str = "") -> MemorySnapshot:
        """拍摄内存快照"""
        allocated, reserved, free, total = self.get_memory_info(device_id)
        utilization = (reserved / total * 100) if total > 0 else 0
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            device_id=device_id,
            allocated_mb=allocated,
            reserved_mb=reserved,
            free_mb=free,
            total_mb=total,
            utilization=utilization,
            description=description
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def start_monitoring(self, device_id: int = 0, interval: float = 1.0):
        """开始连续监控"""
        self.monitor_interval = interval
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    self.take_snapshot(device_id, "auto_monitor")
                    time.sleep(self.monitor_interval)
                except Exception as e:
                    logger.error(f"Monitor loop error: {e}")
                    time.sleep(1)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started memory monitoring for device {device_id} (interval: {interval}s)")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("Memory monitoring stopped")
    
    def print_current_status(self, device_id: int = 0):
        """打印当前内存状态"""
        snapshot = self.take_snapshot(device_id, "status_check")
        
        print(f"\n📊 Memory Status - Device {device_id} ({self.device_type.upper()})")
        print("=" * 60)
        print(f"Allocated:   {snapshot.allocated_mb:8.1f} MB")
        print(f"Reserved:    {snapshot.reserved_mb:8.1f} MB")
        print(f"Free:        {snapshot.free_mb:8.1f} MB")
        print(f"Total:       {snapshot.total_mb:8.1f} MB")
        print(f"Utilization: {snapshot.utilization:8.1f}%")
        
        # 系统内存
        sys_mem = self.get_system_memory()
        if sys_mem:
            print(f"\n💻 System Memory")
            print("-" * 20)
            print(f"Used:        {sys_mem.get('used_gb', 0):8.1f} GB")
            print(f"Available:   {sys_mem.get('available_gb', 0):8.1f} GB") 
            print(f"Total:       {sys_mem.get('total_gb', 0):8.1f} GB")
            print(f"Utilization: {sys_mem.get('percent', 0):8.1f}%")
    
    def print_memory_analysis(self):
        """分析并打印内存使用情况"""
        if not self.snapshots:
            print("No memory snapshots available")
            return
        
        print(f"\n📈 Memory Analysis ({len(self.snapshots)} snapshots)")
        print("=" * 80)
        
        # 按设备分组
        device_snapshots = {}
        for snapshot in self.snapshots:
            device_id = snapshot.device_id
            if device_id not in device_snapshots:
                device_snapshots[device_id] = []
            device_snapshots[device_id].append(snapshot)
        
        for device_id, snapshots in device_snapshots.items():
            print(f"\nDevice {device_id} ({self.device_type.upper()}):")
            print("-" * 40)
            
            # 统计信息
            allocated_values = [s.allocated_mb for s in snapshots]
            reserved_values = [s.reserved_mb for s in snapshots]
            utilization_values = [s.utilization for s in snapshots]
            
            print(f"Allocated Memory:")
            print(f"  Min:     {min(allocated_values):8.1f} MB")
            print(f"  Max:     {max(allocated_values):8.1f} MB")
            print(f"  Avg:     {sum(allocated_values)/len(allocated_values):8.1f} MB")
            print(f"  Current: {allocated_values[-1]:8.1f} MB")
            
            print(f"Reserved Memory:")
            print(f"  Min:     {min(reserved_values):8.1f} MB")
            print(f"  Max:     {max(reserved_values):8.1f} MB")
            print(f"  Avg:     {sum(reserved_values)/len(reserved_values):8.1f} MB")
            print(f"  Current: {reserved_values[-1]:8.1f} MB")
            
            print(f"Utilization:")
            print(f"  Min:     {min(utilization_values):8.1f}%")
            print(f"  Max:     {max(utilization_values):8.1f}%")
            print(f"  Avg:     {sum(utilization_values)/len(utilization_values):8.1f}%")
            print(f"  Current: {utilization_values[-1]:8.1f}%")
            
            # 检查内存泄漏（保留内存持续增长）
            if len(reserved_values) >= 10:
                recent_trend = reserved_values[-5:]
                early_trend = reserved_values[:5]
                avg_recent = sum(recent_trend) / len(recent_trend)
                avg_early = sum(early_trend) / len(early_trend)
                
                if avg_recent > avg_early * 1.1:  # 增长超过10%
                    print(f"⚠️  Potential memory leak detected!")
                    print(f"   Early avg: {avg_early:.1f} MB")
                    print(f"   Recent avg: {avg_recent:.1f} MB")
                    print(f"   Growth: {((avg_recent - avg_early) / avg_early * 100):+.1f}%")
    
    def print_memory_timeline(self, max_entries: int = 20):
        """打印内存使用时间线"""
        if not self.snapshots:
            print("No memory snapshots available")
            return
        
        print(f"\n📅 Memory Timeline (last {max_entries} entries)")
        print("=" * 100)
        print(f"{'Time':<19} {'Device':<6} {'Allocated':<10} {'Reserved':<10} {'Util%':<6} {'Description':<20}")
        print("-" * 100)
        
        recent_snapshots = self.snapshots[-max_entries:]
        for snapshot in recent_snapshots:
            timestamp_str = datetime.fromtimestamp(snapshot.timestamp).strftime("%H:%M:%S.%f")[:-3]
            print(f"{timestamp_str:<19} {snapshot.device_id:<6} "
                  f"{snapshot.allocated_mb:8.1f}MB {snapshot.reserved_mb:8.1f}MB "
                  f"{snapshot.utilization:5.1f}% {snapshot.description:<20}")
    
    def export_csv(self, filepath: str):
        """导出内存数据到 CSV"""
        try:
            import csv
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 'datetime', 'device_id', 'allocated_mb', 
                    'reserved_mb', 'free_mb', 'total_mb', 'utilization', 'description'
                ])
                
                for snapshot in self.snapshots:
                    datetime_str = datetime.fromtimestamp(snapshot.timestamp).isoformat()
                    writer.writerow([
                        snapshot.timestamp, datetime_str, snapshot.device_id,
                        snapshot.allocated_mb, snapshot.reserved_mb, snapshot.free_mb,
                        snapshot.total_mb, snapshot.utilization, snapshot.description
                    ])
            
            logger.info(f"Memory data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
    
    def clear_snapshots(self):
        """清空快照历史"""
        self.snapshots.clear()
        logger.info("Memory snapshots cleared")

def test_memory_during_model_load():
    """测试模型加载期间的内存使用"""
    monitor = MemoryMonitor()
    
    print("🧪 Testing memory usage during model loading...")
    
    # 初始内存状态
    monitor.print_current_status(0)
    monitor.take_snapshot(0, "before_import")
    
    try:
        # 开始监控
        monitor.start_monitoring(0, 0.5)  # 每0.5秒监控一次
        
        # 模拟模型加载
        print("\n1. Importing torch modules...")
        import torch
        if monitor.device_type == "npu":
            import torch_npu
        monitor.take_snapshot(0, "after_torch_import")
        
        print("2. Loading WAN model...")
        import wan
        monitor.take_snapshot(0, "after_wan_import")
        
        print("3. Creating model config...")
        cfg = wan.configs.WAN_CONFIGS["i2v-14B"]
        monitor.take_snapshot(0, "after_config")
        
        print("4. Initializing model (this may take time)...")
        # 这里可以添加实际的模型初始化代码
        time.sleep(2)  # 模拟加载时间
        monitor.take_snapshot(0, "after_model_init")
        
        print("5. Testing inference...")
        time.sleep(1)  # 模拟推理
        monitor.take_snapshot(0, "after_inference")
        
        # 停止监控
        monitor.stop_monitoring()
        
        # 分析结果
        monitor.print_memory_analysis()
        monitor.print_memory_timeline()
        
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        monitor.stop_monitoring()
        return False
    
    return True

def stress_test_memory():
    """内存压力测试"""
    monitor = MemoryMonitor()
    
    print("💪 Memory stress test...")
    
    try:
        import torch
        if monitor.device_type == "npu":
            import torch_npu
            device = torch.device('npu:0')
        elif monitor.device_type == "cuda":
            device = torch.device('cuda:0')
        else:
            print("No GPU/NPU available for stress test")
            return False
        
        monitor.start_monitoring(0, 0.2)  # 高频监控
        
        tensors = []
        
        # 逐步分配内存
        for i in range(10):
            print(f"Allocating tensor {i+1}/10...")
            tensor_size = (1024, 1024, 50)  # 约200MB
            tensor = torch.randn(tensor_size, device=device)
            tensors.append(tensor)
            monitor.take_snapshot(0, f"alloc_tensor_{i+1}")
            time.sleep(0.5)
        
        print("All tensors allocated, holding for 5 seconds...")
        time.sleep(5)
        monitor.take_snapshot(0, "peak_usage")
        
        # 逐步释放内存
        for i, tensor in enumerate(tensors):
            print(f"Releasing tensor {i+1}/10...")
            del tensor
            if monitor.device_type == "npu":
                torch_npu.npu.empty_cache()
            elif monitor.device_type == "cuda":
                torch.cuda.empty_cache()
            monitor.take_snapshot(0, f"release_tensor_{i+1}")
            time.sleep(0.2)
        
        del tensors
        
        print("Final cleanup...")
        if monitor.device_type == "npu":
            torch_npu.npu.empty_cache()
        elif monitor.device_type == "cuda":
            torch.cuda.empty_cache()
        
        monitor.take_snapshot(0, "final_cleanup")
        monitor.stop_monitoring()
        
        # 分析结果
        monitor.print_memory_analysis()
        
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        monitor.stop_monitoring()
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Memory Monitoring Debug Tool")
    parser.add_argument("--device-type", choices=["auto", "npu", "cuda", "cpu"], 
                       default="auto", help="Device type to monitor")
    parser.add_argument("--device-id", type=int, default=0,
                       help="Device ID to monitor")
    parser.add_argument("--mode", choices=["status", "monitor", "model-test", "stress-test"],
                       default="status", help="Monitoring mode")
    parser.add_argument("--duration", type=int, default=10,
                       help="Monitoring duration in seconds")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Monitoring interval in seconds")
    parser.add_argument("--export", type=str, help="Export data to CSV file")
    
    args = parser.parse_args()
    
    print("🔍 Memory Monitoring Debug Tool")
    print("=" * 50)
    
    try:
        if args.mode == "status":
            # 显示当前状态
            monitor = MemoryMonitor(args.device_type)
            monitor.print_current_status(args.device_id)
            
        elif args.mode == "monitor":
            # 连续监控
            monitor = MemoryMonitor(args.device_type)
            print(f"Starting {args.duration}s monitoring (interval: {args.interval}s)...")
            
            monitor.start_monitoring(args.device_id, args.interval)
            time.sleep(args.duration)
            monitor.stop_monitoring()
            
            monitor.print_memory_analysis()
            monitor.print_memory_timeline()
            
            if args.export:
                monitor.export_csv(args.export)
                
        elif args.mode == "model-test":
            # 模型加载测试
            success = test_memory_during_model_load()
            if not success:
                return 1
                
        elif args.mode == "stress-test":
            # 压力测试
            success = stress_test_memory()
            if not success:
                return 1
        
        print("\n✅ Memory monitoring completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Memory monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())