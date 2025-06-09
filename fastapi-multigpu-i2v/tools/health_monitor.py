#!/usr/bin/env python3
"""
健康监控工具
实时监控服务健康状态、资源使用和性能指标
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import signal

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import aiohttp
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HealthMetrics:
    """健康指标"""
    timestamp: float
    status: str
    uptime: float
    response_time: float
    active_tasks: int
    total_requests: int
    success_rate: float
    error_rate: float
    device_utilization: float
    memory_usage: Dict[str, float]

class HealthMonitor:
    """健康监控器"""
    
    def __init__(self, base_url: str = "http://localhost:8088"):
        self.base_url = base_url.rstrip('/')
        self.metrics_history: List[HealthMetrics] = []
        self.monitoring = False
        self.alert_thresholds = {
            "response_time": 30.0,  # 30秒
            "error_rate": 10.0,     # 10%
            "memory_usage": 90.0,   # 90%
        }
        self.alert_callbacks = []
    
    def add_alert_callback(self, callback):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    async def check_health(self) -> Optional[HealthMetrics]:
        """检查服务健康状态"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                # 获取健康状态
                async with session.get(f"{self.base_url}/health", timeout=10) as response:
                    health_data = await response.json() if response.status == 200 else {}
                    response_time = time.time() - start_time
                
                # 获取详细指标
                metrics_data = {}
                try:
                    async with session.get(f"{self.base_url}/metrics", timeout=5) as metrics_response:
                        if metrics_response.status == 200:
                            metrics_data = await metrics_response.json()
                except Exception as e:
                    logger.warning(f"Failed to get metrics: {str(e)}")
                
                # 解析数据
                status = health_data.get("status", "unknown")
                uptime = health_data.get("uptime", 0)
                
                # 服务统计
                service_stats = metrics_data.get("service", {})
                resource_stats = metrics_data.get("resources", {})
                
                total_requests = service_stats.get("total_submitted", 0)
                total_completed = service_stats.get("total_completed", 0)
                total_failed = service_stats.get("total_failed", 0)
                active_tasks = resource_stats.get("concurrent_tasks", 0)
                
                # 计算成功率和错误率
                if total_requests > 0:
                    success_rate = (total_completed / total_requests) * 100
                    error_rate = (total_failed / total_requests) * 100
                else:
                    success_rate = 100.0
                    error_rate = 0.0
                
                # 设备利用率（模拟值，实际需要从具体监控API获取）
                device_utilization = 0.0
                
                # 内存使用情况
                memory_usage = {}
                
                metrics = HealthMetrics(
                    timestamp=time.time(),
                    status=status,
                    uptime=uptime,
                    response_time=response_time,
                    active_tasks=active_tasks,
                    total_requests=total_requests,
                    success_rate=success_rate,
                    error_rate=error_rate,
                    device_utilization=device_utilization,
                    memory_usage=memory_usage
                )
                
                return metrics
                
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return HealthMetrics(
                timestamp=time.time(),
                status="error",
                uptime=0,
                response_time=999.0,
                active_tasks=0,
                total_requests=0,
                success_rate=0.0,
                error_rate=100.0,
                device_utilization=0.0,
                memory_usage={}
            )
    
    def check_alerts(self, metrics: HealthMetrics):
        """检查告警条件"""
        alerts = []
        
        # 响应时间告警
        if metrics.response_time > self.alert_thresholds["response_time"]:
            alerts.append(f"High response time: {metrics.response_time:.2f}s")
        
        # 错误率告警
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(f"High error rate: {metrics.error_rate:.2f}%")
        
        # 服务状态告警
        if metrics.status != "healthy":
            alerts.append(f"Service unhealthy: {metrics.status}")
        
        # 内存使用告警
        for device, usage in metrics.memory_usage.items():
            if usage > self.alert_thresholds["memory_usage"]:
                alerts.append(f"High memory usage on {device}: {usage:.1f}%")
        
        # 触发告警回调
        if alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(metrics, alerts)
                except Exception as e:
                    logger.error(f"Alert callback failed: {str(e)}")
        
        return alerts
    
    def print_current_status(self, metrics: HealthMetrics):
        """打印当前状态"""
        timestamp_str = datetime.fromtimestamp(metrics.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # 状态颜色
        status_emoji = "✅" if metrics.status == "healthy" else "❌"
        
        print(f"\n🏥 Health Status - {timestamp_str}")
        print("=" * 60)
        print(f"Status:          {status_emoji} {metrics.status}")
        print(f"Uptime:          {self._format_duration(metrics.uptime)}")
        print(f"Response Time:   {metrics.response_time:.3f}s")
        print(f"Active Tasks:    {metrics.active_tasks}")
        print(f"Total Requests:  {metrics.total_requests}")
        print(f"Success Rate:    {metrics.success_rate:.1f}%")
        print(f"Error Rate:      {metrics.error_rate:.1f}%")
        
        if metrics.memory_usage:
            print("Memory Usage:")
            for device, usage in metrics.memory_usage.items():
                print(f"  {device}: {usage:.1f}%")
    
    def _format_duration(self, seconds: float) -> str:
        """格式化时间长度"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes:.0f}m {seconds % 60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def print_statistics(self, window_minutes: int = 60):
        """打印统计信息"""
        if not self.metrics_history:
            print("No metrics history available")
            return
        
        # 获取时间窗口内的数据
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            print(f"No metrics in the last {window_minutes} minutes")
            return
        
        print(f"\n📊 Statistics (last {window_minutes} minutes)")
        print("=" * 60)
        
        # 响应时间统计
        response_times = [m.response_time for m in recent_metrics]
        print(f"Response Time:")
        print(f"  Average:       {sum(response_times) / len(response_times):.3f}s")
        print(f"  Min:           {min(response_times):.3f}s")
        print(f"  Max:           {max(response_times):.3f}s")
        
        # 成功率统计
        success_rates = [m.success_rate for m in recent_metrics if m.total_requests > 0]
        if success_rates:
            print(f"Success Rate:")
            print(f"  Average:       {sum(success_rates) / len(success_rates):.1f}%")
            print(f"  Min:           {min(success_rates):.1f}%")
        
        # 活跃任务统计
        active_tasks = [m.active_tasks for m in recent_metrics]
        print(f"Active Tasks:")
        print(f"  Average:       {sum(active_tasks) / len(active_tasks):.1f}")
        print(f"  Max:           {max(active_tasks)}")
        
        # 健康状态统计
        status_counts = {}
        for m in recent_metrics:
            status_counts[m.status] = status_counts.get(m.status, 0) + 1
        
        print(f"Status Distribution:")
        for status, count in status_counts.items():
            percentage = (count / len(recent_metrics)) * 100
            print(f"  {status:10} {count:3d} ({percentage:.1f}%)")
    
    async def monitor_continuously(self, interval: int = 30, duration: Optional[int] = None):
        """持续监控"""
        print(f"🔄 Starting continuous monitoring (interval: {interval}s)")
        if duration:
            print(f"Will run for {duration} seconds")
        
        self.monitoring = True
        start_time = time.time()
        
        try:
            while self.monitoring:
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # 检查健康状态
                metrics = await self.check_health()
                if metrics:
                    self.metrics_history.append(metrics)
                    
                    # 检查告警
                    alerts = self.check_alerts(metrics)
                    
                    # 显示状态
                    self.print_current_status(metrics)
                    
                    if alerts:
                        print("\n🚨 ALERTS:")
                        for alert in alerts:
                            print(f"  ⚠️  {alert}")
                
                # 限制历史记录大小
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # 等待下次检查
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.monitoring = False
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
    
    def export_metrics(self, filepath: str, window_hours: int = 24):
        """导出指标数据"""
        try:
            # 获取时间窗口内的数据
            cutoff_time = time.time() - (window_hours * 3600)
            export_metrics = [
                asdict(m) for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
            
            data = {
                "metrics": export_metrics,
                "export_time": datetime.now().isoformat(),
                "window_hours": window_hours,
                "total_records": len(export_metrics)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Metrics exported to {filepath} ({len(export_metrics)} records)")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")

def alert_callback(metrics: HealthMetrics, alerts: List[str]):
    """默认告警回调"""
    timestamp = datetime.fromtimestamp(metrics.timestamp).strftime("%H:%M:%S")
    print(f"\n🚨 ALERT at {timestamp}:")
    for alert in alerts:
        print(f"  ⚠️  {alert}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Health Monitor Tool")
    parser.add_argument("--url", default="http://localhost:8088",
                       help="API base URL")
    parser.add_argument("--mode", choices=["check", "monitor", "stats"],
                       default="check", help="Monitoring mode")
    parser.add_argument("--interval", type=int, default=30,
                       help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int,
                       help="Monitoring duration in seconds")
    parser.add_argument("--export", type=str,
                       help="Export metrics to JSON file")
    parser.add_argument("--window", type=int, default=60,
                       help="Statistics window in minutes")
    
    args = parser.parse_args()
    
    print("🏥 FastAPI Multi-GPU I2V Health Monitor")
    print("=" * 50)
    print(f"Target URL: {args.url}")
    
    monitor = HealthMonitor(args.url)
    monitor.add_alert_callback(alert_callback)
    
    try:
        if args.mode == "check":
            # 单次健康检查
            metrics = await monitor.check_health()
            if metrics:
                monitor.print_current_status(metrics)
                alerts = monitor.check_alerts(metrics)
                if alerts:
                    print("\n🚨 ALERTS:")
                    for alert in alerts:
                        print(f"  ⚠️  {alert}")
        
        elif args.mode == "monitor":
            # 持续监控
            await monitor.monitor_continuously(args.interval, args.duration)
            
            # 显示统计信息
            if monitor.metrics_history:
                monitor.print_statistics(args.window)
        
        elif args.mode == "stats":
            # 仅显示统计信息（需要先有历史数据）
            print("Please run monitoring first to collect data")
        
        # 导出数据
        if args.export and monitor.metrics_history:
            monitor.export_metrics(args.export)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nHealth monitoring interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Health monitoring failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))