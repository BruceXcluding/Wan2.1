#!/usr/bin/env python3
"""
性能基准测试工具
测试视频生成服务的性能指标、吞吐量和资源使用情况
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import concurrent.futures
import statistics

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import aiohttp
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    timestamp: str

@dataclass
class LoadTestResult:
    """负载测试结果"""
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    throughput: float
    error_rate: float
    resource_usage: Dict[str, float]

class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8088"):
        self.base_url = base_url.rstrip('/')
        self.results = []
        self.test_image_url = "https://picsum.photos/1280/720"  # 测试图片
    
    def get_test_request(self) -> Dict:
        """获取测试请求数据"""
        return {
            "prompt": "A beautiful sunset over the ocean with gentle waves",
            "image_url": self.test_image_url,
            "image_size": "1280*720",
            "num_frames": 41,  # 较少帧数以加快测试
            "quality_preset": "fast",
            "seed": 42
        }
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=10) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def single_request_test(self, session: aiohttp.ClientSession) -> Tuple[bool, float, str]:
        """单个请求测试"""
        start_time = time.time()
        
        try:
            # 提交任务
            async with session.post(
                f"{self.base_url}/video/submit",
                json=self.get_test_request(),
                timeout=30
            ) as response:
                if response.status != 202:
                    return False, time.time() - start_time, f"Submit failed: {response.status}"
                
                result = await response.json()
                request_id = result.get("requestId")
                
                if not request_id:
                    return False, time.time() - start_time, "No request ID returned"
            
            # 轮询状态直到完成
            max_wait_time = 300  # 5分钟超时
            while time.time() - start_time < max_wait_time:
                async with session.post(
                    f"{self.base_url}/video/status",
                    json={"requestId": request_id},
                    timeout=10
                ) as response:
                    if response.status != 200:
                        return False, time.time() - start_time, f"Status check failed: {response.status}"
                    
                    status_result = await response.json()
                    status = status_result.get("status")
                    
                    if status == "Succeed":
                        return True, time.time() - start_time, "Success"
                    elif status in ["Failed", "Cancelled"]:
                        reason = status_result.get("reason", "Unknown error")
                        return False, time.time() - start_time, f"Task failed: {reason}"
                    
                    # 等待后继续轮询
                    await asyncio.sleep(2)
            
            # 超时
            return False, time.time() - start_time, "Timeout"
            
        except Exception as e:
            return False, time.time() - start_time, f"Exception: {str(e)}"
    
    async def response_time_test(self, num_requests: int = 10) -> BenchmarkResult:
        """响应时间测试"""
        print(f"🚀 Running response time test ({num_requests} requests)...")
        
        response_times = []
        successful = 0
        failed = 0
        errors = []
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                print(f"Request {i+1}/{num_requests}")
                success, response_time, error = await self.single_request_test(session)
                
                response_times.append(response_time)
                
                if success:
                    successful += 1
                    print(f"  ✅ Success in {response_time:.2f}s")
                else:
                    failed += 1
                    errors.append(error)
                    print(f"  ❌ Failed in {response_time:.2f}s: {error}")
        
        total_time = time.time() - start_time
        
        # 计算统计信息
        avg_response_time = statistics.mean(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        sorted_times = sorted(response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p99_index = int(len(sorted_times) * 0.99)
        p95_response_time = sorted_times[p95_index] if sorted_times else 0
        p99_response_time = sorted_times[p99_index] if sorted_times else 0
        
        rps = successful / total_time if total_time > 0 else 0
        error_rate = failed / num_requests * 100 if num_requests > 0 else 0
        
        result = BenchmarkResult(
            test_name="Response Time Test",
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=rps,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            error_rate=error_rate,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    async def concurrent_load_test(self, concurrent_users: int = 5, requests_per_user: int = 2) -> LoadTestResult:
        """并发负载测试"""
        print(f"🔥 Running concurrent load test ({concurrent_users} users, {requests_per_user} requests each)...")
        
        total_requests = concurrent_users * requests_per_user
        successful = 0
        failed = 0
        all_response_times = []
        
        start_time = time.time()
        
        async def user_simulation(user_id: int):
            """模拟单个用户"""
            user_response_times = []
            user_successful = 0
            user_failed = 0
            
            async with aiohttp.ClientSession() as session:
                for req_num in range(requests_per_user):
                    print(f"User {user_id+1} - Request {req_num+1}")
                    success, response_time, error = await self.single_request_test(session)
                    
                    user_response_times.append(response_time)
                    
                    if success:
                        user_successful += 1
                    else:
                        user_failed += 1
                        print(f"  User {user_id+1} request failed: {error}")
            
            return user_successful, user_failed, user_response_times
        
        # 并发执行所有用户
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 汇总结果
        for result in results:
            if isinstance(result, Exception):
                failed += requests_per_user
                print(f"User simulation failed: {str(result)}")
            else:
                user_successful, user_failed, user_times = result
                successful += user_successful
                failed += user_failed
                all_response_times.extend(user_times)
        
        total_time = time.time() - start_time
        avg_response_time = statistics.mean(all_response_times) if all_response_times else 0
        throughput = successful / total_time if total_time > 0 else 0
        error_rate = failed / total_requests * 100 if total_requests > 0 else 0
        
        # 获取资源使用情况
        resource_usage = await self.get_resource_usage()
        
        return LoadTestResult(
            concurrent_users=concurrent_users,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            avg_response_time=avg_response_time,
            throughput=throughput,
            error_rate=error_rate,
            resource_usage=resource_usage
        )
    
    async def get_resource_usage(self) -> Dict[str, float]:
        """获取服务器资源使用情况"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/metrics", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "active_tasks": data.get("resources", {}).get("concurrent_tasks", 0),
                            "total_tasks": data.get("service", {}).get("total_submitted", 0),
                            "uptime": data.get("system", {}).get("uptime", 0)
                        }
        except Exception as e:
            logger.warning(f"Failed to get resource usage: {str(e)}")
        
        return {}
    
    def print_benchmark_result(self, result: BenchmarkResult):
        """打印基准测试结果"""
        print(f"\n📊 {result.test_name} Results")
        print("=" * 60)
        print(f"Total Requests:      {result.total_requests}")
        print(f"Successful:          {result.successful_requests}")
        print(f"Failed:              {result.failed_requests}")
        print(f"Error Rate:          {result.error_rate:.2f}%")
        print(f"Total Time:          {result.total_time:.2f}s")
        print(f"Requests/Second:     {result.requests_per_second:.2f}")
        print(f"Avg Response Time:   {result.avg_response_time:.2f}s")
        print(f"Min Response Time:   {result.min_response_time:.2f}s")
        print(f"Max Response Time:   {result.max_response_time:.2f}s")
        print(f"95th Percentile:     {result.p95_response_time:.2f}s")
        print(f"99th Percentile:     {result.p99_response_time:.2f}s")
    
    def print_load_test_result(self, result: LoadTestResult):
        """打印负载测试结果"""
        print(f"\n🔥 Load Test Results")
        print("=" * 60)
        print(f"Concurrent Users:    {result.concurrent_users}")
        print(f"Total Requests:      {result.total_requests}")
        print(f"Successful:          {result.successful_requests}")
        print(f"Failed:              {result.failed_requests}")
        print(f"Error Rate:          {result.error_rate:.2f}%")
        print(f"Throughput:          {result.throughput:.2f} req/s")
        print(f"Avg Response Time:   {result.avg_response_time:.2f}s")
        
        if result.resource_usage:
            print(f"Active Tasks:        {result.resource_usage.get('active_tasks', 0)}")
            print(f"Total Tasks:         {result.resource_usage.get('total_tasks', 0)}")
    
    def export_results(self, filepath: str):
        """导出测试结果"""
        try:
            data = {
                "benchmark_results": [asdict(result) for result in self.results],
                "timestamp": datetime.now().isoformat(),
                "base_url": self.base_url
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Performance Benchmark Tool")
    parser.add_argument("--url", default="http://localhost:8088", 
                       help="API base URL")
    parser.add_argument("--response-test", type=int, default=5,
                       help="Number of requests for response time test")
    parser.add_argument("--load-test", action="store_true",
                       help="Run load test")
    parser.add_argument("--concurrent-users", type=int, default=3,
                       help="Number of concurrent users for load test")
    parser.add_argument("--requests-per-user", type=int, default=2,
                       help="Requests per user in load test")
    parser.add_argument("--export", type=str,
                       help="Export results to JSON file")
    parser.add_argument("--skip-health-check", action="store_true",
                       help="Skip initial health check")
    
    args = parser.parse_args()
    
    print("🚀 FastAPI Multi-GPU I2V Performance Benchmark")
    print("=" * 60)
    print(f"Target URL: {args.url}")
    
    benchmark = PerformanceBenchmark(args.url)
    
    try:
        # 健康检查
        if not args.skip_health_check:
            print("🏥 Checking service health...")
            if not await benchmark.health_check():
                print("❌ Service health check failed. Is the service running?")
                return 1
            print("✅ Service is healthy")
        
        # 响应时间测试
        if args.response_test > 0:
            result = await benchmark.response_time_test(args.response_test)
            benchmark.print_benchmark_result(result)
        
        # 负载测试
        if args.load_test:
            load_result = await benchmark.concurrent_load_test(
                args.concurrent_users, 
                args.requests_per_user
            )
            benchmark.print_load_test_result(load_result)
        
        # 导出结果
        if args.export:
            benchmark.export_results(args.export)
        
        print("\n🎉 Benchmark completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))