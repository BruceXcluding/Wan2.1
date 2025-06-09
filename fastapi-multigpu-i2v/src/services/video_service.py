import asyncio
import time
import uuid
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from ..schemas.video import (
    VideoSubmitRequest, 
    TaskStatus, 
    VideoResults,
    VideoStatusResponse
)

logger = logging.getLogger(__name__)

class VideoService:
    """视频生成服务"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_timeout = 3600  # 1小时超时
        self._lock = asyncio.Lock()
        
        # 统计信息
        self.stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_cancelled": 0,
            "active_tasks": 0
        }
    
    async def submit_video_task(self, request: VideoSubmitRequest) -> str:
        """提交视频生成任务"""
        task_id = uuid.uuid4().hex
        
        async with self._lock:
            # 创建任务记录
            self.tasks[task_id] = {
                "id": task_id,
                "status": TaskStatus.IN_QUEUE,
                "request": request,
                "created_at": time.time(),
                "updated_at": time.time(),
                "progress": 0.0,
                "queue_position": len([t for t in self.tasks.values() 
                                     if t["status"] in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]]),
                "results": None,
                "reason": None
            }
            
            self.stats["total_submitted"] += 1
            self.stats["active_tasks"] += 1
        
        # 异步执行任务
        asyncio.create_task(self._execute_task(task_id))
        
        logger.info(f"Task {task_id} submitted to queue")
        return task_id
    
    async def _execute_task(self, task_id: str):
        """执行视频生成任务"""
        try:
            async with self._lock:
                if task_id not in self.tasks:
                    return
                
                task = self.tasks[task_id]
                if task["status"] == TaskStatus.CANCELLED:
                    return
                
                # 更新状态为进行中
                task["status"] = TaskStatus.IN_PROGRESS
                task["updated_at"] = time.time()
                task["progress"] = 0.1
            
            logger.info(f"Starting task {task_id}")
            
            # 调用管道生成视频
            request = task["request"]
            video_path = await self.pipeline.generate_video(request, task_id)
            
            # 检查任务是否被取消
            async with self._lock:
                if task_id not in self.tasks or self.tasks[task_id]["status"] == TaskStatus.CANCELLED:
                    # 清理生成的文件
                    try:
                        Path(video_path).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup cancelled task file: {e}")
                    return
            
            # 获取视频信息
            video_info = await self._get_video_info(video_path)
            
            # 更新任务状态为成功
            async with self._lock:
                if task_id in self.tasks and self.tasks[task_id]["status"] != TaskStatus.CANCELLED:
                    self.tasks[task_id].update({
                        "status": TaskStatus.SUCCEED,
                        "updated_at": time.time(),
                        "progress": 1.0,
                        "results": VideoResults(
                            video_url=f"/videos/{Path(video_path).name}",
                            video_path=video_path,
                            **video_info
                        ).dict()
                    })
                    
                    self.stats["total_completed"] += 1
                    self.stats["active_tasks"] -= 1
            
            logger.info(f"Task {task_id} completed successfully")
            
        except asyncio.CancelledError:
            # 任务被取消
            async with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].update({
                        "status": TaskStatus.CANCELLED,
                        "updated_at": time.time(),
                        "reason": "Task was cancelled"
                    })
                    self.stats["total_cancelled"] += 1
                    self.stats["active_tasks"] -= 1
            
            logger.info(f"Task {task_id} was cancelled")
            
        except Exception as e:
            # 任务失败
            error_msg = str(e)
            logger.error(f"Task {task_id} failed: {error_msg}")
            
            async with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].update({
                        "status": TaskStatus.FAILED,
                        "updated_at": time.time(),
                        "reason": error_msg
                    })
                    self.stats["total_failed"] += 1
                    self.stats["active_tasks"] -= 1
    
    async def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """获取视频文件信息"""
        try:
            path = Path(video_path)
            if not path.exists():
                return {}
            
            file_size = path.stat().st_size
            
            # 尝试获取视频时长和帧数（需要 ffprobe 或类似工具）
            duration = None
            frame_count = None
            resolution = None
            
            try:
                import subprocess
                import json
                
                # 使用 ffprobe 获取视频信息
                cmd = [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", str(path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    info = json.loads(result.stdout)
                    
                    # 查找视频流
                    for stream in info.get("streams", []):
                        if stream.get("codec_type") == "video":
                            duration = float(stream.get("duration", 0))
                            frame_count = int(stream.get("nb_frames", 0))
                            width = stream.get("width", 0)
                            height = stream.get("height", 0)
                            if width and height:
                                resolution = f"{width}*{height}"
                            break
                            
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, 
                    json.JSONDecodeError, ImportError, FileNotFoundError):
                # ffprobe 不可用或执行失败，使用默认值
                pass
            
            return {
                "file_size": file_size,
                "duration": duration,
                "frame_count": frame_count,
                "resolution": resolution
            }
            
        except Exception as e:
            logger.warning(f"Failed to get video info for {video_path}: {e}")
            return {}
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            
            # 更新队列位置
            if task["status"] == TaskStatus.IN_QUEUE:
                queue_position = len([
                    t for t in self.tasks.values() 
                    if t["status"] == TaskStatus.IN_QUEUE and t["created_at"] < task["created_at"]
                ]) + 1
                task["queue_position"] = queue_position
                
                # 估计剩余时间
                if queue_position > 0:
                    avg_processing_time = 120  # 平均2分钟
                    estimated_time = queue_position * avg_processing_time
                    task["estimated_time"] = estimated_time
            
            return task.copy()
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task["status"] in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            # 标记为已取消
            task["status"] = TaskStatus.CANCELLED
            task["updated_at"] = time.time()
            task["reason"] = "User cancelled"
            
            self.stats["total_cancelled"] += 1
            if task["status"] in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]:
                self.stats["active_tasks"] -= 1
            
            logger.info(f"Task {task_id} cancelled by user")
            return True
    
    async def cleanup_expired_tasks(self) -> int:
        """清理过期任务"""
        current_time = time.time()
        expired_tasks = []
        
        async with self._lock:
            for task_id, task in self.tasks.items():
                # 删除超过24小时的已完成任务
                if (task["status"] in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    current_time - task["updated_at"] > 86400):  # 24小时
                    expired_tasks.append(task_id)
                
                # 删除超时的进行中任务
                elif (task["status"] == TaskStatus.IN_PROGRESS and
                      current_time - task["updated_at"] > self.task_timeout):
                    task["status"] = TaskStatus.FAILED
                    task["reason"] = "Task timeout"
                    task["updated_at"] = current_time
                    self.stats["total_failed"] += 1
                    self.stats["active_tasks"] -= 1
                    expired_tasks.append(task_id)
            
            # 删除过期任务
            for task_id in expired_tasks:
                del self.tasks[task_id]
        
        if expired_tasks:
            logger.info(f"Cleaned up {len(expired_tasks)} expired tasks")
        
        return len(expired_tasks)
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        async with self._lock:
            current_stats = self.stats.copy()
            
            # 添加当前状态统计
            status_counts = {}
            for status in TaskStatus:
                status_counts[status.value] = len([
                    t for t in self.tasks.values() if t["status"] == status
                ])
            
            current_stats["status_breakdown"] = status_counts
            current_stats["total_tasks"] = len(self.tasks)
            
            return current_stats