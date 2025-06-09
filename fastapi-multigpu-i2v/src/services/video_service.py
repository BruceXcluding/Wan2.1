import asyncio
import time
import uuid
import logging
from typing import Dict, Optional

from schemas import VideoSubmitRequest, VideoResults, TaskStatus
from multigpu_pipeline import MultiGPUVideoPipeline

logger = logging.getLogger(__name__)

class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self.tasks: Dict[str, dict] = {}
        self.lock = asyncio.Lock()
    
    async def create_task(self, task_id: str) -> dict:
        """创建新任务"""
        task_data = {
            "status": TaskStatus.IN_QUEUE,
            "reason": None,
            "results": None,
            "progress": 0.0,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        async with self.lock:
            self.tasks[task_id] = task_data
        
        return task_data
    
    async def update_task(self, task_id: str, **kwargs) -> bool:
        """更新任务状态"""
        async with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
                self.tasks[task_id]["updated_at"] = time.time()
                return True
            return False
    
    async def get_task(self, task_id: str) -> Optional[dict]:
        """获取任务信息"""
        async with self.lock:
            return self.tasks.get(task_id)
    
    async def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        async with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                return True
            return False
    
    async def get_task_count(self) -> int:
        """获取任务总数"""
        async with self.lock:
            return len(self.tasks)
    
    async def get_queue_position(self, task_id: str) -> Optional[int]:
        """获取任务在队列中的位置"""
        async with self.lock:
            if task_id not in self.tasks:
                return None
            
            # 计算排在当前任务前面的 InQueue 任务数量
            task = self.tasks[task_id]
            if task["status"] != TaskStatus.IN_QUEUE:
                return None
            
            position = 1
            for tid, t in self.tasks.items():
                if (t["status"] == TaskStatus.IN_QUEUE and 
                    t["created_at"] < task["created_at"]):
                    position += 1
            
            return position
    
    async def cleanup_old_tasks(self, max_age: int = 3600) -> int:
        """清理超过指定时间的任务"""
        current_time = time.time()
        cleaned_count = 0
        
        async with self.lock:
            expired_tasks = [
                task_id for task_id, task in self.tasks.items()
                if (current_time - task["created_at"] > max_age and
                    task["status"] in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCELLED])
            ]
            
            for task_id in expired_tasks:
                del self.tasks[task_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired tasks")
        
        return cleaned_count

class VideoService:
    """视频生成服务"""
    
    def __init__(self, pipeline: MultiGPUVideoPipeline):
        self.pipeline = pipeline
        self.task_manager = TaskManager()
        self.processing_lock = asyncio.Lock()
    
    async def submit_video_task(self, request: VideoSubmitRequest) -> str:
        """提交视频生成任务"""
        task_id = uuid.uuid4().hex
        
        # 创建任务
        await self.task_manager.create_task(task_id)
        
        # 异步处理任务
        asyncio.create_task(self._process_video_task(request, task_id))
        
        logger.info(f"Video task submitted: {task_id}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[dict]:
        """获取任务状态"""
        task = await self.task_manager.get_task(task_id)
        if not task:
            return None
        
        # 如果任务在队列中，计算队列位置
        if task["status"] == TaskStatus.IN_QUEUE:
            queue_position = await self.task_manager.get_queue_position(task_id)
            task = task.copy()
            task["queue_position"] = queue_position
        
        return task
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = await self.task_manager.get_task(task_id)
        if not task:
            return False
        
        if task["status"] in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        await self.task_manager.update_task(
            task_id,
            status=TaskStatus.CANCELLED,
            reason="用户取消"
        )
        
        logger.info(f"Task cancelled: {task_id}")
        return True
    
    async def get_service_stats(self) -> dict:
        """获取服务统计信息"""
        task_count = await self.task_manager.get_task_count()
        return {
            "total_tasks": task_count,
            "pipeline_world_size": self.pipeline.world_size,
            "pipeline_rank": self.pipeline.rank
        }
    
    async def cleanup_expired_tasks(self) -> int:
        """清理过期任务"""
        return await self.task_manager.cleanup_old_tasks()
    
    async def _process_video_task(self, request: VideoSubmitRequest, task_id: str):
        """处理视频生成任务的内部方法"""
        try:
            # 更新状态为处理中
            await self.task_manager.update_task(
                task_id,
                status=TaskStatus.IN_PROGRESS,
                progress=0.1
            )
            
            logger.info(f"Starting video generation for task: {task_id}")
            
            # 执行视频生成
            video_path = await self.pipeline.generate_video(request, task_id)
            
            # 创建结果对象
            results = VideoResults(
                video_url=f"/videos/{task_id}.mp4",
                video_path=video_path
            )
            
            # 更新状态为成功
            await self.task_manager.update_task(
                task_id,
                status=TaskStatus.SUCCEED,
                progress=1.0,
                results=results.dict()
            )
            
            logger.info(f"Video generation completed for task: {task_id}")
            
        except Exception as e:
            logger.error(f"Video generation failed for task {task_id}: {str(e)}")
            await self.task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                reason=str(e)
            )