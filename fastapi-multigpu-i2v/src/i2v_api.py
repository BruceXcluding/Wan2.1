"""
视频生成服务
负责管理视频生成任务的生命周期
"""
import asyncio
import time
import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# 修复导入 - 使用绝对导入
from schemas.video import (
    VideoSubmitRequest,
    VideoStatusResponse,
    VideoResults,
    TaskStatus
)

logger = logging.getLogger(__name__)

class VideoService:
    """视频生成服务"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_lock = asyncio.Lock()
        
        # 统计信息
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.cancelled_tasks = 0
        
        logger.info("VideoService initialized")
    
    async def submit_video_task(self, request: VideoSubmitRequest) -> str:
        """提交视频生成任务"""
        task_id = f"req_{uuid.uuid4().hex[:16]}"
        
        async with self.task_lock:
            # 创建任务记录
            task_data = {
                "requestId": task_id,
                "status": TaskStatus.PENDING,
                "progress": 0,
                "message": "Task submitted successfully",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "request_params": request.model_dump(),
                "results": None,
                "reason": None,
                "elapsed_time": None,
                "queue_position": len([t for t in self.tasks.values() if t["status"] == TaskStatus.PENDING])
            }
            
            self.tasks[task_id] = task_data
            self.total_tasks += 1
        
        # 异步启动任务
        asyncio.create_task(self._process_video_task(task_id, request))
        
        logger.info(f"Task {task_id} submitted successfully")
        return task_id
    
    async def _process_video_task(self, task_id: str, request: VideoSubmitRequest):
        """处理视频生成任务"""
        start_time = time.time()
        
        try:
            # 更新任务状态为运行中
            await self._update_task_status(
                task_id, 
                TaskStatus.RUNNING, 
                10, 
                "Starting video generation..."
            )
            
            # 准备参数
            generation_params = self._prepare_generation_params(request)
            
            # 更新进度
            await self._update_task_status(
                task_id, 
                TaskStatus.RUNNING, 
                20, 
                "Initializing models..."
            )
            
            # 调用管道生成视频
            logger.info(f"Starting video generation for task {task_id}")
            result = await self._generate_video_with_pipeline(task_id, generation_params)
            
            # 生成成功
            elapsed_time = int(time.time() - start_time)
            
            # 创建结果对象
            video_results = VideoResults(
                video_url=f"http://localhost:8088/videos/{result['filename']}",
                video_path=result['output_path'],
                duration=result.get('duration', 3.4),
                frames=generation_params['num_frames'],
                size=generation_params['image_size'],
                file_size=result.get('file_size', 0)
            )
            
            async with self.task_lock:
                if task_id in self.tasks:
                    self.tasks[task_id].update({
                        "status": TaskStatus.SUCCEED,
                        "progress": 100,
                        "message": "Video generation completed successfully",
                        "updated_at": datetime.utcnow().isoformat() + "Z",
                        "results": video_results.model_dump(),
                        "elapsed_time": elapsed_time
                    })
                    self.successful_tasks += 1
            
            logger.info(f"Task {task_id} completed successfully in {elapsed_time}s")
            
        except asyncio.CancelledError:
            # 任务被取消
            await self._update_task_status(
                task_id, 
                TaskStatus.CANCELLED, 
                0, 
                "Task cancelled by user",
                reason="Task cancelled"
            )
            self.cancelled_tasks += 1
            logger.info(f"Task {task_id} was cancelled")
            
        except Exception as e:
            # 任务失败
            elapsed_time = int(time.time() - start_time)
            error_msg = str(e)
            
            await self._update_task_status(
                task_id, 
                TaskStatus.FAILED, 
                0, 
                f"Video generation failed: {error_msg}",
                reason=error_msg,
                elapsed_time=elapsed_time
            )
            self.failed_tasks += 1
            logger.error(f"Task {task_id} failed after {elapsed_time}s: {error_msg}")
    
    def _prepare_generation_params(self, request: VideoSubmitRequest) -> Dict[str, Any]:
        """准备生成参数"""
        return {
            "prompt": request.prompt,
            "image_url": request.image_url,
            "image_size": request.image_size or "1280*720",
            "num_frames": request.num_frames or 81,
            "guidance_scale": request.guidance_scale or 3.0,
            "infer_steps": request.infer_steps or 30,
            "seed": request.seed,
            "negative_prompt": request.negative_prompt,
            
            # 分布式参数
            "vae_parallel": request.vae_parallel or False,
            "ulysses_size": request.ulysses_size or 1,
            "dit_fsdp": request.dit_fsdp or False,
            "t5_fsdp": request.t5_fsdp or False,
            "cfg_size": request.cfg_size or 1,
            
            # 性能优化参数
            "use_attentioncache": request.use_attentioncache or False,
            "start_step": request.start_step or 12,
            "attentioncache_interval": request.attentioncache_interval or 4,
            "end_step": request.end_step or 37,
            "sample_solver": request.sample_solver or "unipc",
            "sample_shift": request.sample_shift or 5.0,
        }
    
    async def _generate_video_with_pipeline(self, task_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """使用管道生成视频"""
        
        # 进度回调函数
        async def progress_callback(step: int, total_steps: int, message: str = ""):
            progress = int(20 + (step / total_steps) * 70)  # 20-90%的进度
            await self._update_task_status(task_id, TaskStatus.RUNNING, progress, message)
        
        # 调用管道生成
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.pipeline.generate_video(
                **params,
                progress_callback=progress_callback
            )
        )
        
        # 最终进度更新
        await self._update_task_status(
            task_id, 
            TaskStatus.RUNNING, 
            95, 
            "Finalizing video..."
        )
        
        return result
    
    async def _update_task_status(
        self, 
        task_id: str, 
        status: TaskStatus, 
        progress: int, 
        message: str,
        reason: Optional[str] = None,
        elapsed_time: Optional[int] = None
    ):
        """更新任务状态"""
        async with self.task_lock:
            if task_id in self.tasks:
                self.tasks[task_id].update({
                    "status": status,
                    "progress": progress,
                    "message": message,
                    "updated_at": datetime.utcnow().isoformat() + "Z"
                })
                
                if reason is not None:
                    self.tasks[task_id]["reason"] = reason
                
                if elapsed_time is not None:
                    self.tasks[task_id]["elapsed_time"] = elapsed_time
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        async with self.task_lock:
            return self.tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        async with self.task_lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task["status"] in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            # 更新状态为已取消
            task.update({
                "status": TaskStatus.CANCELLED,
                "progress": 0,
                "message": "Task cancelled by user",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "reason": "Cancelled by user"
            })
            
            self.cancelled_tasks += 1
            
        logger.info(f"Task {task_id} cancelled")
        return True
    
    async def cleanup_expired_tasks(self, max_age_hours: int = 24) -> int:
        """清理过期任务"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        async with self.task_lock:
            expired_tasks = []
            
            for task_id, task in self.tasks.items():
                # 解析创建时间
                try:
                    created_at = datetime.fromisoformat(task["created_at"].replace("Z", "+00:00"))
                    task_age = current_time - created_at.timestamp()
                    
                    if task_age > max_age_seconds:
                        expired_tasks.append(task_id)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse timestamp for task {task_id}: {e}")
                    expired_tasks.append(task_id)  # 清理无效时间戳的任务
            
            # 删除过期任务
            for task_id in expired_tasks:
                del self.tasks[task_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired tasks")
        
        return cleaned_count
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        async with self.task_lock:
            active_tasks = len([t for t in self.tasks.values() 
                              if t["status"] in [TaskStatus.PENDING, TaskStatus.RUNNING]])
            
            return {
                "total_tasks": self.total_tasks,
                "active_tasks": active_tasks,
                "successful_tasks": self.successful_tasks,
                "failed_tasks": self.failed_tasks,
                "cancelled_tasks": self.cancelled_tasks,
                "success_rate": (self.successful_tasks / max(self.total_tasks, 1)) * 100,
                "tasks_in_memory": len(self.tasks)
            }
    
    async def get_task_list(self, status_filter: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """获取任务列表"""
        async with self.task_lock:
            tasks = list(self.tasks.values())
            
            if status_filter:
                tasks = [t for t in tasks if t["status"] == status_filter]
            
            # 按创建时间倒序排列
            tasks.sort(key=lambda x: x["created_at"], reverse=True)
            
            return tasks