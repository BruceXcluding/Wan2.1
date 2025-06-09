"""
FastAPI Multi-GPU I2V API Server
"""
import sys
import os
import logging
import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import time
import json
from contextlib import asynccontextmanager
import traceback
from datetime import datetime

# 确保能找到项目根目录的 utils - 在所有其他导入之前
def setup_project_paths():
    """设置项目路径，确保能找到所有模块"""
    current_file = Path(__file__).resolve()
    
    # 计算路径：src/i2v_api.py -> 项目根目录
    project_root = current_file.parent.parent
    src_root = current_file.parent
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

# PyTorch 相关导入
import torch
import torch.distributed as dist

# FastAPI 相关导入
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # 添加这个导入
import uvicorn

# 导入项目模块 - 使用修复的导入方式
from schemas import (
    VideoSubmitRequest, VideoSubmitResponse,
    VideoStatusRequest, VideoStatusResponse,
    VideoCancelRequest, VideoCancelResponse,
    TaskStatus, VideoResults, HealthResponse, MetricsResponse
)

from pipelines import PipelineFactory, get_available_pipelines

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
            # 如果都失败了，打印调试信息并退出
            print(f"❌ Failed to import device_detector in i2v_api.py: {e}")
            print(f"Project root: {project_root}")
            print(f"Utils root: {utils_root}")
            print(f"Utils exists: {utils_root.exists()}")
            print(f"device_detector.py exists: {(utils_root / 'device_detector.py').exists()}")
            print(f"Current sys.path: {sys.path[:5]}")
            print(f"Current working directory: {os.getcwd()}")
            sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 分布式相关全局变量
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
startup_time = time.time()

# 服务统计
service_stats = {
    "total_tasks": 0,
    "successful_tasks": 0,
    "failed_tasks": 0,
    "cancelled_tasks": 0
}

# 全局变量
pipeline = None
task_manager = None
app_metrics = {
    "start_time": time.time(),
    "total_requests": 0,
    "active_tasks": 0,
    "completed_tasks": 0,
    "failed_tasks": 0,
    "total_generation_time": 0.0
}

# 视频生成服务类
class VideoService:
    """视频生成服务"""
    
    def __init__(self, pipeline_instance):
        self.pipeline = pipeline_instance
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_lock = asyncio.Lock()
        
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
                "elapsed_time": None
            }
            
            self.tasks[task_id] = task_data
            service_stats['total_tasks'] += 1
        
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
            
            # 进度回调函数
            async def progress_callback(progress: int, total: int, message: str = ""):
                await self._update_task_status(task_id, TaskStatus.RUNNING, progress, message)
            
            # 调用管道生成视频 - 修复调用方式
            logger.info(f"Starting video generation for task {task_id}")
            
            # 使用正确的管道调用方式
            result_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline.generate_video(request, task_id, progress_callback)
            )
            
            # 生成成功
            elapsed_time = int(time.time() - start_time)
            
            # 创建结果对象
            video_results = VideoResults(
                video_url=f"http://localhost:8088/videos/{os.path.basename(result_path)}",
                video_path=result_path,
                duration=3.4,  # 默认时长
                frames=request.num_frames or 81,
                size=request.image_size or "1280*720",
                file_size=os.path.getsize(result_path) if os.path.exists(result_path) else 0
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
                    service_stats['successful_tasks'] += 1
            
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
            service_stats['cancelled_tasks'] += 1
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
            service_stats['failed_tasks'] += 1
            logger.error(f"Task {task_id} failed after {elapsed_time}s: {error_msg}")
    
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
            
            service_stats['cancelled_tasks'] += 1
            
        logger.info(f"Task {task_id} cancelled")
        return True
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        async with self.task_lock:
            active_tasks = len([t for t in self.tasks.values() 
                              if t["status"] in [TaskStatus.PENDING, TaskStatus.RUNNING]])
            
            return {
                **service_stats,
                "active_tasks": active_tasks,
                "success_rate": (service_stats['successful_tasks'] / max(service_stats['total_tasks'], 1)) * 100,
                "tasks_in_memory": len(self.tasks)
            }

# 全局服务实例
video_service = None

def init_distributed():
    """初始化分布式环境"""
    if world_size == 1:
        logger.info("Single device mode")
        return True
    
    try:
        # 检测设备类型 - 添加错误处理
        try:
            device_type, device_count = device_detector.detect_device()
            logger.info(f"Detected device: {device_type.value} x {device_count}")
        except Exception as e:
            logger.error(f"Device detection failed: {e}")
            # 使用 CPU 作为备用
            device_type = device_detector.DeviceType.CPU  # 或者导入 DeviceType
            device_count = 1
     
        # 设置后端
        if device_type.value == "npu":
            import torch_npu
            torch_npu.npu.set_device(local_rank)
            backend = "hccl"
        elif device_type.value == "cuda":
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        
        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size
            )
        
        # 同步所有进程
        dist.barrier()
        logger.info(f"Distributed initialized with {backend}")
        return True
        
    except Exception as e:
        logger.error(f"Distributed initialization failed: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global pipeline, video_service
    
    try:
        logger.info("🔄 Starting application...")
        
        # 初始化分布式
        if not init_distributed():
            logger.warning("Distributed initialization failed, continuing...")
        
        # 只在主进程中初始化服务
        if rank == 0:
            # 创建管道
            logger.info("🏭 Creating pipeline...")
            
            # 获取配置
            ckpt_dir = os.environ.get('MODEL_CKPT_DIR', '/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P')
            config = {
                'ckpt_dir': ckpt_dir,
                't5_cpu': os.environ.get('T5_CPU', 'true').lower() == 'true',
                'dit_fsdp': os.environ.get('DIT_FSDP', 'true').lower() == 'true',
                'vae_parallel': os.environ.get('VAE_PARALLEL', 'true').lower() == 'true'
            }
            
            pipeline = PipelineFactory.create_pipeline(**config)
            video_service = VideoService(pipeline)
            
            logger.info("✅ Service initialized successfully")
        else:
            logger.info(f"⏳ Worker {rank} waiting...")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        traceback.print_exc()
        yield
        
    finally:
        # 清理资源
        if pipeline:
            try:
                pipeline.cleanup()
            except Exception as e:
                logger.warning(f"Pipeline cleanup warning: {e}")
        
        # 清理分布式
        if world_size > 1 and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                logger.warning(f"Distributed cleanup warning: {e}")

# 创建应用
app = FastAPI(
    title="Multi-GPU I2V Generation API",
    description="Fast and scalable image-to-video generation service",
    version="1.0.0",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
os.makedirs("generated_videos", exist_ok=True)
app.mount("/videos", StaticFiles(directory="generated_videos"), name="videos")

# API 路由
@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        device_type, device_count = device_detector.detect_device()
    except Exception as e:
        logger.warning(f"Device detection failed in health check: {e}")
        device_type = None
        device_count = 0
    
    return HealthResponse(
        status="healthy" if video_service else "initializing",
        timestamp=time.time(),
        uptime=time.time() - startup_time,
        config={
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device_type": device_type.value if device_type else "unknown",
            "device_count": device_count
        },
        service=await video_service.get_service_stats() if video_service else {},
        resources={"memory": "unknown", "cpu_count": os.cpu_count()}
    )

@app.get("/metrics")
async def get_metrics():
    """获取监控指标"""
    if not video_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    stats = await video_service.get_service_stats()
    
    return MetricsResponse(
        timestamp=time.time(),
        system={"uptime": time.time() - startup_time, "rank": rank},
        service=stats,
        tasks={"total": stats["total_tasks"], "active": stats["active_tasks"]},
        performance={"success_rate": stats["success_rate"]}
    )

@app.post("/submit", response_model=VideoSubmitResponse)
async def submit_video_generation(request: VideoSubmitRequest):
    """提交视频生成任务"""
    if rank != 0:
        raise HTTPException(status_code=503, detail="Only available on main process")
    
    if not video_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        task_id = await video_service.submit_video_task(request)
        return VideoSubmitResponse(
            requestId=task_id,
            status=TaskStatus.PENDING,
            message="Task submitted successfully",
            estimated_time=180
        )
    except Exception as e:
        logger.error(f"Submit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/status", response_model=VideoStatusResponse)
async def get_video_status(request: VideoStatusRequest):
    """查询任务状态"""
    if rank != 0:
        raise HTTPException(status_code=503, detail="Only available on main process")
    
    if not video_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    task_data = await video_service.get_task_status(request.requestId)
    
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return VideoStatusResponse(**task_data)

@app.post("/cancel", response_model=VideoCancelResponse)
async def cancel_video_generation(request: VideoCancelRequest):
    """取消任务"""
    if rank != 0:
        raise HTTPException(status_code=503, detail="Only available on main process")
    
    if not video_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    success = await video_service.cancel_task(request.requestId)
    
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
    
    return VideoCancelResponse(
        requestId=request.requestId,
        status="cancelled",
        message="Task cancelled successfully"
    )

@app.get("/tasks")
async def get_task_list(status: Optional[str] = None):
    """获取任务列表"""
    if rank != 0:
        raise HTTPException(status_code=503, detail="Only available on main process")
    
    if not video_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # 这里简化实现，实际应该在VideoService中实现
    async with video_service.task_lock:
        tasks = list(video_service.tasks.values())
        
        if status:
            try:
                status_filter = TaskStatus(status)
                tasks = [t for t in tasks if t["status"] == status_filter]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        tasks.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {"tasks": tasks, "total": len(tasks), "filtered_by": status if status else "none"}

# 主函数
def main():
    if rank == 0:
        # 主进程运行HTTP服务
        host = os.getenv("SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("SERVER_PORT", 8088))
        
        logger.info(f"🌐 Starting server on {host}:{port}")
        
        uvicorn.run(app, host=host, port=port, log_level="info", workers=1)
    else:
        # 工作进程等待
        logger.info(f"⏳ Worker {rank} ready, waiting...")
        try:
            import time
            while True:
                time.sleep(60)  # 心跳
                logger.debug(f"💓 Worker {rank} alive")
        except KeyboardInterrupt:
            logger.info(f"🛑 Worker {rank} stopping")

if __name__ == "__main__":
    main()