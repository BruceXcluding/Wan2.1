import os
import asyncio
import time
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from schemas import (
    VideoSubmitRequest,
    VideoStatusRequest,
    VideoStatusResponse,
    VideoCancelRequest
)
from services.video_service import VideoService
from multigpu_pipeline import MultiGPUVideoPipeline

# ==================== 配置和常量 ====================

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 环境变量
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

# 服务配置
SERVER_CONFIG = {
    "host": os.environ.get("SERVER_HOST", "0.0.0.0"),
    "port": int(os.environ.get("SERVER_PORT", 8088)),
    "workers": 1,  # 分布式环境下必须是1
    "reload": False,
    "access_log": True,
}

# 模型配置
MODEL_CONFIG = {
    "ckpt_dir": os.environ.get(
        "MODEL_CKPT_DIR", 
        "/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"
    ),
    "task": "i2v-14B",
    "size": "1280*720",
    "frame_num": 81,
    "sample_steps": 40,
    "dit_fsdp": True,
    "t5_fsdp": True,
    "cfg_size": 1,
    "ulysses_size": 8,
    "vae_parallel": True,
    "use_attentioncache": True,
    "start_step": 12,
    "attentioncache_interval": 4,
    "end_step": 37,
}

# 业务配置
BUSINESS_CONFIG = {
    "max_concurrent_tasks": int(os.environ.get("MAX_CONCURRENT_TASKS", 5)),
    "task_timeout": int(os.environ.get("TASK_TIMEOUT", 1800)),  # 30分钟
    "cleanup_interval": int(os.environ.get("CLEANUP_INTERVAL", 300)),  # 5分钟
    "max_video_output_dir_size": int(os.environ.get("MAX_OUTPUT_DIR_SIZE", 50)),  # 50GB
    "allowed_hosts": os.environ.get("ALLOWED_HOSTS", "*").split(","),
}

# ==================== 全局异常处理 ====================

class VideoGenerationException(Exception):
    """视频生成自定义异常"""
    def __init__(self, message: str, error_code: str = "VIDEO_GENERATION_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

async def validation_exception_handler(request: Request, exc: ValueError):
    """参数验证异常处理"""
    logger.warning(f"Validation error from {request.client.host}: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"error": "VALIDATION_ERROR", "message": str(exc)}
    )

async def video_generation_exception_handler(request: Request, exc: VideoGenerationException):
    """视频生成异常处理"""
    logger.error(f"Video generation error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"error": exc.error_code, "message": exc.message}
    )

async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_SERVER_ERROR", "message": "服务器内部错误"}
    )

# ==================== 资源管理 ====================

class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        self.concurrent_tasks = 0
        self.max_concurrent = BUSINESS_CONFIG["max_concurrent_tasks"]
        self._lock = asyncio.Lock()
    
    async def acquire_task_slot(self) -> bool:
        """获取任务槽位"""
        async with self._lock:
            if self.concurrent_tasks >= self.max_concurrent:
                return False
            self.concurrent_tasks += 1
            return True
    
    async def release_task_slot(self):
        """释放任务槽位"""
        async with self._lock:
            if self.concurrent_tasks > 0:
                self.concurrent_tasks -= 1
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取资源统计"""
        async with self._lock:
            return {
                "concurrent_tasks": self.concurrent_tasks,
                "max_concurrent": self.max_concurrent,
                "available_slots": self.max_concurrent - self.concurrent_tasks
            }

# ==================== 初始化 ====================

logger.info(f"Initializing pipeline on rank {LOCAL_RANK}/{WORLD_SIZE}")

try:
    pipeline = MultiGPUVideoPipeline(**MODEL_CONFIG)
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {str(e)}")
    if LOCAL_RANK == 0:
        sys.exit(1)
    raise

# ==================== FastAPI 应用 ====================

if LOCAL_RANK == 0:
    # 主进程启动 HTTP 服务
    
    # 初始化服务组件
    video_service = VideoService(pipeline)
    resource_manager = ResourceManager()
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        logger.info("FastAPI application starting up")
        
        # 创建视频输出目录
        os.makedirs("generated_videos", exist_ok=True)
        
        # 启动后台任务
        cleanup_task = asyncio.create_task(periodic_cleanup())
        monitor_task = asyncio.create_task(resource_monitor())
        
        # 设置应用状态
        app.state.video_service = video_service
        app.state.resource_manager = resource_manager
        app.state.start_time = time.time()
        
        yield
        
        # 清理资源
        logger.info("Shutting down FastAPI application...")
        cleanup_task.cancel()
        monitor_task.cancel()
        
        try:
            await asyncio.gather(cleanup_task, monitor_task, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        
        pipeline.cleanup()
        logger.info("FastAPI application shut down completed")

    async def periodic_cleanup():
        """定期清理任务"""
        while True:
            try:
                await asyncio.sleep(BUSINESS_CONFIG["cleanup_interval"])
                
                # 清理过期任务
                cleaned_count = await video_service.cleanup_expired_tasks()
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} expired tasks")
                
                # 清理旧视频文件（可选）
                await cleanup_old_videos()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {str(e)}")

    async def cleanup_old_videos():
        """清理旧视频文件"""
        try:
            import shutil
            from pathlib import Path
            
            video_dir = Path("generated_videos")
            if not video_dir.exists():
                return
            
            # 获取目录大小
            total_size = sum(f.stat().st_size for f in video_dir.rglob('*') if f.is_file())
            max_size = BUSINESS_CONFIG["max_video_output_dir_size"] * 1024 * 1024 * 1024  # GB to bytes
            
            if total_size > max_size:
                logger.info(f"Video directory size ({total_size / 1024 / 1024 / 1024:.2f}GB) exceeds limit, cleaning up...")
                
                # 删除最旧的文件
                files = [(f, f.stat().st_mtime) for f in video_dir.rglob('*') if f.is_file()]
                files.sort(key=lambda x: x[1])  # 按修改时间排序
                
                for file_path, _ in files:
                    file_path.unlink()
                    total_size -= file_path.stat().st_size
                    if total_size <= max_size * 0.8:  # 清理到80%
                        break
                
                logger.info("Video cleanup completed")
        
        except Exception as e:
            logger.error(f"Error cleaning up videos: {str(e)}")

    async def resource_monitor():
        """资源监控任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟监控一次
                
                # 记录资源使用情况
                stats = await resource_manager.get_stats()
                service_stats = await video_service.get_service_stats()
                
                logger.info(f"Resource stats: {stats}")
                logger.info(f"Service stats: {service_stats}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitor error: {str(e)}")

    # 创建 FastAPI 应用
    app = FastAPI(
        title="Multi-GPU Video Generation API",
        description="基于 Wan2.1-I2V-14B-720P 的多卡分布式视频生成服务",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # 添加异常处理器
    app.add_exception_handler(ValueError, validation_exception_handler)
    app.add_exception_handler(VideoGenerationException, video_generation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    # 添加中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=BUSINESS_CONFIG["allowed_hosts"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    
    if BUSINESS_CONFIG["allowed_hosts"] != ["*"]:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=BUSINESS_CONFIG["allowed_hosts"]
        )
    
    # 静态文件服务
    app.mount("/videos", StaticFiles(directory="generated_videos"), name="videos")

    # ==================== 中间件 ====================

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        """请求日志中间件"""
        start_time = time.time()
        
        # 记录请求
        logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")
        
        response = await call_next(request)
        
        # 记录响应
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} in {process_time:.3f}s")
        
        return response

    # ==================== API 路由 ====================

    @app.post("/video/submit", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
    async def submit_video_task(request: VideoSubmitRequest, background_tasks: BackgroundTasks):
        """提交视频生成任务"""
        # 检查资源可用性
        if not await resource_manager.acquire_task_slot():
            raise HTTPException(
                status_code=429, 
                detail="服务器繁忙，请稍后重试"
            )
        
        try:
            task_id = await video_service.submit_video_task(request)
            
            # 添加资源释放任务
            background_tasks.add_task(
                release_task_slot_after_completion, 
                task_id, 
                resource_manager
            )
            
            return {"requestId": task_id}
            
        except Exception as e:
            # 释放资源
            await resource_manager.release_task_slot()
            logger.error(f"Failed to submit task: {str(e)}")
            raise VideoGenerationException("任务提交失败", "TASK_SUBMIT_ERROR")

    async def release_task_slot_after_completion(task_id: str, resource_manager: ResourceManager):
        """任务完成后释放资源槽位"""
        try:
            # 等待任务完成
            while True:
                task = await video_service.get_task_status(task_id)
                if not task or task["status"] in ["Succeed", "Failed", "Cancelled"]:
                    break
                await asyncio.sleep(5)
            
            await resource_manager.release_task_slot()
            logger.info(f"Released resource slot for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error releasing resource slot for task {task_id}: {str(e)}")

    @app.post("/video/status", response_model=VideoStatusResponse)
    async def get_task_status(request: VideoStatusRequest):
        """查询任务状态"""
        task = await video_service.get_task_status(request.requestId)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return VideoStatusResponse(**task)

    @app.post("/video/cancel", response_model=dict)
    async def cancel_task(request: VideoCancelRequest):
        """取消任务"""
        success = await video_service.cancel_task(request.requestId)
        if not success:
            task = await video_service.get_task_status(request.requestId)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            return {"status": "Already finished"}
        
        return {"status": "Cancelled"}

    @app.get("/health")
    async def health_check():
        """健康检查"""
        try:
            service_stats = await video_service.get_service_stats()
            resource_stats = await resource_manager.get_stats()
            
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "uptime": time.time() - app.state.start_time,
                "rank": LOCAL_RANK,
                "world_size": WORLD_SIZE,
                "service": service_stats,
                "resources": resource_stats
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise HTTPException(status_code=503, detail="服务不可用")

    @app.get("/metrics")
    async def get_metrics():
        """获取详细指标"""
        try:
            service_stats = await video_service.get_service_stats()
            resource_stats = await resource_manager.get_stats()
            
            return {
                "timestamp": time.time(),
                "service": service_stats,
                "resources": resource_stats,
                "system": {
                    "rank": LOCAL_RANK,
                    "world_size": WORLD_SIZE,
                    "uptime": time.time() - app.state.start_time
                }
            }
        except Exception as e:
            logger.error(f"Metrics collection failed: {str(e)}")
            raise HTTPException(status_code=500, detail="指标收集失败")

    @app.get("/")
    async def root():
        """根路径"""
        return {
            "service": "Multi-GPU Video Generation API",
            "version": "2.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics"
        }

    # ==================== 启动服务 ====================

    def run_server():
        """启动服务器"""
        try:
            logger.info(f"Starting FastAPI server on {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
            uvicorn.run(
                app,
                host=SERVER_CONFIG["host"],
                port=SERVER_CONFIG["port"],
                workers=SERVER_CONFIG["workers"],
                reload=SERVER_CONFIG["reload"],
                access_log=SERVER_CONFIG["access_log"],
                log_level="info"
            )
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            sys.exit(1)

    # 优雅关闭处理
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, gracefully shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动服务
    run_server()

else:
    # 非主进程只参与分布式推理
    logger.info(f"Rank {LOCAL_RANK} ready for distributed inference")
    
    def signal_handler(sig, frame):
        logger.info(f"Rank {LOCAL_RANK} shutting down")
        pipeline.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 保持进程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info(f"Rank {LOCAL_RANK} interrupted")
        pipeline.cleanup()