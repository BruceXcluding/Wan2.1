import os
import asyncio
import time
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

# 在导入其他模块之前设置环境变量
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, HTTPException, status, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# 修复导入 - 使用新的模块结构
from schemas.video import (
    VideoSubmitRequest,
    VideoStatusRequest,
    VideoStatusResponse,
    VideoCancelRequest
)
from services.video_service import VideoService
from pipelines.pipeline_factory import PipelineFactory
from utils.device_detector import device_detector

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

# 获取设备信息（在创建配置之前）
try:
    device_info = PipelineFactory.get_available_devices()
    logger.info(f"Detected device: {device_info}")
except Exception as e:
    logger.error(f"Failed to detect device: {str(e)}")
    device_info = {"device_type": "unknown", "device_count": 0, "backend": "unknown"}

# 模型配置 - 通过环境变量控制，自动适配设备
MODEL_CONFIG = {
    "ckpt_dir": os.environ.get(
        "MODEL_CKPT_DIR", 
        "/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"
    ),
    "task": os.environ.get("MODEL_TASK", "i2v-14B"),
    "size": os.environ.get("DEFAULT_SIZE", "1280*720"),
    "frame_num": int(os.environ.get("DEFAULT_FRAME_NUM", "81")),
    "sample_steps": int(os.environ.get("DEFAULT_SAMPLE_STEPS", "40")),
    
    # 分布式配置 - 自动适配设备
    "dit_fsdp": os.environ.get("DIT_FSDP", "true").lower() == "true",
    "t5_fsdp": os.environ.get("T5_FSDP", "false").lower() == "true",
    "t5_cpu": os.environ.get("T5_CPU", "false").lower() == "true",
    "cfg_size": int(os.environ.get("CFG_SIZE", "1")),
    "ulysses_size": int(os.environ.get("ULYSSES_SIZE", "8")),
    "vae_parallel": os.environ.get("VAE_PARALLEL", "true").lower() == "true",
    
    # 性能优化
    "use_attentioncache": os.environ.get("USE_ATTENTION_CACHE", "true").lower() == "true",
    "start_step": int(os.environ.get("CACHE_START_STEP", "12")),
    "attentioncache_interval": int(os.environ.get("CACHE_INTERVAL", "4")),
    "end_step": int(os.environ.get("CACHE_END_STEP", "37")),
}

# 业务配置 - 根据设备类型和 T5 CPU 模式调整
t5_cpu_mode = MODEL_CONFIG["t5_cpu"]
is_npu = device_info.get("device_type") == "npu"

BUSINESS_CONFIG = {
    "max_concurrent_tasks": int(os.environ.get(
        "MAX_CONCURRENT_TASKS", 
        "2" if (t5_cpu_mode or is_npu) else "5"
    )),
    "task_timeout": int(os.environ.get(
        "TASK_TIMEOUT", 
        "2400" if t5_cpu_mode else "1800"
    )),
    "cleanup_interval": int(os.environ.get("CLEANUP_INTERVAL", "300")),
    "max_video_output_dir_size": int(os.environ.get("MAX_OUTPUT_DIR_SIZE", "50")),
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
        self.task_start_times = {}
    
    async def acquire_task_slot(self, task_id: str = None) -> bool:
        """获取任务槽位"""
        async with self._lock:
            if self.concurrent_tasks >= self.max_concurrent:
                return False
            self.concurrent_tasks += 1
            if task_id:
                self.task_start_times[task_id] = time.time()
            return True
    
    async def release_task_slot(self, task_id: str = None):
        """释放任务槽位"""
        async with self._lock:
            if self.concurrent_tasks > 0:
                self.concurrent_tasks -= 1
            if task_id and task_id in self.task_start_times:
                duration = time.time() - self.task_start_times[task_id]
                logger.info(f"Task {task_id} completed in {duration:.2f} seconds")
                del self.task_start_times[task_id]
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取资源统计"""
        async with self._lock:
            return {
                "concurrent_tasks": self.concurrent_tasks,
                "max_concurrent": self.max_concurrent,
                "available_slots": self.max_concurrent - self.concurrent_tasks,
                "active_tasks": len(self.task_start_times)
            }

# ==================== 初始化 ====================

logger.info(f"Initializing pipeline on rank {LOCAL_RANK}/{WORLD_SIZE}")
logger.info(f"Device info: {device_info}")
logger.info(f"Model configuration: {MODEL_CONFIG}")
logger.info(f"T5 CPU mode: {t5_cpu_mode}")

try:
    # 使用工厂模式创建管道，自动适配设备
    pipeline = PipelineFactory.create_pipeline(**MODEL_CONFIG)
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {str(e)}")
    if LOCAL_RANK == 0:
        sys.exit(1)
    raise

# ==================== FastAPI 应用（仅在主进程） ====================

if LOCAL_RANK == 0:
    
    # 初始化服务组件
    video_service = VideoService(pipeline)
    resource_manager = ResourceManager()
    
    async def periodic_cleanup():
        """定期清理任务"""
        while True:
            try:
                await asyncio.sleep(BUSINESS_CONFIG["cleanup_interval"])
                
                # 清理过期任务
                cleaned_count = await video_service.cleanup_expired_tasks()
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} expired tasks")
                
                # 清理旧视频文件
                await cleanup_old_videos()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {str(e)}")

    async def cleanup_old_videos():
        """清理旧视频文件"""
        try:
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
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        total_size -= file_size
                        if total_size <= max_size * 0.8:  # 清理到80%
                            break
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {str(e)}")
                
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
        app.state.pipeline = pipeline
        
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

    # 创建 FastAPI 应用
    app = FastAPI(
        title="Multi-GPU Video Generation API",
        description=f"基于 Wan2.1-I2V-14B-720P 的多设备分布式视频生成服务 (Device: {device_info.get('device_type', 'unknown').upper()}, T5 CPU: {t5_cpu_mode})",
        version="3.0.0",
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

    async def release_task_slot_after_completion(task_id: str, resource_manager: ResourceManager):
        """任务完成后释放资源槽位"""
        try:
            max_wait_time = BUSINESS_CONFIG["task_timeout"]
            start_time = time.time()
            
            while True:
                if time.time() - start_time > max_wait_time:
                    logger.warning(f"Task {task_id} monitoring timeout after {max_wait_time}s")
                    break
                
                task = await video_service.get_task_status(task_id)
                if not task or task["status"] in ["Succeed", "Failed", "Cancelled"]:
                    break
                await asyncio.sleep(5)
            
            await resource_manager.release_task_slot(task_id)
            logger.info(f"Released resource slot for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error releasing resource slot for task {task_id}: {str(e)}")

    @app.post("/video/submit", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
    async def submit_video_task(request: VideoSubmitRequest, background_tasks: BackgroundTasks):
        """提交视频生成任务"""
        task_id = None
        try:
            import uuid
            task_id = str(uuid.uuid4()).replace('-', '')
            
            if not await resource_manager.acquire_task_slot(task_id):
                raise HTTPException(
                    status_code=429, 
                    detail=f"服务器繁忙，请稍后重试。当前最大并发数: {BUSINESS_CONFIG['max_concurrent_tasks']}"
                )
            
            # 提交任务
            task_id = await video_service.submit_video_task(request)
            
            # 添加资源释放任务
            background_tasks.add_task(
                release_task_slot_after_completion, 
                task_id, 
                resource_manager
            )
            
            return {"requestId": task_id}
            
        except HTTPException:
            if task_id:
                await resource_manager.release_task_slot(task_id)
            raise
        except Exception as e:
            if task_id:
                await resource_manager.release_task_slot(task_id)
            logger.error(f"Failed to submit task: {str(e)}")
            raise VideoGenerationException("任务提交失败，请检查参数或稍后重试", "TASK_SUBMIT_ERROR")

    @app.post("/video/status", response_model=VideoStatusResponse)
    async def get_task_status(request: VideoStatusRequest):
        """查询任务状态"""
        task = await video_service.get_task_status(request.requestId)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        
        return VideoStatusResponse(**task)

    @app.post("/video/cancel", response_model=dict)
    async def cancel_task(request: VideoCancelRequest):
        """取消任务"""
        success = await video_service.cancel_task(request.requestId)
        if not success:
            task = await video_service.get_task_status(request.requestId)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            return {"status": "Already finished", "message": "任务已完成，无法取消"}
        
        return {"status": "Cancelled", "message": "任务已取消"}

    @app.get("/device_info")
    async def get_device_info():
        """获取设备信息"""
        return {
            "device_info": device_info,
            "model_config": MODEL_CONFIG,
            "business_config": BUSINESS_CONFIG,
            "rank": LOCAL_RANK,
            "world_size": WORLD_SIZE
        }
    
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
                "config": {
                    "t5_cpu": MODEL_CONFIG["t5_cpu"],
                    "dit_fsdp": MODEL_CONFIG["dit_fsdp"],
                    "vae_parallel": MODEL_CONFIG["vae_parallel"],
                    "max_concurrent": BUSINESS_CONFIG["max_concurrent_tasks"],
                    "task_timeout": BUSINESS_CONFIG["task_timeout"],
                    "device_type": device_info.get("device_type", "unknown")
                },
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
                "config": {
                    "model": MODEL_CONFIG,
                    "business": BUSINESS_CONFIG
                },
                "system": {
                    "rank": LOCAL_RANK,
                    "world_size": WORLD_SIZE,
                    "uptime": time.time() - app.state.start_time,
                    "device_type": device_info.get("device_type", "unknown")
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
            "version": "3.0.0",
            "status": "running",
            "config": {
                "t5_cpu": MODEL_CONFIG["t5_cpu"],
                "distributed_inference": WORLD_SIZE > 1,
                "concurrent_tasks": BUSINESS_CONFIG["max_concurrent_tasks"],
                "device_type": device_info.get("device_type", "unknown")
            },
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "metrics": "/metrics",
                "device_info": "/device_info"
            }
        }

    # ==================== 启动服务 ====================

    def run_server():
        """启动服务器"""
        try:
            logger.info(f"Starting FastAPI server on {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
            logger.info(f"Configuration: Device={device_info.get('device_type', 'unknown')}, T5 CPU={MODEL_CONFIG['t5_cpu']}, Max concurrent={BUSINESS_CONFIG['max_concurrent_tasks']}")
            
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
        pipeline.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动服务
    run_server()

else:
    # 非主进程只参与分布式推理
    logger.info(f"Rank {LOCAL_RANK} ready for distributed inference (Device: {device_info.get('device_type', 'unknown')}, T5 CPU: {t5_cpu_mode})")
    
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