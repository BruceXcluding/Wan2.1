import os
import asyncio
import time
import logging
import signal
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    VideoSubmitRequest,
    VideoStatusRequest,
    VideoStatusResponse,
    VideoCancelRequest
)
from services.video_service import VideoService
from multigpu_pipeline import MultiGPUVideoPipeline

# ==================== 配置 ====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

MODEL_CONFIG = {
    "ckpt_dir": "/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P",
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

# ==================== 初始化 ====================

logger.info(f"Initializing pipeline on rank {LOCAL_RANK}")
pipeline = MultiGPUVideoPipeline(**MODEL_CONFIG)

# ==================== FastAPI 应用 ====================

if LOCAL_RANK == 0:
    # 主进程启动 HTTP 服务
    
    video_service = VideoService(pipeline)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        logger.info("FastAPI application starting up")
        
        # 启动定期清理任务
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
        yield
        
        # 清理资源
        cleanup_task.cancel()
        pipeline.cleanup()
        logger.info("FastAPI application shut down")

    async def periodic_cleanup():
        """定期清理过期任务"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟清理一次
                await video_service.cleanup_expired_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {str(e)}")

    # 创建 FastAPI 应用
    app = FastAPI(
        title="Multi-GPU Video Generation API",
        description="基于 Wan2.1-I2V-14B-720P 的多卡分布式视频生成服务",
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
    app.mount("/videos", StaticFiles(directory="generated_videos"), name="videos")

    # ==================== API 路由 ====================

    @app.post("/video/submit", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
    async def submit_video_task(request: VideoSubmitRequest):
        """提交视频生成任务"""
        try:
            task_id = await video_service.submit_video_task(request)
            return {"requestId": task_id}
        except Exception as e:
            logger.error(f"Failed to submit task: {str(e)}")
            raise HTTPException(status_code=500, detail="任务提交失败")

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
        stats = await video_service.get_service_stats()
        return {
            "status": "healthy",
            "rank": LOCAL_RANK,
            **stats
        }

    @app.get("/")
    async def root():
        """根路径"""
        return {
            "service": "Multi-GPU Video Generation API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }

    # 启动服务
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)

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