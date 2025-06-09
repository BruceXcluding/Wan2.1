import os
from typing import Optional, Dict
from multigpu_pipeline import MultiGPUVideoPipeline

local_rank = int(os.environ.get("LOCAL_RANK", 0))

# 所有进程都初始化分布式模型
ckpt_dir = "/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P"
model_args = dict(
    task="i2v-14B",
    size="1280*720",
    frame_num=81,
    sample_steps=40,
    dit_fsdp=True,
    t5_fsdp=True,
    cfg_size=1,
    ulysses_size=8,
    vae_parallel=True,
    use_attentioncache=True,
    start_step=12,
    attentioncache_interval=4,
    end_step=37,
)
pipeline = MultiGPUVideoPipeline(ckpt_dir=ckpt_dir, **model_args)

# 只有 rank 0 启动 HTTP 服务
if local_rank == 0:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    import uuid
    import time
    import asyncio
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        app.state.pipe = pipeline
        yield

    app = FastAPI(lifespan=lifespan)
    app.mount("/videos", StaticFiles(directory="generated_videos"), name="videos")

    # 任务状态全局变量
    TASKS: Dict[str, dict] = {}
    TASKS_LOCK = asyncio.Lock()

    class VideoSubmitRequest(BaseModel):
        model: str = Field(default="Wan2.1-I2V-14B-720P", description="模型版本")
        prompt: str = Field(..., min_length=10, max_length=500, description="视频描述提示词，10-500个字符")
        image_url: str = Field(..., description="输入图像URL，需支持HTTP/HTTPS协议")
        image_size: str = Field(default="auto", description="输出分辨率，格式：宽x高 或 auto（自动计算）")
        negative_prompt: Optional[str] = Field(default=None, max_length=500, description="排除不需要的内容")
        seed: Optional[int] = Field(default=None, ge=0, le=2147483647, description="随机数种子")
        num_frames: int = Field(default=81, ge=24, le=120, description="视频帧数")
        guidance_scale: float = Field(default=3.0, ge=1.0, le=20.0, description="引导系数")
        infer_steps: int = Field(default=30, ge=20, le=100, description="推理步数")
        t5_fsdp: Optional[bool] = Field(default=False, description="是否启用T5 FSDP")
        dit_fsdp: Optional[bool] = Field(default=False, description="是否启用DiT FSDP")
        cfg_size: Optional[int] = Field(default=1, description="CFG组数")
        ulysses_size: Optional[int] = Field(default=1, description="Ulysses组数")
        vae_parallel: Optional[bool] = Field(default=False, description="是否VAE并行")
        use_attentioncache: Optional[bool] = Field(default=False, description="是否使用AttentionCache")
        start_step: Optional[int] = Field(default=12, description="AttentionCache起始步")
        attentioncache_interval: Optional[int] = Field(default=4, description="AttentionCache间隔")
        end_step: Optional[int] = Field(default=37, description="AttentionCache结束步")
        sample_shift: Optional[float] = Field(default=5.0, description="采样shift")
        sample_solver: Optional[str] = Field(default="unipc", description="采样算法")

    class VideoStatusRequest(BaseModel):
        requestId: str = Field(..., min_length=32, max_length=32, description="32位任务ID")

    class VideoStatusResponse(BaseModel):
        status: str = Field(..., description="任务状态: Succeed, InQueue, InProgress, Failed, Cancelled")
        reason: Optional[str] = Field(None, description="失败原因")
        results: Optional[dict] = Field(None, description="生成结果")
        queue_position: Optional[int] = Field(None, description="队列位置")

    class VideoCancelRequest(BaseModel):
        requestId: str = Field(..., min_length=32, max_length=32, description="32位任务ID")

    @app.post("/video/submit", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
    async def submit_task(request: VideoSubmitRequest):
        task_id = uuid.uuid4().hex
        async with TASKS_LOCK:
            TASKS[task_id] = {
                "status": "InQueue",
                "reason": None,
                "results": None,
                "queue_position": None,
                "created_at": time.time()
            }
        asyncio.create_task(process_video_generation(request, task_id, app))
        return {"requestId": task_id}

    async def process_video_generation(request: VideoSubmitRequest, task_id: str, app: FastAPI):
        async with TASKS_LOCK:
            TASKS[task_id]["status"] = "InProgress"
        try:
            print(f"[DEBUG] Starting video generation for task {task_id}")
        
            # 传递 task_id 参数给 generate_video
            video_path = await app.state.pipe.generate_video(request, task_id)
        
            print(f"[DEBUG] Video generation completed for task {task_id}: {video_path}")
        
            async with TASKS_LOCK:
                TASKS[task_id]["status"] = "Succeed"
                TASKS[task_id]["results"] = {
                    "video_url": f"/videos/{task_id}.mp4",
                    "video_path": video_path
                }
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {str(e)}")
            import traceback
            traceback.print_exc()

            async with TASKS_LOCK:
                TASKS[task_id]["status"] = "Failed"
                TASKS[task_id]["reason"] = str(e)
    
    @app.post("/video/status", response_model=VideoStatusResponse)
    async def get_status(request: VideoStatusRequest):
        async with TASKS_LOCK:
            task = TASKS.get(request.requestId)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            return VideoStatusResponse(
                status=task["status"],
                reason=task["reason"],
                results=task["results"],
                queue_position=None
            )

    @app.post("/video/cancel", response_model=dict)
    async def cancel_task(request: VideoCancelRequest):
        async with TASKS_LOCK:
            task = TASKS.get(request.requestId)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            if task["status"] in ["Succeed", "Failed", "Cancelled"]:
                return {"status": "Already finished"}
            task["status"] = "Cancelled"
        return {"status": "Cancelled"}

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)
else:
    # 非0号卡进程只做分布式推理，不启动HTTP服务
    pass