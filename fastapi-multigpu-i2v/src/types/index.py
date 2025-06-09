from pydantic import BaseModel, Field
from typing import Optional, List

class VideoSubmitRequest(BaseModel):
    model: str = Field(
        default="Wan2.1-I2V-14B-720P",
        description="模型版本"
    )
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="视频描述提示词，10-500个字符"
    )
    image_url: str = Field(
        ...,
        description="输入图像URL，需支持HTTP/HTTPS协议"
    )
    image_size: str = Field(
        default="auto",
        description="输出分辨率，格式：宽x高 或 auto（自动计算）"
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        max_length=500,
        description="排除不需要的内容"
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="随机数种子，范围0-2147483647"
    )
    num_frames: int = Field(
        default=81,
        ge=24,
        le=120,
        description="视频帧数，24-120帧"
    )
    guidance_scale: float = Field(
        default=3.0,
        ge=1.0,
        le=20.0,
        description="引导系数，1.0-20.0"
    )
    infer_steps: int = Field(
        default=30,
        ge=20,
        le=100,
        description="推理步数，20-100步"
    )
    # 新增参数，覆盖命令行所有可调项
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
    # 你可以继续补充其它命令行参数...

class VideoStatusRequest(BaseModel):
    requestId: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="32位任务ID"
    )

class VideoStatusResponse(BaseModel):
    status: str = Field(..., description="任务状态: Succeed, InQueue, InProgress, Failed, Cancelled")
    reason: Optional[str] = Field(None, description="失败原因")
    results: Optional[dict] = Field(None, description="生成结果")
    queue_position: Optional[int] = Field(None, description="队列位置")

class VideoCancelRequest(BaseModel):
    requestId: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="32位任务ID"
    )