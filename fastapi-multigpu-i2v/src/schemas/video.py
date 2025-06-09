from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum

class TaskStatus(str, Enum):
    """任务状态枚举"""
    IN_QUEUE = "InQueue"
    IN_PROGRESS = "InProgress"
    SUCCEED = "Succeed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

class SampleSolver(str, Enum):
    """采样算法枚举"""
    UNIPC = "unipc"
    DPMPP = "dpmpp"
    EULER = "euler"

class VideoSubmitRequest(BaseModel):
    """视频生成请求"""
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
        default="1280*720",  # 改为具体默认值，避免 "auto" 问题
        description="输出分辨率，格式：宽*高，如 1280*720"
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
    
    # 视频生成参数
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
    
    # 分布式训练参数
    t5_fsdp: Optional[bool] = Field(default=False, description="是否启用T5 FSDP")
    dit_fsdp: Optional[bool] = Field(default=False, description="是否启用DiT FSDP")
    cfg_size: Optional[int] = Field(default=1, ge=1, description="CFG组数")
    ulysses_size: Optional[int] = Field(default=1, ge=1, description="Ulysses组数")
    vae_parallel: Optional[bool] = Field(default=False, description="是否VAE并行")
    
    # 注意力缓存参数
    use_attentioncache: Optional[bool] = Field(default=False, description="是否使用AttentionCache")
    start_step: Optional[int] = Field(default=12, ge=0, description="AttentionCache起始步")
    attentioncache_interval: Optional[int] = Field(default=4, ge=1, description="AttentionCache间隔")
    end_step: Optional[int] = Field(default=37, ge=1, description="AttentionCache结束步")
    
    # 采样参数
    sample_shift: Optional[float] = Field(default=5.0, ge=0.1, le=10.0, description="采样shift")
    sample_solver: Optional[SampleSolver] = Field(default=SampleSolver.UNIPC, description="采样算法")
    
    @validator('image_size')
    def validate_image_size(cls, v):
        """验证和标准化图像尺寸"""
        if v == "auto":
            return "1280*720"  # 自动转换为默认值
        
        if '*' not in v:
            raise ValueError("图像尺寸格式应为 '宽*高'，如 '1280*720'")
        
        try:
            width, height = map(int, v.split('*'))
            if width <= 0 or height <= 0:
                raise ValueError("宽度和高度必须为正数")
            if width % 8 != 0 or height % 8 != 0:
                raise ValueError("宽度和高度必须是8的倍数")
        except ValueError as e:
            if "must be a multiple of 8" in str(e):
                raise e
            raise ValueError("图像尺寸格式无效，应为数字格式如 '1280*720'")
        
        return v
    
    class Config:
        use_enum_values = True

class VideoStatusRequest(BaseModel):
    """查询状态请求"""
    requestId: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="32位任务ID"
    )

class VideoResults(BaseModel):
    """生成结果"""
    video_url: str = Field(..., description="视频下载链接")
    video_path: Optional[str] = Field(None, description="服务器文件路径")
    duration: Optional[float] = Field(None, description="视频时长(秒)")
    file_size: Optional[int] = Field(None, description="文件大小(字节)")

class VideoStatusResponse(BaseModel):
    """状态查询响应"""
    status: TaskStatus = Field(..., description="任务状态")
    reason: Optional[str] = Field(None, description="失败原因")
    results: Optional[VideoResults] = Field(None, description="生成结果")
    queue_position: Optional[int] = Field(None, description="队列位置")
    progress: Optional[float] = Field(None, ge=0, le=1, description="进度百分比")
    created_at: Optional[float] = Field(None, description="创建时间戳")
    updated_at: Optional[float] = Field(None, description="更新时间戳")

class VideoCancelRequest(BaseModel):
    """取消任务请求"""
    requestId: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="32位任务ID"
    )

class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误描述")
    details: Optional[dict] = Field(None, description="详细信息")