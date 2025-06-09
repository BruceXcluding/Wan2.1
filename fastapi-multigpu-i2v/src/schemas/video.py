"""
视频生成相关的数据模型定义
"""
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum

class QualityPreset(str, Enum):
    """质量预设"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    CUSTOM = "custom"

class SampleSolver(str, Enum):
    """采样器类型"""
    UNIPC = "unipc"
    DPMPP = "dpmpp"
    EULER = "euler"
    HEUN = "heun"

class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEED = "Succeed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

class VideoSubmitRequest(BaseModel):
    """视频生成请求模型"""
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "prompt": "A serene lake with a swan gracefully swimming",
                "image_url": "https://example.com/image.jpg",
                "num_frames": 81,
                "guidance_scale": 3.0,
                "infer_steps": 30,
                "quality_preset": "balanced"
            }
        }
    )
    
    # 基础参数
    prompt: str = Field(
        ..., 
        min_length=10, 
        max_length=500,
        description="视频描述提示词"
    )
    image_url: str = Field(
        ..., 
        description="输入图像URL地址"
    )
    
    # 视频参数
    image_size: Optional[str] = Field(
        default="1280*720",
        description="输出视频分辨率",
        pattern=r"^\d+\*\d+$"
    )
    num_frames: Optional[int] = Field(
        default=81,
        ge=24,
        le=121,
        description="视频帧数"
    )
    
    # 生成参数
    guidance_scale: Optional[float] = Field(
        default=3.0,
        ge=1.0,
        le=20.0,
        description="CFG引导系数"
    )
    infer_steps: Optional[int] = Field(
        default=30,
        ge=20,
        le=100,
        description="去噪推理步数"
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="随机数种子"
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        max_length=500,
        description="负面提示词"
    )
    
    # 质量预设
    quality_preset: Optional[QualityPreset] = Field(
        default=QualityPreset.BALANCED,
        description="质量预设模式"
    )
    
    # 高级参数 - 分布式配置
    vae_parallel: Optional[bool] = Field(
        default=False,
        description="VAE并行编解码"
    )
    ulysses_size: Optional[int] = Field(
        default=1,
        ge=1,
        le=8,
        description="Ulysses序列并行组数"
    )
    dit_fsdp: Optional[bool] = Field(
        default=False,
        description="DiT模型FSDP分片"
    )
    t5_fsdp: Optional[bool] = Field(
        default=False,
        description="T5编码器FSDP分片"
    )
    cfg_size: Optional[int] = Field(
        default=1,
        ge=1,
        le=8,
        description="CFG并行组数"
    )
    
    # 性能优化参数
    use_attentioncache: Optional[bool] = Field(
        default=False,
        description="启用注意力缓存"
    )
    start_step: Optional[int] = Field(
        default=12,
        ge=0,
        le=50,
        description="缓存起始步数"
    )
    attentioncache_interval: Optional[int] = Field(
        default=4,
        ge=1,
        le=10,
        description="缓存更新间隔"
    )
    end_step: Optional[int] = Field(
        default=37,
        ge=10,
        le=100,
        description="缓存结束步数"
    )
    sample_solver: Optional[SampleSolver] = Field(
        default=SampleSolver.UNIPC,
        description="采样算法"
    )
    sample_shift: Optional[float] = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="采样偏移"
    )
    
    @field_validator('image_url')
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        """验证图像URL"""
        if not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError('image_url must be a valid HTTP/HTTPS URL')
        return v
    
    @field_validator('image_size')
    @classmethod
    def validate_image_size(cls, v: Optional[str]) -> Optional[str]:
        """验证图像尺寸"""
        if v:
            try:
                width, height = map(int, v.split('*'))
                if width < 512 or height < 512:
                    raise ValueError('Image size must be at least 512x512')
                if width > 2048 or height > 2048:
                    raise ValueError('Image size must not exceed 2048x2048')
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError('Image size format must be WIDTH*HEIGHT')
                raise
        return v

class VideoSubmitResponse(BaseModel):
    """视频生成提交响应"""
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "requestId": "req_1234567890abcdef",
                "status": "Pending",
                "message": "Task submitted successfully",
                "estimated_time": 180
            }
        }
    )
    
    requestId: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    message: str = Field(..., description="状态消息")
    estimated_time: Optional[int] = Field(None, description="预估完成时间(秒)")

class VideoStatusRequest(BaseModel):
    """查询任务状态请求"""
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "requestId": "req_1234567890abcdef"
            }
        }
    )
    
    requestId: str = Field(..., description="任务ID")

class VideoResults(BaseModel):
    """视频生成结果"""
    model_config = ConfigDict(populate_by_name=True)
    
    video_url: str = Field(..., description="生成的视频URL")
    video_path: str = Field(..., description="视频文件路径")
    duration: float = Field(..., description="视频时长(秒)")
    frames: int = Field(..., description="视频帧数")
    size: str = Field(..., description="视频分辨率")
    file_size: int = Field(..., description="文件大小(字节)")

class VideoStatusResponse(BaseModel):
    """查询任务状态响应"""
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "requestId": "req_1234567890abcdef",
                "status": "Succeed",
                "progress": 100,
                "message": "Video generation completed",
                "results": {
                    "video_url": "http://localhost:8088/videos/output_video.mp4",
                    "video_path": "generated_videos/output_video.mp4",
                    "duration": 3.4,
                    "frames": 81,
                    "size": "1280*720",
                    "file_size": 5242880
                },
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T10:03:00Z",
                "elapsed_time": 180,
                "queue_position": 0
            }
        }
    )
    
    requestId: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    progress: int = Field(default=0, ge=0, le=100, description="完成进度百分比")
    message: str = Field(..., description="状态消息")
    results: Optional[VideoResults] = Field(None, description="生成结果")
    reason: Optional[str] = Field(None, description="失败原因")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    elapsed_time: Optional[int] = Field(None, description="已用时间(秒)")
    queue_position: Optional[int] = Field(None, description="队列位置")

class VideoCancelRequest(BaseModel):
    """取消任务请求"""
    model_config = ConfigDict(populate_by_name=True)
    
    requestId: str = Field(..., description="任务ID")
    reason: Optional[str] = Field(None, max_length=200, description="取消原因")

class VideoCancelResponse(BaseModel):
    """取消任务响应"""
    model_config = ConfigDict(populate_by_name=True)
    
    requestId: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    message: str = Field(..., description="状态消息")
    cancelled_at: str = Field(..., description="取消时间")

class ErrorResponse(BaseModel):
    """错误响应"""
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Invalid input parameters",
                "details": ["prompt must be at least 10 characters long"],
                "timestamp": "2024-01-01T10:00:00Z"
            }
        }
    )
    
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    details: Optional[List[str]] = Field(None, description="详细错误信息")
    timestamp: str = Field(..., description="错误时间")
    requestId: Optional[str] = Field(None, description="关联的请求ID")

class HealthResponse(BaseModel):
    """健康检查响应"""
    model_config = ConfigDict(populate_by_name=True)
    
    status: str = Field(..., description="服务状态")
    timestamp: float = Field(..., description="检查时间戳")
    uptime: float = Field(..., description="运行时间(秒)")
    config: Dict[str, Any] = Field(..., description="服务配置")
    service: Dict[str, Any] = Field(..., description="服务统计")
    resources: Dict[str, Any] = Field(..., description="资源使用")

class MetricsResponse(BaseModel):
    """监控指标响应"""
    model_config = ConfigDict(populate_by_name=True)
    
    timestamp: float = Field(..., description="指标时间戳")
    system: Dict[str, Any] = Field(..., description="系统指标")
    service: Dict[str, Any] = Field(..., description="服务指标")
    tasks: Dict[str, Any] = Field(..., description="任务指标")
    performance: Dict[str, Any] = Field(..., description="性能指标")

class DeviceInfoResponse(BaseModel):
    """设备信息响应"""
    model_config = ConfigDict(populate_by_name=True)
    
    device_type: str = Field(..., description="设备类型")
    device_count: int = Field(..., description="设备数量")
    devices: List[Dict[str, Any]] = Field(..., description="设备详情")
    driver_version: Optional[str] = Field(None, description="驱动版本")
    runtime_version: Optional[str] = Field(None, description="运行时版本")