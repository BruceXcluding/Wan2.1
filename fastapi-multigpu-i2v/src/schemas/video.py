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

class QualityPreset(str, Enum):
    """质量预设"""
    FAST = "fast"           # 快速：20步，简单参数
    BALANCED = "balanced"   # 平衡：30步，默认参数
    HIGH = "high"           # 高质量：50步，优化参数
    CUSTOM = "custom"       # 自定义：使用用户参数

class VideoSubmitRequest(BaseModel):
    """视频生成请求"""
    # === 基础参数 ===
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
    
    # === 输出参数 ===
    image_size: str = Field(
        default="1280*720",
        description="输出分辨率，支持: 1280*720, 1920*1080, 1024*576 等"
    )
    num_frames: int = Field(
        default=81,
        ge=24,
        le=121,
        description="视频帧数，建议值: 41, 61, 81, 121"
    )
    
    # === 质量控制 ===
    quality_preset: QualityPreset = Field(
        default=QualityPreset.BALANCED,
        description="质量预设：fast(快速), balanced(平衡), high(高质量), custom(自定义)"
    )
    
    # === 创意参数 ===
    negative_prompt: Optional[str] = Field(
        default="",
        max_length=500,
        description="负面提示词，描述不想要的内容"
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2147483647,
        description="随机种子，固定种子可复现结果"
    )
    
    # === 高级参数（quality_preset为custom时生效） ===
    guidance_scale: Optional[float] = Field(
        default=3.0,
        ge=1.0,
        le=20.0,
        description="引导系数，控制提示词影响程度，1.0-20.0"
    )
    infer_steps: Optional[int] = Field(
        default=30,
        ge=20,
        le=100,
        description="推理步数，影响质量和速度，20-100步"
    )
    sample_shift: Optional[float] = Field(
        default=5.0, 
        ge=0.1, 
        le=10.0, 
        description="采样shift参数，影响生成风格"
    )
    sample_solver: Optional[SampleSolver] = Field(
        default=SampleSolver.UNIPC, 
        description="采样算法，unipc(推荐), dpmpp, euler"
    )
    
    # === 系统参数（专家模式，隐藏在文档中）===
    # 这些参数普通用户不需要修改，但保留给高级用户
    _t5_fsdp: Optional[bool] = Field(
        default=None, 
        alias="t5_fsdp",
        description="[专家] T5模型FSDP分片，None为自动"
    )
    _dit_fsdp: Optional[bool] = Field(
        default=None,
        alias="dit_fsdp", 
        description="[专家] DiT模型FSDP分片，None为自动"
    )
    _cfg_size: Optional[int] = Field(
        default=None,
        alias="cfg_size",
        ge=1, 
        description="[专家] CFG并行组数，None为自动"
    )
    _ulysses_size: Optional[int] = Field(
        default=None,
        alias="ulysses_size",
        ge=1, 
        description="[专家] Ulysses序列并行组数，None为自动"
    )
    _vae_parallel: Optional[bool] = Field(
        default=None,
        alias="vae_parallel",
        description="[专家] VAE并行处理，None为自动"
    )
    
    # === 性能优化参数（专家模式）===
    _use_attentioncache: Optional[bool] = Field(
        default=None,
        alias="use_attentioncache",
        description="[专家] 注意力缓存优化，None为自动"
    )
    _cache_start_step: Optional[int] = Field(
        default=None,
        alias="cache_start_step",
        ge=0, 
        description="[专家] 缓存起始步数"
    )
    _cache_interval: Optional[int] = Field(
        default=None,
        alias="cache_interval",
        ge=1, 
        description="[专家] 缓存间隔步数"
    )
    _cache_end_step: Optional[int] = Field(
        default=None,
        alias="cache_end_step",
        ge=1, 
        description="[专家] 缓存结束步数"
    )
    
    @validator('image_size')
    def validate_image_size(cls, v):
        """验证图像尺寸"""
        if v == "auto":
            return "1280*720"
        
        if '*' not in v:
            raise ValueError("图像尺寸格式应为 '宽*高'，如 '1280*720'")
        
        try:
            width, height = map(int, v.split('*'))
            if width <= 0 or height <= 0:
                raise ValueError("宽度和高度必须为正数")
            if width % 8 != 0 or height % 8 != 0:
                raise ValueError("宽度和高度必须是8的倍数")
            if width > 3840 or height > 2160:
                raise ValueError("分辨率过高，最大支持 3840*2160")
        except ValueError as e:
            if "必须是8的倍数" in str(e) or "分辨率过高" in str(e):
                raise e
            raise ValueError("图像尺寸格式无效，应为数字格式如 '1280*720'")
        
        return v
    
    @validator('num_frames')
    def validate_num_frames(cls, v):
        """验证并调整帧数为有效值"""
        valid_frames = [41, 61, 81, 121]
        if v not in valid_frames:
            closest = min(valid_frames, key=lambda x: abs(x - v))
            return closest
        return v
    
    def get_effective_params(self) -> dict:
        """根据质量预设获取有效参数"""
        # 质量预设参数
        preset_configs = {
            QualityPreset.FAST: {
                "guidance_scale": 2.5,
                "infer_steps": 20,
                "sample_shift": 4.0,
                "sample_solver": SampleSolver.UNIPC
            },
            QualityPreset.BALANCED: {
                "guidance_scale": 3.0,
                "infer_steps": 30,
                "sample_shift": 5.0,
                "sample_solver": SampleSolver.UNIPC
            },
            QualityPreset.HIGH: {
                "guidance_scale": 4.0,
                "infer_steps": 50,
                "sample_shift": 6.0,
                "sample_solver": SampleSolver.DPMPP
            }
        }
        
        if self.quality_preset == QualityPreset.CUSTOM:
            # 使用用户自定义参数
            return {
                "guidance_scale": self.guidance_scale or 3.0,
                "infer_steps": self.infer_steps or 30,
                "sample_shift": self.sample_shift or 5.0,
                "sample_solver": self.sample_solver or SampleSolver.UNIPC
            }
        else:
            # 使用预设参数
            return preset_configs[self.quality_preset]
    
    def get_system_params(self) -> dict:
        """获取系统参数（过滤None值）"""
        return {
            k.lstrip('_'): v for k, v in {
                '_t5_fsdp': self._t5_fsdp,
                '_dit_fsdp': self._dit_fsdp,
                '_cfg_size': self._cfg_size,
                '_ulysses_size': self._ulysses_size,
                '_vae_parallel': self._vae_parallel,
                '_use_attentioncache': self._use_attentioncache,
                '_cache_start_step': self._cache_start_step,
                '_cache_interval': self._cache_interval,
                '_cache_end_step': self._cache_end_step,
            }.items() if v is not None
        }
    
    class Config:
        use_enum_values = True
        allow_population_by_field_name = True  # 允许使用别名
        schema_extra = {
            "example": {
                "prompt": "A beautiful sunset over the ocean with gentle waves",
                "image_url": "https://example.com/input.jpg",
                "image_size": "1280*720",
                "num_frames": 81,
                "quality_preset": "balanced",
                "negative_prompt": "blurry, low quality",
                "seed": 42
            }
        }

# === 其他模型保持不变 ===

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
    frame_count: Optional[int] = Field(None, description="实际帧数")
    resolution: Optional[str] = Field(None, description="实际分辨率")
    
    # 添加生成参数记录
    used_params: Optional[dict] = Field(None, description="实际使用的生成参数")

class VideoStatusResponse(BaseModel):
    """状态查询响应"""
    status: TaskStatus = Field(..., description="任务状态")
    reason: Optional[str] = Field(None, description="失败原因")
    results: Optional[VideoResults] = Field(None, description="生成结果")
    queue_position: Optional[int] = Field(None, description="队列位置")
    progress: Optional[float] = Field(None, ge=0, le=1, description="进度百分比 0-1")
    created_at: Optional[float] = Field(None, description="创建时间戳")
    updated_at: Optional[float] = Field(None, description="更新时间戳")
    estimated_time: Optional[int] = Field(None, description="预计剩余时间(秒)")

class VideoCancelRequest(BaseModel):
    """取消任务请求"""
    requestId: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="32位任务ID"
    )