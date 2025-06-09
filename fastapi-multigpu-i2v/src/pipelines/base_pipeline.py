"""
重构后的基础管道类 - 使用模板方法模式和混入类
"""
import abc
import os
import time
import logging
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Optional, Callable, Dict, Any
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

class DistributedMixin:
    """分布式功能混入类"""
    
    def _init_distributed_common(self, backend: str, timeout_seconds: int = 1800):
        """通用分布式初始化逻辑"""
        if self.world_size <= 1:
            logger.info(f"Single {self.device_type} mode, skipping distributed initialization")
            return
        
        # 设备特定配置
        self._set_device()
        
        if not dist.is_initialized():
            logger.info(f"Initializing distributed process group with {backend} backend")
            try:
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=timedelta(seconds=timeout_seconds)
                )
                
                # 测试通信
                self._test_communication()
                
            except Exception as e:
                logger.error(f"Failed to initialize {backend} process group: {str(e)}")
                raise
    
    def _test_communication(self):
        """测试分布式通信"""
        logger.info(f"Testing {self.device_type} distributed communication...")
        test_tensor = torch.tensor([self.rank], dtype=torch.float32)
        test_tensor = self._move_to_device(test_tensor)
        dist.all_reduce(test_tensor)
        logger.info(f"{self.device_type} communication test passed, sum: {test_tensor.item()}")
    
    def _broadcast_image_path_common(self, image_path: Optional[str]) -> str:
        """通用图片路径广播逻辑"""
        if self.world_size == 1:
            return image_path
        
        try:
            # 创建路径字符串的张量
            if self.rank == 0:
                path_str = image_path or ""
                path_bytes = path_str.encode('utf-8')
                size_tensor = torch.tensor([len(path_bytes)], dtype=torch.long)
                path_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8)
            else:
                size_tensor = torch.tensor([0], dtype=torch.long)
                path_tensor = None
            
            # 移动到设备
            size_tensor = self._move_to_device(size_tensor)
            
            # 广播大小
            dist.broadcast(size_tensor, 0)
            
            # 创建接收张量
            if self.rank != 0:
                path_tensor = torch.zeros(size_tensor[0].item(), dtype=torch.uint8)
            
            # 移动到设备并广播路径
            if size_tensor[0].item() > 0:
                path_tensor = self._move_to_device(path_tensor)
                dist.broadcast(path_tensor, 0)
            
            # 转换回字符串
            if size_tensor[0].item() > 0:
                received_path = bytes(path_tensor.cpu().tolist()).decode('utf-8')
            else:
                received_path = ""
            
            logger.info(f"Rank {self.rank} received image path: {received_path}")
            return received_path
            
        except Exception as e:
            logger.error(f"Image path broadcast failed: {e}")
            raise

class ModelLoaderMixin:
    """模型加载功能混入类"""
    
    def _load_model_common(self, model_config: Dict[str, Any]):
        """通用模型加载逻辑"""
        try:
            import wan
            cfg = wan.configs.WAN_CONFIGS[model_config.get("task", "i2v-14B")]
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Rank {self.rank}: Loading {self.device_type} model attempt {attempt + 1}/{max_retries}")
                    
                    model = wan.WanI2V(
                        config=cfg,
                        checkpoint_dir=self.ckpt_dir,
                        device_id=self.local_rank,
                        rank=self.rank,
                        **model_config
                    )
                    
                    # 设备特定的模型配置
                    self._configure_model(model)
                    
                    # T5 CPU模式预热
                    if self.t5_cpu:
                        self._warmup_t5_cpu(model)
                    
                    logger.info(f"Rank {self.rank}: {self.device_type} model loaded successfully")
                    return model
                    
                except Exception as e:
                    logger.warning(f"Rank {self.rank}: Model loading attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    
                    time.sleep(5 * (attempt + 1))  # 递增延迟
            
        except Exception as e:
            logger.error(f"Failed to load {self.device_type} model: {str(e)}")
            raise
    
    def _warmup_t5_cpu(self, model):
        """T5 CPU模式预热"""
        if not self.t5_cpu:
            return
        
        try:
            logger.info(f"Rank {self.rank}: Starting T5 CPU warmup...")
            
            # 执行预热推理
            dummy_text = "warmup text for t5 encoder"
            with torch.no_grad():
                _ = model.text_encoder.encode([dummy_text])
            
            # 分布式同步
            if self.world_size > 1:
                logger.info(f"Rank {self.rank}: T5 warmup sync...")
                dist.barrier()
            
            logger.info(f"Rank {self.rank}: T5 warmup completed")
            
        except Exception as e:
            logger.warning(f"Rank {self.rank}: T5 warmup failed: {str(e)}")
            if self.world_size > 1:
                try:
                    dist.barrier()
                except Exception as sync_e:
                    logger.error(f"Rank {self.rank}: Failed to sync after warmup failure: {str(sync_e)}")

class VideoGenerationMixin:
    """视频生成功能混入类"""
    
    def _generate_video_common(self, request, image_path: str, output_path: str) -> str:
        """通用视频生成逻辑"""
        try:
            self._log_memory_usage()
            
            # 处理参数
            size = self._normalize_size(request.image_size or "1280*720")
            img = Image.open(image_path).convert("RGB")
            frame_num = self._validate_frame_num(request.num_frames or 81)
            
            logger.info(f"Generating video with size={size}, frames={frame_num}")
            logger.info(f"{self.device_type} T5 CPU mode: {self.t5_cpu}")
            
            start_time = time.time()
            
            # 分布式同步
            if self.world_size > 1:
                logger.info(f"Rank {self.rank} waiting at {self.device_type} generation sync barrier")
                dist.barrier()
                logger.info(f"Rank {self.rank} passed {self.device_type} generation sync barrier")
            
            # 调用设备特定的生成方法
            video_tensor = self._generate_video_device_specific(request, img, size, frame_num)
            
            # 保存视频
            if video_tensor is not None:
                logger.info(f"Generated video tensor shape: {video_tensor.shape}")
                self._save_video(video_tensor, output_path, frame_num)
                logger.info(f"Video saved to {output_path}")
            else:
                logger.info(f"Non-master rank {self.rank}, video not saved")
            
            # 最终同步
            if self.world_size > 1:
                logger.info(f"Rank {self.rank} final {self.device_type} barrier")
                dist.barrier()
            
            generation_time = time.time() - start_time
            logger.info(f"Rank {self.rank}: {self.device_type} generation completed in {generation_time:.2f}s")
            
            self._log_memory_usage()
            return output_path
            
        except Exception as e:
            logger.error(f"{self.device_type} video generation failed: {str(e)}")
            self._empty_cache()
            raise
    
    def _save_video(self, video_tensor, output_path: str, frame_num: int):
        """保存视频"""
        try:
            from wan.utils.utils import cache_video
            cache_video(
                tensor=video_tensor[None],
                save_file=output_path,
                fps=max(8, frame_num // 10),
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            raise
    
    def _validate_frame_num(self, frame_num: int) -> int:
        """验证和调整帧数"""
        valid_frames = [41, 61, 81, 121]
        if frame_num in valid_frames:
            return frame_num
        
        closest_frame = min(valid_frames, key=lambda x: abs(x - frame_num))
        logger.warning(f"Frame number {frame_num} adjusted to {closest_frame}")
        return closest_frame
    
    def _normalize_size(self, size: str) -> str:
        """标准化尺寸参数"""
        if size == "auto":
            return "1280*720"
        
        try:
            if '*' in size:
                width, height = map(int, size.split('*'))
                if width > 0 and height > 0 and width <= 4096 and height <= 4096:
                    return size
        except:
            pass
        
        logger.warning(f"Invalid size {size}, using default 1280*720")
        return "1280*720"

class BasePipeline(DistributedMixin, ModelLoaderMixin, VideoGenerationMixin):
    """重构后的基础管道类 - 使用模板方法模式"""
    
    def __init__(self, ckpt_dir: str, **model_args):
        self.ckpt_dir = ckpt_dir
        self.model_args = model_args
        self.model = None
        self.device_type = "unknown"  # 子类需要设置
        
        # 分布式环境
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # T5 CPU配置
        self.t5_cpu = model_args.get('t5_cpu', False)
        
        logger.info(f"Initializing {self.__class__.__name__} on rank {self.rank}")
    
    # 模板方法 - 定义算法骨架
    def initialize(self):
        """初始化管道 - 模板方法"""
        self._setup_environment()
        self._init_distributed()
        self.model = self._load_model()
        logger.info(f"{self.device_type} Pipeline initialized successfully")
    
    def generate_video(self, request, task_id: str, progress_callback: Optional[Callable] = None) -> str:
        """生成视频 - 模板方法"""
        logger.info(f"Starting video generation for task {task_id} on rank {self.rank}")
        
        try:
            # 下载图片（只在 rank 0 执行）
            image_path = None
            if self.rank == 0:
                image_path = self._download_image_sync(request.image_url, task_id)
            
            # 广播图片路径
            image_path = self._broadcast_image_path(image_path)
            
            # 生成视频
            output_path = self._get_output_path(task_id)
            result_path = self._generate_video_common(request, image_path, output_path)
            
            logger.info(f"Video generation completed for task {task_id}: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Video generation failed for task {task_id}: {str(e)}")
            self._empty_cache()
            raise
    
    def cleanup(self):
        """清理资源 - 模板方法"""
        try:
            logger.info(f"Cleaning up {self.device_type} pipeline on rank {self.rank}")
            
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            self._empty_cache()
            
            if self.world_size > 1 and dist.is_initialized():
                dist.destroy_process_group()
            
            logger.info(f"{self.device_type} pipeline cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during {self.device_type} cleanup: {str(e)}")
    
    # 钩子方法 - 子类可以重写这些方法来提供特定实现
    @abc.abstractmethod
    def _setup_environment(self):
        """设置环境变量 - 子类实现"""
        pass
    
    @abc.abstractmethod
    def _set_device(self):
        """设置设备 - 子类实现"""
        pass
    
    @abc.abstractmethod
    def _move_to_device(self, tensor):
        """移动张量到设备 - 子类实现"""
        pass
    
    @abc.abstractmethod
    def _configure_model(self, model):
        """配置模型 - 子类实现"""
        pass
    
    @abc.abstractmethod
    def _generate_video_device_specific(self, request, img, size: str, frame_num: int):
        """设备特定的视频生成 - 子类实现"""
        pass
    
    @abc.abstractmethod
    def _log_memory_usage(self):
        """记录内存使用 - 子类实现"""
        pass
    
    @abc.abstractmethod
    def _empty_cache(self):
        """清理缓存 - 子类实现"""
        pass
    
    # 提供默认实现的方法
    def _init_distributed(self):
        """初始化分布式 - 子类可重写"""
        backend = self._get_backend()
        timeout_seconds = 2400 if self.t5_cpu else 1800
        self._init_distributed_common(backend, timeout_seconds)
    
    def _load_model(self):
        """加载模型 - 子类可重写"""
        model_config = {
            't5_fsdp': self.model_args.get("t5_fsdp", False),
            'dit_fsdp': self.model_args.get("dit_fsdp", False),
            'use_usp': (self.model_args.get("ulysses_size", 1) > 1),
            'ulysses_size': self.model_args.get("ulysses_size", 1),
            'vae_parallel': self.model_args.get("vae_parallel", False),
            'task': self.model_args.get("task", "i2v-14B")
        }
        return self._load_model_common(model_config)
    
    def _broadcast_image_path(self, image_path: Optional[str]) -> str:
        """广播图片路径 - 子类可重写"""
        return self._broadcast_image_path_common(image_path)
    
    def _download_image_sync(self, image_url: str, task_id: str) -> str:
        """同步下载图片"""
        output_dir = Path("generated_videos")
        output_dir.mkdir(exist_ok=True)
        image_path = output_dir / f"{task_id}_input.jpg"
        
        try:
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # 调整大小
            max_size = (2048, 2048)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            image.save(image_path, quality=95, optimize=True)
            logger.info(f"Image downloaded and saved: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Failed to download image for task {task_id}: {str(e)}")
            raise
    
    def _get_output_path(self, task_id: str) -> str:
        """获取输出视频路径"""
        output_dir = Path("generated_videos")
        output_dir.mkdir(exist_ok=True)
        return str(output_dir / f"{task_id}.mp4")
    
    @abc.abstractmethod
    def _get_backend(self) -> str:
        """获取分布式后端 - 子类实现"""
        pass