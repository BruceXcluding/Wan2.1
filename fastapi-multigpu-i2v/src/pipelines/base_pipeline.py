import abc
import asyncio
import logging
import time
import signal
import sys
from typing import Any, Optional
from pathlib import Path
from PIL import Image
from io import BytesIO
import aiohttp

logger = logging.getLogger(__name__)

class BasePipeline(abc.ABC):
    """视频生成管道基类"""
    
    def __init__(self, ckpt_dir: str, **model_args):
        self.ckpt_dir = ckpt_dir
        self.model_args = model_args
        self.model = None
        
        # 分布式环境变量
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        logger.info(f"Initializing {self.__class__.__name__}: rank={self.rank}, world_size={self.world_size}")
        
        # 注册信号处理器
        self._register_signal_handlers()
    
    @abc.abstractmethod
    def _init_distributed(self):
        """初始化分布式环境 - 子类实现"""
        pass
    
    @abc.abstractmethod
    def _load_model(self):
        """加载模型 - 子类实现"""
        pass
    
    @abc.abstractmethod
    def _log_memory_usage(self):
        """记录内存使用 - 子类实现"""
        pass
    
    @abc.abstractmethod
    def _empty_cache(self):
        """清理缓存 - 子类实现"""
        pass
    
    def _register_signal_handlers(self):
        """注册信号处理器"""
        def signal_handler(sig, frame):
            logger.info(f"Rank {self.rank} received shutdown signal {sig}")
            self.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def generate_video(self, request, task_id: str) -> str:
        """生成视频的主入口"""
        try:
            logger.info(f"Starting video generation for task {task_id} on rank {self.rank}")
            
            # 1. 下载并保存图片（只在 rank 0 执行）
            image_path = None
            if self.rank == 0:
                image_path = await self._download_image(request.image_url, task_id)
            
            # 2. 广播图片路径到所有 rank
            image_path = await self._broadcast_image_path(image_path)
            
            # 3. 准备输出路径
            output_path = self._get_output_path(task_id)
            
            # 4. 异步执行视频生成
            result_path = await self._execute_generation(request, image_path, output_path, task_id)
            
            logger.info(f"Video generation completed for task {task_id}: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Video generation failed for task {task_id}: {str(e)}")
            self._empty_cache()
            raise
    
    async def _broadcast_image_path(self, image_path: Optional[str]) -> str:
        """广播图片路径到所有进程"""
        if self.world_size == 1:
            return image_path
        
        # 子类实现具体的广播逻辑
        return await self._do_broadcast_image_path(image_path)
    
    @abc.abstractmethod
    async def _do_broadcast_image_path(self, image_path: Optional[str]) -> str:
        """执行图片路径广播 - 子类实现"""
        pass
    
    async def _download_image(self, image_url: str, task_id: str) -> str:
        """异步下载图片"""
        output_dir = Path("generated_videos")
        output_dir.mkdir(exist_ok=True)

        image_path = output_dir / f"{task_id}_input.jpg"

        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download image: HTTP {response.status}")

                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith('image/'):
                        logger.warning(f"Unexpected content type: {content_type}")

                    content = await response.read()

                    if len(content) == 0:
                        raise Exception("Downloaded image is empty")

                    if len(content) > 50 * 1024 * 1024:
                        raise Exception("Image file too large (>50MB)")

                    image = Image.open(BytesIO(content)).convert("RGB")

                    # 调整图像大小
                    max_size = (2048, 2048)
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                        logger.info(f"Image resized to {image.size}")

                    image.save(image_path, quality=95, optimize=True)

            logger.info(f"Image downloaded and saved: {image_path}")
            return str(image_path)

        except Exception as e:
            logger.error(f"Failed to download image for task {task_id}: {str(e)}")
            raise
    
    def _get_output_path(self, task_id: str) -> str:
        """获取输出视频路径"""
        return f"generated_videos/{task_id}.mp4"
    
    async def _execute_generation(self, request, image_path: str, output_path: str, task_id: str) -> str:
        """在线程池中执行同步的视频生成"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._generate_sync, 
            request, image_path, output_path, task_id
        )
    
    @abc.abstractmethod
    def _generate_sync(self, request, image_path: str, output_path: str, task_id: str) -> str:
        """同步视频生成核心逻辑 - 子类实现"""
        pass
    
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

        # 导入配置
        try:
            from wan.configs import MAX_AREA_CONFIGS
            if size not in MAX_AREA_CONFIGS:
                logger.warning(f"Size {size} not in MAX_AREA_CONFIGS, using 1280*720")
                return "1280*720"
        except ImportError:
            logger.warning("Could not import MAX_AREA_CONFIGS, using default size")
            return "1280*720"

        return size

    def _save_video(self, video_tensor, output_path: str, frame_num: int):
        """保存视频张量为文件"""
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
            logger.error(f"Failed to save video to {output_path}: {str(e)}")
            raise
    
    @abc.abstractmethod
    def cleanup(self):
        """清理资源 - 子类实现"""
        pass