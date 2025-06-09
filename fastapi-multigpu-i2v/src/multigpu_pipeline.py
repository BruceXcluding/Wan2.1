import torch
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format=False
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist
import wan
from wan.configs import MAX_AREA_CONFIGS
from wan.utils.utils import cache_video
from io import BytesIO
from PIL import Image
from pathlib import Path
import os
import asyncio
import aiohttp
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiGPUVideoPipeline:
    """多GPU分布式视频生成管道"""
    
    def __init__(self, ckpt_dir: str, **model_args):
        self.ckpt_dir = ckpt_dir
        self.model_args = model_args
        
        # 获取分布式环境
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        logger.info(f"Initializing pipeline: rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}")
        
        # 初始化分布式环境
        self._init_distributed()
        
        # 加载模型
        self.model = self._load_model()
        logger.info("Pipeline initialized successfully")

    def _init_distributed(self):
        """初始化分布式环境"""
        if self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                logger.info("Initializing distributed process group")
                dist.init_process_group(
                    backend="hccl",
                    init_method="env://",
                    rank=self.rank,
                    world_size=self.world_size
                )

    def _load_model(self):
        """加载分布式模型"""
        cfg = wan.configs.WAN_CONFIGS[self.model_args.get("task", "i2v-14B")]
        return wan.WanI2V(
            config=cfg,
            checkpoint_dir=self.ckpt_dir,
            device_id=self.local_rank,
            rank=self.rank,
            t5_fsdp=self.model_args.get("t5_fsdp", False),
            dit_fsdp=self.model_args.get("dit_fsdp", False),
            use_usp=(self.model_args.get("ulysses_size", 1) > 1),
            t5_cpu=self.model_args.get("t5_cpu", False),
            use_vae_parallel=self.model_args.get("vae_parallel", False),
        )

    async def generate_video(self, request, task_id: str) -> str:
        """生成视频的主入口"""
        try:
            logger.info(f"Starting video generation for task {task_id}")
            
            # 1. 下载并保存图片
            image_path = await self._download_image(request.image_url, task_id)
            
            # 2. 准备输出路径
            output_path = self._get_output_path(task_id)
            
            # 3. 异步执行视频生成
            result_path = await self._execute_generation(request, image_path, output_path, task_id)
            
            logger.info(f"Video generation completed for task {task_id}: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Video generation failed for task {task_id}: {str(e)}")
            raise

    # 在 _download_image 方法中添加更多错误处理
    async def _download_image(self, image_url: str, task_id: str) -> str:
        """异步下载图片"""
        output_dir = Path("generated_videos")
        output_dir.mkdir(exist_ok=True)

        image_path = output_dir / f"{task_id}_input.jpg"

        try:
            # 添加超时和更多错误处理
            timeout = aiohttp.ClientTimeout(total=30)  # 30秒超时
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download image: HTTP {response.status}")

                    # 检查 Content-Type
                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith('image/'):
                        logger.warning(f"Unexpected content type: {content_type}")

                    content = await response.read()

                    # 检查文件大小
                    if len(content) == 0:
                        raise Exception("Downloaded image is empty")

                    if len(content) > 50 * 1024 * 1024:  # 50MB 限制
                        raise Exception("Image file too large (>50MB)")

                    image = Image.open(BytesIO(content)).convert("RGB")

                    # 可选：调整图像大小以节省内存
                    max_size = (2048, 2048)
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                        logger.info(f"Image resized to {image.size}")

                    image.save(image_path, quality=95, optimize=True)

            logger.info(f"Image downloaded and saved: {image_path}")
            return str(image_path)

        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error downloading image for task {task_id}: {str(e)}")
            raise Exception(f"Network error downloading image: {str(e)}")
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

    def _generate_sync(self, request, image_path: str, output_path: str, task_id: str) -> str:
        """同步视频生成核心逻辑"""
        try:
            # 处理参数
            size = self._normalize_size(request.image_size)
            img = Image.open(image_path).convert("RGB")
            
            logger.info(f"Generating video with size={size}, frames={request.num_frames}")
            
            # 调用模型生成
            video_tensor = self.model.generate(
                request.prompt,
                img,
                max_area=MAX_AREA_CONFIGS[size],
                frame_num=request.num_frames,
                shift=getattr(request, 'sample_shift', 5.0),
                sample_solver=getattr(request, 'sample_solver', 'unipc'),
                sampling_steps=request.infer_steps,
                guide_scale=request.guidance_scale,
                n_prompt=getattr(request, 'negative_prompt', ""),
                seed=request.seed if request.seed is not None else -1,
                offload_model=False,
            )
            
            # 保存视频（只有 rank 0 会有数据）
            if video_tensor is not None:
                self._save_video(video_tensor, output_path, request.num_frames)
                logger.info(f"Video saved to {output_path}")
            else:
                logger.info(f"Non-master rank {self.rank}, video not saved")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Sync generation failed for task {task_id}: {str(e)}")
            raise

    def _normalize_size(self, size: str) -> str:
        """标准化尺寸参数"""
        if size == "auto":
            return "1280*720"
        return size

    def _save_video(self, video_tensor, output_path: str, frame_num: int):
        """保存视频张量为文件"""
        try:
            cache_video(
                tensor=video_tensor[None],
                save_file=output_path,
                fps=max(8, frame_num // 10),  # 动态FPS，最小8
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
        except Exception as e:
            logger.error(f"Failed to save video to {output_path}: {str(e)}")
            raise

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
        logger.info("Pipeline cleaned up")
