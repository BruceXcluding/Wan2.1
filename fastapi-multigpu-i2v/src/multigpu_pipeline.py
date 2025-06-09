import torch
import torch_npu
import os
import time
import signal
import sys
from datetime import timedelta

# 设置关键环境变量
os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "0")  # 异步模式
os.environ.setdefault("HCCL_TIMEOUT", "1800")  # 30分钟超时
os.environ.setdefault("HCCL_BUFFSIZE", "512")  # 缓冲区大小
os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "600")  # 10分钟连接超时

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
import asyncio
import aiohttp
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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
        
        # 从参数中获取 T5 CPU 配置
        self.t5_cpu = model_args.get('t5_cpu', False)
        
        logger.info(f"Initializing pipeline: rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}")
        logger.info(f"Configuration: t5_cpu={self.t5_cpu}, dit_fsdp={model_args.get('dit_fsdp', False)}, vae_parallel={model_args.get('vae_parallel', False)}")
        
        # 调整超时时间（T5 CPU 模式需要更长时间）
        if self.t5_cpu:
            os.environ["HCCL_TIMEOUT"] = str(int(os.environ.get("HCCL_TIMEOUT", "1800")) + 600)  # 额外10分钟
            logger.info("T5 CPU mode enabled, extended HCCL timeout")
        
        # 初始化分布式环境
        self._init_distributed()
        
        # 加载模型
        self.model = self._load_model()
        
        # 注册信号处理器
        self._register_signal_handlers()
        
        logger.info("Pipeline initialized successfully")

    def _init_distributed(self):
        """初始化分布式环境"""
        if self.world_size > 1:
            # 设置NPU设备
            torch.npu.set_device(self.local_rank)
            
            if not dist.is_initialized():
                logger.info("Initializing distributed process group with HCCL backend")
                try:
                    # 根据 T5 CPU 模式调整超时
                    timeout_seconds = 2400 if self.t5_cpu else 1800
                    
                    dist.init_process_group(
                        backend="hccl",
                        init_method="env://",
                        rank=self.rank,
                        world_size=self.world_size,
                        timeout=timedelta(seconds=timeout_seconds)
                    )
                    
                    # 测试通信
                    logger.info("Testing distributed communication...")
                    test_tensor = torch.tensor([self.rank], dtype=torch.float32).npu()
                    dist.all_reduce(test_tensor)
                    logger.info(f"Communication test passed, sum: {test_tensor.item()}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize distributed process group: {str(e)}")
                    raise
        else:
            logger.info("Single GPU mode, skipping distributed initialization")

    def _load_model(self):
        """加载分布式模型 - 改进版本"""
        try:
            cfg = wan.configs.WAN_CONFIGS[self.model_args.get("task", "i2v-14B")]
            
            # 添加模型加载重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Rank {self.rank}: Loading model attempt {attempt + 1}/{max_retries}")
                    
                    model = wan.WanI2V(
                        config=cfg,
                        checkpoint_dir=self.ckpt_dir,
                        device_id=self.local_rank,
                        rank=self.rank,
                        t5_fsdp=self.model_args.get("t5_fsdp", False),
                        dit_fsdp=self.model_args.get("dit_fsdp", False),
                        use_usp=(self.model_args.get("ulysses_size", 1) > 1),
                        t5_cpu=self.t5_cpu,
                        use_vae_parallel=self.model_args.get("vae_parallel", False),
                    )
                    
                    logger.info(f"Rank {self.rank}: Model loaded successfully")
                    
                    # 先同步模型加载完成
                    if self.world_size > 1:
                        logger.info(f"Rank {self.rank}: Waiting at model loading barrier")
                        dist.barrier(timeout=timedelta(seconds=600))
                        logger.info(f"Rank {self.rank}: Passed model loading barrier")
                    
                    # T5 CPU 模式下的预热（修复后的版本）
                    if self.t5_cpu:
                        logger.info(f"Rank {self.rank}: Starting T5 CPU warmup process")
                        self._warmup_t5_cpu(model)
                    
                    logger.info(f"Rank {self.rank}: All initialization completed")
                    return model
                    
                except Exception as e:
                    logger.warning(f"Rank {self.rank}: Model loading attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    
                    # 清理和等待
                    torch.npu.empty_cache()
                    time.sleep(10)
                    
        except Exception as e:
            logger.error(f"Rank {self.rank}: Failed to load model: {str(e)}")
            raise
        
    def _warmup_t5_cpu(self, model):
        """T5 CPU 模式预热 - 修复版本"""
        try:
            # 只让 rank 0 进行 T5 预热，避免多进程竞争
            if self.rank == 0:
                logger.info(f"Rank {self.rank}: Warming up T5 on CPU")

                # 预热 T5 编码器
                dummy_prompts = [
                    "warm up text",
                    "a simple test prompt for warmup"
                ]

                with torch.no_grad():
                    for i, prompt in enumerate(dummy_prompts):
                        logger.info(f"Rank {self.rank}: T5 warmup step {i+1}/{len(dummy_prompts)}")
                        try:
                            _ = model.text_encoder([prompt], torch.device('cpu'))
                            logger.info(f"Rank {self.rank}: T5 warmup step {i+1} completed")
                        except Exception as e:
                            logger.warning(f"Rank {self.rank}: T5 warmup step {i+1} failed: {str(e)}")
                            break
                        
                logger.info(f"Rank {self.rank}: T5 warmup completed successfully")
            else:
                logger.info(f"Rank {self.rank}: Skipping T5 warmup (only rank 0 performs warmup)")

            # 同步所有进程，等待 rank 0 完成预热
            if self.world_size > 1:
                logger.info(f"Rank {self.rank}: Waiting for T5 warmup synchronization")
                dist.barrier(timeout=timedelta(seconds=300))  # 5分钟超时
                logger.info(f"Rank {self.rank}: T5 warmup synchronization completed")

        except Exception as e:
            logger.warning(f"Rank {self.rank}: T5 warmup failed: {str(e)}")
            # 即使预热失败也要尝试同步，避免其他进程永远等待
            if self.world_size > 1:
                try:
                    dist.barrier(timeout=timedelta(seconds=60))
                except Exception as sync_e:
                    logger.error(f"Rank {self.rank}: Failed to sync after warmup failure: {str(sync_e)}")

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
            
            # 广播图片路径到所有 rank
            if self.world_size > 1:
                if self.rank == 0:
                    image_path_tensor = torch.tensor([len(image_path)], dtype=torch.int32).npu()
                    dist.broadcast(image_path_tensor, src=0)
                    
                    # 广播字符串
                    image_path_bytes = image_path.encode('utf-8')
                    image_path_tensor = torch.frombuffer(
                        image_path_bytes, dtype=torch.uint8
                    ).npu()
                    dist.broadcast(image_path_tensor, src=0)
                else:
                    # 接收长度
                    length_tensor = torch.tensor([0], dtype=torch.int32).npu()
                    dist.broadcast(length_tensor, src=0)
                    
                    # 接收字符串
                    image_path_tensor = torch.zeros(
                        length_tensor.item(), dtype=torch.uint8
                    ).npu()
                    dist.broadcast(image_path_tensor, src=0)
                    
                    image_path = image_path_tensor.cpu().numpy().tobytes().decode('utf-8')
            
            # 2. 准备输出路径
            output_path = self._get_output_path(task_id)
            
            # 3. 异步执行视频生成
            result_path = await self._execute_generation(request, image_path, output_path, task_id)
            
            logger.info(f"Video generation completed for task {task_id}: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Video generation failed for task {task_id}: {str(e)}")
            # 清理NPU缓存
            torch.npu.empty_cache()
            raise

    async def _download_image(self, image_url: str, task_id: str) -> str:
        """异步下载图片（只在 rank 0 执行）"""
        output_dir = Path("generated_videos")
        output_dir.mkdir(exist_ok=True)

        image_path = output_dir / f"{task_id}_input.jpg"

        try:
            # 添加超时和更多错误处理
            timeout = aiohttp.ClientTimeout(total=60)  # 60秒超时
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
            # 记录内存使用
            self._log_memory_usage()
            
            # 处理参数
            size = self._normalize_size(request.image_size)
            img = Image.open(image_path).convert("RGB")

            # 验证和调整帧数
            frame_num = self._validate_frame_num(request.num_frames)
        
            logger.info(f"Generating video with size={size}, frames={frame_num}")
            logger.info(f"Image size: {img.size}, max_area: {MAX_AREA_CONFIGS.get(size, 'Unknown')}")
            logger.info(f"T5 CPU mode: {self.t5_cpu}")
            
            # 生成开始时间
            start_time = time.time()
            
            # 分布式同步点 - T5 CPU 模式需要更长的同步时间
            if self.world_size > 1:
                timeout_seconds = 900 if self.t5_cpu else 300
                logger.info(f"Rank {self.rank} waiting at generation sync barrier (timeout: {timeout_seconds}s)")
                dist.barrier(timeout=timedelta(seconds=timeout_seconds))
                logger.info(f"Rank {self.rank} passed generation sync barrier")
            
            # 调用模型生成
            try:
                logger.info(f"Rank {self.rank} starting model generation")
                video_tensor = self.model.generate(
                    request.prompt,
                    img,
                    max_area=MAX_AREA_CONFIGS[size],
                    frame_num=frame_num,
                    shift=getattr(request, 'sample_shift', 5.0),
                    sample_solver=getattr(request, 'sample_solver', 'unipc'),
                    sampling_steps=request.infer_steps,
                    guide_scale=request.guidance_scale,
                    n_prompt=getattr(request, 'negative_prompt', ""),
                    seed=request.seed if request.seed is not None else -1,
                    offload_model=self.t5_cpu,  # T5 CPU 模式时启用 offload
                )
                
                generation_time = time.time() - start_time
                logger.info(f"Rank {self.rank} completed generation in {generation_time:.2f}s")
                
            except RuntimeError as e:
                if "HcclAllGather" in str(e) or "Remote Rank" in str(e) or "timeout" in str(e).lower():
                    logger.error(f"HCCL communication error on rank {self.rank}: {str(e)}")
                    self._reset_distributed_state()
                    raise Exception(f"分布式通信失败，建议重启服务: {str(e)}")
                elif "out of memory" in str(e).lower():
                    logger.error(f"NPU out of memory on rank {self.rank}: {str(e)}")
                    torch.npu.empty_cache()
                    raise Exception(f"显存不足，请降低并发数或帧数: {str(e)}")
                else:
                    raise
            
            # 保存视频（只有 rank 0 会有数据）
            if video_tensor is not None:
                logger.info(f"Generated video tensor shape: {video_tensor.shape}")
                self._save_video(video_tensor, output_path, frame_num)
                logger.info(f"Video saved to {output_path}")
            else:
                logger.info(f"Non-master rank {self.rank}, video not saved")
            
            # 最终同步
            if self.world_size > 1:
                logger.info(f"Rank {self.rank} final barrier")
                dist.barrier(timeout=timedelta(seconds=120))
            
            # 记录最终内存使用
            self._log_memory_usage()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Sync generation failed for task {task_id}: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            # 清理资源
            torch.npu.empty_cache()
            raise

    def _reset_distributed_state(self):
        """重置分布式状态"""
        try:
            logger.warning(f"Attempting to reset distributed state on rank {self.rank}")
            
            # 清理NPU缓存
            torch.npu.empty_cache()
            
            # 记录状态但不重新初始化（需要重启整个服务）
            logger.warning("Distributed state reset completed. Service restart recommended.")
            
        except Exception as e:
            logger.error(f"Failed to reset distributed state: {str(e)}")

    def _validate_frame_num(self, frame_num: int) -> int:
        """验证和调整帧数"""
        # Wan2.1 模型通常要求帧数是特定倍数
        valid_frames = [41, 61, 81, 121]

        if frame_num in valid_frames:
            return frame_num

        # 找到最接近的有效帧数
        closest_frame = min(valid_frames, key=lambda x: abs(x - frame_num))
        logger.warning(f"Frame number {frame_num} adjusted to {closest_frame}")
        return closest_frame

    def _normalize_size(self, size: str) -> str:
        """标准化尺寸参数"""
        if size == "auto":
            return "1280*720"

        # 验证尺寸是否在支持列表中
        if size not in MAX_AREA_CONFIGS:
            logger.warning(f"Size {size} not in MAX_AREA_CONFIGS, using 1280*720")
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

    def _log_memory_usage(self):
        """记录内存使用情况"""
        try:
            memory_allocated = torch.npu.memory_allocated(self.local_rank) / 1024**3  # GB
            memory_reserved = torch.npu.memory_reserved(self.local_rank) / 1024**3   # GB
            
            logger.info(f"Rank {self.rank} NPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
        except Exception as e:
            logger.warning(f"Failed to get memory info: {str(e)}")

    def cleanup(self):
        """清理资源"""
        try:
            logger.info(f"Cleaning up pipeline on rank {self.rank}")
            
            if hasattr(self, 'model'):
                del self.model
            
            # 清理NPU缓存
            torch.npu.empty_cache()
            
            # 销毁进程组
            if dist.is_initialized():
                dist.destroy_process_group()
                
            logger.info("Pipeline cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")