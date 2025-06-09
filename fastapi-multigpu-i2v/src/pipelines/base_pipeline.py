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
    
    def _init_distributed(self):
        """初始化分布式环境"""
        try:
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.rank = int(os.environ.get("RANK", 0))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            if self.world_size > 1:
                logger.info(f"Initializing distributed: rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}")
                
                # 🔧 设备特定的分布式初始化
                if self.device_type == "npu":
                    # NPU 分布式初始化
                    import torch_npu
                    
                    # 设置当前设备
                    torch_npu.npu.set_device(self.local_rank)
                    
                    # 🔧 关键修复：使用 HCCL 后端并增加超时
                    dist.init_process_group(
                        backend="hccl",  # NPU 使用 HCCL 后端
                        init_method=f"env://",
                        world_size=self.world_size,
                        rank=self.rank,
                        timeout=timedelta(seconds=3600)  # 增加超时时间
                    )
                    
                    logger.info(f"✅ NPU distributed initialized: rank={self.rank}")
                    
                elif self.device_type == "cuda":
                    # CUDA 分布式初始化
                    torch.cuda.set_device(self.local_rank)
                    
                    dist.init_process_group(
                        backend="nccl",
                        init_method="env://",
                        world_size=self.world_size,
                        rank=self.rank,
                        timeout=timedelta(seconds=1800)
                    )
                    
                    logger.info(f"✅ CUDA distributed initialized: rank={self.rank}")
                    
                else:
                    # CPU 分布式初始化
                    dist.init_process_group(
                        backend="gloo",
                        init_method="env://", 
                        world_size=self.world_size,
                        rank=self.rank,
                        timeout=timedelta(seconds=1800)
                    )
                    
                    logger.info(f"✅ CPU distributed initialized: rank={self.rank}")
            else:
                logger.info("Single device mode, skipping distributed initialization")
                
        except Exception as e:
            logger.error(f"Distributed initialization failed: {e}")
            # 🔧 重要：不要 raise，继续以单设备模式运行
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            logger.warning("Falling back to single device mode")
        
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
            # 确保能找到 wan 模块
            import sys
            from pathlib import Path

            # 添加 wan 模块路径
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            wan_project_root = project_root.parent

            if str(wan_project_root) not in sys.path:
                sys.path.insert(0, str(wan_project_root))
                logger.info(f"Added wan project path: {wan_project_root}")

            # 导入 wan 模块
            import wan
            logger.info("✅ Successfully imported wan module")

            # 获取配置
            cfg = wan.configs.WAN_CONFIGS[model_config.get("task", "i2v-14B")]

            # 获取分布式参数
            ulysses_size = model_config.get('ulysses_size', 1)
            ring_size = model_config.get('ring_size', 1)
            use_usp = ulysses_size > 1 or ring_size > 1

            logger.info(f"Distributed params: ulysses_size={ulysses_size}, ring_size={ring_size}, use_usp={use_usp}")

            # 分布式环境验证和初始化 (参照 generate.py:333-346)
            world_size = getattr(self, 'world_size', 1)
            rank = getattr(self, 'rank', 0)

            if world_size > 1:
                if use_usp:
                    # 验证参数 (generate.py:334)
                    assert ulysses_size * ring_size == world_size, \
                        f"ulysses_size({ulysses_size}) * ring_size({ring_size}) != world_size({world_size})"

                    # 验证注意力头数兼容性 (generate.py:359)
                    if ulysses_size > 1:
                        assert cfg.num_heads % ulysses_size == 0, \
                            f"`{cfg.num_heads=}` cannot be divided evenly by `{ulysses_size=}`."

                    try:
                        # 初始化 xfuser 分布式环境 (generate.py:335-346)
                        import torch.distributed as dist
                        from xfuser.core.distributed import (
                            init_distributed_environment,
                            initialize_model_parallel,
                        )

                        if dist.is_initialized():
                            logger.info(f"Initializing xfuser: ulysses_size={ulysses_size}, ring_size={ring_size}")

                            init_distributed_environment(
                                rank=dist.get_rank(), 
                                world_size=dist.get_world_size()
                            )

                            initialize_model_parallel(
                                sequence_parallel_degree=dist.get_world_size(),
                                ring_degree=ring_size,
                                ulysses_degree=ulysses_size,
                            )

                            logger.info("✅ xfuser initialized successfully")
                        else:
                            logger.warning("PyTorch distributed not initialized, disabling USP")
                            use_usp = False

                    except ImportError as e:
                        logger.warning(f"xfuser not available: {e}, disabling USP")
                        use_usp = False
                    except Exception as e:
                        logger.warning(f"xfuser initialization failed: {e}, disabling USP") 
                        use_usp = False
            else:
                # 单进程环境验证 (generate.py:327-332)
                assert not (model_config.get('t5_fsdp', False) or model_config.get('dit_fsdp', False)), \
                    "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
                assert not use_usp, \
                    "context parallel are not supported in non-distributed environments."

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Rank {rank}: Loading {self.device_type} model attempt {attempt + 1}/{max_retries}")

                    # 构建 WanI2V 参数 - 严格按照 generate.py:421-425 的方式
                    wan_config = {
                        'config': cfg,
                        'checkpoint_dir': self.ckpt_dir,
                        'device_id': getattr(self, 'local_rank', 0),
                        'rank': rank,
                        't5_fsdp': model_config.get('t5_fsdp', False),
                        'dit_fsdp': model_config.get('dit_fsdp', True),
                        'use_usp': use_usp,  # 这个是关键参数
                        't5_cpu': model_config.get('t5_cpu', True),
                    }

                    logger.info(f"Creating WanI2V with config: {list(wan_config.keys())}")
                    logger.info(f"Final distributed config: use_usp={use_usp}")

                    # 创建 WanI2V 模型
                    model = wan.WanI2V(**wan_config)

                    # 设备特定的模型配置
                    self._configure_model(model)

                    logger.info(f"Rank {rank}: {self.device_type} model loaded successfully")
                    return model

                except Exception as e:
                    logger.warning(f"Rank {rank}: Model loading attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    
                    time.sleep(5 * (attempt + 1))

        except ImportError as e:
            logger.error(f"❌ Failed to import wan module: {e}")
            raise
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
            
            # 🔧 关键修改：传递正确的参数给设备特定的生成方法
            video_tensor = self._generate_video_device_specific(
                request, 
                img, 
                size, 
                frame_num
            )
            
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