import os
import time
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Optional
from .base_pipeline import BasePipeline
from ..utils.device_detector import device_detector, DeviceType
from PIL import Image

import logging
logger = logging.getLogger(__name__)

class CUDAPipeline(BasePipeline):
    """NVIDIA CUDA GPU 视频生成管道"""
    
    def __init__(self, ckpt_dir: str, **model_args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, cannot use CUDA pipeline")
        
        # 设置 CUDA 特定环境变量
        os.environ.setdefault("NCCL_TIMEOUT", "1800")
        
        # 从参数中获取 T5 CPU 配置
        self.t5_cpu = model_args.get('t5_cpu', False)
        
        super().__init__(ckpt_dir, **model_args)
        
        # 初始化分布式和模型
        self._init_distributed()
        self.model = self._load_model()
        
        logger.info("CUDA Pipeline initialized successfully")
    
    def _init_distributed(self):
        """初始化 CUDA 分布式环境"""
        if self.world_size > 1:
            # 设置 CUDA 设备
            torch.cuda.set_device(self.local_rank)
            
            if not dist.is_initialized():
                logger.info("Initializing distributed process group with NCCL backend")
                try:
                    timeout_seconds = 2400 if self.t5_cpu else 1800
                    
                    dist.init_process_group(
                        backend="nccl",
                        init_method="env://",
                        rank=self.rank,
                        world_size=self.world_size,
                        timeout=timedelta(seconds=timeout_seconds)
                    )
                    
                    # 测试通信
                    logger.info("Testing CUDA distributed communication...")
                    test_tensor = torch.tensor([self.rank], dtype=torch.float32).cuda()
                    dist.all_reduce(test_tensor)
                    logger.info(f"CUDA communication test passed, sum: {test_tensor.item()}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize NCCL process group: {str(e)}")
                    raise
        else:
            logger.info("Single CUDA GPU mode, skipping distributed initialization")
    
    def _load_model(self):
        """加载 CUDA 分布式模型"""
        try:
            import wan
            cfg = wan.configs.WAN_CONFIGS[self.model_args.get("task", "i2v-14B")]
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Rank {self.rank}: Loading CUDA model attempt {attempt + 1}/{max_retries}")
                    
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
                    
                    logger.info(f"Rank {self.rank}: CUDA model loaded successfully")
                    
                    # 同步模型加载完成
                    if self.world_size > 1:
                        logger.info(f"Rank {self.rank}: Waiting at CUDA model loading barrier")
                        dist.barrier()
                        logger.info(f"Rank {self.rank}: Passed CUDA model loading barrier")
                    
                    # T5 CPU 模式下的预热
                    if self.t5_cpu:
                        logger.info(f"Rank {self.rank}: Starting T5 CPU warmup process")
                        self._warmup_t5_cpu(model)
                    
                    logger.info(f"Rank {self.rank}: All CUDA initialization completed")
                    return model
                    
                except Exception as e:
                    logger.warning(f"Rank {self.rank}: CUDA model loading attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    
                    # 清理和等待
                    torch.cuda.empty_cache()
                    time.sleep(10)
                    
        except Exception as e:
            logger.error(f"Rank {self.rank}: Failed to load CUDA model: {str(e)}")
            raise
    
    def _warmup_t5_cpu(self, model):
        """T5 CPU 模式预热"""
        try:
            # 只让 rank 0 进行 T5 预热
            if self.rank == 0:
                logger.info(f"Rank {self.rank}: Warming up T5 on CPU")

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

            # 同步所有进程
            if self.world_size > 1:
                logger.info(f"Rank {self.rank}: Waiting for T5 warmup synchronization")
                dist.barrier()
                logger.info(f"Rank {self.rank}: T5 warmup synchronization completed")

        except Exception as e:
            logger.warning(f"Rank {self.rank}: T5 warmup failed: {str(e)}")
            if self.world_size > 1:
                try:
                    dist.barrier()
                except Exception as sync_e:
                    logger.error(f"Rank {self.rank}: Failed to sync after warmup failure: {str(sync_e)}")
    
    async def _do_broadcast_image_path(self, image_path: Optional[str]) -> str:
        """CUDA 图片路径广播"""
        if self.rank == 0:
            image_path_tensor = torch.tensor([len(image_path)], dtype=torch.int32).cuda()
            dist.broadcast(image_path_tensor, src=0)
            
            # 广播字符串
            image_path_bytes = image_path.encode('utf-8')
            image_path_tensor = torch.frombuffer(
                image_path_bytes, dtype=torch.uint8
            ).cuda()
            dist.broadcast(image_path_tensor, src=0)
            return image_path
        else:
            # 接收长度
            length_tensor = torch.tensor([0], dtype=torch.int32).cuda()
            dist.broadcast(length_tensor, src=0)
            
            # 接收字符串
            image_path_tensor = torch.zeros(
                length_tensor.item(), dtype=torch.uint8
            ).cuda()
            dist.broadcast(image_path_tensor, src=0)
            
            return image_path_tensor.cpu().numpy().tobytes().decode('utf-8')
    
    def _generate_sync(self, request, image_path: str, output_path: str, task_id: str) -> str:
        """CUDA 同步视频生成"""
        try:
            self._log_memory_usage()
            
            # 处理参数
            size = self._normalize_size(request.image_size)
            img = Image.open(image_path).convert("RGB")
            frame_num = self._validate_frame_num(request.num_frames)
            
            from wan.configs import MAX_AREA_CONFIGS
        
            logger.info(f"Generating video with size={size}, frames={frame_num}")
            logger.info(f"CUDA T5 CPU mode: {self.t5_cpu}")
            
            start_time = time.time()
            
            # 分布式同步
            if self.world_size > 1:
                logger.info(f"Rank {self.rank} waiting at CUDA generation sync barrier")
                dist.barrier()
                logger.info(f"Rank {self.rank} passed CUDA generation sync barrier")
            
            # 调用模型生成
            try:
                logger.info(f"Rank {self.rank} starting CUDA model generation")
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
                    offload_model=self.t5_cpu,
                )
                
                generation_time = time.time() - start_time
                logger.info(f"Rank {self.rank} completed CUDA generation in {generation_time:.2f}s")
                
            except RuntimeError as e:
                if "nccl" in str(e).lower() or "timeout" in str(e).lower():
                    logger.error(f"NCCL communication error on rank {self.rank}: {str(e)}")
                    raise Exception(f"CUDA 分布式通信失败，建议重启服务: {str(e)}")
                elif "out of memory" in str(e).lower():
                    logger.error(f"CUDA out of memory on rank {self.rank}: {str(e)}")
                    torch.cuda.empty_cache()
                    raise Exception(f"CUDA 显存不足，请降低并发数或帧数: {str(e)}")
                else:
                    raise
            
            # 保存视频
            if video_tensor is not None:
                logger.info(f"Generated video tensor shape: {video_tensor.shape}")
                self._save_video(video_tensor, output_path, frame_num)
                logger.info(f"Video saved to {output_path}")
            else:
                logger.info(f"Non-master rank {self.rank}, video not saved")
            
            # 最终同步
            if self.world_size > 1:
                logger.info(f"Rank {self.rank} final CUDA barrier")
                dist.barrier()
            
            self._log_memory_usage()
            return output_path
            
        except Exception as e:
            logger.error(f"CUDA sync generation failed for task {task_id}: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            torch.cuda.empty_cache()
            raise
    
    def _log_memory_usage(self):
        """记录 CUDA 内存使用"""
        try:
            memory_allocated = torch.cuda.memory_allocated(self.local_rank) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.local_rank) / 1024**3
            
            logger.info(f"Rank {self.rank} CUDA Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
        except Exception as e:
            logger.warning(f"Failed to get CUDA memory info: {str(e)}")
    
    def _empty_cache(self):
        """清理 CUDA 缓存"""
        torch.cuda.empty_cache()
    
    def cleanup(self):
        """清理 CUDA 资源"""
        try:
            logger.info(f"Cleaning up CUDA pipeline on rank {self.rank}")
            
            if hasattr(self, 'model'):
                del self.model
            
            torch.cuda.empty_cache()
            
            if dist.is_initialized():
                dist.destroy_process_group()
                
            logger.info("CUDA pipeline cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during CUDA cleanup: {str(e)}")