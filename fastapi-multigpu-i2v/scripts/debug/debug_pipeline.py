#!/usr/bin/env python3
"""
管道调试工具
测试管道创建和基本功能
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from pipelines.pipeline_factory import PipelineFactory

def main():
    print("🔧 Pipeline Debug Tool")
    print("=" * 50)
    
    try:
        # 获取设备信息
        device_info = PipelineFactory.get_available_devices()
        print(f"Available devices: {device_info}")
        
        # 测试管道创建
        print("\nTesting pipeline creation...")
        pipeline = PipelineFactory.create_pipeline(
            ckpt_dir="/data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P",
            t5_cpu=True
        )
        
        print("✅ Pipeline created successfully")
        print(f"Pipeline type: {type(pipeline).__name__}")
        
        # 清理
        pipeline.cleanup()
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()