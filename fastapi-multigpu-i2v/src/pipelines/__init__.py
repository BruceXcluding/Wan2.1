"""
推理管道包
"""

from pipelines.base_pipeline import BasePipeline
from pipelines.pipeline_factory import PipelineFactory

__all__ = ["BasePipeline", "PipelineFactory"]