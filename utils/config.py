#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration file for DNN optimization project.
"""


from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from omegaconf import OmegaConf


# General settings
@dataclass
class GeneralConfig:
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "logs"
    results_dir: str = "results"


# Dataset settings
@dataclass
class DatasetConfig:
    name: str = "cifar10"
    batch_size: int = 128
    num_workers: int = 4
    data_dir: str = "data"


# Model settings
@dataclass
class ModelConfig:
    name: str = "resnet18"
    pretrained: bool = True
    checkpoint: Optional[str] = None
    

# BaselineModel Train settings
@dataclass
class BaselineTrainConfig:
    epochs: int = 5
    lr: float = 0.01

# Compression settings
@dataclass
class WeightSharingConfig:
    n_clusters: int = 32
    layers_to_skip: List[str] = field(default_factory=lambda: ["conv1", "fc"])


@dataclass
class LowRankConfig:
    rank_percent: float = 0.25
    layers_to_skip: List[str] = field(default_factory=lambda: ["conv1", "fc"])


@dataclass
class CompressionConfig:
    enabled: bool = False
    method: str = "weight_sharing"
    weight_sharing: WeightSharingConfig = field(default_factory=WeightSharingConfig)
    low_rank: LowRankConfig = field(default_factory=LowRankConfig)


# Pruning settings
@dataclass
class PruningConfig:
    enabled: bool = False
    method: str = "magnitude"
    sparsity: float = 0.5
    layers_to_skip: List[str] = field(default_factory=lambda: ["conv1", "fc"])
    fine_tune: bool = True
    epochs: int = 10
    lr: float = 0.01


# Distillation settings
@dataclass
class DistillationConfig:
    enabled: bool = False
    teacher_model: str = "resnet50"
    teacher_model_checkpoint: Optional[str] = None
    student_model: str = "resnet18"
    alpha: float = 0.5
    temperature: float = 4.0
    epochs: int = 100
    lr: float = 0.01


# Quantization settings
@dataclass
class QuantizationConfig:
    enabled: bool = False
    method: str = "static"
    bit_width: int = 8
    layers_to_skip: List[str] = field(default_factory=lambda: ["conv1", "fc"])


# Optimization (汇总)
@dataclass
class OptimizationConfig:
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


# Evaluation settings
@dataclass
class EvaluationConfig:
    batch_size: int = 128
    metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy",
            "inference_time",
            "flops",
            "size",
            "parameters",
        ]
    )
    plot_results: bool = True


# 总配置
@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: BaselineTrainConfig = field(default_factory=BaselineTrainConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(
    path: Optional[str] = None, overrides: Optional[List[str]] = None
) -> Config:
    base_cfg = OmegaConf.structured(Config)

    if path:
        file_cfg = OmegaConf.load(path)
        base_cfg = OmegaConf.merge(base_cfg, file_cfg)

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    return OmegaConf.to_object(base_cfg)


def save_config_to_file(config, file_path):
    """
    Save configuration to a file.

    Args:
        config (dict): Configuration to save
        file_path (str): Path to save configuration to

    Returns:
        None
    """
    OmegaConf.save(config, file_path)

if __name__ == "__main__":
    config = load_config()
    print(config)
