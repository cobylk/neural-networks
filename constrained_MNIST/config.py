"""
Configuration settings for MNIST experiments.

This module provides configurations for running various MNIST experiments,
including default settings and plot configurations.
"""

import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Any, Type, Optional, Union, Tuple
from experiment_utils import PlotConfig
from file_utils import EXPERIMENT_TYPES, BASE_RESULTS_DIR

@dataclass
class BaseExperimentConfig:
    """Base configuration for MNIST experiments."""
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    batch_size: int = 128
    learning_rate: float = 0.001
    epochs: int = 30
    early_stopping_patience: int = 5
    save_dir: str = BASE_RESULTS_DIR
    experiment_type: str = EXPERIMENT_TYPES['default']
    plot_config: Optional[PlotConfig] = None
    
    def get_default_plot_config(self):
        """Get default plot configuration if none is provided."""
        if self.plot_config is None:
            self.plot_config = PlotConfig(
                training_curves=True,
                weight_distributions=True,
                learning_curves=True,
                experiment_comparison=True,
                save_format='png',
                dpi=300
            )
        return self.plot_config

@dataclass
class VanillaExperimentConfig(BaseExperimentConfig):
    """Configuration for standard MLP experiments."""
    dropout_prob: float = 0.0
    activation_class: Type = nn.ReLU
    activation_kwargs: Dict[str, Any] = field(default_factory=dict)
    experiment_type: str = EXPERIMENT_TYPES['vanilla']

@dataclass
class StochasticExperimentConfig(BaseExperimentConfig):
    """Configuration for stochastic layer experiments."""
    activation_class: Optional[Type] = None
    activation_kwargs: Dict[str, Any] = field(default_factory=dict)
    layer_kwargs: Dict[str, Any] = field(default_factory=dict)
    experiment_type: str = EXPERIMENT_TYPES['stochastic']
    
    def __post_init__(self):
        """Initialize default layer kwargs if none provided."""
        if not self.layer_kwargs:
            self.layer_kwargs = {'temperature': 1.0} 