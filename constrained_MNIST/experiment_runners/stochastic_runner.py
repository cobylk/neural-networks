"""
Stochastic layer experiment runner for MNIST dataset.

This module provides functionality for running experiments with stochastic layers
on the MNIST dataset with various activation functions and configurations.
"""

import torch
from stochastic_layer import StochasticLayer
from experiment_runners.base_runner import run_base_experiment
from config import StochasticExperimentConfig
from experiment_utils import MNISTTrainer

class SimplexMNISTTrainer(MNISTTrainer):
    """Extended MNIST trainer that normalizes inputs to lie on the simplex"""
    
    @staticmethod
    def _normalize_to_simplex(x):
        """
        Normalize input images to lie on the probability simplex.
        Each image is normalized so its pixels sum to 1 and are non-negative.
        """
        # Flatten images first
        x = x.view(x.size(0), -1)
        
        # Shift values to make them non-negative (per example)
        # Find minimum value for each example and subtract it
        min_vals, _ = torch.min(x, dim=1, keepdim=True)
        x = x - min_vals
        
        # Add small epsilon to avoid division by zero
        x = x + 1e-8
        # Normalize each image to sum to 1
        return 100 * x / x.sum(dim=1, keepdim=True)

def run_stochastic_experiment(config=None, **kwargs):
    """
    Run a stochastic layer experiment with the given configuration.
    
    Args:
        config: StochasticExperimentConfig object (created from kwargs if not provided)
        **kwargs: Configuration parameters that override config settings
    
    Returns:
        trainer: The trainer instance
        history: Training history
        activation_name: Name of the activation function used
    """
    # Create config from kwargs if not provided
    if config is None:
        config_kwargs = {k: v for k, v in kwargs.items() 
                         if k in StochasticExperimentConfig.__annotations__}
        config = StochasticExperimentConfig(**config_kwargs)
    
    # Get activation name for directory organization
    from experiment_runners.base_runner import get_activation_name
    activation_name = get_activation_name(
        config.activation_class,
        config.activation_kwargs or {}
    )
    
    # Add layer parameters to directory name if present
    if config.layer_kwargs:
        activation_name += "_layer_" + "_".join(f"{k}_{v}" for k, v in config.layer_kwargs.items())
    
    # Create experiment directory
    experiment_dir = f"{config.save_dir}/{activation_name}"
    
    # Update save_dir in config
    config.save_dir = experiment_dir
    
    # Run the experiment with stochastic layer specifics
    trainer, history, _ = run_base_experiment(
        config,
        model_kwargs={'layer_class': StochasticLayer},
        trainer_class=SimplexMNISTTrainer,
        preprocess_fn=SimplexMNISTTrainer._normalize_to_simplex
    )
    
    return trainer, history, activation_name 