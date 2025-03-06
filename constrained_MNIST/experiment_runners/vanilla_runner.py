"""
Standard MLP experiment runner for MNIST dataset.

This module provides functionality for running vanilla MLP experiments 
on the MNIST dataset with various activation functions and configurations.
"""

from experiment_runners.base_runner import run_base_experiment
from config import VanillaExperimentConfig

def run_vanilla_experiment(config=None, **kwargs):
    """
    Run a vanilla MLP experiment with the given configuration.
    
    Args:
        config: VanillaExperimentConfig object (created from kwargs if not provided)
        **kwargs: Configuration parameters that override config settings
    
    Returns:
        trainer: The trainer instance
        history: Training history
    """
    # Create config from kwargs if not provided
    if config is None:
        config_kwargs = {k: v for k, v in kwargs.items() 
                         if k in VanillaExperimentConfig.__annotations__}
        config = VanillaExperimentConfig(**config_kwargs)
    
    # Run the experiment
    trainer, history, _ = run_base_experiment(config)
    
    return trainer, history 