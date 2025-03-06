"""
Base experiment runner for MNIST experiments.

This module provides base functionality for running MNIST experiments with various
configurations and models.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from base_MLP import BaseMLP
from experiment_utils import MNISTTrainer, ExperimentManager
from config import BaseExperimentConfig

def get_activation_name(activation_class, activation_kwargs):
    """Get a string representation of an activation function with its parameters."""
    if activation_class is None:
        return "NoActivation"
    
    activation_name = activation_class.__name__
    
    if activation_kwargs:
        activation_name += "_" + "_".join(f"{k}_{v}" for k, v in activation_kwargs.items())
    
    return activation_name

def run_base_experiment(
    config, 
    model_class=BaseMLP, 
    trainer_class=MNISTTrainer,
    model_kwargs=None,
    trainer_kwargs=None,
    preprocess_fn=None
):
    """
    Base function for running an experiment with the given configuration.
    
    Args:
        config: Experiment configuration object
        model_class: Model class to use (default: BaseMLP)
        trainer_class: Trainer class to use (default: MNISTTrainer)
        model_kwargs: Additional model parameters
        trainer_kwargs: Additional trainer parameters
        preprocess_fn: Optional preprocessing function for data
    
    Returns:
        trainer: The trainer instance
        history: Training history
        exp_manager: Experiment manager instance
    """
    model_kwargs = model_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}
    
    # Ensure we have plot config
    plot_config = config.get_default_plot_config()
    
    # Create experiment manager
    exp_manager = ExperimentManager(config.save_dir, plot_config=plot_config)
    
    # Create activation instance if needed
    activation = None
    if hasattr(config, 'activation_class') and config.activation_class is not None:
        activation_kwargs = getattr(config, 'activation_kwargs', {}) or {}
        activation = config.activation_class(**activation_kwargs)
    
    # Get activation name for tagging
    activation_name = get_activation_name(
        getattr(config, 'activation_class', None),
        getattr(config, 'activation_kwargs', {})
    )
    
    # Create model
    model_params = {
        'input_dim': 784,
        'hidden_dims': config.hidden_dims,
        'output_dim': 10,
        'activation': activation,
        'store_activations': True
    }
    
    # Add dropout if in config
    if hasattr(config, 'dropout_prob'):
        model_params['dropout_prob'] = config.dropout_prob
    
    # Add layer class and kwargs if in config
    if hasattr(config, 'layer_class'):
        model_params['layer_class'] = config.layer_class
    
    if hasattr(config, 'layer_kwargs'):
        model_params['layer_kwargs'] = config.layer_kwargs
    
    # Override with any additional model parameters
    model_params.update(model_kwargs)
    
    # Create the model
    model = model_class(**model_params)
    
    # Create tags
    tags = []
    
    # Include number of layers
    tags.append(f'layers_{len(config.hidden_dims)}')
    
    # Include dropout if applicable
    if hasattr(config, 'dropout_prob'):
        tags.append(f'dropout_{config.dropout_prob}')
    
    # Include activation
    tags.append(f'activation_{activation_name}')
    
    # Include any layer info if applicable
    if hasattr(config, 'layer_kwargs') and config.layer_kwargs:
        for k, v in config.layer_kwargs.items():
            tags.append(f'layer_{k}_{v}')
    
    # Create trainer parameters
    trainer_params = {
        'model': model,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'save_dir': config.save_dir,
        'plot_config': plot_config,
        'tags': tags
    }
    
    # Add preprocessing function if provided
    if preprocess_fn:
        trainer_params['preprocess_fn'] = preprocess_fn
    
    # Override with any additional trainer parameters
    trainer_params.update(trainer_kwargs)
    
    # Create trainer
    trainer = trainer_class(**trainer_params)
    
    # Train model
    history = trainer.train(
        epochs=config.epochs, 
        early_stopping_patience=config.early_stopping_patience
    )
    
    # Create comparison plot if we have more than one experiment
    df = exp_manager.compare_experiments()
    if len(df) > 1:
        comparison_fig = exp_manager.plot_comparison('dropout_prob', 'test_acc')
        if comparison_fig:
            try:
                comparison_fig.savefig(
                    f'{config.save_dir}/experiment_comparison.{plot_config.save_format}',
                    dpi=plot_config.dpi
                )
            except Exception as e:
                print(f"Warning: Could not save comparison plot: {e}")
            finally:
                plt.close(comparison_fig)
    
    return trainer, history, exp_manager 