"""
Experiment Runner Utilities

This module provides utilities for managing MNIST experiments, including:
- ExperimentConfig: For defining experiment configurations
- ExperimentRunner: For running experiments with different configurations
- ExperimentTracker: For tracking and displaying experiment results
"""

from typing import List, Dict, Any, Tuple, Type, Optional, Union, Callable
import types
import torch
import torch.nn as nn
import pandas as pd
from tabulate import tabulate
from dataclasses import dataclass, asdict, fields

from config import (
    VanillaExperimentConfig, 
    StochasticExperimentConfig, 
    EXPERIMENT_TYPES
)
from base_MLP import BaseMLP
from stochastic_layer import StochasticLayer
from experiment_utils import MNISTTrainer, PlotConfig


def normalize_to_simplex(x):
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


def get_activation_name(activation_class, activation_kwargs):
    """Get a string representation of an activation function with its parameters."""
    if activation_class is None:
        return "NoActivation"
    
    activation_name = activation_class.__name__
    
    if activation_kwargs:
        activation_name += "_" + "_".join(f"{k}_{v}" for k, v in activation_kwargs.items())
    
    return activation_name


def run_experiment(config, 
                  model_class=BaseMLP, 
                  model_kwargs=None,
                  trainer_kwargs=None,
                  preprocess_fn=None):
    """
    Unified function for running an experiment with the given configuration.
    This replaces the separate vanilla and stochastic experiment runners.
    
    Args:
        config: Configuration object (VanillaExperimentConfig or StochasticExperimentConfig)
        model_class: Model class to use (default: BaseMLP)
        model_kwargs: Additional model parameters
        trainer_kwargs: Additional trainer parameters
        preprocess_fn: Optional preprocessing function for data
    
    Returns:
        trainer: The trainer instance
        history: Training history with threshold values if applicable
    """
    model_kwargs = model_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}
    
    # Check if this is a stochastic experiment
    is_stochastic = isinstance(config, StochasticExperimentConfig)
    
    # For stochastic experiments, set defaults if not provided
    if is_stochastic:
        # Set layer_class if not already set
        if 'layer_class' not in model_kwargs:
            model_kwargs['layer_class'] = StochasticLayer
        
        # Set preprocess_fn for simplex normalization if not already set
        if preprocess_fn is None:
            preprocess_fn = normalize_to_simplex
    
    # Ensure we have plot config
    plot_config = config.get_default_plot_config()
    
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
    
    # Add model type tag
    tags.append(f"model:{model.__class__.__name__}")
    
    # Add activation tag
    tags.append(f"activation:{activation_name}")
    
    # Add hidden dims tag
    hidden_dims_str = "_".join(str(d) for d in config.hidden_dims)
    tags.append(f"hidden_dims:{hidden_dims_str}")
    
    # Add dropout tag if applicable
    if hasattr(config, 'dropout_prob'):
        tags.append(f"dropout:{config.dropout_prob}")
    
    # Add layer type tag if applicable
    if hasattr(config, 'layer_class'):
        tags.append(f"layer:{config.layer_class.__name__}")
    
    # Prepare trainer kwargs
    base_trainer_kwargs = {
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'save_dir': config.save_dir,
        'experiment_type': config.experiment_type,
        'plot_config': plot_config,
        'tags': tags,
        'preprocess_fn': preprocess_fn,
        'total_epochs': config.epochs  # Pass the total_epochs for epoch-aware modules
    }
    
    # Override with any additional trainer parameters
    base_trainer_kwargs.update(trainer_kwargs)
    
    # Create trainer
    trainer = MNISTTrainer(model, **base_trainer_kwargs)
    
    # Train the model
    history = trainer.train(
        epochs=config.epochs,
        early_stopping_patience=config.early_stopping_patience
    )
    
    # After training, capture threshold values if using threshold activation
    from activations import ThresholdActivation, JumpReLU, RescaledJumpReLU, FixedRescaledJumpReLU
    
    # First check if the main activation is a ThresholdActivation, JumpReLU, or *RescaledJumpReLU
    if isinstance(activation, (ThresholdActivation, JumpReLU, RescaledJumpReLU, FixedRescaledJumpReLU)):
        history['final_threshold'] = activation.get_threshold()
        
    # Also check if any modules in the network are ThresholdActivation, JumpReLU, or *RescaledJumpReLU
    if hasattr(model, 'network'):
        # Track which hidden layer we're on
        hidden_layer_count = 0
        
        for i, layer in enumerate(model.network):
            # Linear layers are followed by activation functions in BaseMLP
            if isinstance(layer, nn.Linear):
                hidden_layer_count += 1
                
            if isinstance(layer, (ThresholdActivation, JumpReLU, RescaledJumpReLU, FixedRescaledJumpReLU)):
                # Use the hidden layer number instead of raw index
                history[f'final_threshold_hidden_layer_{hidden_layer_count-1}'] = layer.get_threshold()
            # Also check for activation in sequential modules
            elif isinstance(layer, nn.Sequential):
                for j, sublayer in enumerate(layer):
                    if isinstance(sublayer, (ThresholdActivation, JumpReLU, RescaledJumpReLU, FixedRescaledJumpReLU)):
                        history[f'final_threshold_hidden_layer_{hidden_layer_count-1}_sub_{j}'] = sublayer.get_threshold()
    
    return trainer, history


class ExperimentTracker:
    """Tracks and summarizes experiment results."""
    
    def __init__(self, columns: List[str] = None):
        """
        Initialize experiment tracker.
        
        Args:
            columns: List of column names to track (keys in result dictionaries)
        """
        self.results = []
        # Only set default columns if none provided
        if columns is None:
            # We'll dynamically extend these based on the actual results
            self.columns = [
                'experiment', 
                'activation', 
                'architecture', 
                'temperature',
                'test_acc', 
                'train_acc'
            ]
        else:
            self.columns = columns
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a result to the tracker.
        
        Args:
            result: Dictionary containing experiment result data
        """
        self.results.append(result)
        
        # Update the columns list to include any new activation or layer parameters
        # This ensures all parameters appear in the summary table
        for key in result.keys():
            if key not in self.columns and (
                key.startswith('activation_') or 
                key.startswith('layer_') or
                key.startswith('final_threshold') or
                key.startswith('threshold_hidden_layer_') or
                key in ['temperature', 'dropout_prob']
            ):
                self.columns.append(key)
    
    def get_results_df(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame.
        
        Returns:
            DataFrame with all tracked results
        """
        return pd.DataFrame(self.results)
    
    def print_summary(self, 
                      format_spec: Dict[str, Dict[str, Any]] = None, 
                      tablefmt: str = 'grid') -> None:
        """
        Print a formatted summary of experiment results.
        
        Args:
            format_spec: Dictionary mapping column names to formatting specifications
            tablefmt: Table format for tabulate
        """
        if not self.results:
            print("No results to display.")
            return
        
        # Default formatting if none provided
        if format_spec is None:
            format_spec = DEFAULT_FORMAT_SPEC
            
        # Generate formatted data
        formatted_data = []
        headers = []
        
        # First, determine all columns that actually have data
        all_columns = []
        for col in self.columns:
            # Only include columns that exist in at least one result
            if any(col in result for result in self.results):
                all_columns.append(col)
                
        # Now generate the headers for these columns
        for col in all_columns:
            if col in format_spec:
                formatter = format_spec[col]
                # Modify header if there's a suffix
                suffix = formatter.get('suffix', '')
                prefix = formatter.get('prefix', '')
                column_name = col
                
                # Format activation_* and layer_* parameter column names for better readability
                if col.startswith('activation_'):
                    column_name = col.replace('activation_', '')
                elif col.startswith('layer_'):
                    column_name = col.replace('layer_', '')
                elif col.startswith('final_threshold_hidden_layer_'):
                    # Format as "Threshold Layer X"
                    layer_num = col.replace('final_threshold_hidden_layer_', '')
                    column_name = f"Threshold Hidden Layer {layer_num}"
                elif col.startswith('threshold_hidden_layer_'):
                    # Format as "Threshold Layer X"
                    layer_num = col.replace('threshold_hidden_layer_', '')
                    column_name = f"Threshold Hidden Layer {layer_num}"
                elif col == 'final_threshold':
                    column_name = "Final Threshold"
                
                headers.append(f"{prefix}{column_name.replace('_', ' ').title()}{suffix}")
            else:
                column_name = col
                # Format activation_* and layer_* parameter column names for better readability
                if col.startswith('activation_'):
                    column_name = col.replace('activation_', '')
                elif col.startswith('layer_'):
                    column_name = col.replace('layer_', '')
                elif col.startswith('final_threshold_hidden_layer_'):
                    # Format as "Threshold Layer X"
                    layer_num = col.replace('final_threshold_hidden_layer_', '')
                    column_name = f"Threshold Hidden Layer {layer_num}"
                elif col.startswith('threshold_hidden_layer_'):
                    # Format as "Threshold Layer X"
                    layer_num = col.replace('threshold_hidden_layer_', '')
                    column_name = f"Threshold Hidden Layer {layer_num}"
                elif col == 'final_threshold':
                    column_name = "Final Threshold"
                    
                headers.append(column_name.replace('_', ' ').title())
        
        # Format each row of data
        for result in self.results:
            row = []
            for col in all_columns:
                # Add empty string if column not in this result
                if col not in result:
                    row.append("")
                else:
                    value = result[col]
                    
                    # Apply formatting if specified
                    if col in format_spec:
                        formatter = format_spec[col]
                        fmt = formatter.get('fmt', '')
                        prefix = formatter.get('prefix', '')
                        suffix = formatter.get('suffix', '')
                        
                        # Handle different value types
                        if isinstance(value, (int, float)):
                            formatted = f"{prefix}{value:{fmt}}{suffix}"
                        elif isinstance(value, list):
                            formatted = str(value)
                        else:
                            formatted = str(value)
                    else:
                        formatted = str(value)
                    
                    row.append(formatted)
            
            formatted_data.append(row)
        
        # Print table
        print("\nExperiment Summary:")
        print(tabulate(formatted_data, headers=headers, tablefmt=tablefmt))
        
        # Also return the DataFrame for further analysis if needed
        return pd.DataFrame(data=formatted_data, columns=headers)


class ExperimentDefinition:
    """Defines a set of experiments to run."""
    
    def __init__(self, 
                 name: str,
                 config_class: Type, 
                 base_params: Dict[str, Any] = None):
        """
        Initialize experiment definition.
        
        Args:
            name: Name for this set of experiments
            config_class: Configuration class to use
            base_params: Base parameters for all experiments
        """
        self.name = name
        self.config_class = config_class
        self.base_params = base_params or {}
        
        # Ensure experiment_type is set correctly
        if 'experiment_type' not in self.base_params:
            if self.config_class == VanillaExperimentConfig:
                self.base_params['experiment_type'] = EXPERIMENT_TYPES['vanilla']
            elif self.config_class == StochasticExperimentConfig:
                self.base_params['experiment_type'] = EXPERIMENT_TYPES['stochastic']
    
    def create_config(self, params: Dict[str, Any] = None) -> Any:
        """
        Create a configuration object.
        
        Args:
            params: Parameters to override base parameters
            
        Returns:
            Configuration object
        """
        # Merge base params with provided params
        config_params = self.base_params.copy()
        if params:
            config_params.update(params)
        
        # Create config object
        return self.config_class(**config_params)


class ExperimentRunner:
    """Runs experiments with different configurations."""
    
    def __init__(self, experiment_def: ExperimentDefinition):
        """
        Initialize experiment runner.
        
        Args:
            experiment_def: Definition of the experiment to run
        """
        self.experiment_def = experiment_def
        self.tracker = ExperimentTracker()
    
    def run_grid_search(self,
                        activations: List[Tuple] = None,
                        architectures: List[List[int]] = None,
                        temperatures: List[float] = None,
                        result_keys: List[str] = None) -> ExperimentTracker:
        """
        Run a grid search across multiple parameters. This is the main method for running experiments 
        and can handle all experiment types by adjusting which parameters vary and which remain fixed.
        
        Examples:
        - For activation experiments: provide multiple activations, single architecture, no temperatures
        - For architecture experiments: provide single activation, multiple architectures, no temperatures
        - For stochastic temperature experiments: provide single activation, single architecture, multiple temperatures
        
        Args:
            activations: List of (activation_class, activation_kwargs) tuples
            architectures: List of hidden layer dimensions
            temperatures: List of temperature values (for stochastic layers)
            result_keys: Keys to extract from history for tracking
            
        Returns:
            ExperimentTracker with results
        """
        # Default result keys if none provided
        result_keys = result_keys or ['test_acc', 'train_acc']
        
        # Use default values if not provided
        activations = activations or [(None, {})]
        architectures = architectures or [[512, 256, 128]]
        
        # Loop through parameter combinations
        for activation_class, activation_kwargs in activations:
            activation_name = activation_class.__name__ if activation_class else "NoActivation"
            
            for hidden_dims in architectures:
                # For stochastic experiments, also loop through temperatures
                if self.experiment_def.config_class == StochasticExperimentConfig and temperatures:
                    for temp in temperatures:
                        # Create layer kwargs with temperature
                        layer_kwargs = {'temperature': temp}
                        
                        # Create config for this experiment
                        config = self.experiment_def.create_config({
                            'hidden_dims': hidden_dims,
                            'activation_class': activation_class,
                            'activation_kwargs': activation_kwargs,
                            'layer_kwargs': layer_kwargs
                        })
                        
                        # Get all base configuration parameters
                        base_config_params = {
                            key: getattr(config, key) 
                            for key in dir(config) 
                            if not key.startswith('_') and not callable(getattr(config, key))
                            and key not in ['activation_class', 'activation_kwargs', 'layer_kwargs', 'hidden_dims']
                        }
                        
                        # Print experiment info
                        print(f"\n{'-'*80}")
                        print(f"Running experiment with:")
                        print(f"  Architecture: {hidden_dims}")
                        print(f"  Activation: {activation_name}")
                        
                        # Print activation parameters if any
                        if activation_kwargs:
                            print(f"  Activation parameters:")
                            for key, value in activation_kwargs.items():
                                print(f"    {key}: {value}")
                                
                        # Print stochastic layer parameters
                        print(f"  StochasticLayer parameters:")
                        for key, value in layer_kwargs.items():
                            print(f"    {key}: {value}")
                            
                        # Print base configuration parameters
                        print(f"  Configuration parameters:")
                        for key, value in base_config_params.items():
                            if not isinstance(value, (type, types.ModuleType)):
                                print(f"    {key}: {value}")
                                
                        print(f"{'-'*80}")
                        
                        # Run experiment using unified run_experiment function
                        trainer, history = run_experiment(config)
                        
                        # Create result dictionary
                        result = {
                            'experiment': self.experiment_def.name,
                            'architecture': str(hidden_dims),
                            'activation': activation_name,
                            'temperature': temp
                        }
                        
                        # Add activation parameters to result
                        if activation_kwargs:
                            for key, value in activation_kwargs.items():
                                result[f'activation_{key}'] = value
                        
                        # Add layer kwargs to result
                        for key, value in layer_kwargs.items():
                            if key != 'temperature':  # We already added this separately
                                result[f'layer_{key}'] = value
                        
                        # Add configuration parameters to result
                        for key, value in base_config_params.items():
                            if not isinstance(value, (type, types.ModuleType)):
                                result[key] = value
                                
                        # Add requested history values to result
                        for key in result_keys:
                            if key in history:
                                # For list values, get the last element
                                if isinstance(history[key], list) and key != 'hidden_dims':
                                    result[key] = history[key][-1]
                                else:
                                    result[key] = history[key]
                        
                        # Add any threshold values from history
                        for key, value in history.items():
                            if (key == 'final_threshold' or 
                                key.startswith('final_threshold_hidden_layer_') or 
                                key.startswith('threshold_hidden_layer_')):
                                result[key] = value
                        
                        # Add result to tracker
                        self.tracker.add_result(result)
                        
                        # Print immediate result
                        print(f"Test accuracy: {history['test_acc']:.2f}%")
                else:
                    # Create config for this experiment
                    config = self.experiment_def.create_config({
                        'hidden_dims': hidden_dims,
                        'activation_class': activation_class,
                        'activation_kwargs': activation_kwargs
                    })
                    
                    # Get all base configuration parameters
                    base_config_params = {
                        key: getattr(config, key) 
                        for key in dir(config) 
                        if not key.startswith('_') and not callable(getattr(config, key))
                        and key not in ['activation_class', 'activation_kwargs', 'layer_kwargs', 'hidden_dims']
                    }
                    
                    # Print experiment info
                    print(f"\n{'-'*80}")
                    print(f"Running experiment with:")
                    print(f"  Architecture: {hidden_dims}")
                    print(f"  Activation: {activation_name}")
                    
                    # Print activation parameters if any
                    if activation_kwargs:
                        print(f"  Activation parameters:")
                        for key, value in activation_kwargs.items():
                            print(f"    {key}: {value}")
                    
                    # Print base configuration parameters
                    print(f"  Configuration parameters:")
                    for key, value in base_config_params.items():
                        if not isinstance(value, (type, types.ModuleType)):
                            print(f"    {key}: {value}")
                            
                    print(f"{'-'*80}")
                    
                    # Run experiment using unified run_experiment function
                    trainer, history = run_experiment(config)
                    
                    # Create result dictionary
                    result = {
                        'experiment': self.experiment_def.name,
                        'architecture': str(hidden_dims),
                        'activation': activation_name
                    }
                    
                    # Add activation parameters to result
                    if activation_kwargs:
                        for key, value in activation_kwargs.items():
                            result[f'activation_{key}'] = value
                    
                    # Add configuration parameters to result
                    for key, value in base_config_params.items():
                        if not isinstance(value, (type, types.ModuleType)):
                            result[key] = value
                    
                    # Add requested history values to result
                    for key in result_keys:
                        if key in history:
                            # For list values, get the last element
                            if isinstance(history[key], list) and key != 'hidden_dims':
                                result[key] = history[key][-1]
                            else:
                                result[key] = history[key]
                    
                    # Add any threshold values from history
                    for key, value in history.items():
                        if (key == 'final_threshold' or 
                            key.startswith('final_threshold_hidden_layer_') or 
                            key.startswith('threshold_hidden_layer_')):
                            result[key] = value
                    
                    # Add result to tracker
                    self.tracker.add_result(result)
                    
                    # Print immediate result
                    print(f"Test accuracy: {history['test_acc']:.2f}%")
        
        return self.tracker


# Commonly used format specifications for result display
DEFAULT_FORMAT_SPEC = {
    # Performance metrics
    'test_acc': {'fmt': '.2f', 'suffix': '%'},
    'train_acc': {'fmt': '.2f', 'suffix': '%'},
    'val_acc': {'fmt': '.2f', 'suffix': '%'},
    'test_loss': {'fmt': '.4f'},
    'train_loss': {'fmt': '.4f'},
    'val_loss': {'fmt': '.4f'},
    
    # Model parameters
    'temperature': {'fmt': '.2f'},
    'dropout_prob': {'fmt': '.2f'},
    
    # Activation parameters
    'activation_eps': {'fmt': '.1e'},
    'activation_temperature': {'fmt': '.2f'},
    'activation_initial_threshold': {'fmt': '.4f'},
    
    # Final threshold values after training
    'final_threshold': {'fmt': '.4f', 'prefix': '(final) '},
    'final_threshold_hidden_layer_0': {'fmt': '.4f', 'prefix': 'HL0: '},
    'final_threshold_hidden_layer_1': {'fmt': '.4f', 'prefix': 'HL1: '},
    'final_threshold_hidden_layer_2': {'fmt': '.4f', 'prefix': 'HL2: '},
    'final_threshold_hidden_layer_3': {'fmt': '.4f', 'prefix': 'HL3: '},
    'threshold_hidden_layer_0': {'fmt': '.4f', 'prefix': 'HL0: '},
    'threshold_hidden_layer_1': {'fmt': '.4f', 'prefix': 'HL1: '},
    'threshold_hidden_layer_2': {'fmt': '.4f', 'prefix': 'HL2: '},
    'threshold_hidden_layer_3': {'fmt': '.4f', 'prefix': 'HL3: '},
    
    # Layer parameters
    'layer_temperature': {'fmt': '.2f'},
    
    # Other parameters that might be added
    'learning_rate': {'fmt': '.5f'},
    'epochs': {'fmt': 'd'},
    'batch_size': {'fmt': 'd'},
} 