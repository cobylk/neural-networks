"""
MNIST Experiment Runner

This script runs vanilla MLP experiments on the MNIST dataset
with different activation functions and configurations.
"""

import sys
import torch
import torch.nn as nn
from activations import (
    RescaledReLU, 
    LayerwiseSoftmax, 
    PowerNormalization, 
    TemperaturePowerNorm,
    ThresholdActivation,
    JumpReLU,
    RescaledJumpReLU,
    FixedRescaledJumpReLU,
    HectoRescaledReLU,
    HectoRescaledJumpReLU,
    HectoFixedRescaledJumpReLU
)
from config import VanillaExperimentConfig, EXPERIMENT_TYPES
from experiment_runner_utils import (
    ExperimentDefinition, 
    ExperimentRunner, 
    DEFAULT_FORMAT_SPEC
)

def main():
    """Run experiments with different configurations"""
    # Define activation functions to test
    activations = [
        # (nn.ReLU, {}),  
        # (JumpReLU, {'initial_threshold': 0.5}),  
        # (RescaledReLU, {'eps': 1e-10}),  
        # (RescaledJumpReLU, {'initial_threshold': 0.5, 'eps': 1e-10}),
        # (HectoRescaledReLU, {}),
        # (HectoRescaledJumpReLU, {'initial_threshold': 0.5}),
        (HectoFixedRescaledJumpReLU, {'threshold': 0.7})
    ]
    
    # Create experiment definition
    experiment_def = ExperimentDefinition(
        name="Vanilla MLP",
        config_class=VanillaExperimentConfig,
        base_params={
            'hidden_dims': [512, 256, 128, 64],
            'dropout_prob': 0.2,
            'epochs': 100,
            'learning_rate': 0.001,
            'early_stopping_patience': 20,
            'experiment_type': EXPERIMENT_TYPES['vanilla'],
            'save_dir': 'results/rescaled_convergence_speed/run_3'
        }
    )
    
    runner = ExperimentRunner(experiment_def)
    
    tracker = runner.run_grid_search(
        activations=activations,
        architectures=[[512, 256, 128, 64]], 
        result_keys=['test_acc', 'train_acc', 'final_threshold']
    )
    
    tracker.print_summary(format_spec=DEFAULT_FORMAT_SPEC)

if __name__ == '__main__':
    main() 