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
    HectoFixedRescaledJumpReLU,
    DecayingRescaledJumpReLU,
    DecayingRescaledReLU,
    DecayingFixedRescaledJumpReLU
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
        (RescaledJumpReLU, {'initial_threshold': 0.5, 'eps': 1e-10}),
        (HectoRescaledReLU, {}),
        (HectoRescaledJumpReLU, {'initial_threshold': 0.5}),
        # (HectoFixedRescaledJumpReLU, {'threshold': 0.7})
        # (DecayingRescaledJumpReLU, {'initial_threshold': 0.5, 'initial_scale': 100, 'final_scale': 1, 'decay_epochs': 10, 'decay_schedule': 'exponential', 'decay_rate': 0.95}),
        # (DecayingRescaledReLU, {'initial_scale': 100, 'final_scale': 1, 'decay_epochs': 10, 'decay_schedule': 'exponential', 'decay_rate': 0.95}),
        # (DecayingFixedRescaledJumpReLU, {'threshold': 0.5, 'initial_scale': 100, 'final_scale': 1, 'decay_epochs': 10, 'decay_schedule': 'exponential', 'decay_rate': 0.95})
    ]
    
    # Create experiment definition
    experiment_def = ExperimentDefinition(
        name="Vanilla MLP",
        config_class=VanillaExperimentConfig,
        base_params={
            'hidden_dims': [512, 256, 128, 64],
            'dropout_prob': 0.0,
            'epochs': 100,
            'learning_rate': 0.001,
            'early_stopping_patience': 20,
            'experiment_type': EXPERIMENT_TYPES['vanilla'],
            'save_dir': 'results/rescaled_convergence_speed/run_6'
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