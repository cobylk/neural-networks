"""
Architecture Experiment Runner

This script runs experiments with different network architectures
using a fixed activation function.
"""

import torch.nn as nn
from activations import RescaledReLU, LayerwiseSoftmax
from config import VanillaExperimentConfig, EXPERIMENT_TYPES
from experiment_runner_utils import (
    ExperimentDefinition, 
    ExperimentRunner, 
    DEFAULT_FORMAT_SPEC
)

def main():
    """Run experiments with different architectures"""
    # Define architectures to test
    architectures = [
        [512],                  # Single layer
        [512, 256],             # Two layers
        [512, 256, 128],        # Three layers
        [512, 256, 128, 64],    # Four layers
        [1024, 512, 256, 128]   # Four layers, wider
    ]
    
    # Choose a fixed activation function
    activation = (LayerwiseSoftmax, {'temperature': 0.5})
    
    # Create experiment definition
    experiment_def = ExperimentDefinition(
        name="Architecture Comparison",
        config_class=VanillaExperimentConfig,
        base_params={
            'dropout_prob': 0.2,
            'epochs': 100,
            'learning_rate': 0.001,
            'experiment_type': EXPERIMENT_TYPES['vanilla']
        }
    )
    
    # Create experiment runner
    runner = ExperimentRunner(experiment_def)
    
    # Run architecture experiments using grid search
    # (single activation, multiple architectures)
    tracker = runner.run_grid_search(
        activations=[activation],
        architectures=architectures,
        result_keys=['test_acc', 'train_acc', 'test_loss']
    )
    
    # Print formatted summary
    tracker.print_summary(format_spec=DEFAULT_FORMAT_SPEC)
    
    # Optional: Show more detailed tracking of specific parameters
    format_spec = DEFAULT_FORMAT_SPEC.copy()
    format_spec.update({
        'test_loss': {'fmt': '.4f'},
        'architecture': {}
    })
    
    # Use a different table format
    tracker.print_summary(format_spec=format_spec, tablefmt='pretty')

if __name__ == '__main__':
    main() 