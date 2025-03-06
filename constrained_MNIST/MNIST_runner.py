"""
MNIST Experiment Runner

This script runs vanilla MLP experiments on the MNIST dataset
with different activation functions and configurations.
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
    """Run experiments with different configurations"""
    # Define activation functions to test
    activations = [
        (nn.ReLU, {}),  # Standard ReLU
        (RescaledReLU, {'eps': 1e-10}),  # ReLU with rescaling
        (LayerwiseSoftmax, {'temperature': 1.0}),  # Softmax with temperature
        (LayerwiseSoftmax, {'temperature': 0.5}),  # Softmax with lower temperature
        # (LayerwiseSoftmax, {'temperature': 0.2}),  # Softmax with lower temperature
        # (LayerwiseSoftmax, {'temperature': 0.1}),  # Softmax with lower temperature
    ]
    
    # Create experiment definition
    experiment_def = ExperimentDefinition(
        name="Vanilla MLP",
        config_class=VanillaExperimentConfig,
        base_params={
            'hidden_dims': [512, 256, 128, 64],
            'dropout_prob': 0.2,
            'epochs': 5,
            'learning_rate': 0.001,
            'experiment_type': EXPERIMENT_TYPES['vanilla']
        }
    )
    
    # Create experiment runner
    runner = ExperimentRunner(experiment_def)
    
    # Run activation experiments using grid search
    # (multiple activations, single architecture)
    tracker = runner.run_grid_search(
        activations=activations,
        architectures=[[512, 256, 128, 64]],  # Fixed architecture
        result_keys=['test_acc', 'train_acc']
    )
    
    # Print formatted summary
    tracker.print_summary(format_spec=DEFAULT_FORMAT_SPEC)

if __name__ == '__main__':
    main() 