"""
MNIST Experiment Runner

This script runs vanilla MLP experiments on the MNIST dataset
with different activation functions and configurations.
"""

import torch.nn as nn
from activations import RescaledReLU, LayerwiseSoftmax
from experiment_runners import run_vanilla_experiment
from config import VanillaExperimentConfig

def main():
    """Run experiments with different configurations"""
    # Define activation functions to test
    activations = [
        (nn.ReLU, {}),  # Standard ReLU
        (RescaledReLU, {'eps': 1e-10}),  # ReLU with rescaling
        (LayerwiseSoftmax, {'temperature': 1.0}),  # Softmax with temperature
        (LayerwiseSoftmax, {'temperature': 0.5}),  # Softmax with lower temperature
        (LayerwiseSoftmax, {'temperature': 0.2}),  # Softmax with lower temperature
        (LayerwiseSoftmax, {'temperature': 0.1}),  # Softmax with lower temperature
    ]
    
    # Base configuration
    base_config = VanillaExperimentConfig(
        hidden_dims=[512, 256, 128, 64],
        dropout_prob=0.2,
        epochs=100,
        learning_rate=0.001
    )
    
    # Run experiments for each activation function
    for activation_class, activation_kwargs in activations:
        # Update the base config for this specific experiment
        config = VanillaExperimentConfig(
            **base_config.__dict__,
            activation_class=activation_class,
            activation_kwargs=activation_kwargs
        )
        
        print(f"\nRunning experiment with activation: {activation_class.__name__}")
        if activation_kwargs:
            print(f"Activation parameters: {activation_kwargs}")
        
        trainer, history = run_vanilla_experiment(config)
        print(f"Test accuracy: {history['test_acc']:.2f}%")

if __name__ == '__main__':
    main() 