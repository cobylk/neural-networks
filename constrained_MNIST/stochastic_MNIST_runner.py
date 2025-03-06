"""
Stochastic MNIST Experiment Runner

This script runs experiments with stochastic layers on the MNIST dataset
with different activation functions and configurations.
"""

from activations import ThresholdActivation, SparseMax, PowerNormalization
from experiment_runners import run_stochastic_experiment
from utils import track_threshold_evolution
from config import StochasticExperimentConfig

def main():
    """Run experiments with different architectures and activations"""
    # Define activation functions to test
    activations = [
        (None, None),  # No activation (just stochastic layers)
        # (PowerNormalization, None),
        # (SparseMax, None),
        (ThresholdActivation, {
            'initial_threshold': 0.1, 
            'sharpness': 15.0,  # Higher sharpness for steeper thresholding
        }),
        # (ThresholdActivation, {
        #     'initial_threshold': 0.5,
        #     'sharpness': 15.0,
        # }),
        # (ThresholdActivation, {'initial_threshold': 0.05})
    ]
    
    # Test different network architectures
    architectures = [
        # [512],          # Single layer
        # [512, 256],     # Two layers
        [512, 256, 128]  # Three layers
    ]
    
    # Test different temperatures for the stochastic layer
    temperatures = [1.0]  # Lower temperatures for sharper distributions
    
    results = []
    
    # Run experiments for each combination
    for temp in temperatures:
        for hidden_dims in architectures:
            for activation_class, activation_kwargs in activations:
                activation_name = activation_class.__name__ if activation_class else "NoActivation"
                print(f"\nRunning experiment with:")
                print(f"Architecture: {hidden_dims}")
                print(f"Activation: {activation_name}")
                print(f"StochasticLayer Temperature: {temp}")
                if activation_kwargs:
                    print(f"Activation parameters: {activation_kwargs}")
                
                # Create configuration
                config = StochasticExperimentConfig(
                    hidden_dims=hidden_dims,
                    epochs=100,
                    early_stopping_patience=20,
                    learning_rate=0.001, 
                    batch_size=128,
                    activation_class=activation_class,
                    activation_kwargs=activation_kwargs,
                    layer_kwargs={'temperature': temp}
                )
                
                trainer, history, act_name = run_stochastic_experiment(config)
                
                # Track threshold evolution if it's a ThresholdActivation
                if activation_class == ThresholdActivation:
                    track_threshold_evolution(trainer, act_name)
                
                results.append({
                    'architecture': hidden_dims,
                    'activation': act_name,
                    'temperature': temp,
                    'test_acc': history['test_acc'],
                    'train_acc': history['train_acc'][-1]
                })
                print(f"Test accuracy: {history['test_acc']:.2f}%")
    
    # Print comprehensive summary
    print("\nExperiment Summary:")
    print("-" * 100)
    print(f"{'Activation':<20} {'Architecture':<15} {'Temp':>5} {'Test Acc':>10} {'Train Acc':>10}")
    print("-" * 100)
    for result in results:
        print(f"{result['activation']:<20} {str(result['architecture']):<15} "
              f"{result['temperature']:>5.2f} {result['test_acc']:>9.2f}% {result['train_acc']:>9.2f}%")
    print("-" * 100)

if __name__ == '__main__':
    main() 