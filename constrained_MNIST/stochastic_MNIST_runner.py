from experiment_utils import MNISTTrainer, ExperimentManager, PlotConfig
from activations import *
from base_MLP import BaseMLP
from stochastic_layer import StochasticLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplexMNISTTrainer(MNISTTrainer):
    """Extended MNIST trainer that normalizes inputs to lie on the simplex"""
    
    def _normalize_to_simplex(self, x):
        """
        Normalize input images to lie on the probability simplex.
        Each image is normalized so its pixels sum to 1 and are non-negative.
        """
        # Ensure non-negativity (should already be true for MNIST)
        x = F.relu(x)
        # Flatten images
        x = x.view(x.size(0), -1)
        # Normalize each image to sum to 1
        return x / (x.sum(dim=1, keepdim=True) + 1e-8)

    def train_step(self, batch):
        """Override train step to normalize inputs"""
        images, labels = batch
        images = self._normalize_to_simplex(images)
        return super().train_step((images, labels))

    def eval_step(self, batch):
        """Override eval step to normalize inputs"""
        images, labels = batch
        images = self._normalize_to_simplex(images)
        return super().eval_step((images, labels))

def run_stochastic_experiment(
    hidden_dims=[512, 256, 128, 64],
    batch_size=128,
    learning_rate=0.001,
    epochs=30,
    early_stopping_patience=5,
    save_dir='stochastic_experiments',
    activation_class=None,
    activation_kwargs=None
):
    """Run experiment with StochasticLayer and simplex-normalized MNIST data"""
    
    # Get activation name for directory organization
    activation_name = activation_class.__name__ if activation_class else "NoActivation"
    if activation_kwargs:
        activation_name += "_" + "_".join(f"{k}_{v}" for k, v in activation_kwargs.items())
    
    # Create experiment directory
    experiment_dir = f"{save_dir}/{activation_name}"
    
    plot_config = PlotConfig(
        training_curves=True,
        weight_distributions=True,
        learning_curves=True,
        experiment_comparison=True,
        save_format='png',
        dpi=300
    )
    
    # Create experiment manager
    exp_manager = ExperimentManager(experiment_dir, plot_config=plot_config)
    
    # Create activation instance
    activation = None
    if activation_class:
        activation_kwargs = activation_kwargs or {}
        activation = activation_class(**activation_kwargs)
    
    # Create model with stochastic layers
    model = BaseMLP(
        input_dim=784,
        hidden_dims=hidden_dims,
        output_dim=10,
        layer_class=StochasticLayer,
        activation=activation,
        store_activations=True
    )
    
    # Create trainer with simplex normalization
    trainer = SimplexMNISTTrainer(
        model=model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=experiment_dir,
        plot_config=plot_config,
        tags=[
            f'stochastic_layers_{len(hidden_dims)}',
            f'activation_{activation_name}'
        ]
    )
    
    # Train model
    history = trainer.train(epochs=epochs, early_stopping_patience=early_stopping_patience)
    
    return trainer, history, activation_name

def main():
    """Run experiments with different architectures and activations"""
    # Define activation functions to test
    activations = [
        (None, None),  # No activation (just stochastic layers)
        # (RescaledReLU, {'eps': 1e-6}),
        # (LayerwiseSoftmax, {'temperature': 0.7}),
        # (LayerwiseSoftmax, {'temperature': 0.5}),
        # (PowerNormalization, {'alpha': 0.5, 'eps': 1e-6}),
        # (TemperaturePowerNorm, {'alpha': 0.5, 'temperature': 0.7, 'eps': 1e-6}),
        (SparseMax, None)
    ]
    
    # Test different network architectures
    architectures = [
        [256],
    ]
    
    results = []
    
    # Run experiments for each combination
    for hidden_dims in architectures:
        for activation_class, activation_kwargs in activations:
            activation_name = activation_class.__name__ if activation_class else "NoActivation"
            print(f"\nRunning experiment with:")
            print(f"Architecture: {hidden_dims}")
            print(f"Activation: {activation_name}")
            if activation_kwargs:
                print(f"Parameters: {activation_kwargs}")
            
            config = {
                'hidden_dims': hidden_dims,
                'epochs': 100,
                'learning_rate': 0.001,
                'batch_size': 128,
                'activation_class': activation_class,
                'activation_kwargs': activation_kwargs
            }
            
            trainer, history, act_name = run_stochastic_experiment(**config)
            results.append({
                'architecture': hidden_dims,
                'activation': act_name,
                'test_acc': history['test_acc'],  # Get final test accuracy
                'train_acc': history['train_acc'][-1]  # Get final train accuracy
            })
            print(f"Test accuracy: {history['test_acc']:.2f}%")
    
    # Print comprehensive summary
    print("\nExperiment Summary:")
    print("-" * 80)
    print(f"{'Activation':<30} {'Architecture':<20} {'Test Acc':>10} {'Train Acc':>10}")
    print("-" * 80)
    for result in results:
        print(f"{result['activation']:<30} {str(result['architecture']):<20} "
              f"{result['test_acc']:>9.2f}% {result['train_acc']:>9.2f}%")
    print("-" * 80)

if __name__ == '__main__':
    main() 