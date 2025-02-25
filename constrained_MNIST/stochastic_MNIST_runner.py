from experiment_utils import MNISTTrainer, ExperimentManager, PlotConfig
from activations import *
from base_MLP import BaseMLP
from stochastic_layer import StochasticLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SimplexMNISTTrainer(MNISTTrainer):
    """Extended MNIST trainer that normalizes inputs to lie on the simplex"""
    
    def _normalize_to_simplex(self, x):
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
        return x / x.sum(dim=1, keepdim=True)

    def train_epoch(self):
        """Override train_epoch to normalize inputs"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in tqdm(self.train_loader, leave=False):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply simplex normalization
            data = self._normalize_to_simplex(data)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, loader):
        """Override evaluate to normalize inputs"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply simplex normalization
            data = self._normalize_to_simplex(data)
            
            output = self.model(data)
            
            total_loss += self.criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

def run_stochastic_experiment(
    hidden_dims=[512, 256, 128, 64],
    batch_size=128,
    learning_rate=0.001,
    epochs=30,
    early_stopping_patience=5,
    save_dir='stochastic_experiments',
    activation_class=None,
    activation_kwargs=None,
    layer_kwargs=None
):
    """Run experiment with StochasticLayer and simplex-normalized MNIST data"""
    
    # Get activation name for directory organization
    activation_name = activation_class.__name__ if activation_class else "NoActivation"
    if activation_kwargs:
        activation_name += "_" + "_".join(f"{k}_{v}" for k, v in activation_kwargs.items())
    
    # Add layer parameters to directory name
    if layer_kwargs:
        activation_name += "_layer_" + "_".join(f"{k}_{v}" for k, v in layer_kwargs.items())
    
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
        layer_kwargs=layer_kwargs,
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
        # (PowerNormalization, None),
        (SparseMax, None)
    ]
    
    # Test different network architectures
    architectures = [
        # [512],          # Single layer
        # [512, 256],     # Two layers
        [512, 256, 128] # Three layers
    ]
    
    # Test different temperatures for the stochastic layer
    temperatures = [0.01]#, 0.1, 0.5, 1.0]  # Lower temperatures for sharper distributions
    
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
                
                # Create model with specific temperature
                config = {
                    'hidden_dims': hidden_dims,
                    'epochs': 50,
                    'learning_rate': 0.001, 
                    'batch_size': 128,
                    'activation_class': activation_class,
                    'activation_kwargs': activation_kwargs,
                    'layer_kwargs': {'temperature': temp}
                }
                
                trainer, history, act_name = run_stochastic_experiment(**config)
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