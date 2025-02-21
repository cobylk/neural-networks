from experiment_utils import MNISTTrainer, ExperimentManager, PlotConfig
from base_MLP import BaseMLP
from activations import RescaledReLU, LayerwiseSoftmax
import matplotlib.pyplot as plt
import torch.nn as nn

def run_vanilla_experiment(
    hidden_dims=[512, 256, 128, 64],
    dropout_prob=0.0,
    batch_size=128,
    learning_rate=0.001,
    epochs=30,
    early_stopping_patience=5,
    save_dir='constrained_activations_dropout',
    plot_config=None,
    activation_class=nn.ReLU,
    activation_kwargs=None
):
    """Run vanilla MLP experiment with specified configuration"""
    if plot_config is None:
        plot_config = PlotConfig(
            training_curves=True,
            weight_distributions=False,
            learning_curves=True,
            experiment_comparison=False,
            save_format='png',
            dpi=300
        )
    
    # Create experiment manager
    exp_manager = ExperimentManager(save_dir, plot_config=plot_config)
    
    # Create activation instance
    activation_kwargs = activation_kwargs or {}
    activation = activation_class(**activation_kwargs)
    
    # Get activation name for tags
    activation_name = activation.__class__.__name__
    if activation_kwargs:
        activation_name += "_" + "_".join(f"{k}_{v}" for k, v in activation_kwargs.items())
    
    # Create model
    model = BaseMLP(
        input_dim=784,  # MNIST image size (28x28)
        hidden_dims=hidden_dims,
        output_dim=10,  # Number of digits
        dropout_prob=dropout_prob,
        activation=activation,
        store_activations=True
    )
    
    # Create trainer
    trainer = MNISTTrainer(
        model=model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=save_dir,
        plot_config=plot_config,
        tags=[f'layers_{len(hidden_dims)}', f'dropout_{dropout_prob}', f'activation_{activation_name}']
    )
    
    # Train model
    history = trainer.train(epochs=epochs, early_stopping_patience=early_stopping_patience)
    
    # Compare with previous experiments
    df = exp_manager.compare_experiments()
    
    # Only create comparison plot if we have more than one experiment
    if len(df) > 1:
        comparison_fig = exp_manager.plot_comparison('dropout_prob', 'test_acc')
        if comparison_fig:
            try:
                comparison_fig.savefig(
                    f'{save_dir}/experiment_comparison.{plot_config.save_format}',
                    dpi=plot_config.dpi
                )
            except Exception as e:
                print(f"Warning: Could not save comparison plot: {e}")
            finally:
                plt.close(comparison_fig)
    
    return trainer, history

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
    base_config = {
        'hidden_dims': [512, 256, 128, 64],
        'dropout_prob': 0.2,
        'epochs': 100,
        'learning_rate': 0.001
    }
    
    # Run experiments for each activation function
    for activation_class, activation_kwargs in activations:
        config = {
            **base_config,
            'activation_class': activation_class,
            'activation_kwargs': activation_kwargs
        }
        print(f"\nRunning experiment with activation: {activation_class.__name__}")
        if activation_kwargs:
            print(f"Activation parameters: {activation_kwargs}")
        
        trainer, history = run_vanilla_experiment(**config)
        print(f"Test accuracy: {history['test_acc']:.2f}%")

if __name__ == '__main__':
    main() 