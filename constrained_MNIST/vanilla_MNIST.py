from MNIST import MNISTTrainer, ExperimentManager, PlotConfig
from base_MLP import BaseMLP
import matplotlib.pyplot as plt

def run_vanilla_experiment(
    hidden_dims=[512, 256, 128, 64],
    dropout_prob=0.0,
    batch_size=128,
    learning_rate=0.001,
    epochs=30,
    early_stopping_patience=5,
    save_dir='mnist_results',
    plot_config=None
):
    """Run vanilla MLP experiment with specified configuration"""
    if plot_config is None:
        plot_config = PlotConfig(
            training_curves=True,
            weight_distributions=True,
            learning_curves=True,
            experiment_comparison=True,
            save_format='png',
            dpi=300
        )
    
    # Create experiment manager
    exp_manager = ExperimentManager(save_dir, plot_config=plot_config)
    
    # Create model
    model = BaseMLP(
        input_dim=784,  # MNIST image size (28x28)
        hidden_dims=hidden_dims,
        output_dim=10,  # Number of digits
        dropout_prob=dropout_prob,
        store_activations=True
    )
    
    # Create trainer
    trainer = MNISTTrainer(
        model=model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=save_dir,
        plot_config=plot_config,
        tags=['vanilla', f'layers_{len(hidden_dims)}', f'dropout_{dropout_prob}']
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
    # Example configurations to test
    configs = [
        {
            'hidden_dims': [512, 256, 128, 64],
            'dropout_prob': x/100,
            'epochs': 10  # Reduced epochs for faster testing
        } for x in range(0, 50, 10)  # Test dropout from 0% to 40% in steps of 10%
    ]
    
    # Run each configuration
    for config in configs:
        print(f"\nRunning experiment with config: {config}")
        trainer, history = run_vanilla_experiment(**config)
        print(f"Test accuracy: {history['test_acc']:.2f}%")

if __name__ == '__main__':
    main() 