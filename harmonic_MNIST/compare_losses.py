import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import numpy as np

def load_stats_files(directory):
    stats_data = []
    for filename in os.listdir(directory):
        if filename.endswith('_stats.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                # Extract key information
                stats = {
                    'loss_type': 'harmonic' if data['hyperparameters']['harmonic'] else 'cross_entropy',
                    'temperature': data['hyperparameters']['temperature'],
                    'hidden_dim': data['hyperparameters']['hidden_dim'],
                    'n': data['hyperparameters']['n'],
                    'test_accuracy': data['training_history']['final_test_accuracy'],
                    'test_loss': data['training_history']['final_test_loss'],
                    'best_train_loss': data['training_history']['best_loss'],
                    'training_losses': data['training_history']['epoch_losses'],
                    'duration': data['timestamps']['training_duration']
                }
                stats_data.append(stats)
    return pd.DataFrame(stats_data)

def plot_accuracy_vs_temperature(df, save_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df[df['loss_type'] == 'cross_entropy'], 
                   x='temperature', y='test_accuracy', label='Cross Entropy')
    sns.scatterplot(data=df[df['loss_type'] == 'harmonic'], 
                   x='temperature', y='test_accuracy', label='Harmonic')
    
    plt.xscale('log')
    plt.xlabel('Temperature')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Temperature by Loss Type')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_vs_temperature.png'))
    plt.close()

    # Create focused plot for hidden_dim=400
    plt.figure(figsize=(10, 6))
    df_400 = df[df['hidden_dim'] == 400]
    
    # Plot points
    sns.scatterplot(data=df_400[df_400['loss_type'] == 'cross_entropy'], 
                   x='temperature', y='test_accuracy', label='Cross Entropy', s=100)
    sns.scatterplot(data=df_400[df_400['loss_type'] == 'harmonic'], 
                   x='temperature', y='test_accuracy', label='Harmonic', s=100)
    
    # Add trend lines
    for loss_type in ['cross_entropy', 'harmonic']:
        subset = df_400[df_400['loss_type'] == loss_type]
        if not subset.empty:
            # Sort by temperature for line plot
            subset = subset.sort_values('temperature')
            plt.plot(subset['temperature'], subset['test_accuracy'], '--', alpha=0.5)
    
    plt.xscale('log')
    plt.xlabel('Temperature')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Temperature (Hidden Dim = 400)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_vs_temperature_hidden400.png'))
    plt.close()

    # Print detailed statistics for hidden_dim=400
    print("\nDetailed Analysis for Hidden Dim = 400:")
    for loss_type in ['cross_entropy', 'harmonic']:
        subset = df_400[df_400['loss_type'] == loss_type]
        if not subset.empty:
            print(f"\n{loss_type.replace('_', ' ').title()} Loss:")
            print(f"Number of runs: {len(subset)}")
            print(f"Temperature range: {subset['temperature'].min():.3f} to {subset['temperature'].max():.3f}")
            print(f"Accuracy range: {subset['test_accuracy'].min():.2f}% to {subset['test_accuracy'].max():.2f}%")
            best_run = subset.loc[subset['test_accuracy'].idxmax()]
            print(f"Best accuracy: {best_run['test_accuracy']:.2f}% (at temperature={best_run['temperature']:.3f})")

def plot_accuracy_vs_hidden_dim(df, save_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df[df['loss_type'] == 'cross_entropy'], 
                   x='hidden_dim', y='test_accuracy', label='Cross Entropy')
    sns.scatterplot(data=df[df['loss_type'] == 'harmonic'], 
                   x='hidden_dim', y='test_accuracy', label='Harmonic')
    
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Hidden Dimension by Loss Type')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_vs_hidden_dim.png'))
    plt.close()

def plot_accuracy_vs_n(df, save_dir):
    harmonic_df = df[df['loss_type'] == 'harmonic']
    if not harmonic_df.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=harmonic_df, x='n', y='test_accuracy')
        plt.xlabel('n (Harmonic Loss Parameter)')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Test Accuracy vs n Parameter for Harmonic Loss')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'accuracy_vs_n.png'))
        plt.close()

def plot_training_curves(df, save_dir):
    plt.figure(figsize=(12, 6))
    
    # Select representative runs for each loss type
    for loss_type in ['cross_entropy', 'harmonic']:
        subset = df[df['loss_type'] == loss_type]
        if not subset.empty:
            # Get the run with median performance
            median_idx = subset['test_accuracy'].argmin()
            losses = subset.iloc[median_idx]['training_losses']
            plt.plot(losses, label=f'{loss_type.replace("_", " ").title()}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves (Representative Runs)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def plot_performance_summary(df, save_dir):
    plt.figure(figsize=(10, 6))
    
    summary_data = df.groupby('loss_type').agg({
        'test_accuracy': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Create box plot
    sns.boxplot(data=df, x='loss_type', y='test_accuracy')
    plt.xlabel('Loss Type')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Performance Distribution by Loss Type')
    plt.savefig(os.path.join(save_dir, 'performance_summary.png'))
    plt.close()
    
    return summary_data

def main():
    # Create output directory for plots
    output_dir = 'MNIST_experiments/analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    stats_dir = 'MNIST_experiments/run_stats/softmax_weights_only'
    df = load_stats_files(stats_dir)
    
    # Generate plots
    plot_accuracy_vs_temperature(df, output_dir)
    plot_accuracy_vs_hidden_dim(df, output_dir)
    plot_accuracy_vs_n(df, output_dir)
    plot_training_curves(df, output_dir)
    summary_data = plot_performance_summary(df, output_dir)
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, 'performance_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Performance Summary:\n\n")
        f.write(str(summary_data))
        
        # Additional statistics
        f.write("\n\nBest Configurations:\n")
        for loss_type in ['cross_entropy', 'harmonic']:
            subset = df[df['loss_type'] == loss_type]
            if not subset.empty:
                best_run = subset.loc[subset['test_accuracy'].idxmax()]
                f.write(f"\nBest {loss_type.replace('_', ' ').title()} Configuration:\n")
                f.write(f"Test Accuracy: {best_run['test_accuracy']:.2f}%\n")
                f.write(f"Temperature: {best_run['temperature']}\n")
                f.write(f"Hidden Dimension: {best_run['hidden_dim']}\n")
                if loss_type == 'harmonic':
                    f.write(f"n parameter: {best_run['n']}\n")

if __name__ == "__main__":
    main() 