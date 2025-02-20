import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional, Tuple, Dict, Any, Type, Union, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import platform
import sys
from tqdm import tqdm
import randomname
import git
import numpy as np
from collections import defaultdict
import pandas as pd
from base_MLP import BaseMLP

def get_git_info() -> Dict[str, str]:
    """Get git repository information"""
    try:
        repo = git.Repo(search_parent_directories=True)
        return {
            'commit_hash': repo.head.object.hexsha,
            'branch': repo.active_branch.name,
            'is_dirty': repo.is_dirty()
        }
    except:
        return {'error': 'Not a git repository or git not installed'}

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    return {
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'platform': platform.platform(),
        'cpu_count': platform.machine(),
        'processor': platform.processor()
    }

class ExperimentManager:
    """Manages multiple experiments and provides analysis tools"""
    def __init__(self, base_dir: str = "mnist_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def load_all_experiments(self) -> pd.DataFrame:
        """Load all experiments into a pandas DataFrame for analysis"""
        experiments = []
        
        for exp_dir in self.base_dir.glob("*"):
            if not exp_dir.is_dir():
                continue
                
            history_file = exp_dir / "history.json"
            if not history_file.exists():
                continue
                
            with open(history_file, 'r') as f:
                history = json.load(f)
                
            # Flatten metadata for DataFrame
            exp_data = {
                'run_name': history['metadata']['run_name'],
                'hidden_dims': str(history['metadata']['model_config']['hidden_dims']),
                'num_parameters': history['metadata']['model_config']['num_parameters'],
                'dropout_prob': history['metadata']['model_config']['dropout_prob'],
                'learning_rate': history['metadata']['training_config']['learning_rate'],
                'batch_size': history['metadata']['training_config']['batch_size'],
                'best_val_acc': history['metadata']['final_metrics']['best_val_acc'],
                'test_acc': history['metadata']['final_metrics']['test_acc'],
                'training_duration': history['metadata']['final_metrics']['training_duration'],
                'total_epochs': history['metadata']['final_metrics']['total_epochs']
            }
            experiments.append(exp_data)
        
        return pd.DataFrame(experiments)
    
    def compare_experiments(self, metric: str = 'test_acc', top_k: int = 5):
        """Compare experiments and show top k performers"""
        df = self.load_all_experiments()
        print(f"\nTop {top_k} experiments by {metric}:")
        print(df.nlargest(top_k, metric)[['run_name', metric, 'hidden_dims', 'dropout_prob', 'learning_rate']])
        
        return df
    
    def plot_comparison(self, metric_x: str, metric_y: str, size_metric: Optional[str] = 'num_parameters'):
        """Create scatter plot comparing two metrics across experiments"""
        df = self.load_all_experiments()
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        if size_metric:
            sizes = 100 * (df[size_metric] - df[size_metric].min()) / (df[size_metric].max() - df[size_metric].min()) + 50
            plt.scatter(df[metric_x], df[metric_y], s=sizes, alpha=0.6)
        else:
            plt.scatter(df[metric_x], df[metric_y], alpha=0.6)
        
        # Get current axis bounds
        x_min, x_max = df[metric_x].min(), df[metric_x].max()
        y_min, y_max = df[metric_y].min(), df[metric_y].max()
        
        # Add padding (5% of range)
        x_padding = 0.05 * (x_max - x_min)
        y_padding = 0.05 * (y_max - y_min)
        
        # Set axis limits with padding
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)
            
        # Format axis labels
        plt.xlabel(metric_x.replace('_', ' ').title())
        plt.ylabel(metric_y.replace('_', ' ').title())
        plt.title(f'Experiment Comparison: {metric_x.replace("_", " ").title()} vs {metric_y.replace("_", " ").title()}')
        
        # Add run names as annotations with smart positioning
        for _, row in df.iterrows():
            # Add annotation with arrow
            plt.annotate(
                row['run_name'], 
                (row[metric_x], row[metric_y]),
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
            )
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        return plt.gcf()

class ModelAnalyzer:
    """Analyzes model properties and behavior"""
    def __init__(self, model: BaseMLP, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device
        
    def analyze_weights(self) -> Dict[str, Dict[str, float]]:
        """Analyze weight statistics for each layer"""
        stats = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                layer_stats = {
                    'mean': float(param.mean()),
                    'std': float(param.std()),
                    'min': float(param.min()),
                    'max': float(param.max()),
                    'norm': float(param.norm()),
                    'sparsity': float((param == 0).float().mean())
                }
                stats[name] = layer_stats
        return stats
    
    def plot_weight_distributions(self):
        """Plot weight distributions for each layer"""
        # Count number of weight tensors
        weight_params = [(name, p) for name, p in self.model.named_parameters() if 'weight' in name]
        num_weight_layers = len(weight_params)
        
        if num_weight_layers == 0:
            return None
        
        fig, axes = plt.subplots(num_weight_layers, 1, figsize=(10, 3*num_weight_layers))
        if num_weight_layers == 1:
            axes = [axes]
            
        for idx, (name, param) in enumerate(weight_params):
            sns.histplot(param.detach().cpu().numpy().flatten(), ax=axes[idx], bins=50)
            axes[idx].set_title(f'{name} Distribution')
            axes[idx].set_xlabel('Weight Value')
            axes[idx].set_ylabel('Count')
        
        plt.tight_layout()
        return fig
    
    def compute_activation_stats(self, loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """Compute activation statistics across a dataset"""
        self.model.store_activations = True
        activation_stats = defaultdict(lambda: {'means': [], 'stds': [], 'sparsity': []})
        
        with torch.no_grad():
            for data, _ in tqdm(loader, desc="Computing activation stats"):
                data = data.to(self.device)
                _ = self.model(data)
                
                for name, activation in self.model.activation_store.activations.items():
                    activation_stats[name]['means'].append(float(activation.mean()))
                    activation_stats[name]['stds'].append(float(activation.std()))
                    activation_stats[name]['sparsity'].append(float((activation == 0).float().mean()))
        
        # Compute aggregate statistics
        for name in activation_stats:
            for stat in ['means', 'stds', 'sparsity']:
                values = activation_stats[name][stat]
                activation_stats[name][f'avg_{stat}'] = float(np.mean(values))
                activation_stats[name][f'std_{stat}'] = float(np.std(values))
                del activation_stats[name][stat]
        
        return dict(activation_stats)

class MNISTTrainer:
    """
    Handles training and evaluation of BaseMLP models on MNIST.
    
    Args:
        model (BaseMLP): The model to train
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimization
        device (str): Device to run on ('cuda' or 'cpu')
        save_dir (str): Directory to save checkpoints and results
        optimizer_class (Type[optim.Optimizer]): Optimizer class to use
        optimizer_kwargs (Dict): Additional optimizer parameters
        criterion (nn.Module): Loss function to use
        tags (List[str]): List of experiment tags
    """
    def __init__(
        self,
        model: BaseMLP,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "results",
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Dict = None,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        tags: List[str] = None  # Add experiment tags
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.criterion = criterion
        
        # Generate unique run name
        self.run_name = randomname.get_name()
        print(f"Run name: {self.run_name}")
        
        # Create run directory
        self.run_dir = self.save_dir / self.run_name
        self.run_dir.mkdir(exist_ok=True)
        
        # Setup training components
        self.optimizer = optimizer_class(
            model.parameters(), 
            lr=learning_rate, 
            **self.optimizer_kwargs
        )
        
        # Load and prepare data
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()
        
        # Training history and metadata
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': None, 'test_acc': None,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'run_name': self.run_name,
                
                # Model configuration
                'model_config': {
                    'class_name': model.__class__.__name__,
                    'input_dim': model.input_dim,
                    'hidden_dims': model.hidden_dims,
                    'output_dim': model.output_dim,
                    'activation_type': model.network[1].__class__.__name__,
                    'dropout_prob': model.network[2].p if isinstance(model.network[2], nn.Dropout) else 0.0,
                    'store_activations': model.store_activations,
                    'num_parameters': sum(p.numel() for p in model.parameters()),
                    'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
                },
                
                # Training configuration
                'training_config': {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'optimizer': {
                        'class_name': optimizer_class.__name__,
                        'parameters': optimizer_kwargs
                    },
                    'criterion': criterion.__class__.__name__,
                    'device': device
                },
                
                # Dataset information
                'dataset_info': {
                    'name': 'MNIST',
                    'train_size': len(self.train_loader.dataset),
                    'val_size': len(self.val_loader.dataset),
                    'test_size': len(self.test_loader.dataset),
                    'input_shape': (1, 28, 28),
                    'num_classes': 10
                },
                
                # System information
                'system_info': get_system_info(),
                
                # Git information
                'git_info': get_git_info()
            }
        }
        
        self.tags = tags or []
        self.analyzer = ModelAnalyzer(model, device)
        
        # Update metadata with tags
        self.history['metadata']['tags'] = self.tags
    
    def _setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup MNIST data loaders"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        # Load datasets
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        # Split training into train and validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in tqdm(self.train_loader, leave=False):
            data, target = data.to(self.device), target.to(self.device)
            
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
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on given loader"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            
            total_loss += self.criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def train(self, epochs: int, early_stopping_patience: Optional[int] = 5) -> Dict:
        """
        Train the model for specified number of epochs
        
        Args:
            epochs (int): Number of epochs to train
            early_stopping_patience (Optional[int]): Patience for early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in (pbar := tqdm(range(epochs))):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            pbar.set_postfix(train_loss=f'{train_loss:.4f}', train_acc=f'{train_acc:.2f}%', val_loss=f'{val_loss:.4f}', val_acc=f'{val_acc:.2f}%')
            
            # Early stopping
            if early_stopping_patience:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f'Early stopping triggered after {epoch+1} epochs')
                        break
        
        # Final test evaluation
        test_loss, test_acc = self.evaluate(self.test_loader)
        self.history['test_loss'] = test_loss
        self.history['test_acc'] = test_acc
        print(f'\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Save final results
        self.save_results()
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        path = self.run_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'run_name': self.run_name
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.run_dir / filename
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.run_name = checkpoint['run_name']
    
    def save_results(self):
        """Save training results and plots"""
        # Update metadata with final metrics
        self.history['metadata']['final_metrics'] = {
            'best_val_loss': min(self.history['val_loss']),
            'best_val_acc': max(self.history['val_acc']),
            'best_epoch': self.history['val_loss'].index(min(self.history['val_loss'])),
            'total_epochs': len(self.history['train_loss']),
            'training_duration': (datetime.now() - datetime.fromisoformat(self.history['metadata']['timestamp'])).total_seconds(),
            'test_loss': self.history['test_loss'],
            'test_acc': self.history['test_acc']
        }
        
        # Save history with metadata
        history_path = self.run_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        # Save a readable summary
        summary_path = self.run_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Run Summary: {self.run_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Model architecture
            f.write("Model Architecture:\n")
            f.write(f"Input dim: {self.history['metadata']['model_config']['input_dim']}\n")
            f.write(f"Hidden dims: {self.history['metadata']['model_config']['hidden_dims']}\n")
            f.write(f"Output dim: {self.history['metadata']['model_config']['output_dim']}\n")
            f.write(f"Parameters: {self.history['metadata']['model_config']['num_parameters']:,}\n")
            f.write("\n")
            
            # Training config
            f.write("Training Configuration:\n")
            f.write(f"Optimizer: {self.history['metadata']['training_config']['optimizer']['class_name']}\n")
            f.write(f"Learning rate: {self.history['metadata']['training_config']['learning_rate']}\n")
            f.write(f"Batch size: {self.history['metadata']['training_config']['batch_size']}\n")
            f.write("\n")
            
            # Results
            f.write("Results:\n")
            f.write(f"Best validation accuracy: {self.history['metadata']['final_metrics']['best_val_acc']:.2f}%\n")
            f.write(f"Test accuracy: {self.history['test_acc']:.2f}%\n")
            f.write(f"Training duration: {self.history['metadata']['final_metrics']['training_duration']:.1f}s\n")
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title(f'Loss vs. Epoch\n{self.run_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train')
        plt.plot(self.history['val_acc'], label='Validation')
        plt.title(f'Accuracy vs. Epoch\n{self.run_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'training_curves.png')
        plt.close()
        
        # Add weight analysis
        weight_stats = self.analyzer.analyze_weights()
        self.history['metadata']['weight_analysis'] = weight_stats
        
        # Save weight distribution plot
        weight_dist_fig = self.analyzer.plot_weight_distributions()
        if weight_dist_fig:
            weight_dist_fig.savefig(self.run_dir / 'weight_distributions.png')
            plt.close(weight_dist_fig)
        
        # Compute and save activation statistics
        activation_stats = self.analyzer.compute_activation_stats(self.val_loader)
        self.history['metadata']['activation_analysis'] = activation_stats
        
        # Update summary.txt with analysis
        with open(self.run_dir / 'summary.txt', 'a') as f:
            f.write("\nNetwork Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write("Weight Statistics:\n")
            for layer_name, stats in weight_stats.items():
                f.write(f"{layer_name}:\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value:.4f}\n")
            
            f.write("\nActivation Statistics (Validation Set):\n")
            for layer_name, stats in activation_stats.items():
                f.write(f"{layer_name}:\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value:.4f}\n")
        
        print(f'Results saved in: {self.run_dir}')

def main():
    """Main training script with experiment management"""
    # Create experiment manager
    exp_manager = ExperimentManager('mnist_results')
    
    # Create and train model
    model = BaseMLP(
        input_dim=784,
        hidden_dims=[512, 256, 128, 64],
        output_dim=10,
        dropout_prob=0.0,
        store_activations=True
    )
    
    # Create trainer with tags
    trainer = MNISTTrainer(
        model=model,
        batch_size=128,
        learning_rate=0.001,
        save_dir='mnist_results',
        tags=['baseline', 'large_model']  # Add relevant tags
    )
    
    # Train model
    trainer.train(epochs=30, early_stopping_patience=5)
    
    # Compare with previous experiments
    exp_manager.compare_experiments()
    
    # Create comparison plots
    comparison_fig = exp_manager.plot_comparison('training_duration', 'test_acc')
    comparison_fig.savefig('mnist_results/experiment_comparison.png')
    plt.close(comparison_fig)

if __name__ == '__main__':
    main()
