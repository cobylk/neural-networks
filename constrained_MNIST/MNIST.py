import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional, Tuple, Dict, Any, Type, Union
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import platform
import sys
from tqdm import tqdm
import randomname
import git
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
        criterion: nn.Module = nn.CrossEntropyLoss()
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
        
        print(f'Results saved in: {self.run_dir}')

def main():
    """Main training script"""
    # Create model with reasonable defaults for MNIST
    model = BaseMLP(
        input_dim=784,  # 28x28 images
        hidden_dims=[512, 256, 128, 64],  # Larger network for better performance
        output_dim=10,  # 10 digit classes
        dropout_prob=0.0,  # Add dropout for regularization
        store_activations=True  # Enable activation storage for analysis
    )
    
    # Create trainer
    trainer = MNISTTrainer(
        model=model,
        batch_size=128,
        learning_rate=0.001,
        save_dir='mnist_results'
    )
    
    # Train model
    trainer.train(
        epochs=30,
        early_stopping_patience=5
    )

if __name__ == '__main__':
    main()
