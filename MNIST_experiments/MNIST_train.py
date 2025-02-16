from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import datetime
import numpy as np
import argparse
from dist_layer import DistLayer
import json
from pathlib import Path

# Define the model class
class SimpleNN(nn.Module):
    def __init__(self, harmonic=False, n=28.):
        super(SimpleNN, self).__init__()
        self.harmonic = harmonic
        self.flatten = nn.Flatten()
        if harmonic:
            self.fc1 = DistLayer(28 * 28, 10, n=n)
        else:
            self.fc1 = nn.Linear(28 * 28, 10)
        nn.init.normal_(self.fc1.weight, mean=0, std=1/28.)

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        if self.harmonic:
            x = self.fc1(x)
            prob = x/torch.sum(x, dim=1, keepdim=True)
            logits = (-1)*torch.log(prob)
            return logits
        return self.fc1(x)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(str(device.type))

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

def train_epoch(model, train_loader, criterion, optimizer, harmonic=False):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        outputs = model(data)
        if harmonic:
            loss = outputs[range(targets.size(0)), targets].mean()
        else:
            loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def test(model, test_loader, criterion, harmonic=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            if harmonic:
                test_loss += outputs[range(targets.size(0)), targets].mean().item()
                outputs = (-1)*outputs  # Negate for correct prediction
            else:
                test_loss += criterion(outputs, targets).item()
            
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def save_run_stats(args, stats, save_dir="run_stats"):
    """Save detailed run statistics to a JSON file."""
    # Create stats directory if it doesn't exist
    stats_dir = Path(save_dir)
    stats_dir.mkdir(exist_ok=True)
    
    # Prepare the statistics dictionary
    run_info = {
        "hyperparameters": {
            "harmonic": args.harmonic,
            "n": args.n if args.harmonic else None,
            "epochs": args.epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "run_name": args.run_name
        },
        "training_history": {
            "epoch_losses": stats["epoch_losses"],
            "early_stop_epoch": stats["early_stop_epoch"],
            "best_epoch": stats["best_epoch"],
            "best_loss": stats["best_loss"],
            "final_test_loss": stats["final_test_loss"],
            "final_test_accuracy": stats["final_test_accuracy"]
        },
        "timestamps": {
            "start_time": stats["start_time"],
            "end_time": stats["end_time"],
            "training_duration": stats["training_duration"]
        }
    }
    
    # Save to JSON file
    timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    if args.run_name is None:
        args.run_name = f"{'harmonic-loss' if args.harmonic else 'cross-entropy-loss'}_{'n=' + str(args.n) + '_' if args.harmonic else ''}epochs={args.epochs}"
    filename = f"{args.run_name}_{timestamp}_stats.json"
    
    with open(stats_dir / filename, 'w') as f:
        json.dump(run_info, f, indent=4)
    print(f"Run statistics saved to {stats_dir / filename}")

def main(args):
    # Setup dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Initialize model
    model = SimpleNN(harmonic=args.harmonic, n=args.n).to(device)
    
    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss() if not args.harmonic else None
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training variables
    best_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None
    best_epoch = 0
    
    # Statistics tracking
    stats = {
        "epoch_losses": [],
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "early_stop_epoch": None,
        "best_epoch": None,
        "best_loss": None,
        "final_test_loss": None,
        "final_test_accuracy": None
    }
    
    # Training loop
    for epoch in tqdm(range(args.epochs)):
        epoch_loss = train_epoch(model, train_loader, criterion, optimizer, args.harmonic)
        stats["epoch_losses"].append(epoch_loss)
        
        # Early stopping check
        if epoch_loss < best_loss - args.min_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
            best_weights = model.state_dict()
            best_epoch = epoch
        else:
            epochs_no_improve += 1

        if epoch + 1 % 10 == 0:
            print(f"Epoch [{epoch + 1}], Loss: {epoch_loss:.4f}")

        if epochs_no_improve >= args.patience:
            print(f"Stopping training. No improvement for {args.patience} epochs.")
            stats["early_stop_epoch"] = epoch
            break
    
    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    # Update statistics
    stats["best_epoch"] = best_epoch
    stats["best_loss"] = best_loss
    
    # Final test
    test_loss, accuracy = test(model, test_loader, criterion, args.harmonic)
    stats["final_test_loss"] = test_loss
    stats["final_test_accuracy"] = accuracy
    
    # Record end time and duration
    stats["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_dt = datetime.datetime.strptime(stats["start_time"], "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.datetime.strptime(stats["end_time"], "%Y-%m-%d %H:%M:%S")
    stats["training_duration"] = str(end_dt - start_dt)
    
    # Save statistics
    save_run_stats(args, stats)
    
    # Save the weights
    weights_dir = "saved_weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    if args.run_name is None:
        args.run_name = f"{'harmonic-loss' if args.harmonic else 'cross-entropy-loss'}_{'n=' + str(args.n) + '_' if args.harmonic else ''}epochs={args.epochs}"
    save_name = f"{args.run_name}_{timestamp}"
    weights_path = os.path.join(weights_dir, f"{save_name}.pt")
    
    # Save only the weights from fc1
    weights = model.fc1.weight.data.cpu()
    torch.save(weights, weights_path)
    print(f"Weights saved to {weights_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST with Harmonic or Cross-Entropy Loss')
    parser.add_argument('--harmonic', action='store_true',
                        help='use harmonic loss instead of cross-entropy')
    parser.add_argument('--n', type=float, default=28.,
                        help='harmonic exponent n (default: 28.0)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='maximum number of epochs to train (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping patience (default: 10)')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                        help='minimum change in loss for early stopping (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='name for this training run')
    
    args = parser.parse_args()
    main(args) 