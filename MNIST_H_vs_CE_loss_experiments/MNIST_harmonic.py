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

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        # Initialize weights with smaller values
        self.linear = nn.Linear(28 * 28, 10)
        # with torch.no_grad():
        #     self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.flatten(x)
        return x, self.linear(x)  # Return both flattened input and logits

class HarmonicLoss(nn.Module):
    def __init__(self, model, input_dim=784, n=None, epsilon=1e-8):
        super(HarmonicLoss, self).__init__()
        self.model = model
        self.n = int(np.sqrt(input_dim)/3) if n is None else n
        self.epsilon = epsilon
        print(f"Initialized HarmonicLoss with n={self.n}, input_dim={input_dim}")
    
    def forward(self, x, targets):
        # Get the weight vectors from the last linear layer
        weights = self.model.linear.weight
        
        # Calculate L2 distances with proper scaling
        x_expanded = x.unsqueeze(1)
        w_expanded = weights.unsqueeze(0)
        
        # Scale the inputs and weights to prevent large distances
        x_norm = torch.nn.functional.normalize(x_expanded, p=2, dim=2)
        w_norm = torch.nn.functional.normalize(w_expanded, p=2, dim=2)
        
        # Calculate cosine distances (ranges from 0 to 2)
        distances = torch.norm(w_norm - x_norm, p=2, dim=2)
        
        log_probs = -self.n * torch.log(distances + self.epsilon)
        log_probs = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
        probs = torch.exp(log_probs)
        
        # Compute loss
        target_probs = torch.gather(probs, 1, targets.unsqueeze(1))
        loss = -torch.log(target_probs + self.epsilon).mean()
        
        return loss

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Training loop
def train(epochs):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            x, output = model(data)  # Get both flattened input and logits
            loss = criterion(x, target)  # Pass flattened input instead of logits
            loss.backward()
            optimizer.step()

        print(f'Train Epoch: {epoch + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing loop
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x, output = model(data)  # Get both flattened input and logits
            test_loss += criterion(x, target).item()
            
            # For accuracy, we still use the model's output
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MNIST with Harmonic Loss')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--n', type=int, default=28,
                        help='harmonic exponent n (default: 28)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='name for this training run (default: harmonic-loss-n={n}-epochs={epochs})')
    
    args = parser.parse_args()
    
    # Set default run name if none provided
    if args.run_name is None:
        args.run_name = f"harmonic-loss-n={args.n}-epochs={args.epochs}"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"{args.run_name}_{timestamp}"
    
    # Initialize model and criterion with specified n
    model = MLP().to(device)
    criterion = HarmonicLoss(model, input_dim=28*28, n=args.n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Use specified learning rate
    
    train(args.epochs)
    test()
    
    # Save the weights
    weights_dir = "saved_weights"
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{save_name}.pt")
    
    # Get the weights from the linear layer
    weights = model.linear.weight.data.cpu()
    torch.save(weights, weights_path)
    print(f"Weights saved to {weights_path}")
