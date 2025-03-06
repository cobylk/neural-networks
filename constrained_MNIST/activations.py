import torch
import torch.nn as nn
import torch.nn.functional as F

class RescaledReLU(nn.Module):
    """ReLU activation where outputs are rescaled to sum to 1"""
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps  # Small constant for numerical stability
        
    def forward(self, x):
        # Apply ReLU
        x = F.relu(x)
        # Add small epsilon to avoid division by zero
        x = x + self.eps
        # Normalize each sample's activations to sum to 1
        return x / x.sum(dim=1, keepdim=True)

class LayerwiseSoftmax(nn.Module):
    """Applies softmax to each layer's outputs"""
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x):
        return F.softmax(x / self.temperature, dim=1)

class PowerNormalization(nn.Module):
    """Applies power transformation while maintaining simplex constraints"""
    def __init__(self, alpha=0.5, eps=1e-6):
        super().__init__()
        self.alpha = alpha  # Power parameter (0 < alpha <= 1)
        self.eps = eps
        
    def forward(self, x):
        # Ensure non-negativity and add small epsilon
        x = F.relu(x) + self.eps
        # Apply power transformation with clipping for stability
        x = torch.clamp(x, min=self.eps, max=1e6)
        x = torch.pow(x, self.alpha)
        # Renormalize to maintain simplex constraint
        return x / (x.sum(dim=1, keepdim=True) + self.eps)

class TemperaturePowerNorm(nn.Module):
    """Power normalization with temperature scaling"""
    def __init__(self, alpha=0.5, temperature=1.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.eps = eps
        
    def forward(self, x):
        # Scale by temperature and ensure non-negativity
        x = F.relu(x / self.temperature) + self.eps
        # Apply power transformation with clipping for stability
        x = torch.clamp(x, min=self.eps, max=1e6)
        x = torch.pow(x, self.alpha)
        # Renormalize to maintain simplex constraint
        return x / (x.sum(dim=1, keepdim=True) + self.eps)

class SparseMax(nn.Module):
    """Sparse alternative to softmax that projects onto simplex"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        sorted_x, _ = torch.sort(x, dim=1, descending=True)
        
        cum_sums = torch.cumsum(sorted_x, dim=1)
        
        k = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
        k = k.view(1, -1)
        
        threshold_values = (cum_sums - 1) / k
        
        rho = torch.sum(sorted_x > threshold_values, dim=1)
        
        threshold = threshold_values[torch.arange(x.shape[0]), (rho - 1).clamp(min=0)]
        
        return torch.maximum(x - threshold.unsqueeze(1), torch.zeros_like(x))

class ThresholdActivation(nn.Module):
    """
    Threshold activation function with a learnable threshold parameter.
    
    This implementation uses a differentiable approximation of the threshold
    operation to ensure gradient flow to the threshold parameter during training.
    
    Args:
        initial_threshold (float): Initial value for the threshold parameter
        constraint_min (float): Minimum allowed value for threshold (default: 0.0)
        constraint_max (float): Maximum allowed value for threshold (default: 1.0)
        sharpness (float): Controls how sharp the threshold transition is (default: 10.0)
                           Higher values make the transition more step-like
        hard_forward (bool): Whether to use the exact (non-differentiable) threshold
                            during forward pass for inference (default: True)
    """
    def __init__(self, initial_threshold=0.5, constraint_min=0.0, constraint_max=1.0, 
                 sharpness=10.0, hard_forward=True):
        super().__init__()
        # Initialize the threshold as a learnable parameter
        self.threshold = nn.Parameter(torch.tensor(float(initial_threshold)))
        self.constraint_min = constraint_min
        self.constraint_max = constraint_max
        self.sharpness = sharpness
        self.hard_forward = hard_forward
        
        # Store the initial value for debugging
        self.initial_value = initial_threshold
    
    def forward(self, x):
        # Constrain the threshold to be within the specified range
        constrained_threshold = torch.clamp(self.threshold, self.constraint_min, self.constraint_max)
        
        if self.training or not self.hard_forward:
            # Differentiable approximation of the threshold function for training
            # Using sigmoid with a sharpness parameter to control the steepness of the transition
            # For high sharpness values, this approaches a step function
            mask = torch.sigmoid(self.sharpness * (x - constrained_threshold))
        else:
            # Hard thresholding for inference (non-differentiable but exact)
            mask = (x >= constrained_threshold).float()
        
        # Apply the mask to scale values according to threshold
        return x * mask
    
    def get_threshold(self):
        """Return the current value of the threshold parameter"""
        with torch.no_grad():
            return torch.clamp(self.threshold, self.constraint_min, self.constraint_max).item()
            
    def extra_repr(self):
        """Return a string with extra information"""
        return f'initial_threshold={self.initial_value}, current_threshold={self.get_threshold():.4f}, sharpness={self.sharpness}'

def test_threshold_activation():
    """Test function to demonstrate ThresholdActivation behavior and learning"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create sample input
    x = torch.linspace(-2, 2, 100)
    
    # Create activations with different thresholds and sharpness values
    thresh_0 = ThresholdActivation(initial_threshold=0.0, sharpness=10.0)  # Similar to ReLU
    thresh_05 = ThresholdActivation(initial_threshold=0.5, sharpness=10.0)
    thresh_1 = ThresholdActivation(initial_threshold=1.0, sharpness=10.0)
    
    # Compare different sharpness values
    thresh_low_sharp = ThresholdActivation(initial_threshold=0.5, sharpness=3.0)
    thresh_med_sharp = ThresholdActivation(initial_threshold=0.5, sharpness=10.0)
    thresh_high_sharp = ThresholdActivation(initial_threshold=0.5, sharpness=30.0)
    
    # Apply activations
    y_0 = thresh_0(x)
    y_05 = thresh_05(x)
    y_1 = thresh_1(x)
    
    # Plot different thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), x.numpy(), 'k--', label='Identity')
    plt.plot(x.numpy(), y_0.detach().numpy(), label=f'Threshold={thresh_0.get_threshold():.2f} (ReLU-like)')
    plt.plot(x.numpy(), y_05.detach().numpy(), label=f'Threshold={thresh_05.get_threshold():.2f}')
    plt.plot(x.numpy(), y_1.detach().numpy(), label=f'Threshold={thresh_1.get_threshold():.2f}')
    
    plt.grid(True)
    plt.legend()
    plt.title('ThresholdActivation with Different Threshold Values')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    
    # Highlight thresholds
    plt.axvline(x=thresh_0.get_threshold(), color='blue', linestyle=':', alpha=0.5)
    plt.axvline(x=thresh_05.get_threshold(), color='orange', linestyle=':', alpha=0.5)
    plt.axvline(x=thresh_1.get_threshold(), color='green', linestyle=':', alpha=0.5)
    
    plt.savefig('threshold_activation_demo.png')
    
    # Plot different sharpness values
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), x.numpy(), 'k--', label='Identity')
    plt.plot(x.numpy(), thresh_low_sharp(x).detach().numpy(), 
             label=f'Low Sharpness ({thresh_low_sharp.sharpness})')
    plt.plot(x.numpy(), thresh_med_sharp(x).detach().numpy(), 
             label=f'Medium Sharpness ({thresh_med_sharp.sharpness})')
    plt.plot(x.numpy(), thresh_high_sharp(x).detach().numpy(), 
             label=f'High Sharpness ({thresh_high_sharp.sharpness})')
    
    # Also show true hard threshold for comparison
    hard_mask = (x >= 0.5).float()
    plt.plot(x.numpy(), (x * hard_mask).numpy(), 'r--', label='Hard Threshold (non-differentiable)')
    
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Sharpness Parameter on Threshold Approximation')
    plt.xlabel('Input')
    plt.ylabel('Output')
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=0.5, color='r', linestyle='-', alpha=0.3)
    
    plt.savefig('threshold_sharpness_demo.png')
    
    print("ThresholdActivation behavior:")
    print(f"  - Values below threshold are gradually attenuated (based on sharpness)")
    print(f"  - Values above threshold approach their original value")
    print(f"  - The threshold is a learnable parameter that starts at the initial value")
    print(f"  - Higher sharpness values make the function more like a hard threshold")
    
    # Demonstrate learning the threshold
    print("\nDemonstrating how the threshold can be learned:")
    # Create a dummy target that rewards a threshold around 0.7
    target_function = lambda x: (x >= 0.7).float() * x
    target_output = target_function(x)
    
    # Create a learnable threshold and optimizer
    learned_threshold = ThresholdActivation(initial_threshold=0.5, sharpness=20.0)
    optimizer = torch.optim.SGD([learned_threshold.threshold], lr=0.1)
    
    # Training loop
    thresholds = []
    losses = []
    steps = 30
    
    for i in range(steps):
        # Forward pass
        output = learned_threshold(x)
        loss = torch.nn.functional.mse_loss(output, target_output)
        
        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track progress
        current_threshold = learned_threshold.get_threshold()
        thresholds.append(current_threshold)
        losses.append(loss.item())
        print(f"  Step {i+1}/{steps}: Threshold = {current_threshold:.4f}, Loss = {loss.item():.6f}")
    
    # Plot threshold learning progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, steps+1), thresholds, marker='o')
    plt.axhline(y=0.7, color='r', linestyle='--', label='Target threshold')
    plt.xlabel('Training Step')
    plt.ylabel('Threshold Value')
    plt.title('Threshold Learning Progress')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, steps+1), losses, marker='o')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('threshold_learning_demo.png')
    
    # Plot the final learned function
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), x.numpy(), 'k--', label='Identity')
    plt.plot(x.numpy(), target_output.numpy(), 'r-', label='Target Function (threshold=0.7)')
    plt.plot(x.numpy(), learned_threshold(x).detach().numpy(), 'g-', 
             label=f'Learned Function (threshold={learned_threshold.get_threshold():.4f})')
    
    plt.grid(True)
    plt.legend()
    plt.title('Learned ThresholdActivation Function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    
    # Add vertical lines at the thresholds
    plt.axvline(x=0.7, color='r', linestyle=':', alpha=0.5, label='Target threshold')
    plt.axvline(x=learned_threshold.get_threshold(), color='g', linestyle=':', alpha=0.5, label='Learned threshold')
    
    plt.savefig('learned_threshold_function.png')
    plt.show()

if __name__ == "__main__":
    test_threshold_activation()

