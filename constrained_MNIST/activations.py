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
        # Sort input in descending order
        sorted_x, _ = torch.sort(x, dim=1, descending=True)
        
        # Calculate cumulative sums
        cum_sums = torch.cumsum(sorted_x, dim=1)
        
        # Calculate indices for thresholding
        k = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
        k = k.view(1, -1)
        
        # Calculate threshold values
        threshold_values = (cum_sums - 1) / k
        
        # Find the last position where threshold_values > sorted_x
        rho = torch.sum(sorted_x > threshold_values, dim=1)
        
        # Calculate final threshold
        threshold = threshold_values[torch.arange(x.shape[0]), (rho - 1).clamp(min=0)]
        
        # Apply projection
        return torch.maximum(x - threshold.unsqueeze(1), torch.zeros_like(x))

