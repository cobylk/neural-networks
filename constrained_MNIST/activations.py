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
    def __init__(self, initial_threshold=0.5, constraint_min=0.0, constraint_max=100.0, 
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
            mask = (x >= self.threshold).float()
        
        # Apply the mask to scale values according to threshold
        return x * mask
    
    def get_threshold(self):
        """Return the current value of the threshold parameter"""
        with torch.no_grad():
            return torch.clamp(self.threshold, self.constraint_min, self.constraint_max).item()
            
    def extra_repr(self):
        """Return a string with extra information"""
        return f'initial_threshold={self.initial_value}, current_threshold={self.get_threshold():.4f}, sharpness={self.sharpness}'

class FixedThreshold(nn.Module):
    """Fixed threshold activation function"""
    def __init__(self):
        super().__init__()
        # self.threshold = threshold
        
    def forward(self, x):
        thresholded_activations = torch.maximum(x, torch.tensor(10.0))
        sum_thresholded_activations = torch.sum(thresholded_activations)
        return (thresholded_activations / sum_thresholded_activations) * 100
    
class JumpReLU(nn.Module):
    """
    JumpReLU activation function with a learnable threshold parameter.
    
    This activation returns 0 for all inputs less than the threshold (κ),
    and returns the original input value for inputs greater than or equal to κ.
    
    Unlike ThresholdActivation which applies the threshold to the y-values (output),
    JumpReLU applies the threshold to the x-values (input).
    
    This implementation uses a straight-through estimator for the gradient
    to enable learning the threshold parameter.
    
    Args:
        initial_threshold (float): Initial value for the threshold parameter (default: 0.0)
        constraint_min (float): Minimum allowed value for threshold (default: -5.0)
        constraint_max (float): Maximum allowed value for threshold (default: 5.0)
    """
    def __init__(self, initial_threshold=0.0, constraint_min=-5.0, constraint_max=5.0):
        super().__init__()
        # Initialize the threshold as a learnable parameter
        self.threshold = nn.Parameter(torch.tensor(float(initial_threshold)))
        self.constraint_min = constraint_min
        self.constraint_max = constraint_max
        
        # Store the initial value for reporting
        self.initial_value = initial_threshold
    
    def forward(self, x):
        # Constrain the threshold to be within the specified range
        constrained_threshold = torch.clamp(self.threshold, self.constraint_min, self.constraint_max)
        
        # Create a binary mask using exact comparison: 1 where x >= threshold, 0 elsewhere
        mask = (x >= constrained_threshold).float()
        
        # Apply the mask to implement the jump behavior
        # For backward pass, we use a straight-through estimator
        # This means we use the exact function in forward pass,
        # but define a custom gradient for backward pass
        return STEFunction.apply(x, mask, constrained_threshold)
    
    def get_threshold(self):
        """Return the current value of the threshold parameter"""
        with torch.no_grad():
            return torch.clamp(self.threshold, self.constraint_min, self.constraint_max).item()
            
    def extra_repr(self):
        """Return a string with extra information"""
        return f'initial_threshold={self.initial_value}, current_threshold={self.get_threshold():.4f}'

class STEFunction(torch.autograd.Function):
    """
    Straight-through estimator for JumpReLU.
    Implements a custom gradient for the threshold parameter.
    """
    @staticmethod
    def forward(ctx, x, mask, threshold):
        ctx.save_for_backward(x, mask, threshold)
        return x * mask
    
    @staticmethod
    def backward(ctx, grad_output):
        x, mask, threshold = ctx.saved_tensors
        
        # Gradient w.r.t. input x: only pass gradients where mask is 1
        grad_x = grad_output * mask
        
        # Gradient w.r.t. threshold: 
        # For values near the threshold, we approximate the derivative
        # This allows the threshold to learn
        near_threshold = torch.abs(x - threshold) < 0.1
        grad_threshold = -torch.sum(grad_output * x * near_threshold)
        
        # No gradient for mask since it's a binary value
        return grad_x, None, grad_threshold
    
class RescaledJumpReLU(nn.Module):
    """
    RescaledJumpReLU activation function combines JumpReLU's threshold with RescaledReLU's normalization.
    
    This activation:
    1. Returns 0 for all inputs less than the threshold (κ)
    2. Returns the original input for inputs greater than or equal to κ
    3. Normalizes the outputs to sum to 1 across each feature dimension
    
    The threshold parameter is learnable and constrained to positive values only.
    
    Args:
        initial_threshold (float): Initial value for the threshold parameter (default: 0.1)
        constraint_min (float): Minimum allowed value for threshold (default: 0.0)
        constraint_max (float): Maximum allowed value for threshold (default: 5.0)
        eps (float): Small epsilon for numerical stability in normalization (default: 1e-10)
    """
    def __init__(self, initial_threshold=0.1, constraint_min=0.0, constraint_max=5.0, eps=1e-10):
        super().__init__()
        # Initialize the threshold as a learnable parameter, ensure positive initial value
        self.threshold = nn.Parameter(torch.tensor(float(max(initial_threshold, 0.0))))
        self.constraint_min = constraint_min  # Min must be non-negative
        self.constraint_max = constraint_max
        self.eps = eps
        
        # Store the initial value for reporting
        self.initial_value = initial_threshold
    
    def forward(self, x):
        # Constrain the threshold to be within the specified range, ensuring it's positive
        constrained_threshold = torch.clamp(self.threshold, self.constraint_min, self.constraint_max)
        
        # Create a binary mask using exact comparison: 1 where x >= threshold, 0 elsewhere
        mask = (x >= constrained_threshold).float()
        
        # Apply the mask to implement the jump behavior with a straight-through estimator
        jumped = STEFunction.apply(x, mask, constrained_threshold)
        
        # Add small epsilon to avoid division by zero
        jumped_plus_eps = jumped + self.eps
        
        # Normalize each sample's activations to sum to 1
        normalized = jumped_plus_eps / jumped_plus_eps.sum(dim=1, keepdim=True)
        
        return normalized
    
    def get_threshold(self):
        """Return the current value of the threshold parameter"""
        with torch.no_grad():
            return torch.clamp(self.threshold, self.constraint_min, self.constraint_max).item()
            
    def extra_repr(self):
        """Return a string with extra information"""
        return f'initial_threshold={self.initial_value}, current_threshold={self.get_threshold():.4f}, eps={self.eps}'
    
class FixedRescaledJumpReLU(nn.Module):
    """
    FixedRescaledJumpReLU activation function with non-trainable threshold.
    
    This activation:
    1. Returns 0 for all inputs less than the threshold (κ)
    2. Returns the original input for inputs greater than or equal to κ
    3. Normalizes the outputs to sum to 1 across each feature dimension
    
    Unlike RescaledJumpReLU, the threshold is fixed and not learnable.
    
    Args:
        threshold (float): Fixed threshold value (default: 0.1)
        eps (float): Small epsilon for numerical stability in normalization (default: 1e-10)
    """
    def __init__(self, threshold=0.1, eps=1e-10):
        super().__init__()
        self.threshold = threshold
        self.eps = eps
    
    def forward(self, x):
        # Create a binary mask using exact comparison: 1 where x >= threshold, 0 elsewhere
        mask = (x >= self.threshold).float()
        
        # Apply the mask to implement the jump behavior
        jumped = x * mask
        
        # Add small epsilon to avoid division by zero
        jumped_plus_eps = jumped + self.eps
        
        # Normalize each sample's activations to sum to 1
        normalized = jumped_plus_eps / jumped_plus_eps.sum(dim=1, keepdim=True)
        
        return normalized
    
    def get_threshold(self):
        """Return the threshold value (for compatibility with other threshold functions)"""
        return self.threshold
            
    def extra_repr(self):
        """Return a string with extra information"""
        return f'threshold={self.threshold}, eps={self.eps}'
    
class HectoRescaledReLU(RescaledReLU):
    """
    HectoRescaledReLU activation function that normalizes to a simplex summing to 100.
    
    This is identical to RescaledReLU but multiplies the final output by 100,
    so activations sum to 100 instead of 1.
    
    Args:
        eps (float): Small epsilon for numerical stability (default: 1e-10)
    """
    def __init__(self, eps=1e-10):
        super().__init__(eps=eps)
    
    def forward(self, x):
        # Get normalized output from parent class
        normalized = super().forward(x)
        # Multiply by 100 to scale to a simplex summing to 100
        return normalized * 100

class HectoRescaledJumpReLU(RescaledJumpReLU):
    """
    HectoRescaledJumpReLU activation function that normalizes to a simplex summing to 100.
    
    This is identical to RescaledJumpReLU but multiplies the final output by 100,
    so activations sum to 100 instead of 1.
    
    Args:
        initial_threshold (float): Initial value for the threshold (default: 0.1)
        constraint_min (float): Minimum allowed value for the threshold (default: 0.0)
        constraint_max (float): Maximum allowed value for the threshold (default: 5.0)
        eps (float): Small epsilon for numerical stability (default: 1e-10)
    """
    def __init__(self, initial_threshold=0.1, constraint_min=0.0, constraint_max=5.0, eps=1e-10):
        super().__init__(
            initial_threshold=initial_threshold,
            constraint_min=constraint_min,
            constraint_max=constraint_max,
            eps=eps
        )
    
    def forward(self, x):
        # Get normalized output from parent class
        normalized = super().forward(x)
        # Multiply by 100 to scale to a simplex summing to 100
        return normalized * 100

class HectoFixedRescaledJumpReLU(FixedRescaledJumpReLU):
    """
    HectoFixedRescaledJumpReLU activation function that normalizes to a simplex summing to 100.
    
    This is identical to FixedRescaledJumpReLU but multiplies the final output by 100,
    so activations sum to 100 instead of 1.
    
    Args:
        threshold (float): Fixed threshold value (default: 0.1)
        eps (float): Small epsilon for numerical stability (default: 1e-10)
    """
    def __init__(self, threshold=0.1, eps=1e-10):
        super().__init__(threshold=threshold, eps=eps)
    
    def forward(self, x):
        # Get normalized output from parent class
        normalized = super().forward(x)
        # Multiply by 100 to scale to a simplex summing to 100
        return normalized * 100
    