import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    
class EpochAwareMixin:
    """
    Mixin that allows modules to be aware of the current training epoch.
    
    This mixin can be added to any activation function to make it epoch-aware,
    enabling functionality that changes over the course of training.
    
    Args:
        total_epochs (int): Total number of epochs for the full training cycle
        decay_epochs (int, optional): Number of epochs over which to decay. After this, 
                                     the value remains at final_value. If None, decays over total_epochs.
        decay_schedule (str, optional): Type of decay schedule to use ('linear', 'exponential', 'cosine')
        decay_rate (float, optional): Rate parameter for exponential decay (only used if decay_schedule='exponential')
    """
    
    def __init__(self, total_epochs=100, decay_epochs=10, decay_schedule='exponential', decay_rate=0.4, **kwargs):
        # Note: We don't call super().__init__() here as it's a mixin
        # The actual parent class init will be called by the concrete class
        self.current_epoch = 0
        self.total_epochs = total_epochs
        self.decay_epochs = total_epochs if decay_epochs is None else decay_epochs
        self.decay_schedule = decay_schedule
        self.decay_rate = decay_rate
        
    def set_epoch(self, epoch):
        """
        Update the current epoch.
        
        Args:
            epoch (int): Current training epoch
        
        Returns:
            self: For method chaining
        """
        self.current_epoch = min(epoch, self.total_epochs)
        return self
        
    def get_training_progress(self):
        """
        Return training progress as a value between 0 and 1.
        
        Returns:
            float: Training progress (0.0 at start, 1.0 at or after decay_epochs)
        """
        return min(1.0, max(0.0, self.current_epoch / self.decay_epochs))
    
    def get_decay_factor(self, initial_value=1.0, final_value=0.0):
        """
        Calculate a decay factor based on the current training progress.
        
        This method implements various decay schedules between initial_value and final_value:
        - linear: straight line interpolation
        - exponential: exponential decay using decay_rate
        - cosine: cosine annealing schedule
        
        The decay occurs over decay_epochs rather than total_epochs, allowing for
        a phase of decay followed by a stable phase.
        
        Args:
            initial_value (float): Starting value at epoch 0
            final_value (float): Final value at or after decay_epochs
            
        Returns:
            float: Current value based on chosen decay schedule
        """
        # If we've passed the decay epochs, return the final value
        if self.current_epoch >= self.decay_epochs:
            return final_value
            
        progress = self.get_training_progress()
        
        if self.decay_schedule == 'linear':
            # Linear interpolation
            return initial_value * (1 - progress) + final_value * progress
            
        elif self.decay_schedule == 'exponential':
            # Exponential decay
            decay = self.decay_rate ** (self.current_epoch)
            normalized_decay = (decay - self.decay_rate ** self.decay_epochs) / (1 - self.decay_rate ** self.decay_epochs)
            return initial_value * normalized_decay + final_value * (1 - normalized_decay)
            
        elif self.decay_schedule == 'cosine':
            # Cosine annealing
            return final_value + 0.5 * (initial_value - final_value) * (1 + math.cos(math.pi * progress))
            
        else:
            # Default to linear
            return initial_value * (1 - progress) + final_value * progress
    
class DecayingRescaledReLU(EpochAwareMixin, RescaledReLU):
    """
    RescaledReLU with a scale factor that decays over training.
    
    This activation function applies RescaledReLU, then multiplies by a factor
    that decays from initial_scale to final_scale during training according
    to the selected decay schedule.
    
    Args:
        initial_scale (float): Initial scale factor at the beginning of training (default: 1000)
        final_scale (float): Final scale factor at the end of decay_epochs (default: 1)
        decay_epochs (int, optional): Number of epochs over which to decay. If None, decays over total_epochs.
        decay_schedule (str): Type of decay ('linear', 'exponential', 'cosine') (default: 'linear')
        decay_rate (float): Rate parameter for exponential decay (default: 0.95)
        eps (float): Small epsilon for numerical stability in normalization (default: 1e-10)
    """
    def __init__(self, initial_scale=100, final_scale=1, decay_epochs=10, 
                 decay_schedule='exponential', decay_rate=0.4, eps=1e-10, total_epochs=100):
        # Initialize both parent classes
        EpochAwareMixin.__init__(self, total_epochs=total_epochs, decay_epochs=decay_epochs,
                                 decay_schedule=decay_schedule, decay_rate=decay_rate)
        RescaledReLU.__init__(self, eps=eps)
        
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.forward_count = 0
    
    def forward(self, x):
        # Get normalized output from RescaledReLU
        normalized = RescaledReLU.forward(self, x)
        
        # Apply decaying scale factor
        current_scale = self.get_decay_factor(self.initial_scale, self.final_scale)
        
        # Debug print randomly (approx. every 100 forward passes)
        self.forward_count += 1
        if self.forward_count % 100 == 0 and torch.rand(1).item() < 0.01:
            print(f"[DEBUG] DecayingRescaledReLU: epoch={self.current_epoch}/{self.decay_epochs}, "
                  f"scale={current_scale:.4f}, initial={self.initial_scale}, final={self.final_scale}")
        
        return normalized * current_scale
    
    def extra_repr(self):
        """Return a string with extra information"""
        current_scale = self.get_decay_factor(self.initial_scale, self.final_scale)
        return (f'initial_scale={self.initial_scale}, final_scale={self.final_scale}, '
                f'current_scale={current_scale:.2f}, decay_schedule={self.decay_schedule}, '
                f'current_epoch={self.current_epoch}/{self.decay_epochs}')


class DecayingRescaledJumpReLU(EpochAwareMixin, RescaledJumpReLU):
    """
    RescaledJumpReLU with a scale factor that decays over training.
    
    This activation function applies RescaledJumpReLU, then multiplies by a factor
    that decays from initial_scale to final_scale during training according
    to the selected decay schedule.
    
    Args:
        initial_scale (float): Initial scale factor at the beginning of training (default: 1000)
        final_scale (float): Final scale factor at the end of decay_epochs (default: 1)
        initial_threshold (float): Initial value for the threshold (default: 0.1)
        constraint_min (float): Minimum allowed value for the threshold (default: 0.0)
        constraint_max (float): Maximum allowed value for the threshold (default: 5.0)
        decay_epochs (int, optional): Number of epochs over which to decay. If None, decays over total_epochs.
        decay_schedule (str): Type of decay ('linear', 'exponential', 'cosine') (default: 'linear')
        decay_rate (float): Rate parameter for exponential decay (default: 0.95)
        eps (float): Small epsilon for numerical stability in normalization (default: 1e-10)
    """
    def __init__(self, initial_scale=100, final_scale=1, initial_threshold=0.1, 
                 constraint_min=0.0, constraint_max=5.0, decay_epochs=10,
                 decay_schedule='exponential', decay_rate=0.4, eps=1e-10, total_epochs=100):
        # Initialize both parent classes
        EpochAwareMixin.__init__(self, total_epochs=total_epochs, decay_epochs=decay_epochs,
                                 decay_schedule=decay_schedule, decay_rate=decay_rate)
        RescaledJumpReLU.__init__(self, initial_threshold=initial_threshold,
                                constraint_min=constraint_min, constraint_max=constraint_max, 
                                eps=eps)
        
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.forward_count = 0
    
    def forward(self, x):
        # Get normalized output from RescaledJumpReLU
        normalized = RescaledJumpReLU.forward(self, x)
        
        # Apply decaying scale factor
        current_scale = self.get_decay_factor(self.initial_scale, self.final_scale)
        
        # Debug print randomly (approx. every 100 forward passes)
        self.forward_count += 1
        if self.forward_count % 100 == 0 and torch.rand(1).item() < 0.01:
            print(f"[DEBUG] DecayingRescaledJumpReLU: epoch={self.current_epoch}/{self.decay_epochs}, "
                  f"scale={current_scale:.4f}, threshold={self.get_threshold():.4f}, "
                  f"initial={self.initial_scale}, final={self.final_scale}")
        
        return normalized * current_scale
    
    def extra_repr(self):
        """Return a string with extra information"""
        current_scale = self.get_decay_factor(self.initial_scale, self.final_scale)
        threshold_info = RescaledJumpReLU.extra_repr(self)
        return (f'{threshold_info}, initial_scale={self.initial_scale}, final_scale={self.final_scale}, '
                f'current_scale={current_scale:.2f}, decay_schedule={self.decay_schedule}, '
                f'current_epoch={self.current_epoch}/{self.decay_epochs}')


class DecayingFixedRescaledJumpReLU(EpochAwareMixin, FixedRescaledJumpReLU):
    """
    FixedRescaledJumpReLU with a scale factor that decays over training.
    
    This activation function applies FixedRescaledJumpReLU, then multiplies by a factor
    that decays from initial_scale to final_scale during training according
    to the selected decay schedule.
    
    Args:
        initial_scale (float): Initial scale factor at the beginning of training (default: 1000)
        final_scale (float): Final scale factor at the end of decay_epochs (default: 1)
        threshold (float): Fixed threshold value (default: 0.1)
        decay_epochs (int, optional): Number of epochs over which to decay. If None, decays over total_epochs.
        decay_schedule (str): Type of decay ('linear', 'exponential', 'cosine') (default: 'linear')
        decay_rate (float): Rate parameter for exponential decay (default: 0.95)
        eps (float): Small epsilon for numerical stability in normalization (default: 1e-10)
    """
    def __init__(self, initial_scale=100, final_scale=1, threshold=0.1, 
                 decay_epochs=10, decay_schedule='exponential', decay_rate=0.4, 
                 eps=1e-10, total_epochs=100):
        # Initialize both parent classes
        EpochAwareMixin.__init__(self, total_epochs=total_epochs, decay_epochs=decay_epochs,
                                 decay_schedule=decay_schedule, decay_rate=decay_rate)
        FixedRescaledJumpReLU.__init__(self, threshold=threshold, eps=eps)
        
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.forward_count = 0
    
    def forward(self, x):
        # Get normalized output from FixedRescaledJumpReLU
        normalized = FixedRescaledJumpReLU.forward(self, x)
        
        # Apply decaying scale factor
        current_scale = self.get_decay_factor(self.initial_scale, self.final_scale)
        
        # Debug print randomly (approx. every 100 forward passes)
        self.forward_count += 1
        if self.forward_count % 100 == 0 and torch.rand(1).item() < 0.01:
            print(f"[DEBUG] DecayingFixedRescaledJumpReLU: epoch={self.current_epoch}/{self.decay_epochs}, "
                  f"scale={current_scale:.4f}, threshold={self.threshold:.4f}, "
                  f"initial={self.initial_scale}, final={self.final_scale}")
        
        return normalized * current_scale
    
    def extra_repr(self):
        """Return a string with extra information"""
        current_scale = self.get_decay_factor(self.initial_scale, self.final_scale)
        threshold_info = FixedRescaledJumpReLU.extra_repr(self)
        return (f'{threshold_info}, initial_scale={self.initial_scale}, final_scale={self.final_scale}, '
                f'current_scale={current_scale:.2f}, decay_schedule={self.decay_schedule}, '
                f'current_epoch={self.current_epoch}/{self.decay_epochs}')
    