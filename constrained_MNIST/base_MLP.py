import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional, Type, Dict
from abc import ABC, abstractmethod
from collections import OrderedDict

class BaseLayer(nn.Module, ABC):
    """Abstract base class for custom layer implementations"""
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class ActivationStore:
    """Stores activations from network layers"""
    def __init__(self):
        self.clear()
        
    def store_activation(self, name: str, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        """Hook function to store activations"""
        self.activations[name] = output.detach()
    
    def get_activation(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve stored activation by name"""
        return self.activations.get(name)
    
    def clear(self):
        """Clear all stored activations"""
        self.activations: Dict[str, torch.Tensor] = {}
    
    def keys(self) -> List[str]:
        """Get names of all stored activations"""
        return list(self.activations.keys())

class BaseMLP(nn.Module):
    """
    Base MLP architecture that can be extended for different constraints and layer types.
    
    Args:
        input_dim (int): Input dimension (784 for MNIST)
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Output dimension (10 for MNIST)
        layer_class (Type[Union[nn.Linear, BaseLayer]]): Layer class to use (default: nn.Linear)
        layer_kwargs (Optional[dict]): Additional keyword arguments for layer initialization
        activation (Optional[nn.Module]): Activation function class (default: None)
        dropout_prob (float): Dropout probability (default: 0.0)
        store_activations (bool): Whether to store activations during forward pass (default: False)
    """
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [128, 64],
        output_dim: int = 10,
        layer_class: Type[Union[nn.Linear, BaseLayer]] = nn.Linear,
        layer_kwargs: Optional[dict] = None,
        activation: Optional[nn.Module] = None,
        dropout_prob: float = 0.0,
        store_activations: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.layer_class = layer_class
        self.store_activations = store_activations
        
        # Build network architecture
        layers = []
        prev_dim = input_dim
        
        # Layer kwargs for initialization
        layer_kwargs = layer_kwargs or {}
        
        # Hidden layers
        for idx, hidden_dim in enumerate(hidden_dims):
            layer = layer_class(prev_dim, hidden_dim, **layer_kwargs)
            layers.append((f"linear_{idx}", layer))
            
            # Only add activation if one is specified
            if activation is not None:
                # Create a NEW activation instance for each layer to ensure unique hooks
                # Create a new instance of the same class
                activation_class = activation.__class__
                
                # Handle specific activation types properly
                if activation_class.__name__ == 'SparseMax':
                    # SparseMax doesn't require any init parameters
                    act = activation_class()
                elif activation_class.__name__ == 'PowerNormalization':
                    # Copy the specific parameters for PowerNormalization
                    if hasattr(activation, 'power'):
                        act = activation_class(power=activation.power)
                    else:
                        act = activation_class()
                elif activation_class.__name__ == 'RescaledReLU':
                    # Copy the scale parameter if it exists
                    if hasattr(activation, 'scale'):
                        act = activation_class(scale=activation.scale)
                    else:
                        act = activation_class()
                # Handle decaying activation types
                elif activation_class.__name__ in ['DecayingRescaledJumpReLU', 'DecayingRescaledReLU', 'DecayingFixedRescaledJumpReLU']:
                    kwargs = {}
                    # Common parameters for all decaying activations
                    for param in ['initial_scale', 'final_scale', 'decay_epochs', 'decay_schedule', 'decay_rate', 'eps', 'total_epochs']:
                        if hasattr(activation, param):
                            kwargs[param] = getattr(activation, param)
                    
                    # Specific parameters for different activation types
                    if activation_class.__name__ == 'DecayingRescaledJumpReLU':
                        for param in ['initial_threshold', 'constraint_min', 'constraint_max']:
                            if hasattr(activation, param):
                                kwargs[param] = getattr(activation, param)
                    elif activation_class.__name__ == 'DecayingFixedRescaledJumpReLU':
                        if hasattr(activation, 'threshold'):
                            kwargs['threshold'] = activation.threshold
                    
                    # Create the activation with proper parameters
                    act = activation_class(**kwargs)
                else:
                    # Default for standard activations
                    act = activation_class()
                
                # Add the activation to the network
                layers.append((f"activation_{idx}", act))
            
            if dropout_prob > 0:
                layers.append((f"dropout_{idx}", nn.Dropout(dropout_prob)))
            prev_dim = hidden_dim
        
        # Output layer
        output_layer = layer_class(prev_dim, output_dim, **layer_kwargs)
        layers.append((f"linear_{len(hidden_dims)}", output_layer))
        
        self.network = nn.Sequential(OrderedDict(layers))
        
        # Setup activation storage
        self.activation_store = ActivationStore()
        if store_activations:
            self.setup_activation_hooks()
    
    def setup_activation_hooks(self):
        """Setup hooks to store activations during forward pass"""
        from torch.nn.modules.activation import ReLU, LeakyReLU, Sigmoid, Tanh, ELU, GELU
        
        for name, module in self.named_modules():
            # Create hooks for linear layers and activations
            if isinstance(module, (nn.Linear, BaseLayer)):
                # Create a hook for linear/layer
                def make_hook(layer_name):
                    def hook(module, input, output):
                        self.activation_store.store_activation(f"{layer_name}_preact", module, input, output)
                    return hook
                
                # Register the hook
                module.register_forward_hook(make_hook(name))
            
            # For activation layers
            elif 'activation_' in name:
                # Create a hook for the activation
                def make_hook(layer_name):
                    def hook(module, input, output):
                        self.activation_store.store_activation(f"{layer_name}_postact", module, input, output)
                    return hook
                
                # Register the hook
                module.register_forward_hook(make_hook(name))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Clear previous activations if storing is enabled
        if self.store_activations:
            self.activation_store.clear()
            
        # Flatten input if necessary (for MNIST)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)
    
    def get_activation(self, name: str) -> Optional[torch.Tensor]:
        """Get stored activation by name"""
        return self.activation_store.get_activation(name)
    
    def get_all_activation_names(self) -> List[str]:
        """Get names of all stored activations"""
        return self.activation_store.keys()
    
    def enable_activation_storage(self):
        """Enable activation storage"""
        if not self.store_activations:
            self.store_activations = True
            self.setup_activation_hooks()
    
    def disable_activation_storage(self):
        """Disable activation storage"""
        self.store_activations = False
        self.activation_store.clear()
    
    def get_layer_weights(self, layer_idx: int) -> torch.Tensor:
        """Get weights of a specific layer"""
        return self._get_layer(layer_idx).weight
    
    def get_layer_biases(self, layer_idx: int) -> torch.Tensor:
        """Get biases of a specific layer"""
        return self._get_layer(layer_idx).bias
    
    def _get_layer(self, layer_idx: int) -> nn.Module:
        """Helper method to get a specific layer"""
        layer_count = 0
        for module in self.network:
            if isinstance(module, (nn.Linear, BaseLayer)):
                if layer_count == layer_idx:
                    return module
                layer_count += 1
        raise IndexError(f"Layer index {layer_idx} out of range")
    
    @property
    def num_layers(self) -> int:
        """Get number of layers (excluding activation and dropout)"""
        return sum(1 for module in self.network if isinstance(module, (nn.Linear, BaseLayer)))
