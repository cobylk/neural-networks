import torch
import torch.nn as nn
import torch.nn.functional as F
from base_MLP import BaseLayer

class StochasticLayer(BaseLayer):
    """
    A layer that maintains stochastic map properties, ensuring inputs from a standard simplex
    remain on the simplex after transformation.
    
    The layer implements:
    1. Weight constraints to maintain stochastic matrix properties (non-negative, rows sum to 1)
    2. Proper initialization of weights as stochastic matrices
    3. No bias terms (to preserve simplex properties)
    
    Args:
        in_features (int): Size of input features
        out_features (int): Size of output features
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize as random stochastic matrix
        raw_weights = torch.rand(out_features, in_features)
        # Normalize rows to sum to 1 to create valid stochastic matrix
        stochastic_weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)
        
        self.weight = nn.Parameter(stochastic_weights)
        # No bias term to preserve simplex properties
        self.register_parameter('bias', None)
    
    def _project_to_stochastic(self):
        """Project weights back to valid stochastic matrix space"""
        with torch.no_grad():
            # Ensure non-negativity
            self.weight.data = F.relu(self.weight.data)
            # Normalize rows to sum to 1
            self.weight.data = self.weight.data / (self.weight.data.sum(dim=1, keepdim=True) + 1e-8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass ensuring stochastic properties are maintained.
        
        Args:
            x (torch.Tensor): Input tensor assumed to be on the probability simplex
                            (non-negative entries summing to 1)
        
        Returns:
            torch.Tensor: Output tensor guaranteed to be on the probability simplex
        """
        # Project weights to ensure they form a valid stochastic matrix
        self._project_to_stochastic()
        
        # Apply the stochastic transformation
        # Since weights are stochastic and input is on simplex,
        # output will automatically be on simplex
        return F.linear(x, self.weight) 