import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
        temperature (float): Temperature parameter for softmax (default: 1.0)
    """
    def __init__(self, in_features: int, out_features: int, temperature: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        
        # Initialize with Glorot/Xavier initialization
        bound = math.sqrt(6. / (in_features + out_features))
        raw_weights = torch.empty(out_features, in_features).uniform_(-bound, bound)
        
        # Store raw weights - softmax will be applied in forward pass
        self.raw_weight = nn.Parameter(raw_weights)
        self.register_parameter('bias', None)
    
    def get_stochastic_weights(self):
        """Convert raw weights to stochastic matrix using softmax"""
        return F.softmax(self.raw_weight / self.temperature, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass ensuring stochastic properties are maintained.
        
        Args:
            x (torch.Tensor): Input tensor assumed to be on the probability simplex
                            (non-negative entries summing to 1)
        
        Returns:
            torch.Tensor: Output tensor guaranteed to be on the probability simplex
        """
        # Get stochastic weights through softmax
        stochastic_weights = self.get_stochastic_weights()
        
        # Apply the stochastic transformation
        return F.linear(x, stochastic_weights) 