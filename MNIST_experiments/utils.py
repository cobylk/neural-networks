import torch
import datetime

class DistLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, n=1., eps=1e-4, bias=False):
        super(DistLayer, self).__init__(in_features, out_features, bias=bias)
        self.n = n
        self.eps = eps
        
    def forward(self, x, scale=False):
        # x: (B, N)
        # w: (V, N)
        # dist_sq: (B, V)
        n_embd = x.size(-1,)
        w = self.weight
        wx = torch.einsum('bn,vn->bv', x, w) # (B, V)
        ww = torch.norm(w, dim=-1)**2 # (V,)
        xx = torch.norm(x, dim=-1)**2 # (B,)

        dist_sq = ww[None,:] + xx[:,None] - 2 * wx + self.eps
        dist_sq = dist_sq / torch.min(dist_sq, dim=-1, keepdim = True)[0]
        return (dist_sq)**(-self.n)

def generate_run_name(args, timestamp=None, subfolder=None):
    """Generate consistent run names for files and directories.
    
    Args:
        args: The argument parser namespace containing run parameters
        timestamp: Optional timestamp string. If None, generates new timestamp.
        subfolder: Optional subfolder path to prepend to the full name.
    
    Returns:
        base_name: The base name without timestamp
        full_name: The complete name with timestamp
        timestamp: The timestamp used
        subfolder_path: The complete subfolder path if provided, otherwise empty string
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        
    # Start with model type identifier
    prefix = "softmax" if hasattr(args, 'temperature') else ""
    
    # Build the base name
    components = [
        f"{prefix}",
        f"{'harmonic-loss' if args.harmonic else 'cross-entropy-loss'}",
        f"n={args.n}" if args.harmonic else None,
        f"epochs={args.epochs}",
        f"temp={args.temperature}" if hasattr(args, 'temperature') else None,
        f"hidden-dim={args.hidden_dim}" if hasattr(args, 'temperature') else None,
        f"ReLU={args.ReLU}" if hasattr(args, 'ReLU') else None,
        f"softmax={args.softmax}" if hasattr(args, 'softmax') else None
    ]
    
    # Filter out empty strings and join
    base_name = '_'.join(filter(lambda x : x is not None, components))
    
    # If a custom run name was provided, use it instead
    if args.run_name is not None:
        base_name = args.run_name
        
    full_name = f"{base_name}_{timestamp}"
    
    # Handle subfolder path
    subfolder_path = ""
    if subfolder:
        # Clean up subfolder path and ensure no leading/trailing slashes
        subfolder_path = '/'.join(filter(None, subfolder.split('/')))
        if subfolder_path:
            subfolder_path = f"{subfolder_path}/"
    
    return base_name, full_name, timestamp, subfolder_path
