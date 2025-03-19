"""
MNIST Training Infrastructure

This module provides infrastructure for training and analyzing neural networks on the MNIST dataset.
It includes classes for experiment management, model analysis, and training configuration.

Key Components:
- FileManager: Handles all file operations and path management consistently
- PlotConfig: Configuration for controlling which plots to generate
- ModelAnalyzer: Analyzes model properties and behavior
- MNISTTrainer: Handles training and evaluation of models on MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional, Tuple, Dict, Any, Type, Union, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import platform
import sys
from tqdm import tqdm
import randomname
import git
import numpy as np
from collections import defaultdict
import pandas as pd
from base_MLP import BaseMLP
from file_utils import (
    EXPERIMENT_TYPES,
    BASE_RESULTS_DIR,
    FILENAMES,
    get_results_path,
    get_run_path,
    get_checkpoint_path,
    get_stats_path,
    get_plot_path,
    list_experiment_dirs,
    list_run_dirs,
    list_all_run_dirs,
)
import os


class FileManager:
    """Centralized file management system for MNIST experiments.

    This class handles all file and directory operations in a consistent manner,
    including path creation, naming conventions, and file saving/loading.
    """

    def __init__(self, base_dir: str = BASE_RESULTS_DIR):
        """Initialize the file manager with a base directory.

        Args:
            base_dir: Base directory for all experiment results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def create_experiment_dir(self, experiment_type: str) -> Path:
        """Create and return a directory for a specific experiment type.

        Args:
            experiment_type: Type of experiment (e.g., 'vanilla', 'stochastic')

        Returns:
            Path to the experiment type directory
        """
        return get_results_path(experiment_type)

    def generate_run_name(self, model_info: Dict[str, Any]) -> str:
        """Generate a standardized run name based on model and experiment info.

        Args:
            model_info: Dictionary containing model information

        Returns:
            A standardized run name string
        """
        # Get timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract model class name
        model_class = model_info.get("model_class", "Unknown")

        # Extract activation name
        activation_name = model_info.get("activation_name", "NoActivation")

        # Get a random name for uniqueness
        random_name = randomname.get_name()

        # Assemble the run name
        run_name = f"{timestamp}_{model_class}_{activation_name}_{random_name}"

        return run_name

    def create_run_dir(self, experiment_type: str, run_name: str) -> Path:
        """Create and return a directory for a specific run.

        Args:
            experiment_type: Type of experiment
            run_name: Name of the run

        Returns:
            Path to the run directory
        """
        # Check if the base_dir is the default or a custom path
        if self.base_dir == Path(BASE_RESULTS_DIR):
            # Use the standard path from get_run_path if using default base_dir
            return get_run_path(experiment_type, run_name)
        else:
            # If a custom base_dir was provided, use it directly
            custom_path = self.base_dir / run_name
            os.makedirs(custom_path, exist_ok=True)
            return custom_path

    def get_checkpoint_path(self, run_dir: Path, filename: str) -> Path:
        """Get path for a model checkpoint.

        Args:
            run_dir: Run directory
            filename: Checkpoint filename

        Returns:
            Path to the checkpoint file
        """
        return get_checkpoint_path(run_dir, filename)

    def get_stats_path(self, run_dir: Path) -> Path:
        """Get path for run statistics JSON file.

        Args:
            run_dir: Run directory

        Returns:
            Path to the statistics file
        """
        return get_stats_path(run_dir)

    def get_plot_path(self, run_dir: Path, plot_name: str, format: str = "png") -> Path:
        """Get path for a plot file.

        Args:
            run_dir: Run directory
            plot_name: Name of the plot
            format: File format for the plot

        Returns:
            Path to the plot file
        """
        return get_plot_path(run_dir, plot_name, format)

    def save_json(self, data: Dict, path: Path) -> None:
        """Save data as JSON.

        Args:
            data: Data to save
            path: Path to save the data to
        """
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load_json(self, path: Path) -> Dict:
        """Load data from JSON.

        Args:
            path: Path to load the data from

        Returns:
            Loaded data as a dictionary
        """
        with open(path, "r") as f:
            return json.load(f)

    def save_torch_model(self, data: Dict, path: Path) -> None:
        """Save PyTorch model data.

        Args:
            data: Model data to save
            path: Path to save the data to
        """
        torch.save(data, path)

    def load_torch_model(self, path: Path) -> Dict:
        """Load PyTorch model data.

        Args:
            path: Path to load the data from

        Returns:
            Loaded model data
        """
        return torch.load(path)

    def save_plot(self, fig: plt.Figure, path: Path, dpi: int = 300) -> None:
        """Save a matplotlib figure.

        Args:
            fig: Figure to save
            path: Path to save the figure to
            dpi: DPI for the saved figure
        """
        fig.savefig(path, dpi=dpi)
        plt.close(fig)

    def list_experiment_dirs(self) -> List[Path]:
        """List all experiment directories.

        Returns:
            List of experiment directory paths
        """
        return list_experiment_dirs()

    def list_run_dirs(self, experiment_type: str) -> List[Path]:
        """List all run directories for a specific experiment type.

        Args:
            experiment_type: Type of experiment

        Returns:
            List of run directory paths
        """
        return list_run_dirs(experiment_type)

    def list_all_run_dirs(self) -> List[Path]:
        """List all run directories across all experiment types.

        Returns:
            List of all run directory paths
        """
        return list_all_run_dirs()


def get_git_info() -> Dict[str, str]:
    """Get git repository information"""
    try:
        repo = git.Repo(search_parent_directories=True)
        return {
            "commit_hash": repo.head.object.hexsha,
            "branch": repo.active_branch.name,
            "is_dirty": repo.is_dirty(),
        }
    except:
        return {"error": "Not a git repository or git not installed"}


def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    return {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "cpu_count": platform.machine(),
        "processor": platform.processor(),
    }


class PlotConfig:
    """Configuration for what plots to generate"""

    def __init__(
        self,
        training_curves: bool = True,
        weight_distributions: bool = True,
        learning_curves: bool = True,
        experiment_comparison: bool = True,
        save_format: str = "png",
        dpi: int = 300,
        style: str = None,
    ):
        self.training_curves = training_curves
        self.weight_distributions = weight_distributions
        self.learning_curves = learning_curves
        self.experiment_comparison = experiment_comparison
        self.save_format = save_format
        self.dpi = dpi
        self.style = style

        # Apply plot style safely
        if style is not None:
            try:
                plt.style.use(style)
            except:
                print(f"Warning: Style '{style}' not found, using default style.")
                plt.style.use("default")
        else:
            # Try to use seaborn-darkgrid style, fall back to default if not available
            try:
                plt.style.use("seaborn-v0_8-darkgrid")
            except:
                try:
                    plt.style.use("seaborn-darkgrid")
                except:
                    plt.style.use("default")


class ModelAnalyzer:
    """Analyzes model properties and behavior"""

    def __init__(
        self,
        model: BaseMLP,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        plot_config: Optional[PlotConfig] = None,
        input_preprocessor: Optional[callable] = None,
    ):
        self.model = model
        self.device = device
        self.plot_config = plot_config or PlotConfig()
        # Store the input preprocessor function, if provided
        self.input_preprocessor = input_preprocessor

    def analyze_weights(self) -> Dict[str, Dict[str, float]]:
        """Analyze weight statistics for each layer"""
        stats = {}
        for name, param in self.model.named_parameters():
            if "weight" in name:
                layer_stats = {
                    "mean": float(param.mean()),
                    "std": float(param.std()),
                    "min": float(param.min()),
                    "max": float(param.max()),
                    "norm": float(param.norm()),
                    "sparsity": float(
                        torch.isclose(param, torch.zeros_like(param), atol=1e-8)
                        .float()
                        .mean()
                    ),
                    "has_negative": float((param < 0).float().mean()) > 0,
                    "percent_negative": float((param < 0).float().mean() * 100),
                    "shape": list(param.shape),
                    "num_params": param.numel(),
                }
                stats[name] = layer_stats
            elif "bias" in name:
                layer_stats = {
                    "mean": float(param.mean()),
                    "std": float(param.std()),
                    "min": float(param.min()),
                    "max": float(param.max()),
                    "has_negative": float((param < 0).float().mean()) > 0,
                    "percent_negative": float((param < 0).float().mean() * 100),
                    "shape": list(param.shape),
                    "num_params": param.numel(),
                }
                stats[name] = layer_stats
        return stats

    def plot_weight_distributions(self):
        """Plot weight distributions for each layer"""
        if not self.plot_config.weight_distributions:
            return None

        # Count number of weight tensors
        weight_params = [
            (name, p) for name, p in self.model.named_parameters() if "weight" in name
        ]
        num_weight_layers = len(weight_params)

        if num_weight_layers == 0:
            return None

        fig, axes = plt.subplots(
            num_weight_layers, 1, figsize=(10, 3 * num_weight_layers)
        )
        if num_weight_layers == 1:
            axes = [axes]

        for idx, (name, param) in enumerate(weight_params):
            sns.histplot(param.detach().cpu().numpy().flatten(), ax=axes[idx], bins=50)
            axes[idx].set_title(f"{name} Distribution")
            axes[idx].set_xlabel("Weight Value")
            axes[idx].set_ylabel("Count")
            # Add vertical line at 0
            axes[idx].axvline(x=0, color="r", linestyle="--", alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_activation_distributions(self, loader: DataLoader, num_batches: int = 5):
        """Plot distributions of pre-activations and post-activations to diagnose issues"""
        if not self.plot_config.weight_distributions:  # Reuse the same config flag
            return None

        self.model.enable_activation_storage()
        all_activations = {}
        all_stats = {}  # To store statistics for each activation

        # Collect activations from a few batches
        with torch.no_grad():
            for i, (data, _) in enumerate(loader):
                if i >= num_batches:
                    break
                data = data.to(self.device)

                # Apply preprocessing if available
                if self.input_preprocessor is not None:
                    data = self.input_preprocessor(data)

                _ = self.model(data)

                # Store activations by type
                for name, activation in self.model.activation_store.activations.items():
                    if name not in all_activations:
                        all_activations[name] = []
                        all_stats[name] = {
                            "mins": [],
                            "maxs": [],
                            "means": [],
                            "stds": [],
                            "sums": [],
                            "sparsity": [],
                        }

                    # Store a sample to avoid memory issues
                    sample = activation.detach().cpu().flatten()[:1000]
                    all_activations[name].append(sample)

                    # Compute batch statistics
                    all_stats[name]["mins"].append(float(activation.min()))
                    all_stats[name]["maxs"].append(float(activation.max()))
                    all_stats[name]["means"].append(float(activation.mean()))
                    all_stats[name]["stds"].append(float(activation.std()))
                    all_stats[name]["sums"].append(float(activation.sum()))
                    all_stats[name]["sparsity"].append(
                        float(
                            torch.isclose(
                                activation, torch.zeros_like(activation), atol=1e-8
                            )
                            .float()
                            .mean()
                        )
                    )

        # Compute average statistics
        for name in all_stats:
            # Get a fixed list of stats before iteration to avoid the "dictionary changed size during iteration" error
            stats_to_process = list(all_stats[name].keys())
            for stat in stats_to_process:
                all_stats[name][f"avg_{stat}"] = np.mean(all_stats[name][stat])

        # Categorize activations based on the new naming convention
        pre_activations = {}
        post_activations = {}
        other_activations = {}

        for name in all_activations:
            if "_postact" in name:
                post_activations[name] = all_activations[name]
            elif "_preact" in name:
                pre_activations[name] = all_activations[name]
            else:
                other_activations[name] = all_activations[name]

        # Print debug info
        if len(post_activations) == 0:
            print(
                "Warning: No post-activation modules detected. Available module names:"
            )
            for name in all_activations:
                print(f"  - {name}")

        # Create subplots based on what we found
        num_plots = (
            len(pre_activations) + len(post_activations) + len(other_activations)
        )
        if num_plots == 0:
            return None

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots))
        if num_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Helper function to create annotation text
        def create_annotation(name, all_acts):
            stats = all_stats[name]
            mean_val = stats["avg_means"]
            std_val = stats["avg_stds"]
            min_val = stats["avg_mins"]
            max_val = stats["avg_maxs"]
            sum_val = stats["avg_sums"]
            neg_pct = (all_acts < 0).mean() * 100
            zero_pct = np.isclose(all_acts, 0, atol=1e-8).mean() * 100
            sum_one_pct = 100 * np.mean(
                np.abs(all_acts.reshape(-1, 1000).sum(axis=1) - 1.0) < 0.01
            )

            result = (
                f"Mean: {mean_val:.4f}, Std: {std_val:.4f}\n"
                f"Min: {min_val:.4f}, Max: {max_val:.4f}\n"
                f"Avg Sum: {sum_val:.4f}\n"
                f"Negative: {neg_pct:.1f}%, Zero: {zero_pct:.1f}%"
            )

            # Add simplex info for stochastic/probability layers
            if (
                "stochastic" in name.lower()
                or "sparsemax" in name.lower()
                or "softmax" in name.lower()
            ):
                result += f"\nSums to 1: {sum_one_pct:.1f}%"

            return result

        # Plot pre-activations (Linear outputs)
        for name, acts in pre_activations.items():
            all_acts = torch.cat(acts, dim=0).numpy()
            sns.histplot(all_acts, ax=axes[plot_idx], bins=50, color="blue", alpha=0.7)
            axes[plot_idx].set_title(f"Pre-Activation: {name}")
            axes[plot_idx].set_xlabel("Activation Value")
            axes[plot_idx].set_ylabel("Count")
            axes[plot_idx].axvline(x=0, color="r", linestyle="--", alpha=0.5)

            # Add stats annotation
            annotation_text = create_annotation(name, all_acts)
            axes[plot_idx].annotate(
                annotation_text,
                xy=(0.02, 0.95),
                xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                va="top",
            )
            plot_idx += 1

        # Plot post-activations (ReLU, SparseMax, etc. outputs)
        for name, acts in post_activations.items():
            all_acts = torch.cat(acts, dim=0).numpy()
            sns.histplot(all_acts, ax=axes[plot_idx], bins=50, color="green", alpha=0.7)
            axes[plot_idx].set_title(f"Post-Activation: {name}")
            axes[plot_idx].set_xlabel("Activation Value")
            axes[plot_idx].set_ylabel("Count")
            axes[plot_idx].axvline(x=0, color="r", linestyle="--", alpha=0.5)

            # Add stats annotation
            annotation_text = create_annotation(name, all_acts)
            axes[plot_idx].annotate(
                annotation_text,
                xy=(0.02, 0.95),
                xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                va="top",
            )

            # Add warning if ReLU has negative values (which should be impossible)
            min_val = all_stats[name]["avg_mins"]
            if min_val < 0 and "relu" in name.lower():
                axes[plot_idx].annotate(
                    "WARNING: ReLU output has negative values!",
                    xy=(0.5, 0.5),
                    xycoords="axes fraction",
                    fontsize=12,
                    color="red",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                )
            plot_idx += 1

        # Plot other activations
        for name, acts in other_activations.items():
            all_acts = torch.cat(acts, dim=0).numpy()
            sns.histplot(
                all_acts, ax=axes[plot_idx], bins=50, color="purple", alpha=0.7
            )
            axes[plot_idx].set_title(f"Other: {name}")
            axes[plot_idx].set_xlabel("Value")
            axes[plot_idx].set_ylabel("Count")
            axes[plot_idx].axvline(x=0, color="r", linestyle="--", alpha=0.5)

            # Add stats annotation
            annotation_text = create_annotation(name, all_acts)
            axes[plot_idx].annotate(
                annotation_text,
                xy=(0.02, 0.95),
                xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                va="top",
            )
            plot_idx += 1

        plt.tight_layout()
        return fig

    def compute_activation_stats(
        self, loader: DataLoader
    ) -> Dict[str, Dict[str, float]]:
        """Compute activation statistics across a dataset"""
        self.model.enable_activation_storage()
        activation_stats = defaultdict(
            lambda: {
                "means": [],
                "stds": [],
                "sparsity": [],
                "mins": [],
                "maxs": [],
                "sums": [],
            }
        )

        with torch.no_grad():
            for data, _ in tqdm(loader, desc="Computing activation stats"):
                data = data.to(self.device)

                # Apply preprocessing if available
                if self.input_preprocessor is not None:
                    data = self.input_preprocessor(data)

                _ = self.model(data)

                for name, activation in self.model.activation_store.activations.items():
                    # Classify based on the modified naming convention
                    if "_postact" in name:
                        name_key = (
                            f"post_activation:{name}"  # Any activation function output
                        )
                    elif "_preact" in name:
                        name_key = f"pre_activation:{name}"  # Linear/layer output (pre-activation)
                    else:
                        # For backwards compatibility and unspecified hooks
                        print(f"Warning: Unclassified module type: {name}")
                        name_key = f"unknown:{name}"

                    activation_stats[name_key]["means"].append(float(activation.mean()))
                    activation_stats[name_key]["stds"].append(float(activation.std()))
                    activation_stats[name_key]["sparsity"].append(
                        float(
                            torch.isclose(
                                activation, torch.zeros_like(activation), atol=1e-8
                            )
                            .float()
                            .mean()
                        )
                    )
                    activation_stats[name_key]["mins"].append(float(activation.min()))
                    activation_stats[name_key]["maxs"].append(float(activation.max()))
                    activation_stats[name_key]["sums"].append(float(activation.sum()))

        # Compute aggregate statistics
        for name in list(
            activation_stats.keys()
        ):  # Use list to avoid modification during iteration
            for stat in ["means", "stds", "sparsity", "mins", "maxs", "sums"]:
                values = activation_stats[name][stat]
                if stat == "sums":
                    # Calculate average sum
                    activation_stats[name]["avg_sum"] = float(np.mean(values))
                else:
                    # For means, stds, sparsity, mins, maxs - calculate average and std
                    activation_stats[name][f"avg_{stat}"] = float(np.mean(values))
                    activation_stats[name][f"std_{stat}"] = float(np.std(values))
                del activation_stats[name][stat]

        # Add validation for ReLU activations - they should never be negative
        for name in list(activation_stats.keys()):
            if "post_activation" in name and "relu" in name.lower():
                if (
                    activation_stats[name]["avg_means"] < 0
                    or activation_stats[name]["avg_mins"] < 0
                ):
                    print(f"WARNING: Found negative values in ReLU activation: {name}")
                    print(
                        f"  Min: {activation_stats[name]['avg_mins']}, Mean: {activation_stats[name]['avg_means']}"
                    )
                    activation_stats[f"{name}_WARNING"] = (
                        "ReLU activations should never be negative!"
                    )

        return dict(activation_stats)


class MNISTTrainer:
    """Handles training and evaluation of BaseMLP models on MNIST"""

    def __init__(
        self,
        model: BaseMLP,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "results",
        experiment_type: str = "default",
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Dict = None,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        tags: List[str] = None,
        plot_config: Optional[PlotConfig] = None,
        preprocess_fn: Optional[callable] = None,
        total_epochs: int = 100,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.experiment_type = experiment_type
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.criterion = criterion
        self.preprocess_fn = preprocess_fn
        self.tags = tags or []
        self.total_epochs = total_epochs

        # Initialize file manager
        self.file_manager = FileManager(save_dir)

        # Create model info for run name generation
        model_info = {
            "model_class": model.__class__.__name__,
            "activation_name": (
                model.network[1].__class__.__name__
                if hasattr(model, "network") and len(model.network) > 1
                else "Unknown"
            ),
        }

        # Generate run name using file manager
        self.run_name = self.file_manager.generate_run_name(model_info)
        print(f"Run name: {self.run_name}")

        # Create run directory using file manager
        self.run_dir = self.file_manager.create_run_dir(
            self.experiment_type, self.run_name
        )

        # Setup model analyzer with new plot config
        self.plot_config = plot_config or PlotConfig()
        self.analyzer = ModelAnalyzer(
            model, device=device, plot_config=self.plot_config
        )

        # Setup training components
        self.optimizer = optimizer_class(
            model.parameters(), lr=learning_rate, **self.optimizer_kwargs
        )

        # Load and prepare data
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()

        # Initialize epoch-aware modules with the total epochs
        self.set_epoch_for_modules(self.model, 0)
        self.set_total_epochs_for_modules(self.model, self.total_epochs)

        # Training history and metadata
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": None,
            "test_acc": None,
            "metadata": {
                "run_name": self.run_name,
                "timestamp": datetime.now().isoformat(),
                "model_config": {
                    "model_class": model.__class__.__name__,
                    "hidden_dims": (
                        model.hidden_dims if hasattr(model, "hidden_dims") else None
                    ),
                    "dropout_prob": (
                        model.dropout_prob if hasattr(model, "dropout_prob") else 0.0
                    ),
                    "activation": (
                        model.network[1].__class__.__name__
                        if hasattr(model, "network") and len(model.network) > 1
                        else "Unknown"
                    ),
                    "num_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                },
                "training_config": {
                    "optimizer": optimizer_class.__name__,
                    "optimizer_kwargs": self.optimizer_kwargs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "device": device,
                    "total_epochs": total_epochs,
                    "tags": self.tags,
                },
                "system_info": get_system_info(),
                "git_info": get_git_info(),
            },
        }

    def _setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup MNIST data loaders"""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )

        # Load datasets
        train_dataset = datasets.MNIST(
            "data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST("data", train=False, transform=transform)

        # Split training into train and validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return train_loader, val_loader, test_loader

    def preprocess_data(self, data):
        """Apply preprocessing to input data if a preprocessing function is defined"""
        if self.preprocess_fn is not None:
            return self.preprocess_fn(data)
        return data

    def train_epoch(self, epoch) -> Tuple[float, float]:
        """Train for one epoch"""
        # Set the current epoch for all epoch-aware modules
        self.set_epoch_for_modules(self.model, epoch)

        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in tqdm(self.train_loader, leave=False):
            data, target = data.to(self.device), target.to(self.device)

            # Apply data preprocessing if defined
            data = self.preprocess_data(data)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def set_epoch_for_modules(self, model, epoch):
        """
        Recursively set the current epoch for all epoch-aware modules in the model.

        Args:
            model: The PyTorch model
            epoch: Current epoch number
        """
        for module in model.modules():  # This traverses the entire model hierarchy
            if hasattr(module, "set_epoch"):
                module.set_epoch(epoch)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on given loader"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)

            # Apply data preprocessing if defined
            data = self.preprocess_data(data)

            output = self.model(data)

            total_loss += self.criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def train(self, epochs: int, early_stopping_patience: Optional[int] = 5) -> Dict:
        """
        Train the model for specified number of epochs

        Args:
            epochs (int): Number of epochs to train
            early_stopping_patience (Optional[int]): Patience for early stopping
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in (pbar := tqdm(range(epochs))):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.evaluate(self.val_loader)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.2f}%",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.2f}%",
            )

            # Early stopping
            if early_stopping_patience:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint("best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break

        # Final test evaluation
        test_loss, test_acc = self.evaluate(self.test_loader)
        self.history["test_loss"] = test_loss
        self.history["test_acc"] = test_acc
        print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # After training is complete, check for JumpReLU and ThresholdActivation instances in the model
        from activations import (
            ThresholdActivation,
            JumpReLU,
            RescaledJumpReLU,
            FixedRescaledJumpReLU,
        )

        # Store the threshold values directly in the history metadata
        threshold_values = {}

        # Check if the main activation is a ThresholdActivation, JumpReLU, RescaledJumpReLU, or FixedRescaledJumpReLU
        if hasattr(self.model, "network"):
            # Track which hidden layer we're on
            hidden_layer_count = 0
            activation_count = 0

            for i, layer in enumerate(self.model.network):
                # Linear layers are followed by activation functions in BaseMLP
                if isinstance(layer, nn.Linear):
                    hidden_layer_count += 1

                if isinstance(
                    layer,
                    (
                        ThresholdActivation,
                        JumpReLU,
                        RescaledJumpReLU,
                        FixedRescaledJumpReLU,
                    ),
                ):
                    # Use the hidden layer number instead of raw index
                    threshold_values[
                        f"threshold_hidden_layer_{hidden_layer_count-1}"
                    ] = layer.get_threshold()
                    activation_count += 1
                # Also check for activation in sequential modules
                elif isinstance(layer, nn.Sequential):
                    for j, sublayer in enumerate(layer):
                        if isinstance(
                            sublayer,
                            (
                                ThresholdActivation,
                                JumpReLU,
                                RescaledJumpReLU,
                                FixedRescaledJumpReLU,
                            ),
                        ):
                            threshold_values[
                                f"threshold_hidden_layer_{hidden_layer_count-1}_sub_{j}"
                            ] = sublayer.get_threshold()
                            activation_count += 1

        # Store the threshold values in metadata
        if threshold_values:
            if "threshold_values" not in self.history["metadata"]:
                self.history["metadata"]["threshold_values"] = {}
            self.history["metadata"]["threshold_values"].update(threshold_values)

        # Save final results
        self.save_results()

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        path = self.file_manager.get_checkpoint_path(self.run_dir, filename)
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "run_name": self.run_name,
        }
        self.file_manager.save_torch_model(checkpoint_data, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.file_manager.get_checkpoint_path(self.run_dir, filename)
        checkpoint = self.file_manager.load_torch_model(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        self.run_name = checkpoint["run_name"]

    def save_results(self):
        """Save training results and plots"""
        # Update metadata with final metrics
        self.history["metadata"]["final_metrics"] = {
            "best_val_loss": min(self.history["val_loss"]),
            "best_val_acc": max(self.history["val_acc"]),
            "best_epoch": self.history["val_loss"].index(min(self.history["val_loss"])),
            "total_epochs": len(self.history["train_loss"]),
            "training_duration": (
                datetime.now()
                - datetime.fromisoformat(self.history["metadata"]["timestamp"])
            ).total_seconds(),
            "test_loss": self.history["test_loss"],
            "test_acc": self.history["test_acc"],
        }

        # Add weight analysis
        weight_stats = self.analyzer.analyze_weights()
        self.history["metadata"]["weight_analysis"] = weight_stats

        # Ensure that the analyzer uses the same preprocessing function
        self.analyzer.input_preprocessor = self.preprocess_fn

        # Compute and save activation statistics
        activation_stats = self.analyzer.compute_activation_stats(self.val_loader)
        self.history["metadata"]["activation_analysis"] = activation_stats

        # Save complete history with all metadata and statistics as JSON
        history_path = self.file_manager.get_stats_path(self.run_dir)
        self.file_manager.save_json(self.history, history_path)

        # Plot training curves if enabled
        if self.plot_config.training_curves:
            # Create the figure
            fig = plt.figure(figsize=(10, 5))

            # Loss plot
            plt.subplot(1, 2, 1)
            plt.plot(self.history["train_loss"], label="Train")
            plt.plot(self.history["val_loss"], label="Validation")
            plt.title(f"Loss vs. Epoch\n{self.run_name}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            # Accuracy plot
            plt.subplot(1, 2, 2)
            plt.plot(self.history["train_acc"], label="Train")
            plt.plot(self.history["val_acc"], label="Validation")
            plt.title(f"Accuracy vs. Epoch\n{self.run_name}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.legend()

            plt.tight_layout()

            # Save using file manager
            plot_path = self.file_manager.get_plot_path(
                self.run_dir, "training_curves", self.plot_config.save_format
            )
            self.file_manager.save_plot(fig, plot_path, self.plot_config.dpi)

        # Save weight distribution plot if enabled
        if self.plot_config.weight_distributions:
            weight_dist_fig = self.analyzer.plot_weight_distributions()
            if weight_dist_fig:
                plot_path = self.file_manager.get_plot_path(
                    self.run_dir, "weight_distributions", self.plot_config.save_format
                )
                self.file_manager.save_plot(
                    weight_dist_fig, plot_path, self.plot_config.dpi
                )

        # Save activation distribution plot
        act_dist_fig = self.analyzer.plot_activation_distributions(
            self.val_loader, num_batches=3
        )
        if act_dist_fig:
            plot_path = self.file_manager.get_plot_path(
                self.run_dir, "activation_distributions", self.plot_config.save_format
            )
            self.file_manager.save_plot(act_dist_fig, plot_path, self.plot_config.dpi)

        print(f"Results saved in: {self.run_dir}")

    def set_total_epochs_for_modules(self, model, total_epochs):
        """
        Set the total epochs for all epoch-aware modules in the model.

        Args:
            model: The PyTorch model
            total_epochs: Total number of epochs for training
        """
        for module in model.modules():
            if hasattr(module, "total_epochs"):
                module.total_epochs = total_epochs
