"""
File Management Utilities

This module provides constants and utility functions for file naming, path management,
and result organization for MNIST experiments.
"""

import os
from pathlib import Path
from typing import Dict, List, Union, Optional

# Base directory for all results
BASE_RESULTS_DIR = "results"

# Standardized experiment types
EXPERIMENT_TYPES = {
    'vanilla': 'vanilla',
    'dropout': 'dropout',
    'stochastic': 'stochastic',
    'constrained': 'constrained',
    'default': 'default'
}

# File extensions
FILE_EXTENSIONS = {
    'model': '.pt',
    'results': '.json',
    'plot': '.png',
    'csv': '.csv'
}

# Standard filenames
FILENAMES = {
    'best_model': 'best_model.pt',
    'final_model': 'final_model.pt',
    'checkpoint': 'checkpoint.pt',
    'run_stats': 'run_stats.json',
    'training_curves': 'training_curves',
    'weight_distributions': 'weight_distributions',
    'activation_distributions': 'activation_distributions',
    'experiment_comparison': 'experiment_comparison'
}

def get_results_path(experiment_type: str = 'default') -> Path:
    """
    Get path to the results directory for a specific experiment type.
    
    Args:
        experiment_type: Type of experiment (must be one of EXPERIMENT_TYPES)
    
    Returns:
        Path to the experiment type directory
    """
    if experiment_type not in EXPERIMENT_TYPES.values():
        raise ValueError(f"Invalid experiment type: {experiment_type}. "
                         f"Must be one of {list(EXPERIMENT_TYPES.values())}")
    
    path = Path(BASE_RESULTS_DIR) / experiment_type
    os.makedirs(path, exist_ok=True)
    return path

def get_run_path(experiment_type: str, run_name: str) -> Path:
    """
    Get path to a specific run directory.
    
    Args:
        experiment_type: Type of experiment
        run_name: Name of the run
    
    Returns:
        Path to the run directory
    """
    path = get_results_path(experiment_type) / run_name
    os.makedirs(path, exist_ok=True)
    return path

def get_checkpoint_path(run_path: Path, filename: str = FILENAMES['checkpoint']) -> Path:
    """
    Get path to a model checkpoint file.
    
    Args:
        run_path: Path to the run directory
        filename: Name of the checkpoint file
    
    Returns:
        Path to the checkpoint file
    """
    return run_path / filename

def get_stats_path(run_path: Path) -> Path:
    """
    Get path to the run statistics file.
    
    Args:
        run_path: Path to the run directory
    
    Returns:
        Path to the run statistics file
    """
    return run_path / FILENAMES['run_stats']

def get_plot_path(run_path: Path, plot_name: str, format: str = 'png') -> Path:
    """
    Get path to a plot file.
    
    Args:
        run_path: Path to the run directory
        plot_name: Name of the plot
        format: File format for the plot
    
    Returns:
        Path to the plot file
    """
    return run_path / f"{plot_name}.{format}"

def list_experiment_dirs() -> List[Path]:
    """
    List all experiment directories.
    
    Returns:
        List of experiment directory paths
    """
    base_dir = Path(BASE_RESULTS_DIR)
    os.makedirs(base_dir, exist_ok=True)
    return [p for p in base_dir.glob("*") if p.is_dir()]

def list_run_dirs(experiment_type: str) -> List[Path]:
    """
    List all run directories for a specific experiment type.
    
    Args:
        experiment_type: Type of experiment
    
    Returns:
        List of run directory paths
    """
    exp_dir = get_results_path(experiment_type)
    return [p for p in exp_dir.glob("*") if p.is_dir()]

def list_all_run_dirs() -> List[Path]:
    """
    List all run directories across all experiment types.
    
    Returns:
        List of all run directory paths
    """
    runs = []
    for exp_dir in list_experiment_dirs():
        runs.extend([p for p in exp_dir.glob("*") if p.is_dir()])
    return runs 