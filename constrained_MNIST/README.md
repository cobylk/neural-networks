# MNIST Neural Network Experiments

This repository contains code for running neural network experiments on the MNIST dataset with various architectures, activations, and training configurations.

The current main files in which I am running experiments are `MNIST_runner.py` and `stochastic_MNIST_runner.py`.

## File Management System

### Directory Structure

The project uses a standardized file management system organized as follows:

```
results/
├── vanilla/               # Vanilla MLP experiments
│   ├── run_20230301_123456_BaseMLP_ReLU_falcon/
│   │   ├── best_model.pt
│   │   ├── run_stats.json
│   │   ├── training_curves.png
│   │   └── ...
│   └── ...
├── stochastic/            # Stochastic layer experiments
│   ├── run_20230301_123457_StochasticMLP_Threshold_hawk/
│   │   ├── best_model.pt
│   │   ├── run_stats.json
│   │   └── ...
│   └── ...
└── ...
```

### File naming conventions

- **Run directories**: `{timestamp}_{model_class}_{activation_name}_{random_name}`
- **Model checkpoints**: `best_model.pt`, `final_model.pt`, `checkpoint.pt`
- **Statistics**: `run_stats.json`
- **Plots**: `training_curves.png`, `weight_distributions.png`, etc.

## Key Components

### Experiment Configuration

Experiment configurations are defined in `config.py` using dataclasses:

- `BaseExperimentConfig`: Base configuration class
- `VanillaExperimentConfig`: Configuration for standard MLP experiments
- `StochasticExperimentConfig`: Configuration for stochastic layer experiments

### File Management

File paths and utilities are centralized in `file_utils.py`:

- Constants for experiment types, directory names, and file extensions
- Path utilities for creating and accessing directories and files
- Functions for listing and retrieving experiment results

The `FileManager` class in `experiment_utils.py` provides a higher-level interface for:

- Creating and managing experiment directories
- Generating standardized run names
- Saving and loading experiment data
- Managing file paths for models, statistics, and plots

### Experiment Utilities

The `experiment_runner_utils.py` module provides utilities for running and analyzing experiments:

- `run_experiment`: Unified function for running experiments with different configurations
- `ExperimentDefinition`: Defines a set of experiments with shared base parameters
- `ExperimentRunner`: Runs experiments with different parameter combinations using grid search
- `ExperimentTracker`: Tracks and displays experiment results in formatted tables

### Runner Scripts

The repository includes several ready-to-use runner scripts:

- `MNIST_runner.py`: Runs vanilla MLP experiments with different activations
- `stochastic_MNIST_runner.py`: Runs experiments with stochastic layers

## Usage

### Basic Usage

To run experiments, use one of the runner scripts:

```bash
# Run vanilla MLP experiments
python MNIST_runner.py

# Run stochastic layer experiments
python stochastic_MNIST_runner.py

# Run architecture comparison experiments
python architecture_runner.py
```

### Custom Experiment Runner

You can create a custom experiment runner using the experiment utilities:

```python
from config import VanillaExperimentConfig, EXPERIMENT_TYPES
from experiment_runner_utils import ExperimentDefinition, ExperimentRunner, DEFAULT_FORMAT_SPEC
import torch.nn as nn

# Define experiment parameters
experiment_def = ExperimentDefinition(
    name="My Experiment",
    config_class=VanillaExperimentConfig,
    base_params={
        'hidden_dims': [512, 256],
        'dropout_prob': 0.2,
        'epochs': 50,
        'experiment_type': EXPERIMENT_TYPES['vanilla']
    }
)

# Create runner and run experiments with grid search
runner = ExperimentRunner(experiment_def)

# Run grid search with specific parameter combinations
tracker = runner.run_grid_search(
    # Multiple activations (varying parameter)
    activations=[
        (nn.ReLU, {}),
        (nn.Tanh, {})
    ],
    # Fixed architecture (constant parameter)
    architectures=[[512, 256]],
    result_keys=['test_acc', 'train_acc']
)

# Display results
tracker.print_summary(format_spec=DEFAULT_FORMAT_SPEC)
```

## Analyzing Results

The MNIST experiments save comprehensive results to the experiment directories:

- Training and validation curves are saved as PNG files
- Complete run statistics are saved as JSON files
- Model parameters are saved as PyTorch checkpoint files

You can analyze the results by:

1. Examining the training curves and statistics in each run directory
2. Loading checkpoint files to further analyze or use trained models
3. Using the `ExperimentTracker` to compare performance across different configurations 