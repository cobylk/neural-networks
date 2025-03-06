"""
Experiment runner module for MNIST experiments.

Provides functionality for running various MNIST experiments
with different configurations and model architectures.
"""

from experiment_runners.vanilla_runner import run_vanilla_experiment
from experiment_runners.stochastic_runner import run_stochastic_experiment, SimplexMNISTTrainer 