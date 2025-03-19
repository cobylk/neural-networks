"""
MNIST Experiment Runner

This script runs vanilla MLP experiments on the MNIST dataset
with different activation functions and configurations.
"""

import sys
import torch
import torch.nn as nn
from activations import (
    RescaledReLU,
    LayerwiseSoftmax,
    PowerNormalization,
    TemperaturePowerNorm,
    ThresholdActivation,
    JumpReLU,
    RescaledJumpReLU,
    FixedRescaledJumpReLU,
    HectoRescaledReLU,
    HectoRescaledJumpReLU,
    HectoFixedRescaledJumpReLU,
    DecayingRescaledJumpReLU,
    DecayingRescaledReLU,
    DecayingFixedRescaledJumpReLU,
)
from config import VanillaExperimentConfig, EXPERIMENT_TYPES
from experiment_runner_utils import (
    ExperimentDefinition,
    ExperimentRunner,
    DEFAULT_FORMAT_SPEC,
)


def main():
    """Run experiments with different configurations"""
    # Define activation functions to test
    activations = [
        (
            DecayingRescaledReLU,
            {
                "initial_scale": 10,
                "final_scale": 1,
                "decay_epochs": 20,
                "decay_schedule": "exponential",
                "decay_rate": 0.6,
            },
        ),
        (
            DecayingRescaledReLU,
            {
                "initial_scale": 25,
                "final_scale": 1,
                "decay_epochs": 20,
                "decay_schedule": "exponential",
                "decay_rate": 0.6,
            },
        ),
        (
            DecayingRescaledReLU,
            {
                "initial_scale": 50,
                "final_scale": 1,
                "decay_epochs": 20,
                "decay_schedule": "exponential",
                "decay_rate": 0.6,
            },
        ),
        (
            DecayingRescaledReLU,
            {
                "initial_scale": 100,
                "final_scale": 1,
                "decay_epochs": 20,
                "decay_schedule": "exponential",
                "decay_rate": 0.6,
            },
        ),
    ]

    # Create experiment definition
    experiment_def = ExperimentDefinition(
        name="Vanilla MLP",
        config_class=VanillaExperimentConfig,
        base_params={
            "hidden_dims": [512, 256, 128, 64],
            "dropout_prob": 0.0,
            "epochs": 30,
            "learning_rate": 0.001,
            "early_stopping_patience": 20,
            "experiment_type": EXPERIMENT_TYPES["vanilla"],
            "save_dir": "results/rescaled_convergence_speed/run_9",
        },
    )

    runner = ExperimentRunner(experiment_def)

    tracker = runner.run_grid_search(
        activations=activations,
        architectures=[[512, 256, 128, 64]],
        result_keys=["test_acc", "train_acc", "final_threshold"],
    )

    tracker.print_summary(format_spec=DEFAULT_FORMAT_SPEC)


if __name__ == "__main__":
    main()
