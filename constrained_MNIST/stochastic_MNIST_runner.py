"""
Stochastic MNIST Experiment Runner

This script runs experiments with stochastic layers on the MNIST dataset
with different activation functions and configurations.
"""

from activations import (
    ThresholdActivation,
    SparseMax,
    PowerNormalization,
    FixedThreshold,
)
from config import StochasticExperimentConfig, EXPERIMENT_TYPES
from experiment_runner_utils import (
    ExperimentDefinition,
    ExperimentRunner,
    DEFAULT_FORMAT_SPEC,
)


def main():
    """Run experiments with different architectures and activations"""
    # Define activation functions to test
    activations = [
        # (None, None),  # No activation (just stochastic layers)
        # (PowerNormalization, None),
        # (SparseMax, None),
        (FixedThreshold, None),
        # (ThresholdActivation, {'initial_threshold': 0.05})
    ]

    # Test different network architectures
    architectures = [
        # [512],          # Single layer
        # [512, 256],     # Two layers
        [512, 256, 128]  # Three layers
    ]

    # Test different temperatures for the stochastic layer
    temperatures = [0.5]

    # Create experiment definition
    experiment_def = ExperimentDefinition(
        name="Stochastic MLP",
        config_class=StochasticExperimentConfig,
        base_params={
            "epochs": 100,
            "early_stopping_patience": 20,
            "learning_rate": 0.001,
            "batch_size": 128,
            "experiment_type": EXPERIMENT_TYPES["stochastic"],
        },
    )

    # Create experiment runner
    runner = ExperimentRunner(experiment_def)

    # Run grid search across all parameter combinations
    tracker = runner.run_grid_search(
        activations=activations,
        architectures=architectures,
        temperatures=temperatures,
        result_keys=["test_acc", "train_acc"],
    )

    # Print formatted summary
    tracker.print_summary(format_spec=DEFAULT_FORMAT_SPEC)


if __name__ == "__main__":
    main()
