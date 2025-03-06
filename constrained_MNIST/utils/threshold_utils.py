"""
Threshold tracking utilities.

This module provides functionality for tracking and visualizing threshold parameters 
in ThresholdActivation modules during training.
"""

import matplotlib.pyplot as plt
import numpy as np
from activations import ThresholdActivation

def track_threshold_evolution(trainer, experiment_name):
    """
    Track and visualize how threshold values evolve during training.
    
    Args:
        trainer: The trainer instance containing the model
        experiment_name: Name of the experiment for labeling
        
    Returns:
        dict: Dictionary containing threshold data
    """
    # Extract all threshold parameters from the model
    thresholds = {}
    initial_thresholds = {}
    
    for name, module in trainer.model.named_modules():
        if isinstance(module, ThresholdActivation):
            thresholds[name] = module.get_threshold()
            initial_thresholds[name] = module.initial_value
    
    if not thresholds:
        print("No ThresholdActivation modules found in the model")
        return {}
    
    # Print final threshold values and compare with initial values
    print("\nThreshold Values - Evolution during training:")
    print(f"{'Layer':<30} {'Initial':<10} {'Final':<10} {'Change':<10}")
    print("-" * 60)
    
    for name, threshold in thresholds.items():
        initial = initial_thresholds[name]
        change = threshold - initial
        change_percent = (change / initial) * 100 if initial != 0 else float('inf')
        
        print(f"{name:<30} {initial:<10.4f} {threshold:<10.4f} {change:+.4f} ({change_percent:+.1f}%)")
    
    # Create a visualization of the threshold values (initial vs final)
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    # Plot bars side by side
    plt.bar(x - width/2, [initial_thresholds[name] for name in thresholds.keys()], 
            width, label='Initial Threshold', color='lightblue')
    plt.bar(x + width/2, [thresholds[name] for name in thresholds.keys()], 
            width, label='Final Threshold', color='salmon')
    
    # Add reference lines
    plt.axhline(y=0.0, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.3)
    plt.axhline(y=1.0, color='b', linestyle='--', alpha=0.3)
    
    # Add labels and legend
    plt.title(f'Learned Threshold Values - {experiment_name}')
    plt.xlabel('Layer')
    plt.ylabel('Threshold Value')
    plt.xticks(x, thresholds.keys(), rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = trainer.run_dir / 'threshold_values.png'
    plt.savefig(save_path)
    print(f"Threshold visualization saved to {save_path}")
    
    return {
        'thresholds': thresholds,
        'initial_thresholds': initial_thresholds,
        'changes': {name: thresholds[name] - initial_thresholds[name] 
                    for name in thresholds.keys()}
    } 