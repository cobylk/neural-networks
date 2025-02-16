import torch
import matplotlib.pyplot as plt
import os
import numpy as np
# import seaborn as sns

def visualize_weights(weights_path, image_dir):
    # Load the weights
    weights = torch.load(weights_path)
    
    # Create figure with more space and better dimensions
    # sns.set_style('darkgrid')
    fig = plt.figure(figsize=(13, 5))
    
    # Create grid with proper spacing
    gs = plt.GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, .1], hspace=0.1, wspace=0.1)
    
    # Create axes for the plots
    axes = []
    for i in range(2):
        for j in range(5):
            axes.append(fig.add_subplot(gs[i, j]))
    
    # Add title with padding
    plottitle = "MNIST weight visualization\n" + ''.join(weights_path.split('/')[-1].rsplit('_', 1)[0]).replace('_', ', ')

    fig.suptitle(plottitle, fontsize=16, y=1.02)
    
    # Reshape and plot weights for each output neuron
    for i in range(10):
        # Get weights for the i-th output neuron and reshape to 28x28
        dim = int(np.sqrt(weights[i].size(0)))
        neuron_weights = weights[i].reshape(dim, dim).numpy()
        
        # Plot with a bit more contrast
        vmin, vmax = np.percentile(neuron_weights, [2, 98])
        im = axes[i].imshow(neuron_weights, cmap='viridis', vmin=vmin, vmax=vmax) 
        axes[i].axis('off')
    
    # Add colorbar in the dedicated space
    cax = fig.add_subplot(gs[:, -1])
    plt.colorbar(im, cax=cax)
    
    # Save with extra padding to prevent cutoff
    save_path = os.path.join(image_dir, weights_path.rsplit('.', 1)[0].split('/')[-1] + '.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.close()
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    # Ask for the weights file to visualize
    weights_dir = "saved_weights"
    image_dir = "saved_images"
    os.makedirs(image_dir, exist_ok=True)
    if not os.path.exists(weights_dir):
        print(f"Error: {weights_dir} directory not found!")
        exit(1)
        
    print("\nAvailable weight files:")
    weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
    weight_files.sort()
    for i, file in enumerate(weight_files):
        print(f"{i+1}. {file}")
    
    if not weight_files:
        print("No weight files found in saved_weights directory!")
        exit(1)
    
    selection = int(input("\nEnter the number of the file to visualize: ")) - 1
    if 0 <= selection < len(weight_files):
        weights_path = os.path.join(weights_dir, weight_files[selection])
        visualize_weights(weights_path, image_dir)
    else:
        print("Invalid selection!") 