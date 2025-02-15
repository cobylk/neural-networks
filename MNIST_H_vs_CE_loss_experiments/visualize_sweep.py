import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

def visualize_sweep(results_path, image_dir):
    # Load the data
    data = pd.read_csv(results_path)
    
    # Create figure
    sns.set_style('darkgrid')
    plt.figure(figsize=(12, 8))
    
    # Create plot with more subdivisions
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))  # Major ticks every 10
    ax.xaxis.set_minor_locator(plt.MultipleLocator(2))   # Minor ticks every 2
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))   # Major ticks every 5%
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))   # Minor ticks every 1%
    
    # Plot data
    sns.lineplot(data=data, x='n', y='accuracy', marker='o')
    
    # Customize plot
    plt.grid(True, which='major', linewidth=0.8)
    plt.grid(True, which='minor', linewidth=0.2)
    plt.xlabel('Harmonic Exponent (n)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('MNIST Test Accuracy vs Harmonic Exponent', fontsize=14, pad=20)
    
    # Add annotations for max accuracy in bottom right
    max_acc_idx = data['accuracy'].idxmax()
    max_n = data.loc[max_acc_idx, 'n']
    max_acc = data.loc[max_acc_idx, 'accuracy']
    
    # Get axis limits for positioning
    ymin, ymax = plt.ylim()
    xmin, xmax = plt.xlim()
    
    # Position box in bottom right
    plt.annotate(f'Max accuracy: {max_acc:.2f}%\nn = {max_n}',
                xy=(xmin + (xmax-xmin)*0.95, ymin + (ymax-ymin)*0.1),  # 95% from left, 10% from bottom
                xytext=(xmin + (xmax-xmin)*0.95, ymin + (ymax-ymin)*0.1),  # Same position as xy
                ha='right',  # Right-align text
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # Save with extra padding to prevent cutoff
    save_path = os.path.join(image_dir, results_path.rsplit('.', 1)[0].split('/')[-1] + '_viz.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    # Ask for the results file to visualize
    results_dir = "sweep_results"
    image_dir = "saved_images"
    os.makedirs(image_dir, exist_ok=True)
    
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} directory not found!")
        exit(1)
        
    print("\nAvailable result files:")
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]
    for i, file in enumerate(result_files):
        print(f"{i+1}. {file}")
    
    if not result_files:
        print("No result files found in sweep_results directory!")
        exit(1)
    
    selection = int(input("\nEnter the number of the file to visualize: ")) - 1
    if 0 <= selection < len(result_files):
        results_path = os.path.join(results_dir, result_files[selection])
        visualize_sweep(results_path, image_dir)
    else:
        print("Invalid selection!") 