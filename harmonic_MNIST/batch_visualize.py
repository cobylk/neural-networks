import os
import re
from harmonic_MNIST.visualize import visualize_weights

def extract_n(filename):
    """Extract n value from the weight filename."""
    match = re.search(r'n=(\d+)', filename)
    return int(match.group(1)) if match else None

def main():
    # Setup directories
    weights_dir = "saved_weights"
    image_dir = "saved_images"
    os.makedirs(image_dir, exist_ok=True)
    
    if not os.path.exists(weights_dir):
        print(f"Error: {weights_dir} directory not found!")
        exit(1)
    
    # Find all weight files from the sweep, including in subfolders
    weight_files = []
    for root, _, files in os.walk(weights_dir):
        for file in files:
            if file.endswith('.pt') and file.startswith('softmax'):
                rel_path = os.path.relpath(os.path.join(root, file), weights_dir)
                weight_files.append(rel_path)
    
    if not weight_files:
        print("No sweep weight files found in saved_weights directory!")
        exit(1)
    
    # Sort files by n value
    # weight_files.sort(key=extract_n)
    
    # Process each file
    print(f"\nProcessing {len(weight_files)} weight files...")
    for file in weight_files:
        weights_path = os.path.join(weights_dir, file)
        visualize_weights(weights_path, image_dir)

if __name__ == "__main__":
    main()