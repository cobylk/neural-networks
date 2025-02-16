import os
import re
from visualize_weights import visualize_weights

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
    
    # Find all weight files from the sweep
    weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]# and 'sweep-n=' in f]
    
    if not weight_files:
        print("No sweep weight files found in saved_weights directory!")
        exit(1)
    
    # Sort files by n value
    weight_files.sort(key=extract_n)
    
    # Process each file
    print(f"\nProcessing {len(weight_files)} weight files...")
    for file in weight_files:
        n = extract_n(file)
        if n is not None:
            weights_path = os.path.join(weights_dir, file)
            print(f"Visualizing weights for n={n}")
            visualize_weights(weights_path, image_dir)

if __name__ == "__main__":
    main()