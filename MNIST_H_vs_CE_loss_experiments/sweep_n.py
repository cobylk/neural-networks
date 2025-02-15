import subprocess
import re
import os
import datetime
import argparse
from tqdm import tqdm

def run_experiment(n):
    """Run MNIST_harmonic.py with specified n value and return test accuracy"""
    cmd = f"python MNIST_harmonic.py --n {n} --epochs 10 --run-name sweep-n={n}"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, _ = process.communicate()
    
    # Extract test accuracy from output using regex
    match = re.search(r'Test set:.*Accuracy: \d+/\d+ \((\d+\.\d+)%\)', output)
    if match:
        return float(match.group(1))
    return None

def main(args):
    # Generate n values
    n_values = list(range(args.start, args.end + 1, args.step))
    accuracies = []
    
    # Create results directory if it doesn't exist
    results_dir = "sweep_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp and filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"n-sweep_{args.start}-{args.end}-step{args.step}_{timestamp}.txt"
    save_path = os.path.join(results_dir, save_name)
    
    # Run experiments
    print(f"Running sweep from n={args.start} to n={args.end} with step={args.step}")
    for n in tqdm(n_values):
        accuracy = run_experiment(n)
        accuracies.append(accuracy)
        print(f"n={n}, accuracy={accuracy:.2f}%")
    
    # Save raw data
    with open(save_path, 'w') as f:
        f.write("n,accuracy\n")
        for n, acc in zip(n_values, accuracies):
            f.write(f"{n},{acc}\n")
    print(f"\nResults saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sweep n values for MNIST Harmonic Loss')
    parser.add_argument('--start', type=int, default=0,
                        help='start value for n (default: 0)')
    parser.add_argument('--end', type=int, default=100,
                        help='end value for n, inclusive (default: 100)')
    parser.add_argument('--step', type=int, default=4,
                        help='step size for n (default: 4)')
    
    args = parser.parse_args()
    main(args) 