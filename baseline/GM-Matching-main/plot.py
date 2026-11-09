import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_theme("poster")

def plot_training_curves(log_file, save_dir=None):
    """Plot training curves from a CSV log file as three separate figures."""
    # Read the CSV file and filter for epoch-end records
    df = pd.read_csv(log_file, skiprows=1)
    df_epoch = df[df['batch'] == 'end']  # Use epoch-end records for test accuracy
    df_batch = df[df['batch'] != 'end']  # Use batch records for training metrics
    
    # Training accuracy plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(df_batch['epoch'].astype(float) + df_batch['batch'].astype(float)/len(df_batch), 
            df_batch['train_acc'], color='blue', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.8)
    plt.minorticks_on()
    plt.grid(True, which='both', alpha=0.8)

    plt.savefig(f"{save_dir}/train_acc.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test accuracy plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(df_epoch['epoch'], df_epoch['test_acc'], color='green', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.8)
    plt.minorticks_on()
    plt.grid(True, which='both', alpha=0.8)

    plt.savefig(f"{save_dir}/test_acc.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Training loss plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(df_batch['epoch'].astype(float) + df_batch['batch'].astype(float)/len(df_batch), 
            df_batch['train_loss'], color='red', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.8)
    plt.minorticks_on()
    plt.grid(True, which='both', alpha=0.8)

    plt.savefig(f"{save_dir}/train_loss.pdf", dpi=300, bbox_inches='tight')
    plt.close()

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Plot training curves from CSV logs.")
    parser.add_argument("--log_file", type=str, help="Path to the CSV log file.", default="logs/cifar10/baseline/logs.csv")
    return parser.parse_args()

if __name__ == "__main__":
    import os
    # Example usage - create both types of plots
    args = parse_args()
    log_file = args.log_file
    save_dir = os.path.dirname(log_file)

    plot_training_curves(log_file=log_file, save_dir=save_dir)