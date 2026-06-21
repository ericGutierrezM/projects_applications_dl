"""
This code file compares the change
predictions from both the naïve and
the pre-trained approaches, by
randomly selecting 10 projects and 
plotting them on the same space.

Authors: Joshua Castillo and Eric Gutiérrez
Barcelona School of Economics, June 2026
"""

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_random_10_comparison(
    baseline_csv="data/dataset_w_baseline.csv", 
    vit_csv="data/dataset_w_changeStar.csv"
):
    print("Loading datasets and merging...")
    
    # Load both datasets
    df_baseline = pd.read_csv(baseline_csv)
    df_vit = pd.read_csv(vit_csv)
    
    # Merge them together using the contractId. 
    df = pd.merge(df_baseline, df_vit, on='contractId')
    
    # Drop any rows that might have failed to process and contain NaNs
    #df = df.dropna(subset=['vit_delta_00_to_25', 'delta_00_to_25'])
    
    print("Extracting 10 random projects...")
    # Randomly sample 10 projects
    sample_df = df.sample(n=10, random_state=42)
    
    intervals_base = ['00_to_25', '25_to_50', '50_to_75', '75_to_10', '10_to_po']
    intervals_vit = ['00_to_25_t0.5', '25_to_50_t0.5', '50_to_75_t0.5', '75_to_10_t0.5', '10_to_po_t0.5']
    
    x_labels = ['0-25%', '25-50%', '50-75%', '75-100%', 'Post']

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    for i, (idx, row) in enumerate(sample_df.iterrows()):
        ax = axes[i]
        
        # Extract the 5 chronological values for both models
        baseline_scores = [row[f'delta_{intv}'] for intv in intervals_base]
        vit_scores = [row[f'vit_delta_{intv}'] for intv in intervals_vit]
        
        # Plot Baseline as a dashed gray line
        ax.plot(x_labels, baseline_scores, marker='o', linestyle='--', 
                color='gray', linewidth=2, label='Baseline (Pixels)')
        
        # Plot ViT as a solid blue line
        ax.plot(x_labels, vit_scores, marker='s', linestyle='-', 
                color='#1f77b4', linewidth=2, label='ViT (Semantic AI)')
        
        # Formatting for each subplot
        ax.set_title(f"Contract: {row['contractId']}", fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        ax.tick_params(axis='x', rotation=45)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)

    fig.suptitle("Construction Curves: Pixel Baseline vs Semantic ViT", 
                 fontsize=20, fontweight='bold', y=1.02)
    fig.supylabel("Mean Change Confidence Score", fontsize=14, x=0.01)
    
    plt.tight_layout()
    
    # Save the master plot
    out_img = "data/baseline_vs_vit_10projects.png"
    plt.savefig(out_img, dpi=600, bbox_inches='tight')
    print(f"Master plot saved to {out_img}")
    
if __name__ == "__main__":
    plot_random_10_comparison(
        baseline_csv="data/dataset_w_baseline.csv", 
        vit_csv="data/dataset_w_changeStar.csv")