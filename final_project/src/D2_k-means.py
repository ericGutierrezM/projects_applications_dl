"""
This code file performs a time series
unsupervised clustering algorithm to
classify the projects in 3 groups.

Authors: Joshua Castillo and Eric Gutiérrez
Barcelona School of Economics, June 2026
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def run_timeseries_clustering(csv_path="data/dataset_w_baseline.csv", n_clusters=3):
    print("Initializing Time-Series Clustering pipeline...")
    
    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    # To run the clustering with the predictions
    # from changeStar, change the file name and
    # replace the column names by the ones below. 
    """
    cols = [
        'vit_delta_00_to_25', 'vit_delta_25_to_50', 
        'vit_delta_50_to_75', 'vit_delta_75_to_10', 'vit_delta_10_to_po'
    ]
    """

    cols = [
        'delta_00_to_25', 'delta_25_to_50', 
        'delta_50_to_75', 'delta_75_to_10', 'delta_10_to_po'
    ]
    
    # Drop rows with missing data
    clean_df = df.dropna(subset=cols).copy()
    
    # Extract the data into a NumPy array
    raw_time_series = clean_df[cols].values
    
    X = raw_time_series.reshape((raw_time_series.shape[0], raw_time_series.shape[1], 1))
    
    # Scale
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    
    # 2. Configure and Run DTW K-Means
    print(f"Running DTW K-Means to find {n_clusters} distinct construction patterns...")
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", 
                             max_iter=10, random_state=42, n_jobs=-1)
    
    # Fit the model and get the cluster labels for each project
    labels = model.fit_predict(X_scaled)
    clean_df['construction_cluster'] = labels
    
    # 3. Visualization
    print("📊 Generating Cluster Visualization...")
    fig, axes = plt.subplots(1, n_clusters, figsize=(18, 5), sharey=True)
    x_labels = ['0-25%', '25-50%', '50-75%', '75-100%', 'Post']
    
    for yi in range(n_clusters):
        ax = axes[yi]
        # Get all time series that fall into this cluster
        cluster_series = X_scaled[labels == yi]
        
        for xx in cluster_series:
            ax.plot(x_labels, xx.ravel(), "k-", alpha=0.05)
            
        barycenter = model.cluster_centers_[yi].ravel()
        ax.plot(x_labels, barycenter, "r-", linewidth=3, label="Cluster Average")
        
        ax.set_title(f"Cluster {yi} (n={len(cluster_series)})", fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        if yi == 0:
            ax.set_ylabel("Normalized Change Score")
            ax.legend()
            
    plt.suptitle("Discovered Construction Lifecycles (DTW K-Means)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    out_img = "data/cluster_discovery.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Save the Labeled Dataset
    # Merge the labels back into the original dataframe 
    final_df = pd.merge(df, clean_df[['contractId', 'construction_cluster']], on='contractId', how='left')
    
    out_csv = "data/dataset_clustered.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"Clustering complete! Labeled data saved to {out_csv}")

if __name__ == "__main__":
    run_timeseries_clustering(n_clusters=3)