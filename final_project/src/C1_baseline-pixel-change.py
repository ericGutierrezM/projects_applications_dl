"""
This code file evaluates changes at the
pixel-level for any given pair of images
using a naïve normalized difference of
pixels. The aggregation at the image
level is done by taking a simple average.

Authors: Joshua Castillo and Eric Gutiérrez
Barcelona School of Economics, June 2026
"""

# Imports
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def compute_interval_changes(csv_path):
    df = pd.read_csv(csv_path)
    
    milestones = [
        "00_pre_start", 
        "25_progress", 
        "50_progress", 
        "75_progress", 
        "100_completed", 
        "post_completion"
    ]
    
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing Pixel Deltas"):
        contract_id = row['contractId']
        folder = Path(row['image_folder_path'])
        mask_path = folder / f"{contract_id}_road_mask.png"
        
        if not mask_path.exists():
            continue
            
        # Load the binary mask (0 = background, 255 = road)
        # Normalize mask to [0, 1] for easy matrix multiplication
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 0).astype(np.float32) 
        
        # Dictionary to hold the raw grayscale arrays for this project
        gray_images = {}
        missing_images = False
        
        for m in milestones:
            img_path = folder / f"{contract_id}_{m}.png"
            if not img_path.exists():
                missing_images = True
                break
            
            # Load image and convert to grayscale
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_images[m] = gray.astype(np.float32)
            
        if missing_images:
            continue
            
        # Compute the changes between consecutive milestones
        project_scores = {'contractId': contract_id}
        
        for i in range(len(milestones) - 1):
            t0_name = milestones[i]
            t1_name = milestones[i+1]
            interval_name = f"delta_{t0_name[:2]}_to_{t1_name[:2]}" # e.g., 'delta_00_to_25'
            
            t0_gray = gray_images[t0_name]
            t1_gray = gray_images[t1_name]
            
            # 1. Absolute Difference
            abs_diff = np.abs(t1_gray - t0_gray)
            
            # 2. Apply OSM Mask
            masked_diff = abs_diff * binary_mask
            
            # 3. Normalize to [0, 1] (Since max grayscale difference is 255)
            normalized_diff = masked_diff / 255.0
            
            # Extract a single aggregate baseline score: 
            # The mean pixel change strictly inside the road corridor
            road_pixels_only = normalized_diff[binary_mask == 1]
            mean_change = np.mean(road_pixels_only) if len(road_pixels_only) > 0 else 0
            
            project_scores[interval_name] = mean_change
            
        results.append(project_scores)

    # Save the baseline scores to a new CSV
    baseline_df = pd.DataFrame(results)
    final_df = pd.merge(df, baseline_df, on='contractId', how='left')
    final_df.to_csv("data/dataset_w_baseline.csv", index=False)
    print("\nBaseline change detection complete!")

if __name__ == "__main__":
    compute_interval_changes("data/dataset_w_roads.csv")