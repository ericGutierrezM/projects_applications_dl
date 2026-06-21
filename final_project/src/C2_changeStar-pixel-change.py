"""
This code file evaluates changes at the
pixel-level for any given pair of images
using a pre-trained neural network. The 
aggregation at the image level is done 
by taking a simple average.

Authors: Joshua Castillo and Eric Gutiérrez
Barcelona School of Economics, June 2026
"""

# Imports
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torchange.models.changen2 import s0_init_s1c1_changestar_vitb_1x256

def compute_and_plot_vit_sensitivity(csv_path="data/dataset_w_roads.csv"):
    print("Initializing End-to-End ViT Pipeline & Plotter...")
    
    # 1. Hardware Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon (MPS) detected! Running on Mac GPU.")
    else:
        device = torch.device("cpu")
        print("Running on standard CPU.")
        
    # 2. Load the Pre-Trained Model
    print("Downloading/Loading Vision Transformer Weights...")
    model = s0_init_s1c1_changestar_vitb_1x256().to(device)
    model.eval()
    
    # Standard ImageNet normalization
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Dataset
    df = pd.read_csv(csv_path)
    milestones = [
        "00_pre_start", "25_progress", "50_progress", 
        "75_progress", "100_completed", "post_completion"
    ]
    
    # Define thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []

    # 4. Iterate through all projects
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="📊 Computing ViT $\Delta$s"):
        contract_id = row['contractId']
        folder = Path(row['image_folder_path'])
        mask_path = folder / f"{contract_id}_road_mask.png"
        
        if not mask_path.exists():
            continue
            
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        binary_osm_mask = torch.from_numpy((raw_mask > 0).astype(np.float32)).to(device)
        
        missing_images = False
        for m in milestones:
            if not (folder / f"{contract_id}_{m}.png").exists():
                missing_images = True
                break
        if missing_images:
            continue
            
        project_scores = {'contractId': contract_id}
        
        # 5. Chronological Sequence Loop
        with torch.no_grad():
            for i in range(len(milestones) - 1):
                t0_name = milestones[i]
                t1_name = milestones[i+1]
                
                img_t0 = cv2.cvtColor(cv2.imread(str(folder / f"{contract_id}_{t0_name}.png")), cv2.COLOR_BGR2RGB)
                img_t1 = cv2.cvtColor(cv2.imread(str(folder / f"{contract_id}_{t1_name}.png")), cv2.COLOR_BGR2RGB)
                
                tensor_t0 = img_transform(img_t0).unsqueeze(0).to(device)
                tensor_t1 = img_transform(img_t1).unsqueeze(0).to(device)
                
                bitemporal_input = torch.cat([tensor_t0, tensor_t1], dim=1)
                preds = model(bitemporal_input)
                
                change_probs = torch.sigmoid(preds.change_prediction).squeeze()
                road_corridor_probs = change_probs[binary_osm_mask == 1.0]
                total_mask_pixels = len(road_corridor_probs)
                
                # Calculate percentage changed for each threshold
                for thresh in thresholds:
                    interval_name = f"vit_delta_{t0_name[:2]}_to_{t1_name[:2]}_t{thresh}"
                    
                    if total_mask_pixels > 0:
                        changed_pixels = (road_corridor_probs > thresh).sum().item()
                        pct_changed = changed_pixels / total_mask_pixels
                    else:
                        pct_changed = 0.0
                        
                    project_scores[interval_name] = pct_changed

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        results.append(project_scores)

    # 6. Save Data
    vit_df = pd.DataFrame(results)
    final_df = pd.merge(df, vit_df, on='contractId', how='left')
    
    out_csv = "data/dataset_w_changeStar.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"\nViT sensitivity data saved to {out_csv}")

    # Plotting
    print("Generating Sensitivity Analysis Plot...")
    
    intervals = ['00_to_25', '25_to_50', '50_to_75', '75_to_10', '10_to_po']
    x_labels = ['0-25%', '25-50%', '50-75%', '75-100%', 'Post']

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds)))

    for idx, thresh in enumerate(thresholds):
        mean_scores = []
        for interval in intervals:
            col_name = f"vit_delta_{interval}_t{thresh}"

            if col_name in final_df.columns:
                mean_scores.append(final_df[col_name].mean())
            else:
                mean_scores.append(0)
                
        plt.plot(x_labels, mean_scores, marker='o', linewidth=2.5, 
                 color=colors[idx], label=f'Threshold {thresh}')

    plt.title("Sensitivity Analysis: Impact of ViT Confidence Threshold", 
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Construction Interval", fontsize=12, fontweight='bold')
    plt.ylabel("Average % of Road Corridor Changed", fontsize=12, fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Confidence Cutoff", bbox_to_anchor=(1.05, 1), 
               loc='upper left', fontsize=11, title_fontsize=12)

    plt.tight_layout()
    
    out_img = "data/sensitivity_analysis_plot.png"
    plt.savefig(out_img, dpi=600, bbox_inches='tight')
    print(f"Sensitivity plot saved to {out_img}")
    plt.show()

if __name__ == "__main__":
    compute_and_plot_vit_sensitivity("data/dataset_w_roads.csv")