import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

# The author's Changen2 Vision Transformer
from torchange.models.changen2 import s0_init_s1c1_changestar_vitb_1x256

def compute_vit_interval_changes(csv_path):
    print("🚀 Initializing ViT Pipeline...")
    
    # 1. Hardware Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("⚡ Apple Silicon (MPS) detected! Running on Mac GPU.")
    else:
        device = torch.device("cpu")
        print("🖥️ Running on standard CPU.")
        
    # 2. Load the Pre-Trained Model ONCE
    print("🧠 Downloading/Loading Vision Transformer Weights...")
    model = s0_init_s1c1_changestar_vitb_1x256().to(device)
    model.eval()
    
    # Standard ImageNet normalization required by the model
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
    
    results = []

    # 4. Iterate through all projects
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="📊 Computing ViT $\Delta$s"):
        contract_id = row['contractId']
        folder = Path(row['image_folder_path'])
        mask_path = folder / f"{contract_id}_road_mask.png"
        
        if not mask_path.exists():
            continue
            
        # Load and prep the OSM mask
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        binary_osm_mask = torch.from_numpy((raw_mask > 0).astype(np.float32)).to(device)
        
        # Verify all 6 images exist before processing
        missing_images = False
        for m in milestones:
            if not (folder / f"{contract_id}_{m}.png").exists():
                missing_images = True
                break
        if missing_images:
            continue
            
        # Dictionary to store the 5 interval scores for this specific project
        project_scores = {'contractId': contract_id}
        
        # 5. Chronological Sequence Loop
        with torch.no_grad(): # Disable gradients for the whole project to save Mac memory
            for i in range(len(milestones) - 1):
                t0_name = milestones[i]
                t1_name = milestones[i+1]
                interval_name = f"vit_delta_{t0_name[:2]}_to_{t1_name[:2]}" 
                
                # Load BGR images and convert to RGB
                img_t0 = cv2.cvtColor(cv2.imread(str(folder / f"{contract_id}_{t0_name}.png")), cv2.COLOR_BGR2RGB)
                img_t1 = cv2.cvtColor(cv2.imread(str(folder / f"{contract_id}_{t1_name}.png")), cv2.COLOR_BGR2RGB)
                
                # Transform to PyTorch tensors and send to GPU
                tensor_t0 = img_transform(img_t0).unsqueeze(0).to(device)
                tensor_t1 = img_transform(img_t1).unsqueeze(0).to(device)
                
                # Concatenate channels [1, 6, 256, 256] and infer
                bitemporal_input = torch.cat([tensor_t0, tensor_t1], dim=1)
                preds = model(bitemporal_input)
                
                # Extract logits and apply sigmoid
                change_logits = preds.change_prediction
                change_probs = torch.sigmoid(change_logits).squeeze()
                
                # MASKING: Extract probabilities ONLY inside the OSM corridor
                road_corridor_probs = change_probs[binary_osm_mask == 1.0]
                
                # Calculate mean (fallback to 0 if mask is weirdly empty)
                if len(road_corridor_probs) > 0:
                    avg_vit_change = road_corridor_probs.mean().item()
                else:
                    avg_vit_change = 0.0
                    
                project_scores[interval_name] = avg_vit_change

                # Optional: Free up memory explicitly for MPS 
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        results.append(project_scores)

    # 6. Save the deep learning scores
    vit_df = pd.DataFrame(results)
    final_df = pd.merge(df, vit_df, on='contractId', how='left')
    
    # Save to a new CSV so we don't overwrite the original baseline
    out_csv = "data/dataset_w_changeStar.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"\nViT chronological processing complete! Saved to {out_csv}")

if __name__ == "__main__":
    compute_vit_interval_changes("data/dataset_w_roads.csv")