import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms

# Use the author's automated wrapper function
from torchange.models.changen2 import s0_init_s1c1_changestar_vitb_1x256

def run_masked_changen2_inference(contract_id="18f00093", base_dir="data/images"):
    print(f"🌟 Running Masked ViT Inference on {contract_id}...")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("⚡ Apple Silicon (MPS) detected!")
    else:
        device = torch.device("cpu")
        
    folder = Path(base_dir) / contract_id
    t0_path = folder / f"{contract_id}_25_progress.png" # Dec 2018
    t1_path = folder / f"{contract_id}_50_progress.png" # Apr 2019
    
    # ─────────────────────────────────────────────────────────
    # NEW: Define the path to your OSM Road Mask
    # ─────────────────────────────────────────────────────────
    mask_path = folder / f"{contract_id}_road_mask.png" 
    
    if not t0_path.exists() or not t1_path.exists() or not mask_path.exists():
        print("Images or OSM mask missing! Please check the folder.")
        return

    # Load Images
    img_t0 = cv2.cvtColor(cv2.imread(str(t0_path)), cv2.COLOR_BGR2RGB)
    img_t1 = cv2.cvtColor(cv2.imread(str(t1_path)), cv2.COLOR_BGR2RGB)
    
    # Load the OSM Mask (Grayscale)
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    # Convert to binary float [0.0 or 1.0] and send to the Mac GPU
    binary_osm_mask = torch.from_numpy((raw_mask > 0).astype(np.float32)).to(device)

    # Transform inputs
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor_t0 = transform(img_t0).unsqueeze(0).to(device)
    tensor_t1 = transform(img_t1).unsqueeze(0).to(device)

    print("Loading Vision Transformer...")
    model = s0_init_s1c1_changestar_vitb_1x256().to(device)
    model.eval()

    print("Passing images through the network...")
    with torch.no_grad():
        bitemporal_input = torch.cat([tensor_t0, tensor_t1], dim=1)
        preds = model(bitemporal_input)
        
        # Extract the raw logits
        change_logits = preds.change_prediction
        
        # 1. Convert logits to probabilities (0.0 to 1.0)
        change_probs = torch.sigmoid(change_logits).squeeze()
        
        # ─────────────────────────────────────────────────────────
        # THE MASKING OPERATION
        # Multiply the model's probabilities by the OSM geographic mask
        # This instantly zeroes out clouds, shadows, and off-road seasonal changes!
        # ─────────────────────────────────────────────────────────
        masked_probs = change_probs * binary_osm_mask
        
        # 2. Apply your strict 0.6 threshold to the masked probabilities
        final_mask_tensor = (masked_probs > 0.6)
                
        binary_output = final_mask_tensor.cpu().numpy().astype(np.uint8) * 255

    out_path = folder / f"{contract_id}_Masked_Changen2_Prediction.png"
    cv2.imwrite(str(out_path), binary_output)
    print(f"✅ Inference complete! Masked prediction saved to {out_path}")

if __name__ == "__main__":
    run_masked_changen2_inference()