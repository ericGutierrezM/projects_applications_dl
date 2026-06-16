import cv2
import numpy as np
from pathlib import Path

def check_alignment(contract_id="18f00093", base_dir="data/images"):
    # Change these paths to match your actual directory structure!
    folder = Path(base_dir) / contract_id
    img_path = folder / f"{contract_id}_post_completion.png"
    mask_path = folder / f"{contract_id}_road_mask.png"
    
    if not img_path.exists() or not mask_path.exists():
        print("Images not found! Check your paths.")
        return
        
    # Load the satellite image (BGR) and the mask (Grayscale)
    sat_img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Create a blank neon red canvas (BGR format: Blue=0, Green=0, Red=255)
    red_layer = np.zeros_like(sat_img)
    red_layer[:, :] = (0, 0, 255) 
    
    # Apply the white parts of the mask to the red layer
    # Only pixels where the mask > 0 will turn red
    road_overlay = cv2.bitwise_and(red_layer, red_layer, mask=mask)
    
    # Blend the original satellite image with the neon red road
    # 0.7 = 70% opacity for satellite, 0.5 = 50% opacity for the red road
    blended = cv2.addWeighted(sat_img, 0.7, road_overlay, 0.5, 0)
    
    # Save the diagnostic image
    out_path = folder / f"{contract_id}_ALIGNMENT_CHECK.png"
    cv2.imwrite(str(out_path), blended)
    print(f"🕵️‍♂️ Diagnostic overlay saved to: {out_path}")
    print("Open it and check if the red line perfectly traces the dirt road!")

if __name__ == "__main__":
    check_alignment()