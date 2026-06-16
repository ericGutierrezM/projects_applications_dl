import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def run_mathematical_road_extraction(csv_path):
    df = pd.read_csv(csv_path)

    df = df.head(1).copy()
    mask_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="📐 Calculating Road Vectors"):
        contract_id = row['contractId']
        folder_path = Path(row['image_folder_path'])
        
        img_path = folder_path / f"{contract_id}_post_completion.png"
        mask_out_path = folder_path / f"{contract_id}_road_mask.png"
        
        if not img_path.exists():
            mask_paths.append(None)
            continue
            
        try:
            # 1. Load Image and convert to Grayscale
            # OpenCV loads images in BGR format by default
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Gaussian Blur (CRUCIAL: Smooths out forest texture noise)
            # A 5x5 or 7x7 kernel works best for Sentinel-2
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # 3. Canny Edge Detection (Finds harsh contrast boundaries)
            edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
            
            # 4. Probabilistic Hough Transform (Finds the straight lines)
            # - minLineLength: Ignores short random lines (like a house roof)
            # - maxLineGap: Connects the road even if a tree shadow breaks the line
            lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=30, 
                minLineLength=40, 
                maxLineGap=25
            )
            
            # 5. Draw the mathematical vectors onto a pure black mask
            mask = np.zeros_like(gray)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Draw a thin white line
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
            
            # ─────────────────────────────────────────────────────────
            # 6. NEW: MORPHOLOGICAL CLEANUP (The "Melting" Phase)
            # ─────────────────────────────────────────────────────────
            # Create a 15x15 pixel block to inflate the lines
            kernel = np.ones((15, 15), np.uint8)
            
            # Dilate (Inflate) the lines so they fuse together
            fused_mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Apply a heavy blur to smooth out the blocky edges
            smoothed = cv2.GaussianBlur(fused_mask, (15, 15), 0)
            
            # Snap it back to a crisp, binary Black/White mask
            _, final_mask = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
            
            # Save the cleanly fused mask
            cv2.imwrite(str(mask_out_path), final_mask)
            mask_paths.append(str(mask_out_path))
            
        except Exception as e:
            tqdm.write(f"Error processing {contract_id}: {e}")
            mask_paths.append(None)

    df['road_mask_path'] = mask_paths
    df.to_csv('data/dataset_w_roads.csv', index=False)
    print("Mathematical vector extraction complete!")

if __name__ == "__main__":
    run_mathematical_road_extraction("data/dataset_w_images.csv")