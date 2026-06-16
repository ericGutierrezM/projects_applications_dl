import os
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from pathlib import Path
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────
# 1. CREDENTIALS & CONFIGURATION
# ─────────────────────────────────────────────────────────
# Replace with your Copernicus Data Space credentials
with open("SH_CLIENT_ID.txt", "r") as f:
    SH_CLIENT_ID = f.read().strip()

with open("SH_CLIENT_SECRET.txt", "r") as f:
    SH_CLIENT_SECRET = f.read().strip()

SH_TOKEN_URL   = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Directories
BASE_DIR      = Path("")
PROCESSED_DIR = BASE_DIR / "processed_s2_tensors"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Processing Parameters
PIXELS_PER_SCENE = 256     # 256x256 is standard for CNNs (e.g., ResNet)
BUFFER_METERS    = 1000    # 1km buffer around the project coordinate
INTERVAL_DAYS    = 60      # 1 month steps to guarantee a cloud-free composite
MAX_CLOUD_COVER  = 25      # Server-side filter to drop totally overcast days

# ─────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────
def get_sh_token():
    r = requests.post(SH_TOKEN_URL, data={
        "client_id":     SH_CLIENT_ID,
        "client_secret": SH_CLIENT_SECRET,
        "grant_type":    "client_credentials"
    })
    r.raise_for_status()
    return r.json()["access_token"]

def point_to_bbox(lat, lon, buffer_m):
    """Converts a center lat/lon and buffer into a bounding box."""
    # 1 degree of latitude is ~111,320 meters
    buffer_deg = buffer_m / 111320.0
    min_lon = lon - buffer_deg
    min_lat = lat - buffer_deg
    max_lon = lon + buffer_deg
    max_lat = lat + buffer_deg
    return [min_lon, min_lat, max_lon, max_lat]

def date_intervals(start_str, end_str, step_days=INTERVAL_DAYS):
    """Splits the project timeline into monthly windows."""
    fmt   = "%Y-%m-%d"
    start = datetime.strptime(start_str[:10], fmt)
    end   = datetime.strptime(end_str[:10], fmt)
    intervals = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=step_days), end)
        intervals.append((cur.strftime(fmt), nxt.strftime(fmt)))
        cur = nxt
    return intervals

def build_s2_evalscript():
    """Extracts RGB, applies a brightness stretch, and outputs standard 8-bit images."""
    return """
    //VERSION=3
    function setup() {
      return {
        input: [{
          bands: ["B04", "B03", "B02", "SCL"],
          units: "DN"
        }],
        output: {
          bands: 3,
          sampleType: "UINT8" // <-- Changed to standard 8-bit format
        }
      };
    }
    function evaluatePixel(sample) {
      let isCloudOrShadow = [3, 8, 9, 10].includes(sample.SCL);
      
      if (isCloudOrShadow) {
        return [0, 0, 0]; 
      } else {
        // 1. Convert DN to reflectance (/ 10000.0)
        // 2. Apply 2.5x brightness stretch
        // 3. Scale to 255 for standard RGB
        // Math.min ensures we don't exceed 255 (prevents visual glitching)
        let r = Math.min(255, (sample.B04 / 10000.0) * 2.5 * 255);
        let g = Math.min(255, (sample.B03 / 10000.0) * 2.5 * 255);
        let b = Math.min(255, (sample.B02 / 10000.0) * 2.5 * 255);
        
        return [r, g, b];
      }
    }
    """

# ─────────────────────────────────────────────────────────
# 3. SENTINEL HUB DOWNLOAD LOGIC
# ─────────────────────────────────────────────────────────
def download_s2_scene(token, bbox, date_from, date_to, out_path):
    min_lon, min_lat, max_lon, max_lat = bbox

    payload = {
        "input": {
            "bounds": {
                "bbox": [min_lon, min_lat, max_lon, max_lat],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "SENTINEL-2-L2A",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{date_from}T00:00:00Z",
                        "to":   f"{date_to}T23:59:59Z"
                    },
                    "maxCloudCoverage": MAX_CLOUD_COVER
                }
            }]
        },
        "output": {
            "width":  PIXELS_PER_SCENE,
            "height": PIXELS_PER_SCENE,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/png"} # <-- Changed from tiff to PNG
            }]
        },
        "evalscript": build_s2_evalscript()
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Accept":        "image/png" # <-- Changed from tiff to PNG
    }

    r = requests.post(SH_PROCESS_URL, json=payload, headers=headers, timeout=120)
    
    if r.status_code == 204 or r.status_code == 400:
        print("      → No clear imagery available for this window.")
        return False
    if r.status_code != 200:
        print(f"      → API Error: {r.text[:200]}")
        return False

    # Force the file extension to be .png
    png_out_path = out_path.with_suffix('.png')
    png_out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Just save the PNG bytes directly to your hard drive
    with open(png_out_path, 'wb') as f:
        f.write(r.content)

    print(f"      ✅ Saved: {png_out_path.name}")
    return True

# ─────────────────────────────────────────────────────────
# 4. MAIN PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────
def run_pipeline(contract_df):
    token = get_sh_token()
    
    for idx, row in contract_df.iterrows():
        project_id = row['project_id']
        print(f"\n{'='*50}\nProcessing Project ID: {project_id}\n{'='*50}")
        
        bbox = point_to_bbox(row['lat'], row['lon'], BUFFER_METERS)
        intervals = date_intervals(row['start_date'], row['end_date'])
        
        project_dir = PROCESSED_DIR / str(project_id)
        
        for date_from, date_to in intervals:
            out_path = project_dir / f"{date_from}_{project_id}_RGB.tif"
            
            if out_path.exists():
                print(f"  ⏭ Cached: {out_path.name}")
                continue
                
            print(f"  ⬇ Window: {date_from} → {date_to}")
            
            try:
                # Attempt download; if token expired, refresh and retry
                success = download_s2_scene(token, bbox, date_from, date_to, out_path)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401: 
                    token = get_sh_token()
                    success = download_s2_scene(token, bbox, date_from, date_to, out_path)
                else:
                    print(f"      ❌ Network Error: {e}")

# ==========================================
# EXAMPLE EXECUTION
# ==========================================
if __name__ == "__main__":
    # Mock dataframe simulating your clean infrastructure dataset
    mock_data = pd.DataFrame({
        'project_id': ['PROJ_WB_1001', 'PROJ_WB_1002'],
        'lat': [11.9934, 4.6097],
        'lon': [105.4645, -74.0817],
        'start_date': ['2020-01-01', '2022-05-01'],
        'end_date': ['2021-06-01', '2024-03-01']
    })
    
    run_pipeline(mock_data)