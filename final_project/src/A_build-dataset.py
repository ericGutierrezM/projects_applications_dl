"""
This code file generates the dataset and downloads the 
satellite images for the project.

Authors: Joshua Castillo and Eric Gutiérrez
Barcelona School of Economics, June 2026
"""

# Imports
import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Credentials and Configuration
with open("SH_CLIENT_ID.txt", "r") as f:
    SH_CLIENT_ID = f.read().strip()

with open("SH_CLIENT_SECRET.txt", "r") as f:
    SH_CLIENT_SECRET = f.read().strip()

SH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Directories
BASE_DIR = Path("")
PROCESSED_DIR = BASE_DIR / "data" / "images"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Image Processing Parameters
PIXELS_PER_SCENE = 256 # 256x256 tensor for the CNN
BUFFER_METERS = 1000 # 1km buffer around the project coordinate
INTERVAL_DAYS = 30 # 1 month steps to guarantee a cloud-free mosaic
MAX_CLOUD_COVER = 100 # Let everything through so the mosaic engine can stitch clear pixels

# Helper function that downloads and filters
# the infrastructure data 
def infrastructure_data():
    data = load_dataset("bettergovph/dpwh-transparency-data")
    raw_df = data["train"].to_pandas()

    df = (raw_df
        .map(lambda x: x.lower() if isinstance(x, str) else x)
        .assign(startDate = lambda x: pd.to_datetime(x.startDate, errors="coerce"),
                completionDate = lambda x: pd.to_datetime(x.completionDate, errors="coerce"),
                province = lambda x: x.location.str.get('province'),
                region = lambda x: x.location.str.get('region'),
                contractorId = lambda x: x.contractor.str.extract(r'\(.*?(\d+).*?\)')
                )

        .loc[lambda x: 
             (x['category'] == 'roads') & 
             (x['description'].str.contains('construction', case=False, na=False)) & 
             (~x['description'].str.contains('reconstruction', case=False, na=False)) &
             (x['startDate'] >= '2019-01-01') &
             (x['completionDate'] - x['startDate'] > pd.Timedelta(days=365))
        ]
        .dropna(subset=["latitude", "longitude", "startDate", "completionDate"])
        .filter(['contractId', 'description', 'category', # project description
                'status', 'progress',  # project progess
                'region', 'province', 'latitude', 'longitude', # project location
                'startDate', 'completionDate', 'infraYear',  # project timeline
                'programName','sourceOfFunds', 'budget', 'amountPaid', 'contractorId' # project financing
                ])
        .copy()
    )
    
    return df

# Helper functions to get the satellite imagery
# for a given infrastructure project
def get_sh_token():
    r = requests.post(SH_TOKEN_URL, data={
        "client_id":     SH_CLIENT_ID,
        "client_secret": SH_CLIENT_SECRET,
        "grant_type":    "client_credentials"
    })
    r.raise_for_status()
    return r.json()["access_token"]

# From a point's lat/long to a bounding box
def point_to_bbox(lat, lon, buffer_m):
    buffer_deg = buffer_m / 111320.0
    min_lon = lon - buffer_deg
    min_lat = lat - buffer_deg
    max_lon = lon + buffer_deg
    max_lat = lat + buffer_deg
    return [min_lon, min_lat, max_lon, max_lat]

# Split the projects in milestone windows
def get_milestone_windows(start_date, end_date, lookback_days=60):
    """
    Calculates the 6 specific project milestones.
    Returns a search window that ends exactly on the milestone date 
    and looks backward by 'lookback_days' to find the most recent clear image.
    """
    duration = end_date - start_date

    milestones = {
        "00_pre_start": start_date - pd.Timedelta(days=30),
        "25_progress": start_date + (duration * 0.25),
        "50_progress": start_date + (duration * 0.50),
        "75_progress": start_date + (duration * 0.75),
        "100_completed": end_date,
        "post_completion": end_date + pd.Timedelta(days=30)
    }

    windows = []
    for milestone_name, target_date in milestones.items():
        # The window ends exactly on the milestone date, and looks backward
        window_start = target_date - pd.Timedelta(days=lookback_days)
        window_end = target_date
        
        # Format for the API
        w_start_str = window_start.strftime("%Y-%m-%d")
        w_end_str = window_end.strftime("%Y-%m-%d")
        
        windows.append((milestone_name, w_start_str, w_end_str))
        
    return windows

# Script
def build_s2_evalscript():
    """Extracts RGB, stretches brightness, and creates a backward-searching mosaic."""
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
          sampleType: "UINT8"
        },
        mosaicking: "ORBIT", 
        mosaickingOrder: "mostRecent" // <-- THIS IS THE MAGIC. It searches backward from the target date!
      };
    }
    
    function evaluatePixel(samples) {
      // Because of 'mostRecent', samples[0] is always the image closest to the milestone date.
      for (let i = 0; i < samples.length; i++) {
        let sample = samples[i];
        
        let isCloudOrShadow = [3, 8, 9, 10].includes(sample.SCL);
        let isNoData = sample.SCL === 0;
        
        // It immediately returns the first (most recent) clear pixel it finds
        if (!isCloudOrShadow && !isNoData) {
            let r = Math.min(255, (sample.B04 / 10000.0) * 2.5 * 255);
            let g = Math.min(255, (sample.B03 / 10000.0) * 2.5 * 255);
            let b = Math.min(255, (sample.B02 / 10000.0) * 2.5 * 255);
            return [r, g, b];
        }
      }
      return [0, 0, 0];
    }
    """

# Download logic
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
                "format": {"type": "image/png"}
            }]
        },
        "evalscript": build_s2_evalscript()
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Accept":        "image/png" 
    }

    r = requests.post(SH_PROCESS_URL, json=payload, headers=headers, timeout=120)
    
    if r.status_code == 204 or r.status_code == 400:
        print("No clear imagery available for this window.")
        return False
    if r.status_code != 200:
        print(f"API Error: {r.text[:200]}")
        return False

    png_out_path = out_path.with_suffix('.png')
    png_out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(png_out_path, 'wb') as f:
        f.write(r.content)

    return True

# Process all 6 images for a single project
def process_single_contract(row, token):
    """Processes all 6 images with a bulletproof retry loop for rate limits."""
    contract_id = row['contractId']
    bbox = point_to_bbox(row['latitude'], row['longitude'], BUFFER_METERS)
    milestones = get_milestone_windows(row['startDate'], row['completionDate'])
    
    project_dir = PROCESSED_DIR / str(contract_id)
    
    with requests.Session() as session:
        for milestone_name, date_from, date_to in milestones:
            out_path = project_dir / f"{contract_id}_{milestone_name}.png"
            
            if out_path.exists():
                continue
            
            payload = {
                "input": {
                    "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                    "data": [{"type": "SENTINEL-2-L2A", "dataFilter": {"timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"}, "maxCloudCoverage": MAX_CLOUD_COVER}}]
                },
                "output": {"width": PIXELS_PER_SCENE, "height": PIXELS_PER_SCENE, "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
                "evalscript": build_s2_evalscript()
            }

            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "image/png"}

            # --- THE RETRY LOOP ---
            max_retries = 4
            for attempt in range(max_retries):
                try:
                    r = session.post(SH_PROCESS_URL, json=payload, headers=headers, timeout=120)
                    
                    # If it works perfectly, save it and break out of the retry loop
                    if r.status_code == 200:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_path, 'wb') as f:
                            f.write(r.content)
                        break 
                    
                    # If we hit the rate limit, sleep and try again
                    elif r.status_code == 429:
                        time.sleep(5 * (attempt + 1)) # Sleeps 5s, then 10s, etc.
                        continue
                        
                    # If there's genuinely no data for that window, give up and move on
                    elif r.status_code == 204 or r.status_code == 400:
                        break
                        
                    else:
                        r.raise_for_status()

                except requests.exceptions.HTTPError as e:
                    # Catch expired tokens
                    if e.response.status_code == 401: 
                        token = get_sh_token()
                        headers["Authorization"] = f"Bearer {token}"
                        continue # Try the request again with the new token
                    else:
                        # For any other weird HTTP error, break the retry loop
                        break

    return contract_id

# Run the pipeline
def run_pipeline(contract_df):
    """Uses multithreading to dramatically speed up the download loop."""
    token = get_sh_token()
    
    MAX_CONCURRENT_PROJECTS = 3
    
    rows_to_process = [row for _, row in contract_df.iterrows()]

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROJECTS) as executor:
        # Submit all projects to the thread pool
        futures = {executor.submit(process_single_contract, row, token): row for row in rows_to_process}
        
        for future in tqdm(as_completed(futures), total=len(rows_to_process), desc="Downloading Projects"):
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"Critical error on a thread: {e}")

# Counts the successfully downloaded PNGs in a directory 
def count_images(folder_path):
    folder = Path(folder_path)
    if folder.exists():
        return len(list(folder.glob("*.png")))
    return 0

# Execution
if __name__ == "__main__":
    df = infrastructure_data()
    print(f"Total valid projects found: {len(df)}")
    
    print("\nStarting satellite download pipeline...")
    run_pipeline(df)

    print("\nLinking image directories to the dataset...")
    df['image_folder_path'] = df['contractId'].apply(lambda x: str(PROCESSED_DIR / str(x)))
    df['downloaded_image_count'] = df['image_folder_path'].apply(count_images)

    print("\nCalculating and appending milestone dates to the dataset...")
    
    # Calculate the total duration of the project
    duration = df['completionDate'] - df['startDate']
    
    # Create 6 new columns with the exact dates for each of the images (formatted as standard strings)
    df['date_00_pre_start'] = (df['startDate'] - pd.Timedelta(days=30)).dt.strftime('%Y-%m-%d')
    df['date_25_progress'] = (df['startDate'] + (duration * 0.25)).dt.strftime('%Y-%m-%d')
    df['date_50_progress'] = (df['startDate'] + (duration * 0.50)).dt.strftime('%Y-%m-%d')
    df['date_75_progress'] = (df['startDate'] + (duration * 0.75)).dt.strftime('%Y-%m-%d')
    df['date_100_completed'] = df['completionDate'].dt.strftime('%Y-%m-%d')
    df['date_post_completion'] = (df['completionDate'] + pd.Timedelta(days=30)).dt.strftime('%Y-%m-%d')

    final_csv_path = "data/dataset_w_images.csv"
    df.to_csv(final_csv_path, index=False)
    
    print(f"\nDone! Enriched dataset saved to {final_csv_path}")