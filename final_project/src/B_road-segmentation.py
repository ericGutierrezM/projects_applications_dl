"""
This code file generates road masks using a vector
of the road infrastructure in the Philippines.

Authors: Joshua Castillo and Eric Gutiérrez
Barcelona School of Economics, June 2026
"""

# Imports
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import requests
import os
import zipfile
from tqdm import tqdm
import cv2

# Helper function that downloads the Philippines
# road infrastructure as a SHP 
def download_and_extract_shapefile(url, extract_dir):
    zip_path = os.path.join(extract_dir, "philippines_shapefile.zip")
    shp_path = os.path.join(extract_dir, "gis_osm_roads_free_1.shp")
    
    if os.path.exists(shp_path):
        print(f"Shapefile already extracted at {shp_path}. Skipping download.")
        return shp_path

    os.makedirs(extract_dir, exist_ok=True)
    print(f"Downloading OpenStreetMap Shapefiles...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    progress = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading")
    
    with open(zip_path, 'wb') as file:
        for data in response.iter_content(1024 * 1024):
            progress.update(len(data))
            file.write(data)
    progress.close()
    
    print("Extracting Shapefiles (This takes a few seconds)...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    print("Download and extraction complete!")
    return shp_path

# Mask configuration
PIXELS_PER_SCENE = 256
LINE_THICKNESS_PIXELS = 5
BUFFER_METERS = 1000 # 1000m radius creates a 2km x 2km bounding box

# Helper function to generate a bounding box
# from lat/lon 
def get_bbox(lat, lon, buffer_meters):
    lat_offset = buffer_meters / 111320.0
    lon_offset = buffer_meters / (111320.0 * np.cos(np.radians(lat)))
    return [lon - lon_offset, lat - lat_offset, lon + lon_offset, lat + lat_offset]

# Helper function to transform lat/long to
# pixels 
def latlon_to_pixel(lat, lon, bbox, img_size):
    min_lon, min_lat, max_lon, max_lat = bbox
    x = int(((lon - min_lon) / (max_lon - min_lon)) * img_size)
    y = int(((max_lat - lat) / (max_lat - min_lat)) * img_size)
    return x, y

# Rasteriztion.: from vector to raster mask
def run_geopandas_rasterizer(csv_path, shp_path):
    df = pd.read_csv(csv_path)
    
    print("Loading road geometries into RAM...")
    print("This takes about 30 to 60 seconds and uses a safe amount of memory...")
    
    all_roads_gdf = gpd.read_file(shp_path)
    
    print(f"Loaded {len(all_roads_gdf)} road segments.")
    
    mask_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rasterizing Masks"):
        contract_id = row['contractId']
        lat = row['latitude']
        lon = row['longitude']
        
        folder_path = Path(row['image_folder_path'])
        mask_out_path = folder_path / f"{contract_id}_road_mask.png"
        
        if mask_out_path.exists():
            mask_paths.append(str(mask_out_path))
            continue
            
        bbox = get_bbox(lat, lon, BUFFER_METERS)
        min_lon, min_lat, max_lon, max_lat = bbox
        
        try:
            proj_roads = all_roads_gdf.cx[min_lon:max_lon, min_lat:max_lat]
            
            mask = np.zeros((PIXELS_PER_SCENE, PIXELS_PER_SCENE), dtype=np.uint8)
            
            if not proj_roads.empty:
                for _, road in proj_roads.iterrows():
                    geom = road.geometry
                    if geom is None:
                        continue
                        
                    if geom.geom_type == 'LineString':
                        lines = [geom]
                    elif geom.geom_type == 'MultiLineString':
                        lines = geom.geoms
                    else:
                        continue
                        
                    for line in lines:
                        pts = []
                        for coord in line.coords:
                            px, py = latlon_to_pixel(coord[1], coord[0], bbox, PIXELS_PER_SCENE)
                            pts.append([px, py])
                        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=LINE_THICKNESS_PIXELS)
            else:
                tqdm.write(f"No roads found in OSM for {contract_id}. Saving blank mask.")
            
            folder_path.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(mask_out_path), mask)
            
            if not success:
                raise RuntimeError(f"OpenCV silently failed to write the image to: {mask_out_path}")
                
            mask_paths.append(str(mask_out_path))
            
        except Exception as e:
            tqdm.write(f"Error on {contract_id}: {e}")
            mask_paths.append(None)

    df['osm_mask_path'] = mask_paths
    df.to_csv("data/dataset_w_roads.csv", index=False)
    print("\nGround truth rasterization complete!")

if __name__ == "__main__":
    csv_file = "data/dataset_w_images.csv"
    shp_url = "https://download.geofabrik.de/asia/philippines-latest-free.shp.zip"
    extract_directory = "data/philippines_shp"
    
    # Download and extract
    download_and_extract_shapefile(shp_url, extract_directory)
    roads_shapefile_path = os.path.join(extract_directory, "gis_osm_roads_free_1.shp")
    
    # Rasterization
    run_geopandas_rasterizer(csv_file, roads_shapefile_path)