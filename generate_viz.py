import rasterio
import numpy as np
import matplotlib.colors as mcolors
import json
import os

def generate_visualization(input_tif, output_dir='output', subsample_skip=5):
    print(f"Reading {input_tif}...")
    
    with rasterio.open(input_tif) as src:
        # Read the first band
        data = src.read(1)
        
        # Get nodata value
        nodata = src.nodata
        
        # Create mask for valid data
        if nodata is not None:
            mask = data != nodata
        else:
            mask = ~np.isnan(data)

        subsample = np.zeros_like(mask, np.bool)
        subsample[::subsample_skip, ::subsample_skip]=True
        mask = mask & subsample
            
        # Get coordinates
        rows, cols = np.where(mask)
        values = data[rows, cols]
        
        # Transform to x, y coordinates
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset='center')
        xs = np.array(xs)
        ys = np.array(ys)
        
        # Stack into (N, 3) array
        points = np.column_stack((xs, ys, values))
        
        print(f"Total valid points: {len(points)}")
        
            
    # Calculate colors based on height
    z = points[:, 2]
    
    # Calculate p5 and p95
    p5 = np.percentile(z, 5)
    p95 = np.percentile(z, 95)
    print(f"Height range: {z.min():.2f} to {z.max():.2f}")
    print(f"Color mapping range (p5-p95): {p5:.2f} to {p95:.2f}")
    
    # Normalize z to 0-1 range based on p5 and p95
    z_norm = np.clip((z - p5) / (p95 - p5), 0, 1)
    
    # Map to HSV (Hue)
    # Blue (low) to Red (high).
    hues = 0.66 - (z_norm * 0.66)
    
    # Saturation and Value = 1
    s = np.ones_like(hues)
    v = np.ones_like(hues)
    
    # Convert to RGB
    hsv = np.column_stack((hues, s, v))
    rgb = mcolors.hsv_to_rgb(hsv)
    
    # Export binary data
    points_bin_path = os.path.join(output_dir, 'points.bin')
    colors_bin_path = os.path.join(output_dir, 'colors.bin')
    
    print(f"Saving binary data to {points_bin_path} and {colors_bin_path}...")
    points.astype(np.float32).tofile(points_bin_path)
    rgb.astype(np.float32).tofile(colors_bin_path)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D visualization data from GeoTIFF.")
    parser.add_argument("--input", type=str, default="data/msn-GD7KCzDIxA_VEGETATION_DEM.tif", help="Input GeoTIFF file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--subsample", type=int, default=5, help="Subsample skip factor (default: 5)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        generate_visualization(args.input, args.output_dir, args.subsample)
