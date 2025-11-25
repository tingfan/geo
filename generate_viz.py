import rasterio
import numpy as np
import matplotlib.colors as mcolors
import json
import os

def generate_visualization(input_tif, output_html, subsample_skip=5):
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
    output_dir = os.path.dirname(os.path.abspath(output_html))
    points_bin_path = os.path.join(output_dir, 'points.bin')
    colors_bin_path = os.path.join(output_dir, 'colors.bin')
    
    print(f"Saving binary data to {points_bin_path} and {colors_bin_path}...")
    points.astype(np.float32).tofile(points_bin_path)
    rgb.astype(np.float32).tofile(colors_bin_path)

    # write path
    path_points = np.zeros((10,3), np.float32)
    path_points[0]= [6.9964438e+05, 4.0342012e+06, 6.5988776e+02]
    path_rgb = np.ones((10,3), np.uint8) * 255
    path_points.astype(np.float32).tofile(os.path.join(output_dir, 'path_points.bin'))
    path_rgb.astype(np.float32).tofile(os.path.join(output_dir, 'path_colors.bin'))
 
    
    # Read template
    with open('viz_template.html', 'r') as f:
        template = f.read()
        
    # Write HTML (no injection needed)
    print("Generating HTML...")
    # We just write the template as is, the JS will load the bin files
    # We might want to ensure the template doesn't have the placeholders anymore or they are ignored
    # The new template won't have them.
    
    with open(output_html, 'w') as f:
        f.write(template)
        
    print(f"Visualization saved to {output_html}")

if __name__ == "__main__":
    input_file = "data/msn-GD7KCzDIxA_VEGETATION_DEM.tif"
    output_file = "output/terrain_viz.html"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
    else:
        generate_visualization(input_file, output_file)
