import numpy as np
import rasterio
from rasterio.transform import from_origin
import argparse
import os

def generate_mock_terrain(width_m, height_m, resolution, output_file):
    """
    Generates a mockup GeoTIFF terrain using IDW interpolation from random points.
    
    Args:
        width_m (float): Width of terrain in meters
        height_m (float): Height of terrain in meters
        resolution (float): Resolution in meters/pixel
        output_file (str): Output filename
    """
    # Calculate grid dimensions
    cols = int(np.ceil(width_m / resolution))
    rows = int(np.ceil(height_m / resolution))
    
    print(f"Generating terrain: {width_m}m x {height_m}m at {resolution}m/px")
    print(f"Grid dimensions: {cols} x {rows} ({cols*rows/1e6:.1f} M pixels)")
    
    # Generate 10 random control points
    # x in [0, width_m], y in [0, height_m]
    n_points = 20
    control_x = np.random.uniform(0, width_m, n_points)
    control_y = np.random.uniform(0, height_m, n_points)
    control_z = np.random.uniform(10, 15, n_points)
    
    print("Control points:")
    for i in range(n_points):
        print(f"  Point {i+1}: ({control_x[i]:.1f}, {control_y[i]:.1f}) -> {control_z[i]:.2f}m")
    
    # Define transform (Top-Left at 0,0)
    # Note: In GeoTIFF, y usually decreases as row index increases (North-Up image)
    # So we map row 0 to y=height_m, row N to y=0
    transform = from_origin(0, height_m, resolution, resolution)
    
    # Prepare metadata
    meta = {
        'driver': 'GTiff',
        'height': rows,
        'width': cols,
        'count': 1,
        'dtype': 'float32',
        'crs': '+proj=latlong', # Dummy CRS
        'transform': transform,
        'nodata': -9999
    }
    
    # Process in chunks to save memory
    chunk_rows = 100 # Process 100 rows at a time
    
    print("Writing to file...")
    with rasterio.open(output_file, 'w', **meta) as dst:
        for r_start in range(0, rows, chunk_rows):
            r_end = min(r_start + chunk_rows, rows)
            n_rows_chunk = r_end - r_start
            
            # Create meshgrid for this chunk
            # Row indices: r_start to r_end-1
            # Col indices: 0 to cols-1
            
            # Convert pixel indices to world coordinates
            # We use the transform to get coordinates for the center of pixels
            # But for simple IDW on a grid, we can just work in local grid coordinates scaled by resolution
            # Let's stick to world coordinates to match control points
            
            # Generate y coordinates for these rows
            # y = top - (row + 0.5) * res
            y_indices = np.arange(r_start, r_end)
            chunk_y = height_m - (y_indices + 0.5) * resolution
            
            # Generate x coordinates
            x_indices = np.arange(cols)
            chunk_x = (x_indices + 0.5) * resolution
            
            # Create 2D grid for chunk
            X, Y = np.meshgrid(chunk_x, chunk_y)
            
            # Inverse Distance Weighting (IDW)
            # Z = sum(z_i / d_i^p) / sum(1 / d_i^p)
            # using p=2
            
            numerator = np.zeros(X.shape, dtype=np.float32)
            denominator = np.zeros(X.shape, dtype=np.float32)
            
            epsilon = 1e-6 # Avoid division by zero
            
            for i in range(n_points):
                # Squared Euclidean distance
                dist_sq = (X - control_x[i])**2 + (Y - control_y[i])**2
                
                # Weight = 1 / dist^2
                # Add epsilon to dist_sq to avoid infinity
                weights = 1.0 / (dist_sq + epsilon)
                
                numerator += weights * control_z[i]
                denominator += weights
            
            chunk_data = numerator / denominator
            
            # Write chunk
            dst.write(chunk_data.astype(np.float32), 1, window=((r_start, r_end), (0, cols)))
            
            # Progress
            if r_end % 1000 == 0 or r_end == rows:
                print(f"  Processed {r_end}/{rows} rows ({(r_end/rows)*100:.1f}%)")

    print(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a mockup GeoTIFF terrain.")
    parser.add_argument("--width", type=float, default=50.0, help="Width in meters (default: 10)")
    parser.add_argument("--height", type=float, default=30.0, help="Height in meters (default: 10)")
    parser.add_argument("--resolution", type=float, default=0.1, help="Resolution in meters/pixel (default: 0.01)")
    parser.add_argument("--output", type=str, default="data/mock_terrain.tif", help="Output filename")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    generate_mock_terrain(args.width, args.height, args.resolution, args.output)
