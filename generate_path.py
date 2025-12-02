import rasterio
import numpy as np
import heapq
from collections import defaultdict
import argparse
import os
import matplotlib.pyplot as plt

def create_height_map(data, transform, nodata, resolution=0.2):
    """
    Create a coarse 2D height map at given resolution using vectorized operations.
    
    Args:
        data: 2D array of elevation data
        transform: rasterio transform
        nodata: nodata value
        resolution: grid resolution in meters (default 0.2m)
    
    Returns:
        height_map: 2D array of mean heights
        grid_transform: transform for the coarse grid
        valid_mask: boolean mask of valid cells
    """
    print(f"Creating height map with resolution {resolution}m...")
    
    # Create mask for valid data
    if nodata is not None:
        mask = data != nodata
    else:
        mask = ~np.isnan(data)
    
    # Get extent in world coordinates
    height, width = data.shape
    
    # Get corner coordinates
    xmin, ymax = transform * (0, 0)
    xmax, ymin = transform * (width, height)
    
    # Calculate grid dimensions
    grid_width = int(np.ceil((xmax - xmin) / resolution))
    grid_height = int(np.ceil((ymax - ymin) / resolution))
    
    print(f"Grid dimensions: {grid_width} x {grid_height}")
    
    # Initialize height map
    height_map = np.full((grid_height, grid_width), np.nan, dtype=np.float32)
    
    # Get indices of valid pixels
    rows, cols = np.where(mask)
    values = data[rows, cols]
    
    if len(values) == 0:
        print("No valid data points found.")
        from rasterio.transform import from_origin
        grid_transform = from_origin(xmin, ymax, resolution, resolution)
        return height_map, grid_transform, np.zeros_like(height_map, dtype=bool)

    print(f"Processing {len(values)} valid points...")

    # Vectorized coordinate transformation
    # rasterio.transform.xy returns lists, convert to arrays
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Vectorized grid index calculation
    grid_cols = ((xs - xmin) / resolution).astype(int)
    grid_rows = ((ymax - ys) / resolution).astype(int)
    
    # Filter out-of-bounds points (just in case)
    valid_indices_mask = (
        (grid_rows >= 0) & (grid_rows < grid_height) &
        (grid_cols >= 0) & (grid_cols < grid_width)
    )
    
    if not np.all(valid_indices_mask):
        grid_rows = grid_rows[valid_indices_mask]
        grid_cols = grid_cols[valid_indices_mask]
        values = values[valid_indices_mask]
    
    # Efficient aggregation using sorting
    # 1. Flatten 2D grid indices to 1D
    flat_indices = grid_rows * grid_width + grid_cols
    
    # 2. Sort values by flat index
    sort_idx = np.argsort(flat_indices)
    sorted_indices = flat_indices[sort_idx]
    sorted_values = values[sort_idx]
    
    # 3. Find boundaries where indices change
    # unique_indices are the flat indices of cells that have data
    # split_indices are the indices in sorted_values where the cell changes
    unique_indices, split_indices = np.unique(sorted_indices, return_index=True)
    
    # 4. Compute mean using reduceat (much faster than split + median)
    # reduceat sums slices of the array defined by indices
    sums = np.add.reduceat(sorted_values, split_indices)
    
    # Calculate counts for each cell
    # We append the total length to get the end of the last group
    counts = np.diff(np.append(split_indices, len(sorted_values)))
    
    # Compute means
    means = sums / counts
    
    # Map back to 2D height map
    # unique_indices contains the flat index for each mean
    rows_out = unique_indices // grid_width
    cols_out = unique_indices % grid_width
    
    height_map[rows_out, cols_out] = means
    
    # Create valid mask
    valid_mask = ~np.isnan(height_map)
    
    # Create transform for the grid
    from rasterio.transform import from_origin
    grid_transform = from_origin(xmin, ymax, resolution, resolution)
    
    print(f"Valid cells: {valid_mask.sum()} / {grid_height * grid_width}")
    
    return height_map, grid_transform, valid_mask

def world_to_grid(x, y, grid_transform, resolution):
    """Convert world coordinates to grid indices."""
    # Get grid origin
    xmin, ymax = grid_transform * (0, 0)
    
    grid_col = int((x - xmin) / resolution)
    grid_row = int((ymax - y) / resolution)
    
    return grid_row, grid_col

def grid_to_world(grid_row, grid_col, grid_transform, resolution):
    """Convert grid indices to world coordinates."""
    xmin, ymax = grid_transform * (0, 0)
    
    x = xmin + (grid_col + 0.5) * resolution
    y = ymax - (grid_row + 0.5) * resolution
    
    return x, y

def dijkstra_search(height_map, valid_mask, start_idx, end_idx, height_threshold=0.2):
    """
    Dijkstra search on the height map.
    
    Args:
        height_map: 2D array of heights
        valid_mask: boolean mask of valid cells
        start_idx: (row, col) tuple for start
        end_idx: (row, col) tuple for end
        height_threshold: maximum height difference for connectivity (default 0.3m)
    
    Returns:
        path: list of (row, col) tuples representing the path
        reached_end: boolean indicating if end was reached
        visited: set of (row, col) tuples of visited nodes
    """
    print(f"Running Dijkstra from {start_idx} to {end_idx}...")
    
    rows, cols = height_map.shape
    
    # Priority queue: (distance, row, col)
    pq = [(0, start_idx[0], start_idx[1])]
    
    # Track distances and previous nodes
    distances = {start_idx: 0}
    previous = {}
    visited = set()
    
    # 4-connectivity (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while pq:
        dist, row, col = heapq.heappop(pq)
        
        if (row, col) in visited:
            continue
        
        visited.add((row, col))
        
        # Check if we reached the end
        if (row, col) == end_idx:
            print(f"Path found! Distance: {dist:.2f}")
            break
        
        current_height = height_map[row, col]
        
        # Explore neighbors
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if not (0 <= new_row < rows and 0 <= new_col < cols):
                continue
            
            # Check if valid cell
            if not valid_mask[new_row, new_col]:
                continue
            
            neighbor_height = height_map[new_row, new_col]
            
            # Check height difference
            if abs(neighbor_height - current_height) > height_threshold:
                continue

            # check clearance
            
            # Calculate distance (Euclidean + height penalty)
            height_diff = abs(neighbor_height - current_height)
            edge_cost = 1.0 + height_diff  # Base cost + height penalty
            
            new_dist = dist + edge_cost
            
            if (new_row, new_col) not in distances or new_dist < distances[(new_row, new_col)]:
                distances[(new_row, new_col)] = new_dist
                previous[(new_row, new_col)] = (row, col)
                heapq.heappush(pq, (new_dist, new_row, new_col))
    
    # Reconstruct path (from end to start, then reverse)
    path = []
    current = end_idx if end_idx in previous or end_idx == start_idx else None
    
    # If end not reached, find the closest visited node
    reached_end = end_idx in visited
    
    if not reached_end and visited:
        print("End not reached, finding closest visited point...")
        # Find the visited node closest to end
        min_dist = float('inf')
        closest = None
        for node in visited:
            dist = abs(node[0] - end_idx[0]) + abs(node[1] - end_idx[1])
            if dist < min_dist:
                min_dist = dist
                closest = node
        current = closest
        print(f"Closest point to end: {closest}, distance: {min_dist}")
    
    # Reconstruct path
    if current:
        while current in previous:
            path.append(current)
            current = previous[current]
        path.append(start_idx)
        path.reverse()
    
    print(f"Path length: {len(path)} cells")
    
    return path, reached_end, visited

def save_height_map_visualization(height_map, valid_mask, output_dir, path_points=None, visited_points=None, cli_command=None):
    """
    Save the height map as a PNG image with color encoding and colorbar.
    
    Args:
        height_map: 2D array of heights
        valid_mask: boolean mask of valid cells
        output_dir: directory to save output
        path_points: optional list of (row, col) tuples to overlay the path
        visited_points: optional set/list of (row, col) tuples to overlay visited points
        cli_command: optional CLI command string to include in title
    """
    print("Generating height map visualization...")
    
    # Create a masked array to handle NaNs/invalid cells properly
    masked_height_map = np.ma.masked_where(~valid_mask, height_map)
    
    plt.figure(figsize=(12, 8))
    # Use a terrain colormap or viridis
    img = plt.imshow(masked_height_map, cmap='terrain', interpolation='nearest')
    cbar = plt.colorbar(img, label='Height (m)')
    title = 'Terrain Height Map'
    if cli_command:
        title += f'\n{cli_command}'
    plt.title(title, fontsize=8)
    plt.xlabel('Grid Column')
    plt.ylabel('Grid Row')
    
    if visited_points:
        # Separate rows and cols
        visited_list = list(visited_points)
        np.random.shuffle(visited_list)
        visited_list = visited_list[:10000]
        visited_rows = [p[0] for p in visited_list]
        visited_cols = [p[1] for p in visited_list]
        # Plot visited points as small semi-transparent dots
        plt.scatter(visited_cols, visited_rows, marker='.', s=0.001,color='black', alpha=0.1, label='Visited')
    
    if path_points:
        # Separate rows and cols
        path_rows = [p[0] for p in path_points]
        path_cols = [p[1] for p in path_points]
        plt.plot(path_cols, path_rows, 'r-', linewidth=2, label='Path')
    
    if path_points or visited_points:
        plt.legend()
    
    output_path = os.path.join(output_dir, 'height_map.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Height map visualization saved to {output_path}")

def generate_path(input_tif, output_dir, start, end, resolution, height_threshold, cli_command=None):
    """
    Generate a path from start to end on the terrain.
    
    Args:
        input_tif: path to input GeoTIFF file
        output_dir: directory to save output files
        start: (x, y) tuple for start point in world coordinates
        end: (x, y) tuple for end point in world coordinates
        resolution: grid resolution in meters
        height_threshold: maximum height difference for connectivity
    """
    print(f"Reading {input_tif}...")
    
    with rasterio.open(input_tif) as src:
        # Read the first band (no subsampling)
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata
        
        # Create height map
        height_map, grid_transform, valid_mask = create_height_map(data, transform, nodata, resolution)
        
        # Save height map visualization (initial)
        save_height_map_visualization(height_map, valid_mask, output_dir, cli_command=cli_command)
        
        # Convert world coordinates to grid indices
        start_idx = world_to_grid(start[0], start[1], grid_transform, resolution)
        end_idx = world_to_grid(end[0], end[1], grid_transform, resolution)
        
        print(f"Start grid: {start_idx}, End grid: {end_idx}")
        
        # Validate indices
        rows, cols = height_map.shape
        if not (0 <= start_idx[0] < rows and 0 <= start_idx[1] < cols and valid_mask[start_idx]):
            print(f"Error: Start point {start_idx} is invalid!")
            return
        
        if not (0 <= end_idx[0] < rows and 0 <= end_idx[1] < cols and valid_mask[end_idx]):
            print(f"Error: End point {end_idx} is invalid!")
            return
        
        # Run Dijkstra
        path, reached_end, visited = dijkstra_search(height_map, valid_mask, start_idx, end_idx, height_threshold)
        
        if not path:
            print("No path found!")
            # Save visualization with visited points if no path found
            save_height_map_visualization(height_map, valid_mask, output_dir, path_points=None, visited_points=visited, cli_command=cli_command)
            return
        
        # Convert path to world coordinates with heights
        path_points = []
        for grid_row, grid_col in path:
            x, y = grid_to_world(grid_row, grid_col, grid_transform, resolution)
            z = height_map[grid_row, grid_col]
            path_points.append([x, y, z])
        
        path_points = np.array(path_points, dtype=np.float32)
        
        print(f"Path has {len(path_points)} points")
        print(f"Height range: {path_points[:, 2].min():.2f} to {path_points[:, 2].max():.2f}")
        
        # Save path to binary file
        path_bin_path = os.path.join(output_dir, 'path_points.bin')
        path_points.tofile(path_bin_path)
        print(f"Path saved to {path_bin_path}")
        
        # Also save as text for debugging
        path_txt_path = os.path.join(output_dir, 'path_points.txt')
        np.savetxt(path_txt_path, path_points, fmt='%.6f', header='x y z')
        print(f"Path (text) saved to {path_txt_path}")
        
        # Save visualization with path
        # If end not reached, also show visited points
        save_height_map_visualization(height_map, valid_mask, output_dir, path, visited, cli_command=cli_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate path on terrain using Dijkstra')
    parser.add_argument('--input', type=str, default="data/msn-GD7KCzDIxA_VEGETATION_DEM.tif",
                        help='Input GeoTIFF file')
    parser.add_argument('--output-dir', type=str, default="output",
                        help='Output directory')
    parser.add_argument('--start', type=float, nargs=2, metavar=('X', 'Y'),
                        help='Start point (x, y) in world coordinates')
    parser.add_argument('--end', type=float, nargs=2, metavar=('X', 'Y'),
                        help='End point (x, y) in world coordinates')
    parser.add_argument('--resolution', type=float, default=0.5,
                        help='Grid resolution in meters (default: 0.5)')
    parser.add_argument('--height-threshold', type=float, default=0.3,
                        help='Maximum height difference for connectivity (default: 0.3)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        
        start = tuple(args.start) if args.start else None
        end = tuple(args.end) if args.end else None
        
        # Build CLI command string for display
        cli_command = f"python generate_path.py --start {args.start[0]} {args.start[1]} --end {args.end[0]} {args.end[1]} --resolution {args.resolution} --height-threshold {args.height_threshold}"
        
        generate_path(args.input, args.output_dir, start, end, args.resolution, args.height_threshold, cli_command=cli_command)
