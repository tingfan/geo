import os
import glob
import json
import subprocess
import sys

def process_all_files(data_dir="data", output_base="output"):
    """
    Scans data directory for TIF files, generates visualization data for each,
    and creates a file list JSON.
    """
    # 1. Regenerate mock terrain
    print("Regenerating mock terrain...")
    try:
        subprocess.run([sys.executable, "generate_testmap.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating mock terrain: {e}")

    # Find all .tif files
    tif_files = glob.glob(os.path.join(data_dir, "*.tif"))
    tif_files.sort()
    
    print(f"Found {len(tif_files)} TIF files. Processing...")
    
    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        # Create a slug/folder name from the filename (remove extension)
        slug = os.path.splitext(filename)[0]
        output_dir = os.path.join(output_base, slug)
        
        print(f"Processing {filename} -> {output_dir}...")
        
        # Run generate_viz.py
        cmd = [
            sys.executable, "generate_viz.py",
            "--input", tif_path,
            "--output-dir", output_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # If this is mock_terrain, generate a test path
            if filename == "mock_terrain.tif":
                print("Generating test path for mock_terrain...")
                path_cmd = [
                    sys.executable, "generate_path.py",
                    "--input", tif_path,
                    "--output-dir", output_dir,
                    "--start", "10", "10",
                    "--end", "50", "50"
                ]
                print(' '.join(path_cmd))
                try:
                    subprocess.run(path_cmd, check=True)
                    print("Test path generated.")
                except subprocess.CalledProcessError as e:
                    print(f"Error generating test path: {e}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing {filename}: {e}")
    
    # Update file list using update_file_list.py
    print("Updating file list...")
    try:
        subprocess.run([sys.executable, "update_file_list.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error updating file list: {e}")
        
    print("Done!")

if __name__ == "__main__":
    process_all_files()
