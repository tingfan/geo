import os
import json
import glob

def update_file_list(data_dir="data", output_base="output", output_file="data/file_list.json"):
    """
    Scans the data directory for .tif files and writes metadata to a JSON file.
    """
    # Find all .tif files in data directory
    tif_files = glob.glob(os.path.join(data_dir, "*.tif"))
    tif_files.sort()
    
    # Create file list with metadata
    file_list = []
    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        slug = os.path.splitext(filename)[0]
        
        file_list.append({
            "name": filename,
            "slug": slug,
            "data_path": f"{output_base}/{slug}"
        })
    
    print(f"Found {len(file_list)} TIF files:")
    for item in file_list:
        print(f"  - {item['name']} -> {item['data_path']}")
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write to JSON
    with open(output_file, 'w') as f:
        json.dump(file_list, f, indent=2)
        
    print(f"Saved file list to {output_file}")

if __name__ == "__main__":
    update_file_list()
