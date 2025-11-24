# Visualize a GeoTIFF with Rerun

This repository includes `visualize_terrain.py`, a small CLI to load a GeoTIFF and log its terrain heights to Rerun as a point cloud.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python visualize_terrain.py /path/to/terrain.tif --max-points 200000
```

Open the Rerun viewer (see Rerun docs) to inspect the `geo_terrain` project.
