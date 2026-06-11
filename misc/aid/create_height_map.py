import os
import requests
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from get_parameters import para

# Load API key from environment
API_KEY = os.getenv("OPENTOPOGRAPHY_TOKEN")
if not API_KEY:
    raise ValueError("Environment variable OPENTOPOGRAPHY_TOKEN not set.")

# center coordinates
anchor_point_lon, anchor_point_lat = para.anchor_point
half_side_deg = 1  # ~100 km

# Define bounding box
south = anchor_point_lat - half_side_deg
north = anchor_point_lat + half_side_deg
west = anchor_point_lon - half_side_deg
east = anchor_point_lon + half_side_deg

# Define file paths
# base_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__)) # .../Simulation/misc/aid
misc_dir = os.path.dirname(script_dir) # .../Simulation/misc

downloads_dir = os.path.join(misc_dir, 'geo_data', 'downloads_from_opentopography')
os.makedirs(downloads_dir, exist_ok=True)
filename = f"aw3d30_{south:.10f}_{north:.10f}_{west:.10f}_{east:.10f}.tif"
file_path = os.path.join(downloads_dir, filename)
npy_file_path = os.path.join(misc_dir, 'geo_data','height_data', 'height_data.npy')

# Check if the GeoTIFF file already exists
if not os.path.exists(file_path):
    # Request data from OpenTopography
    params = {
        "demtype": "AW3D30",
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": API_KEY
    }

    response = requests.get("https://portal.opentopography.org/API/globaldem", params=params)

    if response.ok:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded heightmap from OpenTopography")
    else:
        raise RuntimeError(f"❌ Error {response.status_code}: {response.text}")
else:
    pass
    #print(f"Skip download from OpenTopography (Already exists).")

# Check if the .npy file already exists
if not os.path.exists(npy_file_path):

    # Open the GeoTIFF file using rasterio
    import rasterio
    with rasterio.open(file_path) as src:
        height_data = src.read(1)
        t = src.transform
        # GDAL GetGeoTransform order: c, a, b, f, d, e
        gt = [t.c, t.a, t.b, t.f, t.d, t.e]

    rows, cols = np.indices(height_data.shape)
    lons = gt[0] + cols * gt[1] + rows * gt[2]
    lats = gt[3] + cols * gt[4] + rows * gt[5]

    lons = lons.flatten()
    lats = lats.flatten()
    heights = height_data.flatten()
    data = np.array(list(zip(lons, lats, heights)))

    # Save the numpy array to file
    np.save(npy_file_path, data)
else:
    pass
    #print(f"Skip heightmap creation (Already exists).")