import numpy as np
from geopy.distance import distance
from geopy.distance import geodesic
from pyproj import Geod
from get_parameters import para

def compute_position_from_geo_coordinate(geo_coord, granularity):
    """
    Compute grid position (x, y) from a given geographic coordinate (longitude, latitude).
    Uses the same equirectangular approximation as compute_geo_coordinates_from_grid.
    
    Parameters:
    - geo_coord: (longitude, latitude) tuple of the target geographic coordinate.
    - anchor_point: (longitude, latitude) tuple of the grid origin.
    - granularity: Distance in meters between adjacent grid points.
    
    Returns:
    - (x, y) tuple representing the grid position.
    """
    # Extract coordinates for clarity
    lon0, lat0 = para.anchor_point[0], para.anchor_point[1]
    lon, lat = geo_coord

    deg_per_meter = 1.0 / 111320.0

    # Compute north-south distance (y direction)
    dist_north_south = (lat - lat0) / deg_per_meter
    
    # Compute east-west distance (x direction) accounting for latitude
    dist_east_west = (lon - lon0) * np.cos(np.radians(lat0)) / deg_per_meter

    # Convert distances to grid units
    x = dist_east_west / granularity
    y = dist_north_south / granularity
    return x, y

def compute_geo_coordinate_from_position(position, granularity):
    """
    Compute geographic coordinate from a grid position.
    
    Parameters:
    - position: Tuple (x, y) representing the grid position.
    - anchor_point: Geopy Point object or (longitude,latitude) tuple as the grid origin.
    - granularity: Distance in meters between adjacent grid points.
    
    Returns:
    - (longitude, latitude) tuple of the new geographic coordinate.
    """
    # Adjust longitude granularity based on latitude

    east_point = distance(meters=position[0] * granularity).destination((para.anchor_point[1],para.anchor_point[0]), 90)  # East
    new_point =  distance(meters=position[1] * granularity).destination(east_point, 0)     # North
    return new_point.longitude, new_point.latitude

def compute_geo_coordinates_from_grid(gridsize, anchor_point, granularity):
    """
    Generate a grid of geographic coordinates based on an anchor point and granularity.
    Vectorized via simple equirectangular approximation.

    Parameters:
    - gridsize: Tuple (rows, cols)
    - anchor_point: (latitude, longitude) tuple of the grid origin
    - granularity: meters between adjacent grid points

    Returns:
    - np.ndarray of shape (rows, cols, 2), each entry [lon, lat]
    """
    rows, cols = gridsize
    lon0, lat0 = anchor_point

    deg_per_meter = 1.0 / 111320.0

    # Create index grids
    i = np.arange(rows)[:, None]   # (rows,1)
    j = np.arange(cols)[None, :]   # (1,cols)

    # Compute latitudes and broadcast to (rows,cols)
    lat = lat0 + i * granularity * deg_per_meter       # (rows,1)
    lat_full = np.broadcast_to(lat, (rows, cols))      # (rows,cols)

    # Compute longitudes using lat_full
    lon_full = lon0 + (j * granularity * deg_per_meter) / np.cos(np.radians(lat_full))

    # Stack into shape (rows, cols, 2) with [lon, lat]
    geo_coordinates = np.stack((lon_full,lat_full), axis=2)
    return geo_coordinates

def create_grid(anchor_point, span_point, granularity):
    # Calculate the number of points in each direction
    distance_ns = geodesic((anchor_point[0], anchor_point[1]), (span_point[0], anchor_point[1])).meters
    distance_ew = geodesic((anchor_point[0], anchor_point[1]), (anchor_point[0], span_point[1])).meters
    
    # Calculate the number of steps in each direction
    steps_ns = int(distance_ns / granularity)
    steps_ew = int(distance_ew / granularity)
    
    # Generate arrays of indices
    i = np.arange(steps_ns + 1)
    j = np.arange(steps_ew + 1)
    
    # Use meshgrid to create a grid of indices
    i_grid, j_grid = np.meshgrid(i, j, indexing='ij')
    
    # Calculate latitudes and longitudes
    lat = anchor_point[0] + (i_grid * granularity / 111320)
    lon = anchor_point[1] + (j_grid * granularity / (40075000 * np.cos(np.radians(lat)) / 360))
    
    # Combine lat and lon into a single array of tuples
    grid = np.empty(lat.shape, dtype=object)

    # Use np.frompyfunc to create a vectorized function for tuple creation
    vectorized_tuple_creator = np.frompyfunc(lambda x, y: (x, y), 2, 1)

    # Apply the vectorized function to lat and lon arrays
    grid = vectorized_tuple_creator(lat, lon)
    
    return grid
# File: C:\Users\johan\programming\Simulation\misc\helpful_geo_functions.py

import geopandas as gpd
import numpy as np
from shapely.geometry import box


def get_grid_centroids_and_values(shapefile_path, anchor_point, granularity=150, grid_width=20000, grid_length=20000, time_offset_hours=23):
    # 1. Load the data
    gdf = gpd.read_file(shapefile_path)

    # 2. Define start_time
    start_time = gdf['Time'].min()
    selected_time = start_time + time_offset_hours * 3600

    # Filter data for Time >= selected_time
    time_filtered_gdf = gdf[gdf['Time'] >= selected_time]
    if time_filtered_gdf.empty:
        raise ValueError("No data available after the selected time.")
    else:
        next_available_time = time_filtered_gdf['Time'].min()
        time_filtered_gdf = gdf[gdf['Time'] == next_available_time]

    # 3. Project to a CRS in meters
    gdf_meters = time_filtered_gdf.to_crs(epsg=3857)

    # 3.1 Calculate area of shapefile polygons
    gdf_meters['poly_area'] = gdf_meters.geometry.area

    # 4. Define grid parameters
    granularity = granularity

    # 5. Compute starting point (bottom left)
    minx, miny, maxx, maxy = gdf_meters.total_bounds
    start_x = maxx - grid_width
    start_y = miny

    # Create grid cells as a 2D array
    x_coords = np.arange(start_x, start_x + grid_width, granularity)
    y_coords = np.arange(start_y, start_y + grid_length, granularity)

    grid_cells = np.empty((len(x_coords), len(y_coords)), dtype=object)
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            cell = box(x, y, x + granularity, y + granularity)
            grid_cells[i, j] = cell

    # Flatten grid_cells and create GeoDataFrame
    grid_cells_flat = grid_cells.flatten()
    grid = gpd.GeoDataFrame({'geometry': grid_cells_flat}, crs=gdf_meters.crs)

    # Ensure geometries are valid
    gdf_meters = gdf_meters[gdf_meters.is_valid]
    grid = grid[grid.is_valid]

    # Reset index and assign IDs
    grid = grid.reset_index(drop=True)
    grid['grid_id'] = grid.index
    gdf_meters = gdf_meters.reset_index(drop=True)
    gdf_meters['poly_id'] = gdf_meters.index

    # 6. Perform spatial overlay to find intersections
    intersections = gpd.overlay(grid, gdf_meters, how='intersection')

    # 7. Calculate area of intersections
    intersections['intersection_area'] = intersections.geometry.area

    # 8. Calculate adjusted value for each intersection
    intersections['adjusted_value'] = intersections['Value'] * (intersections['intersection_area'] / intersections['poly_area'])

    # 9. Sum adjusted values for each grid cell
    grouped = intersections.groupby('grid_id')['adjusted_value'].sum()

    # 10. Assign summed values to grid cells
    grid['cell_value'] = grid['grid_id'].map(grouped).fillna(0)

    # Get centroids
    centroids = grid.geometry.centroid

    # Reproject centroids to WGS84
    centroids_wgs84 = centroids.to_crs(epsg=4326)
    lons = centroids_wgs84.x.values
    lats = centroids_wgs84.y.values
    values = grid['cell_value'].values

    # Reshape to 2D arrays
    lons_grid = lons.reshape(len(x_coords), len(y_coords))
    lats_grid = lats.reshape(len(x_coords), len(y_coords))
    values_grid = values.reshape(len(x_coords), len(y_coords))

    # Combine lons and lats into a single array of shape (x, y, 2)
    coordinates_grid = compute_geo_coordinates_from_grid(gridsize=(values_grid.shape[0], values_grid.shape[1]), anchor_point=anchor_point, granularity=granularity)
    return coordinates_grid, values_grid

def create_grid_on_earth(
    grid_size_in_m: float,
    number_of_points: int,
    center_of_grid_as_lon_lat: tuple[float, float],
    direction_given_by_bearing: float
) -> np.ndarray:
    """
    Creates a square geodesic grid on WGS84...
    """    
    lon_center, lat_center = center_of_grid_as_lon_lat
    geod = Geod(ellps="WGS84")
    N = number_of_points

    step = grid_size_in_m / (number_of_points - 1)

    half_size = grid_size_in_m / 2.0
    y_offsets = half_size - np.arange(number_of_points) * step
    x_offsets = -half_size + np.arange(number_of_points) * step
    Y, X = np.meshgrid(y_offsets, x_offsets, indexing='ij')
    
    distances = np.sqrt(X**2 + Y**2)
    theta_radians = np.arctan2(X, Y)
    theta_degrees = np.degrees(theta_radians)
    
    if direction_given_by_bearing != 0:
        azimuths = (theta_degrees + direction_given_by_bearing) % 360
    else:
        azimuths = theta_degrees
    
    lon_centers = np.full((number_of_points, number_of_points), lon_center)
    lat_centers = np.full((number_of_points, number_of_points), lat_center)
    
    lons_new, lats_new, _ = geod.fwd(lon_centers, lat_centers, azimuths, distances)
    grid_points = np.stack([lons_new, lats_new], axis=-1)
    return grid_points