import numpy as np
import math
import random
from skimage.transform import rotate
from misc.aid.helpful_geo_functions import compute_geo_coordinates_from_grid
from get_parameters import para
from geopy.distance import distance
from shapely.geometry import Polygon
from shapely.ops import transform
from shapely.affinity import translate
from shapely.geometry import Point
import random, math
from pyproj import Geod
# --- Geo helpers ---
def random_point_in_polygon(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(20):
        rand_x = np.random.uniform(minx, maxx)
        rand_y = np.random.uniform(miny, maxy)
        p = Point(rand_x, rand_y)
        if polygon.contains(p):
            return p
    return polygon.centroid
def generate_polygon_center(max_distance_km=0):
    return distance(kilometers=random.uniform(0, max_distance_km)).destination((para.anchor_point[1],para.anchor_point[0]), random.uniform(0, 360))

def create_rectangular_target_area(granularity, image_path=None):
    """
    Create a rectangular target area polygon with 4 corners where the anchor point is at the lower left corner.
    Uses the same logic as compute_geo_coordinate_from_position for proper geodesic calculations.
    For jon, the side length is based on the image dimensions.
    
    Parameters:
    - granularity: Distance parameter for grid spacing
    - image_path: Optional path to the image file to use for dimensions
    
    Returns:
    - Shapely Polygon with 4 corners
    """
    # For jon, calculate side length based on image dimensions
    if para.z_rad_type == 'jon':
        import cv2
        import os
        if image_path is None:
            IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval_mowing_9.png")
        else:
            IMAGE_PATH = image_path
        img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")
        
        # Add black border to match what's used in the main environment
        # This ensures coordinate system consistency
        border_size = 5
        img = cv2.copyMakeBorder(
            img, 
            border_size, border_size, border_size, border_size, 
            cv2.BORDER_CONSTANT, 
            value=0  # Black color for grayscale
        )
        
        # Use image dimensions to determine the target area size
        img_height, img_width = img.shape
        # Use the smaller dimension to create a square, scaled by granularity
        side_length_pixels = min(img_height, img_width)
        side_length = side_length_pixels * granularity
    else:
        # Use the same area calculation as the original random_polygon function for other types
        area_values = np.array(para.sizes_of_target_area_km2)
        probabilities = np.array(para.sizes_of_target_area_km2_probabilities)
        if area_values.size > 1 and para.set_episode_length_based_on_first_episode_length:
            raise ValueError("area_values must be a single value when para.set_episode_length_based_on_first_episode_length is True")
        area_m2 = random.choices(area_values, probabilities)[0] * 1_000_000
        side_length = math.sqrt(area_m2)
    
    # Convert side_length from meters to grid units
    side_length_grid_units = side_length / granularity
    
    # Create 4 corner positions in grid coordinates relative to anchor point (lower left corner)
    corner_positions = [
        (0, 0),                                  # Bottom-left (anchor point)
        (side_length_grid_units, 0),             # Bottom-right  
        (side_length_grid_units, side_length_grid_units), # Top-right
        (0, side_length_grid_units)              # Top-left
    ]
    
    # Convert grid positions to geographic coordinates using the same logic as compute_geo_coordinate_from_position
    corners_geo = []
    
    for position in corner_positions:
        # Use the same logic as compute_geo_coordinate_from_position
        east_point = distance(meters=position[0] * granularity).destination((para.anchor_point[1], para.anchor_point[0]), 90)  # East
        new_point = distance(meters=position[1] * granularity).destination(east_point, 0)  # North
        corners_geo.append((new_point.longitude, new_point.latitude))
    
    # Create and return the polygon
    polygon = Polygon(corners_geo)
    return polygon

def random_polygon(center_lon, center_lat):
    """
    Zufälliges Vieleck (4–8 Eckpunkte) um (center_lon,center_lat) mit exakt area_m2 Fläche.
    Rückgabe: [(lon,lat), ...]  (°)
    """
    while True:
        area_values = np.array(para.sizes_of_target_area_km2)
        probabilities = np.array(para.sizes_of_target_area_km2_probabilities)
        if area_values.size > 1 and para.set_episode_length_based_on_first_episode_length:
            raise ValueError("area_values must be a single value when para.set_episode_length_based_on_first_episode_length is True")
        area_m2 = random.choices(area_values, probabilities)[0] * 1_000_000 
        geod = Geod(ellps="WGS84")
        n = random.randint(4, 8)
        bearings = sorted(random.random()*360 for _ in range(n))
        radii    = [random.uniform(300, 1000) for _ in range(n)]   # Startabstände (m)

        # iterativ skalieren, bis Flächenfehler <0.1 %
        scale = 1.0
        while True:
            lons, lats = zip(*[geod.fwd(center_lon, center_lat, az, r*scale)[:2]
                               for az, r in zip(bearings, radii)])
            area, _ = geod.polygon_area_perimeter(lons, lats)
            a = abs(area)
            if abs(a-area_m2)/area_m2 < 1e-10:
                break
            scale *= math.sqrt(area_m2/a)

        polygon = Polygon(zip(lons, lats))
        if polygon.is_valid:  # Check if the polygon is valid
            centroid = polygon.centroid
            dx = center_lon - centroid.x
            dy = center_lat - centroid.y
            poly_shifted = translate(polygon, xoff=dx, yoff=dy)
            return poly_shifted

def vector_length(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)

def points_distance(point1, point2):
    return vector_length((point1[0] - point2[0], (point1[1] - point2[1])))

def clamp(value, minimum, maximum):
    return max(min(value, maximum), minimum)

# --- Image transformation helpers ---

def apply_vignette_effect(grid):
    h, w = grid.shape
    gradient = np.linspace(0, 1, 4)

    # Create gradient masks for rows and columns
    row_mask = np.zeros((h, 1))
    row_mask[:4, 0] = gradient
    row_mask[-4:, 0] = gradient[::-1]

    col_mask = np.zeros((1, w))
    col_mask[0, :4] = gradient
    col_mask[0, -4:] = gradient[::-1]

    # Apply gradients to the grid
    grid[:4, :] *= row_mask[:4]
    grid[-4:, :] *= row_mask[-4:]
    grid[:, :4] *= col_mask[:, :4]
    grid[:, -4:] *= col_mask[:, -4:]

    return grid

def mirror_image(image):
    return np.fliplr(image)

def rotate_image(image):
    angle = random.uniform(0, 360)
    return rotate(image, angle, resize=False, mode='constant', cval=0)

def rescale_image(image):
    scale_factor = random.uniform(0.7, 5.0)
    return np.clip(image * scale_factor, 0, para.radiation_value_to_avoid + 5)

def shift_image(image):
    h, w = image.shape
    shift_x = random.randint(-int(0.25 * w), int(0.25 * w))
    shift_y = random.randint(-int(0.25 * h), int(0.25 * h))
    shifted_image = np.roll(image, shift_x, axis=1)
    shifted_image = np.roll(shifted_image, shift_y, axis=0)
    return shifted_image

def warp_array(values_grid):
    h, w = values_grid.shape
    result = np.zeros_like(values_grid)
    max_idx = np.unravel_index(np.argmax(values_grid), values_grid.shape)
    max_y, max_x = max_idx
    radius = 40
    angle = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0, radius)
    px_ = int(np.clip(max_x + r * np.cos(angle), 0, w - 1))
    py_ = int(np.clip(max_y + r * np.sin(angle), 0, h - 1))
    dx = np.random.randint(-40, 40)
    dy = np.random.randint(-40, 40)
    points = [(px_, py_, dx, dy)]

    # Vectorized grid coordinates
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    offset_x = np.zeros((h, w), dtype=np.float32)
    offset_y = np.zeros((h, w), dtype=np.float32)

    for px_, py_, dx, dy in points:
        shift_vector = np.array([dx, dy], dtype=np.float32)
        sv_len = np.linalg.norm(shift_vector)
        if sv_len == 0:
            continue
        point_position = np.array([px_ + dx, py_ + dy], dtype=np.float32)
        # Compute distances
        dist = np.sqrt((x_grid - point_position[0]) ** 2 + (y_grid - point_position[1]) ** 2)
        helper = 1.0 / (3 * (dist / sv_len) ** 4 + 1)
        offset_x -= helper * shift_vector[0]
        offset_y -= helper * shift_vector[1]

    nx = np.clip(np.round(x_grid + offset_x).astype(int), 0, w - 1)
    ny = np.clip(np.round(y_grid + offset_y).astype(int), 0, h - 1)
    result = values_grid[ny, nx]
    return result
def apply_blur(image):
    """Apply Gaussian blur to create more continuous gradients."""
    from scipy.ndimage import gaussian_filter
    sigma = 0.6
    return gaussian_filter(image, sigma=sigma)
def apply_transformations(data):
    """Apply the complete transformation pipeline to the data."""
    skewed_grid = warp_array(data)
    mirrored_grid = mirror_image(skewed_grid)
    rotated_grid = rotate_image(mirrored_grid)
    rescaled_grid = rescale_image(rotated_grid)
    blurred_grid = apply_blur(rescaled_grid)
    return apply_vignette_effect(blurred_grid)



def too_steep_radiation_gradient(grid):
    """Check if any interior point has problematic neighbors."""
    # Get interior points (not on border)
    interior = grid[1:-1, 1:-1]
    
    # Only consider interior points with values above 25
    interior_mask = interior >= para.radiation_value_to_avoid
    # Get all 8 neighbors for each interior point using vectorized operations
    neighbors = np.stack([
        grid[0:-2, 0:-2],  # top-left
        grid[0:-2, 1:-1],  # top
        grid[0:-2, 2:],    # top-right
        grid[1:-1, 0:-2],  # left
        grid[1:-1, 2:],    # right
        grid[2:, 0:-2],    # bottom-left
        grid[2:, 1:-1],    # bottom
        grid[2:, 2:]       # bottom-right
    ], axis=0)
    threshold = 10
    neighbors_below_threshold = neighbors < threshold  # threshold is now a scalar
    has_outlier_neighbors = np.any(neighbors_below_threshold, axis=0) & interior_mask
    return np.any(has_outlier_neighbors)

def transform_radiation_data(radiation_data):
    # Apply transformations and check for outliers
    while True:
        rescaled_grid = apply_transformations(radiation_data)
        if not too_steep_radiation_gradient(rescaled_grid):
            break
    return rescaled_grid

# --- Single scenario creation function ---
def create_single_radiation_scenario(radiation_data, granularity, image_path=None):
    # 1. First, create geo_coordinates_of_radiation_grid centered at anchor_point
    geo_coordinates_of_radiation_grid = compute_geo_coordinates_from_grid(
        gridsize=(radiation_data.shape[0], radiation_data.shape[1]),
        anchor_point=np.array([para.anchor_point[0], para.anchor_point[1]]),
        granularity=granularity
    )

    # 2. Compute the center of geo_coordinates_of_radiation_grid
    # geo_coordinates_of_radiation_grid shape: (H, W, 2) or similar
    grid_shape = geo_coordinates_of_radiation_grid.shape
    if len(grid_shape) == 3 and grid_shape[2] == 2:
        # Average over all points
        center_lon = np.mean(geo_coordinates_of_radiation_grid[..., 0])
        center_lat = np.mean(geo_coordinates_of_radiation_grid[..., 1])
    else:
        raise ValueError("geo_coordinates_of_radiation_grid must be (H, W, 2)")
    # Use tuple for polygon_center to avoid issues with downstream code expecting subscriptable type
    polygon_center = (center_lon, center_lat)

    # 3. Create target_area polygon centered at polygon_center with 4 corners
    # Use the granularity from parameters for geodesic mapping
    if para.z_rad_type == 'jon':
        target_area = create_rectangular_target_area(para.granularity_cpp_maps_from_paper, image_path)
    elif para.z_rad_type == 'bfs':
        target_area = random_polygon(center_lon, center_lat)
    elif para.z_rad_type == 'mov':
        target_area = random_polygon(center_lon, center_lat)
    measuring_area_scenario = {
        "polygon_center": polygon_center,
        "target_area": target_area,
    }
    if para.z_rad_type == 'mov':
        return geo_coordinates_of_radiation_grid, measuring_area_scenario
    elif para.z_rad_type == 'bfs':
        scenario = transform_radiation_data(radiation_data)
        return scenario, geo_coordinates_of_radiation_grid, measuring_area_scenario
    elif para.z_rad_type == 'jon':
        # For static maps, this function is not used in the main flow
        # Return minimal data for compatibility
        return geo_coordinates_of_radiation_grid, measuring_area_scenario