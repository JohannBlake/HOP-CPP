# Script to step through gym environment and save data for visualization
# This script is called by vis_clone_repo.py with command line arguments
import winsound
import argparse
import json
import wandb
import webbrowser
import time
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import matplotlib.pyplot as plt
import os
import time
import git
import subprocess
import numpy as np
import copy
import yaml
import importlib
import sys
import base64
import datetime
import shutil
from shapely.geometry import mapping, MultiPolygon, Polygon
from tqdm import tqdm
import tempfile
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

# Performance optimizations setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Pre-allocate commonly used constants
DEG_PER_METER = 1.0 / 111320.0

# Set numpy and matplotlib to use optimal settings for performance
if torch.cuda.is_available():
    # Enable memory pool for faster allocation
    torch.cuda.empty_cache()

# Set matplotlib backend for better performance
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster plotting
import re
import threading
import msgpack
from gymnasium.wrappers import AutoResetWrapper

# Add imports for omnisafe model loading
import torch
import importlib.util

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process visualization data')
    parser.add_argument('--sweep-id', type=str, required=True, help='Sweep ID')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs to display')
    parser.add_argument('--cluster', type=str, default='lmu', help='Cluster name')
    parser.add_argument('--main-folder', type=str, required=True, help='Main folder path')
    parser.add_argument('--git-clones-folder', type=str, required=True, help='Git clones folder path')
    parser.add_argument('--html-data-folder', type=str, required=True, help='HTML data folder path')
    parser.add_argument('--timestamp', type=str, required=True, help='Timestamp')
    parser.add_argument('--base-folder', type=str, required=True, help='Base folder path')
    parser.add_argument('--commit-id', type=str, required=True, help='Commit ID')
    parser.add_argument('--run-ids', type=str, default='[]', help='Run IDs as JSON string')
    parser.add_argument('--run-ids-to-be-considered', type=str, default='[]', help='Run IDs to be considered as JSON string')
    parser.add_argument('--constrained-model-number', type=str, default='[]', help='Constrained model number')
    parser.add_argument('--debug-mode', type=str, help='Debug mode as string (True/False)')
    parser.add_argument('--map-tier', type=int, default=1, help='Map tier to set in gymenv')
    parser.add_argument('--num-episodes-to-skip', type=int, default=2, help='Number of episodes to skip before processing')
    parser.add_argument('--visualization-step-interval', type=int, default=1, help='Save visualization data every Nth step (1 = every step)')
    parser.add_argument('--saved-actions-path', type=str, default='', help='Path to saved actions .npy file for replay')
    parser.add_argument('--env-seed', type=int, default=0, help='Seed for environment initialization (for reproducibility)')
    return parser.parse_args()

args = parse_arguments()

# Access the variables from command line arguments
sweep_id = args.sweep_id
num_runs_displayed = args.num_runs
cluster = args.cluster
main_folder = args.main_folder
git_clones_folder = args.git_clones_folder
html_data_folder = args.html_data_folder
timestamp = args.timestamp
base_folder = args.base_folder
commit_id_fitting_to_model = args.commit_id
run_ids = json.loads(args.run_ids)
run_ids_to_be_considered = json.loads(args.run_ids_to_be_considered)
constrained_model_number = args.constrained_model_number
debug_mode = args.debug_mode.lower() == 'true'
num_episodes_to_skip = args.num_episodes_to_skip
visualization_step_interval = args.visualization_step_interval
saved_actions_path = args.saved_actions_path
env_seed = args.env_seed
# Create sweep_ids list from the single sweep_id
sweep_ids = [sweep_id]

# Load saved actions if provided
saved_actions = None
if saved_actions_path and os.path.exists(saved_actions_path):
    saved_actions = np.load(saved_actions_path)
    print(f"Loaded {len(saved_actions)} saved actions from {saved_actions_path}")
    print(f"Actions shape: {saved_actions.shape}, dtype: {saved_actions.dtype}")

# Set up visualization directory structure early
root_dir = main_folder
archive_subdir = 'misc/3d_vis_archive/offline'
archive_dir = os.path.join(root_dir, archive_subdir)
os.makedirs(archive_dir, exist_ok=True)

vis_timestamp_dir = os.path.join(archive_dir, f'vis_{timestamp}')
os.makedirs(vis_timestamp_dir, exist_ok=True)

# Create html_data folder directly in vis_timestamp_dir (replacing the original html_data_folder)
html_data_folder = os.path.join(vis_timestamp_dir, 'html_data')
os.makedirs(html_data_folder, exist_ok=True)

# Load local omnisafe version from main repository
OMNISAFE_DIR = os.path.join(main_folder, "omnisafe")
spec = importlib.util.spec_from_file_location("omnisafe", os.path.join(OMNISAFE_DIR, "__init__.py"))
omnisafe = importlib.util.module_from_spec(spec)
sys.modules["omnisafe"] = omnisafe
spec.loader.exec_module(omnisafe)

from omnisafe.models.actor.actor_builder import ActorBuilder

# Add main_folder to sys.path to import visualization
if main_folder not in sys.path:
    sys.path.insert(0, main_folder)
from misc.aid.visualization import append_output_gymenv_values, append_to_metric_data

# Import geo functions and parameters for PNG handling
misc_path = os.path.join(main_folder, 'misc')
if misc_path not in sys.path:
    sys.path.insert(0, misc_path)

from aid.helpful_geo_functions import compute_geo_coordinate_from_position

from get_parameters import para

def write_polygon_step_file(step_data):
    """
    Helper function for parallel polygon file writing.
    Args:
        step_data: tuple of (step_idx, polygon_data_dir, episode_polygons)
    """
    step_idx, polygon_data_dir, episode_polygons = step_data
    polygon_chunk_file = os.path.join(polygon_data_dir, f'step_{step_idx}.json')
    with open(polygon_chunk_file, 'w') as f:
        json.dump(episode_polygons, f)

def compute_geo_coordinates_gpu(positions, granularity, lon0, lat0):
    """
    GPU-accelerated coordinate conversion using PyTorch CUDA tensors.
    """
    if torch.cuda.is_available() and len(positions) > 1000:  # Only use GPU for large arrays
        positions_tensor = torch.tensor(positions, dtype=torch.float32, device=DEVICE)
        
        # Pre-compute constants on GPU
        cos_lat0 = torch.cos(torch.tensor(np.radians(lat0), device=DEVICE))
        granularity_deg = granularity * DEG_PER_METER
        
        # Vectorized computation on GPU
        x, y = positions_tensor[:, 0], positions_tensor[:, 1]
        lat = lat0 + y * granularity_deg
        lon = lon0 + (x * granularity_deg) / cos_lat0
        
        if positions.shape[1] == 2:
            result = torch.stack([lon, lat], dim=1)
        else:
            result = torch.stack([lon, lat, positions_tensor[:, 2]], dim=1)
        
        return result.cpu().numpy()
    else:
        # Fallback to CPU numpy for small arrays
        return compute_geo_coordinates_vectorized(positions, granularity)

def compute_geo_coordinates_vectorized(positions, granularity):
    """
    Optimized vectorized conversion of flat map positions to geo coordinates.
    Uses equirectangular approximation like compute_geo_coordinates_from_grid.
    
    Args:
        positions: numpy array of shape (N, 2) or (N, 3) with [x, y] or [x, y, z] coordinates
        granularity: Distance in meters between adjacent grid points
        
    Returns:
        numpy array of same shape with [lon, lat] or [lon, lat, z] coordinates
    """
    if len(positions) == 0:
        return positions
    
    positions = np.asarray(positions, dtype=np.float32)  # Use float32 for speed
    
    # Extract anchor point from parameters (cache these values)
    lon0, lat0 = para.anchor_point[0], para.anchor_point[1]
    
    # Pre-compute constants
    cos_lat0 = np.cos(np.radians(lat0))  # Pre-compute cosine
    granularity_deg = granularity * DEG_PER_METER
    
    # Extract x, y coordinates efficiently
    if positions.ndim == 1:
        x, y = positions[0], positions[1]
        # Compute coordinates
        lat = lat0 + y * granularity_deg
        lon = lon0 + (x * granularity_deg) / cos_lat0
        
        if len(positions) == 2:
            return np.array([lon, lat], dtype=np.float32)
        else:
            return np.array([lon, lat, positions[2]], dtype=np.float32)
    else:
        x, y = positions[:, 0], positions[:, 1]
        # Vectorized computation
        lat = lat0 + y * granularity_deg
        lon = lon0 + (x * granularity_deg) / cos_lat0
        
        if positions.shape[1] == 2:
            return np.column_stack([lon, lat]).astype(np.float32)
        else:
            return np.column_stack([lon, lat, positions[:, 2]]).astype(np.float32)

def convert_shapely_geometry_to_geo_coordinates(geometry, granularity):
    """
    Convert Shapely geometry coordinates from flat map positions to geo coordinates.
    Uses vectorized operations for better performance.
    
    Args:
        geometry: Shapely geometry object (Polygon, MultiPolygon, etc.)
        granularity: Granularity parameter for conversion
        
    Returns:
        New Shapely geometry object with geo coordinates
    """
    from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiLineString
    from shapely.ops import transform
    import numpy as np
    
    def vectorized_coord_transformer(x, y, z=None):
        """Vectorized transform function for shapely.ops.transform"""
        # Handle scalar inputs (single coordinate)
        if np.isscalar(x):
            positions = np.array([[x, y]])
            result = compute_geo_coordinates_vectorized(positions, granularity)
            lon, lat = result[0, 0], result[0, 1]
            if z is not None:
                return lon, lat, z
            return lon, lat
        
        # Handle array inputs (multiple coordinates)
        if z is None:
            positions = np.column_stack([x, y])
        else:
            positions = np.column_stack([x, y, z])
        
        result = compute_geo_coordinates_vectorized(positions, granularity)
        
        if z is None:
            return result[:, 0], result[:, 1]
        else:
            return result[:, 0], result[:, 1], result[:, 2]
    
    # Use shapely's transform function which handles all geometry types
    return transform(vectorized_coord_transformer, geometry)

# Prepare lists to store the three different image sizes
images_small_for_3d_visualisation = []
images_medium_for_3d_visualisation = []
images_large_for_3d_visualisation = []

# Prepare lists for radiation data or polygon data depending on radiation type
positions_per_step_list = []
colors_per_step_list = []
black_polygons_per_episode_list = []  # For storing black polygons per episode when z_rad_type is 'jon'

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

# Main processing starts here
print(f"=== SCRIPT STARTED ===")

for sweep_id in sweep_ids:
    print(f"Processing sweep {sweep_id}...")
    
    sweep_path = f"johanndavidblake-ludwig-maximilianuniversity-of-munich/Heli-Logs/{sweep_id}"
    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    runs = sweep.runs
    
    # Use the passed run_ids instead of getting them from sweep
    if not run_ids:
        run_ids = [run.id for run in runs]
    
    # Restrict run_ids to the ones specified in the run_ids list. If the list is empty, all runs are considered
    if run_ids_to_be_considered:
        run_ids = [run_id for run_id in run_ids if run_id in run_ids_to_be_considered]

    # Copy height data file
    copy_start_time = time.time()
    height_data_src = os.path.join(main_folder, 'misc' , 'geo_data','height_data','height_data_for_vis.npy')
    height_data_dest = os.path.join(base_folder, 'misc' , 'geo_data','height_data', 'height_data.npy')
    shutil.copy2(height_data_src, height_data_dest)
    #print("disable next 2 lines in future (just kept because of old shiat run)")
    #height_data_dest = os.path.join(base_folder, 'height_data.npy')
    #shutil.copy2(height_data_src, height_data_dest)

    print(f"Copied {height_data_src} to {height_data_dest}")

    # Process each run
    for run_id in run_ids:
        print(f"\n--- Processing run {run_id} ---")
        
        run_path = f"{sweep_path}/{run_id}"
        run = api.run(run_path)
        run_name = run.name
        config = run.config

        # Update parameters
        from get_parameters import para
        
        # Only download model if we don't have saved actions
        if saved_actions is None:
            # SCP command setup
            if cluster == 'lmu':
                remote_user = 'blake@madeira.dbs.ifi.lmu.de'
                if para.training_library == 'sb3':
                    remote_base_path = f"/home/stud/blake/git_clones/Simulation_{sweep_id}/logs/{run_name}"
                    remote_model_zip_path = f"{remote_base_path}/best_model.zip"
                elif para.training_library == 'omnisafe':
                    if constrained_model_number == "":
                        print("Using latest model, since no constrained_model_number is specified.")
                        list_cmd = (
                            f'ssh blake@madeira.dbs.ifi.lmu.de "ls /home/stud/blake/git_clones/Simulation_{sweep_id}/misc/logs/runs/*/*/torch_save/epoch-*.pt"'
                        )
                    else:
                        print("Using model from epoch ", constrained_model_number)
                        list_cmd = (
                            f'ssh blake@madeira.dbs.ifi.lmu.de "ls /home/stud/blake/git_clones/Simulation_{sweep_id}/misc/logs/runs/*/*/torch_save/epoch-{constrained_model_number}.pt"'
                        )
                    result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True, encoding='utf-8', check=True)
                    files = [f for f in result.stdout.strip().split('\n') if f]
                    # Get file with latest creation/modification time
                    import stat
                    import datetime
                    # Use ssh to get file times
                    if files:
                        # Select file with highest epoch number
                        import re
                        def extract_epoch_num(fname):
                            match = re.search(r'epoch-(\d+)\.pt$', fname)
                            return int(match.group(1)) if match else -1
                        files_with_epoch = [(f, extract_epoch_num(f)) for f in files if extract_epoch_num(f) != -1]
                        if files_with_epoch:
                            latest = max(files_with_epoch, key=lambda x: x[1])[0]
                        else:
                            latest = files[0]
                    else:
                        latest = None
                    remote_model_zip_path = latest


                destination_model_zip_path = os.path.join(base_folder, f"{run_name}-best_model.zip")
                scp_command = (
                    f'scp {remote_user}:"{remote_model_zip_path}" "{destination_model_zip_path}"'
                )
            elif cluster == 'lrz':
                remote_user = 'di97sog@login.ai.lrz.de'
                ssh_key = "C:\\Users\\johan\\.ssh\\id_rsa_lrz"
                if para.training_library == 'sb3':
                    remote_base_path = f"/dss/dsshome1/0C/di97sog/git_clones/Simulation_{sweep_id}/misc/logs/runs/{run_name}"
                    remote_model_zip_path = f"{remote_base_path}/best_model.zip"
                elif para.training_library == 'omnisafe':
                    # Use * for base_run_pattern and get latest model by creation/modification time
                    if constrained_model_number == "":
                        print("Using latest model, since no constrained_model_number is specified.")
                        list_cmd = (
                            f'ssh -i "{ssh_key}" {remote_user} "ls /dss/dsshome1/0C/di97sog/git_clones/Simulation_{sweep_id}/misc/logs/runs/*/*/torch_save/epoch-*.pt"'
                        )
                    else:
                        print("Using model from epoch ", constrained_model_number)
                        list_cmd = (
                            f'ssh -i "{ssh_key}" {remote_user} "ls /dss/dsshome1/0C/di97sog/git_clones/Simulation_{sweep_id}/misc/logs/runs/*/*/torch_save/epoch-{constrained_model_number}.pt"'
                        )
                    result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True, encoding='utf-8', check=True)
                    files = [f for f in result.stdout.strip().split('\n') if f]
                    # Get file with latest creation/modification time
                    import stat
                    import datetime
                    if files:
                        # Select file with highest epoch number
                        import re
                        def extract_epoch_num(fname):
                            match = re.search(r'epoch-(\d+)\.pt$', fname)
                            return int(match.group(1)) if match else -1
                        files_with_epoch = [(f, extract_epoch_num(f)) for f in files if extract_epoch_num(f) != -1]
                        latest = max(files_with_epoch, key=lambda x: x[1])[0]
                    else:
                        latest = None
                    remote_model_zip_path = latest

                destination_model_zip_path = os.path.join(base_folder, f"{run_name}-best_model.zip")
                scp_command = (
                    f'scp -i "{ssh_key}" {remote_user}:"{remote_model_zip_path}" "{destination_model_zip_path}"'
                )
            print(f"Downloading latest model for run '{run_name}' from cluster '{cluster}'...")
            subprocess.run(scp_command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
        else:
            print(f"Skipping model download - using saved actions from {saved_actions_path}")

        # Environment and model setup - broken down for timing analysis
        
        # File setup
        os.chdir(base_folder)
        sys.path.insert(0, base_folder)
        
        # Import environment class
        from class_gymenv import GymEnvOmniSafe

        # Initialize environment with lazy loading optimization
        os.chdir(base_folder)
        
        # Optimize environment creation by disabling radiation grid visualization during init
        # and enabling it only when needed during episode processing
        env = GymEnvOmniSafe(radiation_grid_visualization=False, is_evaluation=True)
        gymenv = env._env.env
        
        # Enable radiation grid visualization after initialization
        gymenv.radiation_grid_visualization = True
        
        # Only build and load policy if we don't have saved actions
        if saved_actions is None:
            # Build policy
            actor_builder = ActorBuilder(env.observation_space, env.action_space, hidden_sizes=[256, 256])
            policy = actor_builder.build_actor(actor_type="cnn")
            
            # Load model weights
            model_file_name = f"{run_name}-best_model.zip"
            model_path = os.path.join(base_folder, model_file_name)
            state = torch.load(model_path, map_location="cpu")
            policy.load_state_dict(state.get("pi", state))
            policy.eval()
            
            # Move policy to GPU if available
            if torch.cuda.is_available():
                policy = policy.to(DEVICE)
        else:
            print("Skipping model loading - will use saved actions")
            policy = None  # Set policy to None when using saved actions
        
        # Set all random seeds for full reproducibility
        import random
        random.seed(env_seed)
        np.random.seed(env_seed)
        torch.manual_seed(env_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(env_seed)
        
        # Environment reset with seed for reproducibility
        # Use high starting seed for skip resets to avoid conflict with episode seeds
        skip_seed_offset = 10000
        obs, _ = env.reset(seed=env_seed + skip_seed_offset)
        
        for _ in range(num_episodes_to_skip*2):
            print("reset", _)
            obs, _ = env.reset(seed=env_seed + skip_seed_offset + _ + 1)   
        images_for_3d_visualisation = []
        paths_for_3d_visualisation = []
        metric_data = {}

        # Track the geo-coordinates at each step
        observed_images_geo_coordinates_history = []
        geo_json = {
            "type": "FeatureCollection",
            "features": []
        }
        current_step_in_animation = 0
        previous_polygon_with_height_target_area = None
        positions_per_step_list = []
        colors_per_step_list = []
        observation_images_list = []  # Store observation images for each step
        episode_step_ranges = []  # Track step ranges for each episode
        current_episode_start_step = 0
        actions_list = []  # Store all actions taken during episodes
        # Access tree and surface_point_cloud from wrapped env
        tree = gymenv.tree_surface_point_cloud_
        surface_point_cloud = np.array(gymenv.surface_point_cloud_)

        # Define evaluation maps 9-14 like in benchmark_local.py
        reset_image_paths = [
            os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_9.png"),
            os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_10.png"),
            os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_11.png"),
            os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_12.png"),
            os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_13.png"),
            os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_14.png")
        ]

        print(f"Starting {num_runs_displayed} episodes...")
        
        for episode_index in tqdm(range(num_runs_displayed), desc="Processing episodes"):
            
            # Reset environment for each episode with specific map cycling through 9-14
            map_index = episode_index % len(reset_image_paths)
            reset_image_path = reset_image_paths[map_index]
            full_reset_path = reset_image_path if os.path.isabs(reset_image_path) else os.path.join(os.getcwd(), reset_image_path)
            
            # Reset with map specification and episode seed
            # Use env_seed as base to ensure reproducibility across runs
            episode_seed = env_seed + episode_index
            obs, _ = env.reset(options={'image_path': full_reset_path}, seed=episode_seed)
            # Create position_as_geo_coordinate attribute for 'jon' type using optimized conversion
            if para.z_rad_type == 'jon':
                # Use vectorized conversion for better performance
                pos_array = np.array([[gymenv.position[0], gymenv.position[1]]], dtype=np.float32)
                geo_coords = compute_geo_coordinates_vectorized(pos_array, para.granularity)
                position_as_geo_coordinate = geo_coords[0]
 
            # Collect black polygons for this episode (for 'jon' type)
            if para.z_rad_type == 'jon':
                # Convert black_polygons to geo coordinates and serializable format
                black_polygons_geo = convert_shapely_geometry_to_geo_coordinates(gymenv.black_polygons, para.granularity)
                polygons_geojson = mapping(black_polygons_geo)
                black_polygons_per_episode_list.append(polygons_geojson)
            terminated = False
            truncated = False
            path_of_episode = []
            lon = position_as_geo_coordinate[0]
            lat = position_as_geo_coordinate[1]
            # --- Use the same height calculation as vis_polygon_model_performance_3d.py ---
            if para.z_rad_type == 'jon':
                # For 'jon' type, always use height 0
                path_of_episode.append([float(lon), float(lat), 0.0])
            elif para.surface_grid_creation_type == 'no_height_map':
                # Get height from the surface point cloud using tree query logic
                _, index = tree.query(np.array([position_as_geo_coordinate[0:2]]))
                surface_grid_height = surface_point_cloud[index[0], 2]
                path_of_episode.append([float(lon), float(lat), float(surface_grid_height + 90)])
            else:
                path_of_episode.append([float(lon), float(lat), float(gymenv.position[2])])
            step_number = 0

            # Pre-compute and cache data for non-jon types to avoid repeated calculations
            if para.z_rad_type != 'jon':
                coords = np.array(gymenv.geo_coordinates_of_radiation_grid, dtype=np.float32)
                positions_array = np.dstack([coords[:, :, 0], coords[:, :, 1]]).astype(np.float32)
                # Pre-allocate colormap for faster lookups
                inferno_cmap = plt.get_cmap('inferno')
                # Pre-compute normalization constants
                rad_min_val, rad_max_val = 0.0, 25.0
                rad_norm_factor = 1.0 / (rad_max_val - rad_min_val) if rad_max_val > rad_min_val else 0.0
            while not terminated and gymenv.current_episode_step <= 10000:
                with torch.no_grad():
                    # Use saved actions if available, otherwise predict from model
                    if saved_actions is not None and step_number < len(saved_actions):
                        # Use saved action
                        action = saved_actions[step_number]
                        # Ensure proper shape
                        if len(action.shape) == 2:
                            action = action  # Already correct shape
                        else:
                            action = action.reshape(1, -1)
                    else:
                        # Predict action from model
                        if torch.cuda.is_available():
                            if isinstance(obs, np.ndarray):
                                obs_tensor = torch.from_numpy(obs).to(DEVICE)
                            elif isinstance(obs, torch.Tensor):
                                obs_tensor = obs.to(DEVICE)
                            else:
                                obs_tensor = obs
                            action = policy.predict(obs_tensor, deterministic=True)
                            # Move action back to CPU for environment step
                            if isinstance(action, torch.Tensor):
                                action = action.cpu()
                        else:
                            action = policy.predict(obs, deterministic=True)
                
                # Save the action for later replay
                if isinstance(action, torch.Tensor):
                    actions_list.append(action.cpu().numpy())
                elif isinstance(action, np.ndarray):
                    actions_list.append(action.copy())
                else:
                    actions_list.append(np.array(action))
                
                output = env.step(action)

                obs = output[0]
                reward = output[1]
                cost = output[2]
                terminated = output[3].item() if hasattr(output[3], "item") else output[3]
                truncated = output[4].item() if hasattr(output[4], "item") else output[4]
                info = output[5]
                
                # Update position_as_geo_coordinate for 'jon' type after each step
                if para.z_rad_type == 'jon':
                    # Use vectorized version for better performance
                    pos_array = np.array([[gymenv.position[0], gymenv.position[1]]], dtype=np.float32)
                    geo_coords = compute_geo_coordinates_vectorized(pos_array, para.granularity)
                    position_as_geo_coordinate = geo_coords[0]
                # Log step number and episode progress every 200th step
                if (step_number + 1) % 200 == 0:
                    tqdm.write(f"Episode {episode_index + 1}/{num_runs_displayed}, Step {step_number + 1}")
                
                # Determine if we should save visualization data this step
                # Save on: interval steps, last step of episode, or if interval is 1 (every step)
                should_save_vis_data = (
                    not terminated and not truncated and
                    (visualization_step_interval == 1 or 
                    gymenv.current_episode_step % visualization_step_interval == 0 or
                    gymenv.percentage_of_target_area_left < 0.02)
                )
                
                if should_save_vis_data:
                    # Handle radiation data based on z_rad_type
                    if para.z_rad_type == 'jon':
                        # For 'jon' type, polygons are stored once per episode, not per step
                        # Create empty positions and colors for consistency (not used)
                        positions_per_step_list.append(None)
                        colors_per_step_list.append(None)
                    else:
                        # Optimized radiation grid processing using pre-computed values
                        radiation_grid = gymenv.radiation_grid.astype(np.float32)  # Ensure float32 for speed
                        
                        # Vectorized normalization using pre-computed factors
                        normalized_radiation = np.clip((radiation_grid - rad_min_val) * rad_norm_factor, 0, 1)
                        
                        # More efficient colormap application
                        colored_array = (inferno_cmap(normalized_radiation) * 255).astype(np.uint8)
                        
                        # Reuse pre-computed positions_array (no need to recalculate)
                        positions_per_step_list.append(positions_array)
                        colors_per_step_list.append(colored_array)

                    # Save observation image
                    # The observation should contain the image data directly
                    if hasattr(obs, 'shape') and len(obs.shape) >= 2:
                        # obs is likely the image directly
                        observation_images_list.append(obs)
                    elif isinstance(obs, dict) and 'image' in obs:
                        # obs is a dict containing image
                        observation_images_list.append(obs['image'])
                    elif 'image' in output[5]:  # info contains image
                        observation_images_list.append(output[5]['image'])
                    else:
                        observation_images_list.append(obs)

                    lon = position_as_geo_coordinate[0]
                    lat = position_as_geo_coordinate[1]
                    if not terminated:
                        if para.z_rad_type == 'jon':
                            # For 'jon' type, always use height 0
                            path_of_episode.append([float(lon), float(lat), 0.0])
                        elif para.surface_grid_creation_type == 'no_height_map':
                            # Get height from the surface point cloud using tree query logic
                            _, index = tree.query(np.array([position_as_geo_coordinate[0:2]]))
                            surface_grid_height = surface_point_cloud[index[0], 2]
                            path_of_episode.append([float(lon), float(lat), float(surface_grid_height + 90)])
                        else:
                            path_of_episode.append([float(lon), float(lat), float(gymenv.position[2])])
                    polygon_with_height_target_area = gymenv.target_area
                    
                    # Convert target_area to geo coordinates for 'jon' type
                    if para.z_rad_type == 'jon':
                        polygon_with_height_target_area = convert_shapely_geometry_to_geo_coordinates(
                            polygon_with_height_target_area, 
                            para.granularity
                        )

                    # --- Ensure target_area is always a MultiPolygon ---
                    # Convert to MultiPolygon if needed
                    if isinstance(polygon_with_height_target_area, Polygon):
                        polygon_with_height_target_area = MultiPolygon([polygon_with_height_target_area])

                    geo_json["features"].append({
                        "type": "Feature",
                        "geometry": mapping(polygon_with_height_target_area),
                        "properties": {
                            "category": "target_area",
                            "timestamp": gymenv.time_rel,
                            "current_step_in_animation": current_step_in_animation
                        }
                    })

                    geo_json["features"].append({
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": copy.deepcopy(path_of_episode)
                        },
                        "properties": {
                            "category": "path_of_episode",
                            "timestamp": gymenv.time_rel,
                            "current_step_in_animation": current_step_in_animation,
                            "run_id": run_id
                        }
                    })
                    previous_polygon_with_height_target_area = polygon_with_height_target_area
                    append_output_gymenv_values(metric_data, gymenv)
                    current_step_in_animation += 1
                else:
                    # Still need to update path for trajectory continuity, even when not saving full vis data
                    lon = position_as_geo_coordinate[0] 
                    lat = position_as_geo_coordinate[1]
                    if not terminated:
                        if para.z_rad_type == 'jon':
                            # For 'jon' type, always use height 0
                            path_of_episode.append([float(lon), float(lat), 0.0])
                        elif para.surface_grid_creation_type == 'no_height_map':
                            # Get height from the surface point cloud using tree query logic
                            _, index = tree.query(np.array([position_as_geo_coordinate[0:2]]))
                            surface_grid_height = surface_point_cloud[index[0], 2]
                            path_of_episode.append([float(lon), float(lat), float(surface_grid_height + 90)])
                        else:
                            path_of_episode.append([float(lon), float(lat), float(gymenv.position[2])])
                    # Still collect basic metrics for analysis even if not saving full visualization
                if terminated or truncated:
                    break
                if debug_mode:
                    break
                step_number += 1
            
            # Track step range for this episode
            episode_step_ranges.append({
                'start': current_episode_start_step,
                'end': current_step_in_animation - 1,  # -1 because current_step_in_animation was incremented after last step
                'episode_index': episode_index
            })
            current_episode_start_step = current_step_in_animation
            
            tqdm.write(f"Episode {episode_index + 1}/{num_runs_displayed} completed with {step_number} steps")

        # Clean up GPU memory after episodes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cache cleared")

        serializable_run_config = convert_to_serializable({
            "action_type": "ta without height change with termination option",
            "include_height_map": False,
            "termination_punishment": -10,
            "surface_grid_creation_type": "no_height_map",
            "include_termination_punishment": True,
            "include_height_in_scalar_observation": False,
            "run_name": "major-sweep-2",
            "radiation_type": para.z_rad_type  # Add radiation type for frontend
        })
        os.chdir(main_folder)
        
        # Data saving phase
        print("Starting data saving phase...")

        run_html_data_folder = os.path.join(html_data_folder, sweep_id, run_id)
        os.makedirs(run_html_data_folder, exist_ok=True)

        run_config_file_path = os.path.join(run_html_data_folder, 'run_config.json')
        with open(run_config_file_path, 'w') as f:
            json.dump(serializable_run_config, f)
        serializable_metric_data = convert_to_serializable(metric_data)
        
        # Save radiation data based on z_rad_type
        print("Processing radiation data...")
        
        if para.z_rad_type == 'jon':
            # Save polygon data instead of PNG paths - split into chunks
            polygon_data_dir = os.path.join(run_html_data_folder, 'polygon_data_chunks')
            os.makedirs(polygon_data_dir, exist_ok=True)
            
            # Create a mapping from episode to polygon data
            episode_polygon_info = {}  # Map episode_index to polygon data
            
            for episode_index, polygons_geojson in enumerate(black_polygons_per_episode_list):
                if polygons_geojson is not None:
                    episode_polygon_info[episode_index] = polygons_geojson
                else:
                    episode_polygon_info[episode_index] = None
            
            # Now create step-based polygon data and save as individual files in parallel
            total_steps = current_step_in_animation
            
            # Prepare all step data for parallel processing
            step_tasks = []
            for episode_range in episode_step_ranges:
                episode_idx = episode_range['episode_index']
                start_step = episode_range['start']
                end_step = episode_range['end']
                
                episode_polygons = episode_polygon_info.get(episode_idx, None)
                
                # Collect all step tasks for this episode
                for step_idx in range(start_step, end_step + 1):
                    if step_idx < total_steps:
                        step_tasks.append((step_idx, polygon_data_dir, episode_polygons))
            
            # Use ThreadPoolExecutor for parallel file I/O operations
            # Use min of available CPU cores and task count for optimal performance
            max_workers = min(os.cpu_count(), len(step_tasks))  # Cap at 16 to avoid too many concurrent files
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Execute all file writes in parallel
                list(executor.map(write_polygon_step_file, step_tasks))
        else:
            # Save all point clouds in one binary file (original logic)
            point_cloud_file_path = os.path.join(run_html_data_folder, 'point_clouds.bin')
            
            # Create consolidated binary data for all steps
            all_steps_data = []
            step_offsets = []  # Track where each step starts in the file
            current_offset = 0
            
            # Optimized batch processing for binary data creation
            all_binary_data = []
            step_offsets = []
            current_offset = 0
            
            # Process in batches for better memory efficiency
            batch_size = 100
            for batch_start in range(0, len(positions_per_step_list), batch_size):
                batch_end = min(batch_start + batch_size, len(positions_per_step_list))
                
                # Process batch
                for step_index in range(batch_start, batch_end):
                    pos_array = positions_per_step_list[step_index]
                    col_array = colors_per_step_list[step_index]
                    
                    # Vectorized reshaping and filtering
                    flattened_positions = pos_array.reshape(-1, 2)
                    flattened_colors = col_array[:, :, :3].reshape(-1, 3)
                    
                    # Vectorized dark color filtering
                    valid_indices = np.any(flattened_colors >= 50, axis=1)
                    
                    if not np.any(valid_indices):
                        step_offsets.append({'offset': current_offset, 'count': 0})
                        continue
                    
                    # Apply filtering using boolean indexing
                    filtered_positions = flattened_positions[valid_indices]
                    filtered_colors = flattened_colors[valid_indices]
                    num_points = len(filtered_positions)
                    
                    step_offsets.append({'offset': current_offset, 'count': num_points})
                    
                    # Optimized binary data creation using numpy's structured arrays
                    # Create structured array with position (3 x float32) and color (3 x uint8)
                    dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
                                    ('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])
                    
                    structured_data = np.empty(num_points, dtype=dtype)
                    structured_data['x'] = filtered_positions[:, 0]
                    structured_data['y'] = filtered_positions[:, 1] 
                    structured_data['z'] = 0.0  # z coordinate is always 0
                    structured_data['r'] = filtered_colors[:, 0]
                    structured_data['g'] = filtered_colors[:, 1]
                    structured_data['b'] = filtered_colors[:, 2]
                    
                    # Convert to bytes more efficiently
                    binary_data = structured_data.tobytes()
                    all_binary_data.append(binary_data)
                    current_offset += len(binary_data)
            
            # Write consolidated binary file in one operation
            with open(point_cloud_file_path, 'wb') as f:
                for binary_data in all_binary_data:
                    f.write(binary_data)
            
            # Save step metadata as JSON for the JS to know where each step's data starts
            step_metadata_path = os.path.join(run_html_data_folder, 'point_cloud_metadata.json')
            with open(step_metadata_path, 'w') as f:
                json.dump(step_offsets, f)
        
        # Optimized observation images processing
        # 
        # MULTI-SCALE MAP HANDLING:
        # When ablation_study_fisheye_instead_of_multi_scale_maps is False, the environment
        # creates 4 observation maps at different scales, concatenated along the channel dimension.
        # 
        # Original structure from gym environment:
        #   - Shape: (32, 32, 8) for 2 base channels × 4 maps
        #   - Channels 0-1: Map 0 (smallest scale - 2500m)
        #   - Channels 2-3: Map 1 (medium scale - 8000m)
        #   - Channels 4-5: Map 2 (medium-large scale)
        #   - Channels 6-7: Map 3 (largest scale - 40000m)
        #   - Base channels are typically: [0] target area, [1] radiation/wall data
        # 
        # This code splits and rearranges them into a 2x2 spatial grid for visualization:
        #   [Map 0] [Map 1]
        #   [Map 2] [Map 3]
        # Final shape: (64, 64, 3) - RGB image with 4 maps arranged spatially
        #
        print("Processing observation images...")
        
        if observation_images_list:
            obs_images_file_path = os.path.join(run_html_data_folder, 'observation_images.bin')
            obs_metadata = []
            
            # Pre-process all images to determine total size and optimize memory allocation
            processed_images = []
            
            # Determine if we're using multi-scale maps (4 maps) or single fisheye map
            # Check the first observation to determine structure
            first_obs = observation_images_list[0]
            if hasattr(first_obs, 'detach'):
                first_obs = first_obs.detach().cpu().numpy()
            
            # Determine number of base channels (e.g., 2 for target_area + radiation)
            if len(first_obs.shape) == 3:
                obs_height, obs_width, total_channels = first_obs.shape
            else:
                obs_height, obs_width = first_obs.shape
                total_channels = 1
            
            # Determine if multi-scale (channels will be a multiple of base channels)
            # Typically: 2 base channels (target area + radiation) * 4 maps = 8 total channels
            # Or with additional channels: (2 + extras) * 4
            num_maps = 1  # Default to single map (fisheye)
            base_channels = 2  # Default assumption
            
            # Check if channels are divisible by 4 (multi-scale) or 2 (single scale)
            if total_channels >= 8 and total_channels % 4 == 0:
                # Likely multi-scale with 4 maps
                num_maps = 4
                base_channels = total_channels // 4
                print(f"Detected multi-scale observations: {num_maps} maps with {base_channels} channels each")
            else:
                # Single scale (fisheye)
                num_maps = 1
                base_channels = total_channels
                print(f"Detected fisheye observations: 1 map with {base_channels} channels")
            
            for step_index, obs_image in enumerate(observation_images_list):
                # Convert tensor to numpy array if needed
                if hasattr(obs_image, 'detach'):
                    obs_image = obs_image.detach().cpu().numpy()
                
                # Ensure correct data type and shape
                if obs_image.dtype != np.uint8:
                    obs_image = obs_image.astype(np.uint8)
                
                # Handle different shape configurations
                if len(obs_image.shape) == 2:
                    height, width = obs_image.shape
                    obs_image = obs_image.reshape(height, width, 1)
                    height, width, channels = obs_image.shape
                else:
                    height, width, channels = obs_image.shape
                
                # Split multi-scale maps and arrange side-by-side
                if num_maps > 1:
                    # Split channels into separate maps
                    # obs_image shape: (32, 32, 8) -> split into 4 maps of (32, 32, 2)
                    maps = []
                    for map_idx in range(num_maps):
                        start_channel = map_idx * base_channels
                        end_channel = (map_idx + 1) * base_channels
                        map_slice = obs_image[:, :, start_channel:end_channel]
                        
                        # Ensure each map has RGB channels (3 channels) for visualization
                        if base_channels == 2:
                            # Add empty third channel
                            map_slice = np.concatenate([map_slice, np.zeros((height, width, 1), dtype=np.uint8)], axis=2)
                        elif base_channels == 1:
                            # Convert to RGB by repeating channel
                            map_slice = np.repeat(map_slice, 3, axis=2)
                        elif base_channels > 3:
                            # Take first 3 channels
                            map_slice = map_slice[:, :, :3]
                        
                        maps.append(map_slice)
                    
                    # Arrange maps in 2x2 grid
                    # Top row: map 0 (smallest) and map 1
                    # Bottom row: map 2 and map 3 (largest)
                    top_row = np.concatenate([maps[0], maps[1]], axis=1)
                    bottom_row = np.concatenate([maps[2], maps[3]], axis=1)
                    combined_image = np.concatenate([top_row, bottom_row], axis=0)
                    
                    # Update dimensions
                    height = combined_image.shape[0]
                    width = combined_image.shape[1]
                    channels = 3  # RGB
                    obs_image = combined_image
                else:
                    # Single map (fisheye) - ensure it has 3 channels
                    if channels == 2:
                        obs_image = np.concatenate([obs_image, np.zeros((height, width, 1), dtype=np.uint8)], axis=2)
                        channels = 3
                    elif channels == 1:
                        obs_image = np.repeat(obs_image, 3, axis=2)
                        channels = 3
                    elif channels > 3:
                        obs_image = obs_image[:, :, :3]
                        channels = 3
                
                processed_images.append((obs_image, height, width, channels))
            
            # Write all images in batches for better I/O performance
            with open(obs_images_file_path, 'wb') as f:
                current_offset = 0
                
                for step_index, (obs_image, height, width, channels) in enumerate(processed_images):
                    # Use numpy's efficient tobytes() for flattening and conversion
                    image_bytes = obs_image.tobytes()
                    f.write(image_bytes)
                    
                    obs_metadata.append({
                        'offset': current_offset,
                        'width': int(width),
                        'height': int(height), 
                        'channels': int(channels),
                        'size': len(image_bytes)
                    })
                    
                    current_offset += len(image_bytes)
            
            # Save observation image metadata
            # Keep backward compatibility with JavaScript - use array format
            obs_metadata_path = os.path.join(run_html_data_folder, 'observation_images_metadata.json')
            with open(obs_metadata_path, 'w') as f:
                json.dump(obs_metadata, f)
            
            # Also save additional metadata separately for future use
            obs_metadata_extra = {
                'is_multi_scale': num_maps > 1,
                'num_maps': num_maps,
                'base_channels': base_channels,
                'original_map_size': obs_height  # Size of each individual map before arrangement
            }
            obs_metadata_extra_path = os.path.join(run_html_data_folder, 'observation_images_metadata_extra.json')
            with open(obs_metadata_extra_path, 'w') as f:
                json.dump(obs_metadata_extra, f)
        
        center_coordinates = {
            "center_lat": para.anchor_point[1],
            "center_lon": para.anchor_point[0],
            "radiation_type": para.z_rad_type  # Add radiation type for frontend
        }

        # Save geo_json data in chunks instead of one large file
        print("Processing GeoJSON data...")
        
        geo_json_chunks_dir = os.path.join(run_html_data_folder, 'geo_json_chunks')
        os.makedirs(geo_json_chunks_dir, exist_ok=True)
        
        # Group features by step
        features_by_step = {}
        for feature in geo_json["features"]:
            step = feature["properties"]["current_step_in_animation"]
            if step not in features_by_step:
                features_by_step[step] = []
            features_by_step[step].append(feature)
        
        # Save each step's features as a separate file
        for step, features in features_by_step.items():
            step_geo_json = {
                "type": "FeatureCollection",
                "features": features
            }
            step_geo_json_file = os.path.join(geo_json_chunks_dir, f'step_{step}.msgpack')
            with open(step_geo_json_file, 'wb') as file:
                msgpack.dump(step_geo_json, file)
        
        # Save metadata about available steps
        geo_json_metadata = {
            "total_steps": len(features_by_step),
            "available_steps": list(features_by_step.keys())
        }
        geo_json_metadata_file = os.path.join(run_html_data_folder, 'geo_json_metadata.json')
        with open(geo_json_metadata_file, 'w') as f:
            json.dump(geo_json_metadata, f)

        metric_data_file_path = os.path.join(run_html_data_folder, 'metric_data.json')
        with open(metric_data_file_path, 'w') as f:
            json.dump(serializable_metric_data, f)

        # Save actions data
        actions_file_path = os.path.join(run_html_data_folder, 'actions.npy')
        np.save(actions_file_path, np.array(actions_list))
        print(f"Saved {len(actions_list)} actions to {actions_file_path}")

        center_coordinates_file_path = os.path.join(run_html_data_folder, 'center_coordinates.json')
        with open(center_coordinates_file_path, 'w') as f:
            json.dump(center_coordinates, f)

        # Clean up GPU memory and model after each run
        if torch.cuda.is_available():
            del policy  # Delete policy from GPU memory
            torch.cuda.empty_cache()
            print("GPU memory cleaned up after run")

# Save sweep data
run_ids_file_path = os.path.join(html_data_folder, sweep_id, 'run_ids.json')
with open(run_ids_file_path, 'w') as f:
    json.dump(run_ids, f)

# Save sweep IDs
sweep_ids_file_path = os.path.join(html_data_folder, 'sweep_ids.json')
with open(sweep_ids_file_path, 'w') as f:
    json.dump(sweep_ids, f)

# Process JavaScript file
original_js = os.path.join(main_folder, 'misc', 'visualization', 'vis_3d_main.js')
with open(original_js, 'r', encoding='utf-8') as f:
    content = f.read()

new_js = os.path.join(vis_timestamp_dir, 'vis_3d_main.js')
with open(new_js, 'w', encoding='utf-8') as f:
    f.write(content)

# Process HTML file
index_file_path = os.path.join(main_folder, 'misc', 'visualization', 'vis_3d_index.html')
with open(index_file_path, 'r', encoding='utf-8') as f:
    index_content = f.read()


new_index_file_path = os.path.join(vis_timestamp_dir, 'vis_3d_index.html')
with open(new_index_file_path, 'w', encoding='utf-8') as f:
    f.write(index_content)

# Copy files
shutil.copy(os.path.join(main_folder, 'misc', 'visualization', 'vis_3d_styles.css'), vis_timestamp_dir)
