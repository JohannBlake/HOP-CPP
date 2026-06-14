import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import AutoResetWrapper
import numpy as np
import cv2
import inspect
import time
from shapely.strtree import STRtree
from shapely.geometry import MultiPolygon
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # headless backend to allow saving without display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely import prepare
from get_parameters import para
import uuid
from misc.aid.helpful_geo_functions import create_grid_on_earth, create_grid, compute_position_from_geo_coordinate, compute_geo_coordinate_from_position
from shapely import vectorized
from shapely import get_parts, union_all, get_type_id
from pyproj import Geod
from shapely.geometry import Polygon, MultiPolygon
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
import numpy as np
from shapely.validation import make_valid
from geopy.distance import distance
import random
import os
import json
import glob
import shutil
import copy
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import nearest_points
import geopy.distance
from misc.aid.simulate_and_precalculate_radiation_data import create_single_radiation_scenario
from scipy.ndimage import binary_dilation
from scipy.ndimage import rotate
from geopy.point import Point as GeopyPoint
from scipy.interpolate import interp1d
from pyproj import Transformer
import cv2
from skimage import measure
from shapely.strtree import STRtree
from shapely.geometry import LineString
from shapely import affinity

class GymnasiumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    if para.z_rad_type != 'jon' and para.base_timestep != 3.0:
        print("set base timestep back to 3")
    def __init__(self, height_data, radiation_grid_visualization = False, logging_enabled = False, render_mode=None, camera_id=None,camera_name=None,width=None,height=None, is_evaluation=False, flip_when_stuck=True, max_stuck_steps=5):
        # ========================================
        # EPISODE TRACKING AND STATISTICS
        # ========================================

        self.position = np.array([0.0, 0.0])  # Initialize position
        self.current_episode_step = 0
        self.current_overall_step = 0
        self.episode_count = 0
        self.episode_history = []
        
        # Episode statistics preservation for logging (to handle AutoResetWrapper timing)
        self.last_episode_collisions = 0
        self.last_episode_total_reward = 0.0
        self.last_episode_steps = 0
        
        # Collision tracking for episode
        self.num_episode_collisions = 0
        
        # ========================================
        # EVALUATION MODE PARAMETERS
        # ========================================
        self.is_evaluation = is_evaluation
        self.eval_episode_counter = 0
        
        # Goal coverage progression (when para.constant_goal_coverage is False)
        if not para.constant_goal_coverage:
            if self.is_evaluation:
                self.start_coverage_goal = 0.9901 
            else:
                self.start_coverage_goal = 0.1
            self.goal_coverage_percentage_currently = min(0.99, self.start_coverage_goal)
        else:
            self.goal_coverage_percentage_currently = para.goal_coverage_percentage
        
        # Early termination episode step limit progression
        if self.is_evaluation:
            self.early_termination_episode_step_limit_currently = 35000
        else:
            self.early_termination_episode_step_limit_currently = para.early_termination_episode_step_limit
        
        # ========================================
        # CURRICULUM LEARNING PARAMETERS
        # ========================================
        if hasattr(para, 'ablation_study_use_curriculum_learning') and para.ablation_study_use_curriculum_learning:
            self.use_curriculum_learning = True
            self.curriculum_level = 1  # Starting level (same as mower_env start_level)
            self.curriculum_goal_coverage = 0.9  # Initial goal coverage for level 1
            self.use_randomized_envs = False  # Whether to use randomized environments
            # Training maps organized by level (similar to mower_env)
            self._initialize_curriculum_training_maps()
            self._set_curriculum_level(self.curriculum_level)
            # Track completed maps for curriculum advancement
            self.completed_maps = [False] * len(self.curriculum_train_maps)
            self.completed_floor_plan = True
            self.completed_obstacles = True
            self.current_map_index = None
            self.next_train_map = 0
        else:
            self.use_curriculum_learning = False
            self.curriculum_level = 8  # Max level when not using curriculum
            self.curriculum_goal_coverage = 0.99
            self.use_randomized_envs = True
        
        # ========================================
        # STUCK DETECTION AND FLIP PARAMETERS
        # ========================================
        self.flip_when_stuck = flip_when_stuck
        self.max_stuck_steps = max_stuck_steps
        self.stuck_steps = 0
        
        # ========================================
        # REWARD AND METRICS TRACKING
        # ========================================
        self.reward_for_little_action_weight_of_summand = para.reward_for_little_action_weight_of_summand
        self.incremental_tv_reward_normalizer = 1
        self.cone_union_last_cone_convex_hull_max_area = 1
        self.rew_little_action_per_episode = 0 
        self.rew_incremental_tv_per_episode = 0
        self.rew_turning_angle_factor = 1
        self.rew_radiation = 0
        self.count_radiation_exposure_exceeded = 0
        self.current_radiation_exposure_previous = 0
        self.reward = 0
        self.total_reward_per_episode = 0
        self.average_reward_per_episode = 0
        self.total_actions_per_episode = 0
        self.rewards = []
        
        # Incremental goal percentage update tracking (last 10000 steps)
        self.reward_history_10k = []  # Store last 10000 step rewards
        self.reward_10k_sum = 0.0     # Running sum for efficiency
        
        # ========================================
        # COVERAGE AND TARGET AREA TRACKING
        # ========================================
        self.target_area_size = 0.0
        self.percentage_of_target_area_left = 1.0
        self.sum_cone_union_last_cone_convex_hull_in_black_polygons_length = 0.0
        self.target_area_remeasured_1_times = MultiPolygon()
        self.target_area_remeasured_2_times = MultiPolygon()
        self.ter_reason_too_much_remeasured_count = 0
        
        # ========================================
        # COORDINATE TRANSFORMERS
        # ========================================
        self.to_merc = Transformer.from_crs(4326, 3857, always_xy=True)
        self.to_wgs  = Transformer.from_crs(3857, 4326, always_xy=True)
        self.geod = Geod(ellps="WGS84")
        
        # ========================================
        # MOWER PARAMETERS (FOR JON RADIATION TYPE)
        # ========================================
        meters_per_pixel_mower = 0.0375  # 3.125% like mower environment
        self.meters_per_pixel_mower = meters_per_pixel_mower
        radius_in_meters_mower = 0.15
        self.radius_of_mower = radius_in_meters_mower / meters_per_pixel_mower
        
        # ========================================
        # LOGGING CONFIGURATION
        # ========================================
        self.logging_enabled = logging_enabled
        
        # ========================================
        # BASE INITIALIZATION
        # ========================================
        super(GymnasiumEnv, self).__init__()
        
        # ========================================
        # TIME AND SIMULATION STATE
        # ========================================
        self.time_rel = 0
        self.time_abs = para.start_time
        self.base_timestep = para.base_timestep
        self.start_time = para.start_time
        self.end_sim = False
        self.end_sim_reason = None
        self.percentage_of_episode_finished_by_time_passed = 0.0
        
        # ========================================
        # SURFACE AND HEIGHT DATA
        # ========================================
        self.radiation_grid_visualization = radiation_grid_visualization
        self.surface_point_cloud_ = height_data
        self.tree_surface_point_cloud_ = cKDTree(self.surface_point_cloud_[:, :2])
        
        # ========================================
        # RADIATION GRID SETUP
        # ========================================
        # Load the base radiation grid from file (ODL simulator output)
        self.radiation_grid_base = np.load(
            os.path.join('.', 'misc', 'radiation_data', 'odl_map_90m_above_ground_from_berlin_radiation_scenario_constant_2025_2026_23h_after_min.npy')
        )
        
        # Initialize radiation scenarios based on type
        # Initialize radiation scenarios based on type
        if para.z_rad_type == 'mov':
            self.radiation_grid_episode_parameters = self.generate_random_radar_params()
            self.geo_coordinates_of_radiation_grid, self.measuring_area_scenario = create_single_radiation_scenario(
                self.radiation_grid_base, para.granularity
            )
            if self.radiation_grid_visualization:
                self.radiation_grid = self.get_time_dependent_radiation_grid(self.time_rel, self.radiation_grid_episode_parameters)
                
        elif para.z_rad_type == 'bfs':
            self.radiation_grid, self.geo_coordinates_of_radiation_grid, self.measuring_area_scenario = create_single_radiation_scenario(
                self.radiation_grid_base, para.granularity
            )
            self.tree_radiation_grid = cKDTree(self.geo_coordinates_of_radiation_grid.reshape(-1, 2))
            
        elif para.z_rad_type == 'jon':
            # Initialize training image management
            self._initialize_training_image_sizes_without_cache()
            self._initialize_training_image_probabilities()
            
            # Generate and save a random map before selecting a training image
            self.generate_random_map()
            
            # Initialize image cache after all images are available
            self._initialize_image_cache()
            
            # Load image based on whether we're in evaluation mode
            if self.is_evaluation:
                self.IMAGE_PATH = self._get_evaluation_image_path()
            else:
                self.IMAGE_PATH = self._get_random_training_image_path()
                
            self.RADIUS = 15  # Search radius in pixels
            self.radius_for_odl_calculation_jon_improved = para.radius_for_odl_calculation_jon_improved
            
            # Load and process the image
            img = self._get_cached_image(self.IMAGE_PATH)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {self.IMAGE_PATH}")
            
            # Store the image for movement restriction checking
            self.current_image_for_movement_restriction = img.copy()
            
            # Extract black polygons from the image
            self.black_polygons, self.black_polygons_boundary, self.black_polygons_dilated, self.black_polygons_dilated_boundary, self.black_polygons_dilated_for_coverage_calc = self.extract_black_polygons_from_png(img)
            
            # Preprocess the image with morphological operations and blur
            processed_img = self.preprocess_image_for_odl(img)
            
            # Create flat coordinate scenario directly from image dimensions  
            img_height, img_width = img.shape
            self.measuring_area_scenario = {
                'polygon_center': [img_width / 2, img_height / 2],
                'target_area': Polygon([
                    (0, 0),
                    (img_width-1, 0),
                    (img_width-1, img_height-1),
                    (0, img_height-1),
                    (0, 0)
                ])
            }
        
        # Initialize path length metric for jon radiation type
        if para.z_rad_type == 'jon':
            self.length_of_path_in_meters = 0.0
        
        # ========================================
        # GRID AND POSITION SETUP
        # ========================================
        # ========================================
        # GRID AND POSITION SETUP
        # ========================================
        self.anchor_point = para.anchor_point
        self.distance_east_from_anchor_point = para.distance_east_from_anchor_point
        self.distance_north_from_anchor_point = para.distance_north_from_anchor_point
        
        # span_point goes distance_north meters north and distance_east meters east from the anchor point
        self.span_point = geopy.distance.distance(meters=para.distance_east_from_anchor_point).destination(
            geopy.distance.distance(meters=para.distance_north_from_anchor_point).destination(self.anchor_point, 0), 90
        )
        
        self.granularity = para.granularity
        self.grid_coordinates = create_grid(self.anchor_point, self.span_point, self.granularity)
        self.gridsize = np.array([self.grid_coordinates.shape[0], self.grid_coordinates.shape[1], para.max_grid_height])
        
        if para.surface_grid_creation_type == 'no_height_map':
            self.surface_grid = np.zeros(self.gridsize[:2], dtype=int) + 40
        
        self.source_position = np.array([np.random.randint(0, self.gridsize[0]), np.random.randint(0, self.gridsize[1])])
        self.normalized_gridsize = self.gridsize / np.max(self.gridsize)
        self.total_amount_of_measurable_points = self.gridsize[0] * self.gridsize[1]
        self.max_flight_time = para.episode_length_added + para.episode_length_factor * self.base_timestep
        
        # ========================================
        # HELICOPTER/MOVEMENT PARAMETERS
        # ========================================
        self.cumulative_area_too_small = 0
        self.distance_to_move_per_step = para.base_timestep * para.speed_in_meters_per_second
        
        # ========================================
        # CONE ENERGY LOOKUP TABLE
        # ========================================
        self.lookup_table_for_cone_energy_2615 = dict({1: 8.7,  # Now used for circle radius
            10: 58.2,
            15: 77.8,
            20: 94.9,
            25: 110.1,
            30: 124.1,
            35: 137.1,
            40: 149.3,
            45: 160.8,
            50: 171.6,
            55: 182.0,
            60: 192.0,
            65: 201.6,
            70: 210.8,
            75: 219.8,
            80: 228.4,
            85: 236.8,
            90: 245.0,
            95: 253.0,
            100: 260.70000000000005,
            105: 268.3,
            110: 275.70000000000005,
            115: 283.00000000000006,
            120: 290.1,
            125: 297.00000000000006,
            130: 303.90000000000003,
            135: 310.6,
            140: 317.20000000000005,
            145: 323.70000000000005,
            150: 330.00000000000006,
            155: 336.30000000000007,
            160: 342.50000000000006,
            165: 348.6,
            170: 354.6,
            175: 360.50000000000006,
            180: 366.30000000000007,
            185: 372.1,
            190: 377.80000000000007,
            195: 383.4,
            200: 388.9,
            205: 394.4,
            210: 399.80000000000007,
            215: 405.2000000000001,
            220: 410.50000000000006,
            225: 415.7000000000001,
            230: 420.9,
            235: 426.00000000000006,
            240: 431.1,
            245: 436.1,
            250: 441.1,
            255: 446.00000000000006,
            260: 450.9,
            265: 455.80000000000007,
            270: 460.6,
            275: 465.30000000000007,
            280: 470.00000000000006,
            285: 474.7000000000001,
            290: 479.30000000000007,
            295: 483.9,
            300: 488.50000000000006,
            305: 493.00000000000006,
            310: 497.50000000000006,
            315: 502.00000000000006,
            320: 506.4,
            325: 510.80000000000007,
            330: 515.1,
            335: 519.4000000000001,
            340: 523.7,
            345: 528.0,
            350: 532.2})
            
        # Interpolate circle radius values for each meter above ground
        heights = np.array(list(self.lookup_table_for_cone_energy_2615.keys()))
        radii = np.array(list(self.lookup_table_for_cone_energy_2615.values()))
        interpolation_function = interp1d(heights, radii, kind='linear', fill_value='extrapolate')
        self.lookup_table_for_cone_energy_2615_for_each_m = interpolation_function(np.arange(1, 350))
        
        # ========================================
        # CONE POLYGON INITIALIZATION
        # ========================================
        # Pre-create base cone polygons at origin (0,0) for optimization
        # These will be translated to actual positions instead of recreating every step
        base_center = Point(0, 0)
        if para.z_rad_type == 'jon':
            self.base_cone_radius = self.radius_of_mower
        else:
            # Use radius for 90m height as default (can be updated when position_z_above_ground changes)
            self.base_cone_radius = self.lookup_table_for_cone_energy_2615_for_each_m[90]
        
        self.base_cone_polygon = base_center.buffer(self.base_cone_radius)
        # Create epsilon variant for collision detection
        epsilon = 1e-9
        self.base_cone_polygon_epsilon = base_center.buffer(self.base_cone_radius - epsilon)
        
        # ========================================
        # LIDAR PARAMETERS (for ablation study)
        # ========================================
        # Lidar obstacle tracking is needed either when lidar is the sensing
        # modality, or when wall gliding uses borders reconstructed from lidar
        # points (the observation itself stays gated by the ablation flag).
        self.lidar_obstacle_tracking_enabled = (
            not para.ablation_study_use_radiation_instead_of_lidar
            or (para.glide_on_collision and para.glide_use_reconstructed_lidar_borders)
        )
        if self.lidar_obstacle_tracking_enabled:
            self.lidar_rays = 24  # Number of lidar rays
            self.lidar_range = 3.5  # Range in meters
            self.lidar_fov = 180  # Field of view in degrees
            self.lidar_noise = 0.05  # Standard deviation for measurement noise
            # Lidar obstacle maps will be initialized in reset()
            self.known_obstacle_map = None
            self.unknown_obstacle_map = None
            # Discretized obstacle segments for efficient detection
            # Each segment is 0.05m and stores: [x, y, discovered_flag]
            self.obstacle_segments = None  # Will be initialized in reset()
            self.obstacle_segments_discovered = None  # Boolean array marking discovered segments
            self.segment_length_meters = 0.05  # Length of each obstacle segment
        
        # ========================================
        # EPISODE STATE PLACEHOLDERS
        # ========================================
        # These will be properly initialized in reset()
        # Set to None to make it clear they need reset() before use
        self.current_bearing = None
        self.observed_radiation_points = None
        self.observed_radiation_points_freshness = None
        self.polygon_center = None
        self.target_area = None
        self.cone_polygon = None
        self.target_area_borders = None
        self.initial_target_area_borders = None
        self.target_area_size_borders = None
        self.anchor_point = None
        self.height_of_surface_grid_at_start_position = None
        self.observed_radiation_map = None
        self.position_init = None
        self.current_radiation_exposure = None
        self.last_cone_polygon = None
        self.cone_union_last_cone_convex_hull = None
        self.target_area_last_step_border_length = None
        self.target_area_last_step_size = None
        self.length_new_measured_minus_length_old_measured = None
        self.percentage_of_new_area_measured_ = None
        self.position_z_above_ground = None
        self.position_z_above_ground = None
        
        # ========================================
        # OBSERVATION GRID PRECOMPUTATION
        # ========================================
        self.number_of_points_for_grids_on_earth = para.observed_image_size  # has to be odd
        self.list_of_grid_sizes = [2500, 8000, 40000]
        
        # Pre-calculate rotated grids for all 0.1-degree increments (3600 total)
        # For fisheye ablation: stores dict of single grids per angle
        # For multi-scale: stores dict of lists (4 grids) per angle
        self.precomputed_rotated_grids = {}
        self.precomputed_kdtrees = {}
        
        if para.z_rad_type == 'jon':
            # For flat coordinate environment
            if para.ablation_study_fisheye_instead_of_multi_scale_maps:
                # Use single fisheye distorted grid
                base_grid = self.generate_flat_distorted_grid()
                grid_shape = base_grid.shape[:2]
                center_row = grid_shape[0] // 2
                center_col = grid_shape[1] // 2
                center_x, center_y = base_grid[center_row, center_col]
                
                # Pre-calculate for all 0.1-degree increments
                for i in range(3600):  # 360 * 10 = 3600 angles (0.0, 0.1, 0.2, ..., 359.9)
                    angle = i * 0.1
                    angle_key = round(angle, 1)
                    grid_coords_flat = base_grid.reshape(-1, 2)
                    rotated_coords = self.rotate_coords_flat(grid_coords_flat, center_x, center_y, angle)
                    rotated_grid = rotated_coords.reshape(base_grid.shape)
                    self.precomputed_rotated_grids[angle_key] = rotated_grid
                    self.precomputed_kdtrees[angle_key] = cKDTree(rotated_coords)
            else:
                # Use 4 multi-scale flat grids (will be initialized in reset after image is loaded)
                # For now, just set up placeholders
                self.precomputed_multi_scale_grids = {}  # Will store {angle: [grid1, grid2, grid3, grid4]}
                self.precomputed_multi_scale_kdtrees = {}  # Will store {angle: [tree1, tree2, tree3, tree4]}
        else:
            # For geodesic coordinate environment ('mov', 'bfs'), use geodesic coordinate grids
            base_grid = self.generate_distorted_grid()
            grid_shape = base_grid.shape[:2]
            center_row = grid_shape[0] // 2
            center_col = grid_shape[1] // 2
            center_lon, center_lat = base_grid[center_row, center_col]
            
            # Pre-calculate for all 0.1-degree increments
            for i in range(3600):  # 360 * 10 = 3600 angles (0.0, 0.1, 0.2, ..., 359.9)
                angle = i * 0.1
                angle_key = round(angle, 1)
                grid_coords_flat = base_grid.reshape(-1, 2)
                rotated_coords = self.rotate_coords_fast(grid_coords_flat, center_lon, center_lat, angle)
                rotated_grid = rotated_coords.reshape(base_grid.shape)
                self.precomputed_rotated_grids[angle_key] = rotated_grid
                self.precomputed_kdtrees[angle_key] = cKDTree(rotated_coords)
        
        # Placeholder for grids (will be set in reset)
        self.list_of_observation_grids = []
        self.grids_kdtrees = []
        
        # Pre-calculate grid center coordinates (constant for all rotations)
        if para.ablation_study_fisheye_instead_of_multi_scale_maps or para.z_rad_type != 'jon':
            # Use the first precomputed grid to get shape
            first_grid = next(iter(self.precomputed_rotated_grids.values()))
            grid_shape = first_grid.shape[:2]
            self.grid_center_row = grid_shape[0] // 2
            self.grid_center_col = grid_shape[1] // 2
        else:
            # For multi-scale grids, center will be calculated after grids are created
            self.grid_center_row = para.observed_image_size // 2
            self.grid_center_col = para.observed_image_size // 2
        
        # ========================================
        # TERMINATION REASON COUNTERS
        # ========================================
        self.ter_reason_cov_thresh_reached_count = 0
        self.ter_reason_oob_count = 0
        self.ter_reason_neg_rew_count = 0
        self.ter_reason_time_limit_count = 0
        self.ter_reason_manual_termination_count = 0
        
        # ========================================
        # RENDERING AND VISUALIZATION
        # ========================================
        self.render_mode = render_mode
        self.last_time_html_logged = None
        self.last_time_html_saved = None
        self.gymenv_id = str(uuid.uuid4())[:4]
        
        # Initialize position tracking for rendering
        self.position_history = []
        self.previous_episode_data = {}
        
        # ========================================
        # INFO DICTIONARY
        # ========================================
        self.info = {}
        self.info['info_avg_last_100_episodes'] = {
            'total reward': 0,
            'avg reward': 0,
            'total actions': 0,
            'radiation exceeded': 0,
            'target area left': 0,
            'avg little action': 0,
            'avg tv': 0,
        }
        
        # ========================================
        # TIME AND MAX FLIGHT TIME
        # ========================================
        self.max_measured_points_in_between_actions_at_90m = 1
        
        # ========================================
        # ACTION SPACE CONFIGURATION
        # ========================================
        # Action is always a single scalar for turning angle (no height change)
        self.action_space = spaces.Box(low=np.array([-1]), 
                                    high=np.array([1]), 
                                    dtype=np.float64)
        
        # ========================================
        # OBSERVATION SPACE CONFIGURATION
        # ========================================
        self.image_channels = 2
        if para.ablation_study_use_visit_frequency:
            self.image_channels += 1
        if para.ablation_study_use_frontier_maps:
            self.image_channels += 1
        if para.include_time_in_observation:
            self.image_channels += 1

        # Determine number of maps based on ablation study setting
        if para.ablation_study_fisheye_instead_of_multi_scale_maps:
            self.num_observation_maps = 1
        else:
            self.num_observation_maps = 4  # Multi-scale grids


        # ALWAYS use Box observation space with uint8 format
        # Lidar is included in observation when ablation_study_fuse_lidar_sensor_data_and_image_data is True
        self.image_observation_space = spaces.Box(
            low=0, high=255,
            shape=(para.observed_image_size, para.observed_image_size, self.image_channels * self.num_observation_maps),
            dtype=np.uint8
        )
        
        # Check if we should fuse lidar data with image data
        self.fuse_lidar_with_image = getattr(para, 'ablation_study_fuse_lidar_sensor_data_and_image_data', False)
        if self.fuse_lidar_with_image and not para.ablation_study_use_radiation_instead_of_lidar:
            # Create lidar observation space
            self.lidar_observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(self.lidar_rays,),
                dtype=np.float32
            )

        if para.training_library == 'sb3':
            self.observation_space = spaces.Dict({
                "image": self.image_observation_space,
            })
        elif para.training_library == 'omnisafe':
            if self.fuse_lidar_with_image and not para.ablation_study_use_radiation_instead_of_lidar:
                # Use Dict observation space with image and lidar
                self.observation_space = spaces.Dict({
                    "image": self.image_observation_space,
                    "lidar": self.lidar_observation_space,
                })
            else:
                self.observation_space = self.image_observation_space
        
        # ========================================
        # LOGGING CONFIGURATION
        # ========================================
        self.difference_from_anchor_to_current_position_geo = None  # Will be set in reset()
        if self.logging_enabled:
            log_dir = os.path.join(os.path.dirname(__file__), "misc", "logs", "gymenv_logs")
            os.makedirs(log_dir, exist_ok=True)
            self.log_file_path = os.path.join(log_dir, f"gymenv_log_{self.gymenv_id}.json")
        self.last_obs_image = np.zeros((para.observed_image_size, para.observed_image_size, self.image_channels), dtype=np.uint8)
        
        # ========================================
        # RESET REQUIREMENT FLAG
        # ========================================
        self._has_reset = False  # Track if reset() has been called

    def _initialize_reward_normalizers(self):
        """
        Calculate reward normalization factors based on cone geometry.
        This is called from reset() after position is initialized.
        """
        # Create 2 cones at positions that are para.base_timestep * para.speed_in_meters_per_second apart
        distance_between_cones = para.base_timestep * para.speed_in_meters_per_second
        
        # Create first cone at current position
        cone1 = self._get_cone_polygon_at_position(self.position)
        
        # Create second cone at position moved by the calculated distance (arbitrary direction, e.g., eastward)
        position2 = self.position + np.array([distance_between_cones, 0.0])
        cone2 = self._get_cone_polygon_at_position(position2)
        
        # Calculate union of the two cones
        cone_union = self.safe_geometry_union(cone1, cone2)
        
        # Calculate area difference for normalization
        area_diff = cone_union.difference(cone1)
        area = area_diff.area
        if abs(area) == 0:
            raise ValueError("Initial area of cone union minus first cone is zero, cannot normalize TV reward.")
        else:
            self.cone_union_last_cone_convex_hull_max_area = abs(area)
        
        # Calculate flat coordinate length of complex boundary differences
        self.border_length_of_cone_union_last_cone_convex_hull_diff_cone_diff_last_cone = abs(area_diff.length) - 2 * abs(cone1.buffer(0.00000001).boundary.intersection(cone_union).length)
        # Handle division by zero exception
        if self.border_length_of_cone_union_last_cone_convex_hull_diff_cone_diff_last_cone != 0:
            self.incremental_tv_reward_normalizer = - para.weight_incremental_tv_reward/self.border_length_of_cone_union_last_cone_convex_hull_diff_cone_diff_last_cone
        else:
            raise ValueError("Initial border length difference is zero, cannot normalize TV reward.")

    def _get_cone_polygon_at_position(self, position, use_epsilon=False):
        """
        Get cone polygon translated to the specified position.
        
        Args:
            position: numpy array [x, y] representing target position
            use_epsilon: if True, returns the epsilon version (slightly smaller for collision detection)
            
        Returns:
            Shapely Polygon translated to the target position
        """
        x, y = position
        dx, dy = x, y  # Translation from origin (0,0) to target position
        
        if use_epsilon:
            return affinity.translate(self.base_cone_polygon_epsilon, xoff=dx, yoff=dy)
        else:
            # Check if we need to update the base polygon radius for non-jon radiation type
            if para.z_rad_type != 'jon':
                current_radius = self.lookup_table_for_cone_energy_2615_for_each_m[np.ceil(self.position_z_above_ground).astype(int)]
                if abs(current_radius - self.base_cone_radius) > 1e-9:  # Radius changed significantly
                    # Update base polygons with new radius
                    base_center = Point(0, 0)
                    self.base_cone_radius = current_radius
                    self.base_cone_polygon = base_center.buffer(self.base_cone_radius)
                    epsilon = 1e-9
                    self.base_cone_polygon_epsilon = base_center.buffer(self.base_cone_radius - epsilon)
            
            return affinity.translate(self.base_cone_polygon, xoff=dx, yoff=dy)

    def _add_black_border_to_image(self, img, border_size=5):
        """
        Add black pixels around the borders of an image.
        
        Args:
            img: Input image (numpy array)
            border_size: Number of pixels to add to each border (default: 4)
            
        Returns:
            Image with black border added
        """
        if img is None:
            return None
        
        # Add black border using cv2.copyMakeBorder
        bordered_img = cv2.copyMakeBorder(
            img, 
            border_size, border_size, border_size, border_size, 
            cv2.BORDER_CONSTANT, 
            value=0  # Black color for grayscale
        )
        return bordered_img
    
    def _initialize_training_image_sizes_without_cache(self):
        """
        Initialize training image sizes without using cache (used during initial setup).
        """
        base_dir = os.path.join(os.path.dirname(__file__), "misc", "radiation_data", "bw_jon_images_from_paper")
        
        # Find all training images that match the pattern train*
        train_image_pattern = os.path.join(base_dir, "train*.png")
        train_image_paths = glob.glob(train_image_pattern)
        
        if not train_image_paths:
            raise FileNotFoundError(f"No training images found matching pattern: {train_image_pattern}")
        
        # Calculate image sizes
        self.train_images_sizes = {}
        for image_path in train_image_paths:
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img = self._add_black_border_to_image(img)
                if img is not None:
                    # Calculate size as number of pixels
                    image_size = img.shape[0] * img.shape[1]
                    self.train_images_sizes[image_path] = image_size
                else:
                    print(f"Warning: Could not load image {image_path}")
            except Exception as e:
                print(f"Warning: Error loading image {image_path}: {e}")
        
        if not self.train_images_sizes:
            raise ValueError("No valid training images found")
    
    def _initialize_training_image_probabilities(self):
        """
        Initialize training image probabilities based on previously calculated sizes.
        """
        if not hasattr(self, 'train_images_sizes') or not self.train_images_sizes:
            raise ValueError("Training image sizes not initialized. Call _initialize_training_image_sizes_without_cache first.")
        
        # Calculate probabilities inversely proportional to image size squared
        # Use 1/(size) as weight, then normalize
        total_inverse_size_sq = sum(1.0 / (size**0.5) for size in self.train_images_sizes.values())

        self.train_image_probabilities = {}
        for image_path, size in self.train_images_sizes.items():
            inverse_sq_probability = (1.0 / (size**0.5)) / total_inverse_size_sq
            self.train_image_probabilities[image_path] = inverse_sq_probability

        # Store lists for easier random selection
        self.train_image_paths_list = list(self.train_image_probabilities.keys())
        self.train_image_probs_list = list(self.train_image_probabilities.values())
        
        # Ensure probabilities sum exactly to 1 (fix floating point precision issues)
        prob_sum = sum(self.train_image_probs_list)
        if prob_sum != 1.0:
            self.train_image_probs_list = [p / prob_sum for p in self.train_image_probs_list]
    
    def get_image_path_where_probability_of_choosing_image_is_inversly_proportional_to_image_size(self):
        """
        Select a training image path where the probability of selection is inversely proportional to image size.
        Smaller images have higher probability of being selected.
        
        Returns:
            str: Path to selected training image
        """
        if not hasattr(self, 'train_image_paths_list') or not self.train_image_paths_list:
            raise ValueError("Training image probabilities not initialized. Call _initialize_training_image_probabilities first.")
        
        # Use numpy's random choice with probabilities
        selected_path = np.random.choice(
            self.train_image_paths_list, 
            p=self.train_image_probs_list
        )
        
        return selected_path

    def _get_random_training_image_path(self, use_size_based_probability=para.use_size_based_probability):
        """
        With 50% probability selects train_random.png, with 50% probability selects 
        one of the other training images.
        
        Args:
            use_size_based_probability (bool): If True, uses probability inversely proportional to image size
        """
        if use_size_based_probability:
            # 50% probability: generate and use random map
            if random.random() < 0.5:
                self.generate_random_map()
                # Return special identifier for in-memory random map
                return "RANDOM_MAP_IN_MEMORY"
            else:
                # 50% probability: use the probability-based selection method
                return self.get_image_path_where_probability_of_choosing_image_is_inversly_proportional_to_image_size()
        
        # Original logic using base training maps directory
        maps_dir = os.path.join(os.path.dirname(__file__), "misc", "radiation_data","bw_jon_images_from_paper")
        
        # Get all training images (excluding train_random.png since we now use in-memory)
        all_train_images = glob.glob(os.path.join(maps_dir, "train*.png"))
        other_train_images = [img for img in all_train_images if not img.endswith("train_random.png")]
        
        # 50% probability logic
        if random.random() < 0.5:
            # 50% chance: generate and use random map
            self.generate_random_map()
            return "RANDOM_MAP_IN_MEMORY"
        elif other_train_images:
            # 50% chance: use one of the other training images
            selected_image = random.choice(other_train_images)
            return selected_image
        else:
            # Fallback: generate random map if no other images exist
            self.generate_random_map()
            return "RANDOM_MAP_IN_MEMORY"
    
    def _get_evaluation_image_path(self):
        """
        Get the specific evaluation image based on eval_episode_counter.
        Cycles through eval_mowing_1.png to eval_mowing_4.png.
        """
        # Use episode counter to determine which evaluation map to use
        eval_map_id = (int(np.floor(self.eval_episode_counter/2))) + 1
        
        base_dir = os.path.join(os.path.dirname(__file__), "misc", "radiation_data", "bw_jon_images_from_paper")
        eval_image_path = os.path.join(base_dir, f"eval_mowing_{eval_map_id}.png")
        
        if not os.path.exists(eval_image_path):
            raise FileNotFoundError(f"Evaluation image not found: {eval_image_path}")
        
        return eval_image_path
    
    def _initialize_image_cache(self):
        """
        Initialize image cache by loading all images from the relevant directories.
        This prevents imread calls during runtime and handles missing files gracefully.
        """
        self.cached_images = {}
        
        # Get all image directories to cache
        image_directories = []
        
        # Base directory with training and evaluation images
        base_dir = os.path.join(os.path.dirname(__file__), "misc", "radiation_data", "bw_jon_images_from_paper")
        if os.path.exists(base_dir):
            image_directories.append(base_dir)
        
        # Cache all PNG images from these directories
        for directory in image_directories:
            try:
                for filename in os.listdir(directory):
                    if filename.lower().endswith('.png'):
                        image_path = os.path.join(directory, filename)
                        try:
                            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                # Apply black border immediately during caching
                                img_with_border = self._add_black_border_to_image(img)
                                self.cached_images[image_path] = img_with_border
                            else:
                                print(f"Warning: Could not load image {image_path}")
                        except Exception as e:
                            print(f"Warning: Error loading image {image_path}: {e}")
            except Exception as e:
                print(f"Warning: Could not access directory {directory}: {e}")
            
    def _get_cached_image(self, image_path):
        """
        Get a cached image by path. If not found in cache, loads from disk and caches it.
        For random maps, returns the in-memory generated map.
        
        Args:
            image_path: Path to the image or "RANDOM_MAP_IN_MEMORY" for random maps
            
        Returns:
            numpy.ndarray or None: The cached image with black border applied, or None if file not found
        """
        # Handle special case for in-memory random maps
        if image_path == "RANDOM_MAP_IN_MEMORY":
            if hasattr(self, 'random_map_image'):
                return self.random_map_image
            else:
                print("Warning: Random map requested but not generated yet")
                return None
        
        # First try exact path match
        if image_path in self.cached_images:
            return self.cached_images[image_path]
        
        # If exact match fails, try normalized path
        normalized_path = os.path.normpath(image_path)
        if normalized_path in self.cached_images:
            return self.cached_images[normalized_path]
        
        # If still not found, try searching by filename in cache
        filename = os.path.basename(image_path)
        for cached_path, cached_image in self.cached_images.items():
            if os.path.basename(cached_path) == filename:
                return cached_image
        
        # If not in cache, try to load from disk
        if os.path.exists(image_path):
            try:
            
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Apply black border (same as in precompute method)
                    border_width = 1
                    img[:border_width, :] = 0  # Top border
                    img[-border_width:, :] = 0  # Bottom border 
                    img[:, :border_width] = 0  # Left border
                    img[:, -border_width:] = 0  # Right border
                    
                    # Cache the image
                    self.cached_images[normalized_path] = img
                    return img
                else:
                    print(f"Warning: Could not load image from disk: {image_path}")
                    return None
            except Exception as e:
                print(f"Warning: Error loading image {image_path}: {e}")
                return None
        else:
            print(f"Warning: Image file not found: {image_path}")
            return None
    
    def _initialize_curriculum_training_maps(self):
        """
        Initialize training maps organized by level for curriculum learning.
        Similar to mower_env.py structure with train_maps_0 through train_maps_5.
        """
        base_dir = os.path.join(os.path.dirname(__file__), "misc", "radiation_data", "maps_curriculum_learning")
        
        # Organize training maps by level (similar to mower_env naming convention)
        self.train_maps_0 = sorted(glob.glob(os.path.join(base_dir, "train_0_*.png")))
        self.train_maps_1 = sorted(glob.glob(os.path.join(base_dir, "train_1_*.png")))
        self.train_maps_2 = sorted(glob.glob(os.path.join(base_dir, "train_2_*.png")))
        self.train_maps_3 = sorted(glob.glob(os.path.join(base_dir, "train_3_*.png")))
        self.train_maps_4 = sorted(glob.glob(os.path.join(base_dir, "train_4_*.png")))
        self.train_maps_5 = sorted(glob.glob(os.path.join(base_dir, "train_5_*.png")))
        
        # Initialize current training maps (will be set by _set_curriculum_level)
        self.curriculum_train_maps = []
    
    def _set_curriculum_level(self, level):
        """
        Set curriculum learning parameters based on level.
        Uses mowing mode progression from mower_env.py (not exploration mode).
        
        Mowing mode levels:
        Level 1: 90% coverage, train_0
        Level 2: 90% coverage, train_0 + train_1
        Level 3: 95% coverage, train_0 + train_1
        Level 4: 95% coverage, train_0 + train_1 + train_2
        Level 5: 97% coverage, train_0 + train_1 + train_2
        Level 6: 99% coverage, train_0 + train_1 + train_2
        Level 7: 99% coverage, train_0 + train_1 + train_2 + train_3
        Level 8+: 99% coverage, train_0 + train_1 + train_2 + train_3, randomized envs
        """
        if level == 1:
            self.curriculum_goal_coverage = 0.9
            self.use_randomized_envs = False
            self.curriculum_train_maps = self.train_maps_0.copy()
        elif level == 2:
            self.curriculum_goal_coverage = 0.9
            self.use_randomized_envs = False
            self.curriculum_train_maps = self.train_maps_0 + self.train_maps_1
        elif level == 3:
            self.curriculum_goal_coverage = 0.95
            self.use_randomized_envs = False
            self.curriculum_train_maps = self.train_maps_0 + self.train_maps_1
        elif level == 4:
            self.curriculum_goal_coverage = 0.95
            self.use_randomized_envs = False
            self.curriculum_train_maps = self.train_maps_0 + self.train_maps_1 + self.train_maps_2
        elif level == 5:
            self.curriculum_goal_coverage = 0.97
            self.use_randomized_envs = False
            self.curriculum_train_maps = self.train_maps_0 + self.train_maps_1 + self.train_maps_2
        elif level == 6:
            self.curriculum_goal_coverage = 0.99
            self.use_randomized_envs = False
            self.curriculum_train_maps = self.train_maps_0 + self.train_maps_1 + self.train_maps_2
        elif level == 7:
            self.curriculum_goal_coverage = 0.99
            self.use_randomized_envs = False
            self.curriculum_train_maps = self.train_maps_0 + self.train_maps_1 + self.train_maps_2 + self.train_maps_3
        else:  # level >= 8
            self.curriculum_goal_coverage = 0.99
            self.use_randomized_envs = True
            self.curriculum_train_maps = self.train_maps_0 + self.train_maps_1 + self.train_maps_2 + self.train_maps_3
        
        # Reset tracking for completed maps
        self.completed_maps = [False] * len(self.curriculum_train_maps) if self.curriculum_train_maps else []
        self.completed_floor_plan = True
        self.completed_obstacles = True
        
        # Reset probabilities for floor plans/obstacles in randomized mode
        if self.use_randomized_envs:
            # Get mower_env values for random map generation
            self.curriculum_p_use_floor_plans = 0.7
            self.curriculum_p_use_known_obstacles = 0.7
            self.curriculum_p_use_unknown_obstacles = 0.7
            self.curriculum_max_known_obstacles = 100
            self.curriculum_max_unknown_obstacles = 100
            
            if self.curriculum_p_use_floor_plans > 0:
                self.completed_floor_plan = False
            use_known_obstacles = self.curriculum_max_known_obstacles > 0 and self.curriculum_p_use_known_obstacles > 0
            use_unknown_obstacles = self.curriculum_max_unknown_obstacles > 0 and self.curriculum_p_use_unknown_obstacles > 0
            if use_known_obstacles or use_unknown_obstacles:
                self.completed_obstacles = False
        
        print(f"Curriculum learning: Set to level {level}, goal coverage {self.curriculum_goal_coverage*100}%, "
              f"{len(self.curriculum_train_maps)} maps, randomized={self.use_randomized_envs}")
    
    def _get_curriculum_training_image_path(self):
        """
        Get training image path based on curriculum learning state.
        Similar to mower_env's _create_training_env logic.
        """
        if not self.curriculum_train_maps:
            # Fallback to random map if no training maps available
            self.generate_random_map()
            return "RANDOM_MAP_IN_MEMORY"
        
        # Similar to mower_env: use randomized envs 50% of the time when enabled
        if self.use_randomized_envs and random.random() < 0.5:
            # Generate random map with mower_env parameters
            self._generate_curriculum_random_map()
            self.current_map_index = None  # Indicate random map
            return "RANDOM_MAP_IN_MEMORY"
        else:
            # Use next training map from curriculum
            self.current_map_index = self.next_train_map
            self.next_train_map = (self.next_train_map + 1) % len(self.curriculum_train_maps)
            return self.curriculum_train_maps[self.current_map_index]
    
    def _generate_curriculum_random_map(self):
        """
        Generate random map using mower_env parameters when curriculum learning is enabled.
        """
        # Use mower_env default values for random map generation
        self.generate_random_map(
            meters_per_pixel=0.0375,
            mower_radius=0.15,
            obstacle_radius=0.25,
            min_size_p=64,  # mower_env min_size_p for mowing mode
            max_size_p=200,  # mower_env max_size_p for mowing mode
            p_use_floor_plans=self.curriculum_p_use_floor_plans if hasattr(self, 'curriculum_p_use_floor_plans') else 0.7,
            p_use_known_obstacles=self.curriculum_p_use_known_obstacles if hasattr(self, 'curriculum_p_use_known_obstacles') else 0.7,
            p_use_unknown_obstacles=self.curriculum_p_use_unknown_obstacles if hasattr(self, 'curriculum_p_use_unknown_obstacles') else 0.7,
            max_known_obstacles=self.curriculum_max_known_obstacles if hasattr(self, 'curriculum_max_known_obstacles') else 100,
            max_unknown_obstacles=self.curriculum_max_unknown_obstacles if hasattr(self, 'curriculum_max_unknown_obstacles') else 100,
        )
        # Track whether floor plan or obstacles were used for this episode
        self._last_random_map_used_floor_plan = hasattr(self, '_random_map_used_floor_plan') and self._random_map_used_floor_plan
        self._last_random_map_used_obstacles = hasattr(self, '_random_map_used_obstacles') and self._random_map_used_obstacles
    
    def _calculate_coverage_for_curriculum(self):
        """
        Calculate coverage percentage using the original mower_env dilation value
        for curriculum learning advancement.
        
        Returns:
            float: Coverage percentage (0.0 to 1.0)
        """
        if not hasattr(self, 'target_area_size_for_curriculum') or self.target_area_size_for_curriculum == 0:
            return 0.0
        
        # Use target_area_size (accumulated covered area) divided by curriculum-specific start area
        coverage = self.target_area_size / self.target_area_size_for_curriculum
        return min(1.0, coverage)
    
    def _check_curriculum_advancement(self):
        """
        Check if curriculum level should advance based on coverage.
        Similar to mower_env's logic in step method.
        """
        if not self.use_curriculum_learning or self.is_evaluation:
            return
        
        # Calculate coverage using mower_env dilation
        coverage = self._calculate_coverage_for_curriculum()
        
        if coverage >= self.curriculum_goal_coverage:
            # Mark current map as completed
            if self.current_map_index is not None and self.current_map_index < len(self.completed_maps):
                self.completed_maps[self.current_map_index] = True
            else:
                # Random map completed
                if hasattr(self, '_last_random_map_used_floor_plan') and self._last_random_map_used_floor_plan:
                    self.completed_floor_plan = True
                if hasattr(self, '_last_random_map_used_obstacles') and self._last_random_map_used_obstacles:
                    self.completed_obstacles = True
            
            # Check if all requirements are met for level advancement
            all_maps_completed = all(self.completed_maps) if self.completed_maps else True
            if all_maps_completed and self.completed_floor_plan and self.completed_obstacles:
                self.curriculum_level += 1
                self.next_train_map = 0
                self._set_curriculum_level(self.curriculum_level)
                print(f"Curriculum learning: Advanced to level {self.curriculum_level}")
    
    def reset(self, seed=None, options=None, image_path=None):
        # ========================================
        # EXTRACT OPTIONS AND RESET EPISODE STEP
        # ========================================
        self.current_episode_step = 0
        self.current_bearing = 0
        
        # Extract image_path from options if not provided directly
        if image_path is None and options is not None:
            image_path = options.get('image_path', None)
        
        # ========================================
        # RESET COVERAGE METRICS
        # ========================================
        self.target_area_size = 0.0
        self.sum_cone_union_last_cone_convex_hull_in_black_polygons_length = 0.0
        self.target_area_remeasured_1_times = MultiPolygon()
        self.target_area_remeasured_2_times = MultiPolygon()
        
        # ========================================
        # HANDLE SPLIT EPISODES OR FULL RESET
        # ========================================
        # ========================================
        # HANDLE SPLIT EPISODES OR FULL RESET
        # ========================================
        # Increment evaluation episode counter if in evaluation mode
        if self.end_sim_reason == "split_episodes_into_sub_episodes" and not self.is_evaluation:
            self.end_sim = False
            self.end_sim_reason = None
            super().reset(seed=seed)
        else:
            # Initialize last observation image
            self.last_obs_image = np.zeros((para.observed_image_size, para.observed_image_size, self.image_channels), dtype=np.uint8)
            
            # Preserve episode statistics before resetting (for logging)
            self.last_episode_collisions = getattr(self, 'num_episode_collisions', 0)
            self.last_episode_total_reward = getattr(self, 'total_reward_per_episode', 0.0)
            self.last_episode_steps = getattr(self, 'current_episode_step', 0)
            
            # Call parent reset
            super().reset(seed=seed)
            
            # Reset reward tracking variables
            self.count_radiation_exposure_exceeded = 0
            self.rew_little_action_per_episode = 0
            self.rew_incremental_tv_per_episode = 0
            self.current_radiation_exposure_previous = 0
            self.rew_turning_angle_factor = 1
            self.rew_radiation = 0
            self.stuck_steps = 0
            self.num_episode_collisions = 0
            
            # Initialize or cache previous episode data for rendering
            if not hasattr(self, 'previous_episode_data'):
                self.previous_episode_data = {}
        
        # ========================================
        # CACHE PREVIOUS EPISODE DATA FOR RENDERING
        # ========================================
            # ========================================
        # CACHE PREVIOUS EPISODE DATA FOR RENDERING
        # ========================================
        # Cache data from previous episode before reset (for rendering)
        if hasattr(self, 'position_history') and len(self.position_history) > 0:
            self.previous_episode_data['position_history'] = self.position_history.copy()
            
            # Cache the actual image data instead of just the path
            # This is critical for RANDOM_MAP_IN_MEMORY which changes on each reset
            current_image_path = getattr(self, 'IMAGE_PATH', None)
            if current_image_path == "RANDOM_MAP_IN_MEMORY":
                # For random maps, cache the actual image data
                if hasattr(self, 'random_map_image'):
                    self.previous_episode_data['cached_image'] = self.random_map_image.copy()
                else:
                    self.previous_episode_data['cached_image'] = None
                self.previous_episode_data['IMAGE_PATH'] = "RANDOM_MAP_IN_MEMORY"
            else:
                # For file-based maps, we can cache the path since the file doesn't change
                self.previous_episode_data['IMAGE_PATH'] = current_image_path
                self.previous_episode_data['cached_image'] = None
            
            self.previous_episode_data['target_area_size'] = getattr(self, 'target_area_size', 0)
            self.previous_episode_data['length_of_path_in_meters'] = getattr(self, 'length_of_path_in_meters', 0)
            self.previous_episode_data['total_degrees_turned'] = getattr(self, 'total_degrees_turned', 0.0)
            self.previous_episode_data['position_init'] = getattr(self, 'position_init', None)
            self.previous_episode_data['position'] = getattr(self, 'position', None)  # final position
            self.previous_episode_data['percentage_of_target_area_left'] = getattr(self, 'percentage_of_target_area_left', 1.0)
            self.previous_episode_data['current_image_for_movement_restriction'] = getattr(self, 'current_image_for_movement_restriction', None)
            self.previous_episode_data['target_area'] = getattr(self, 'target_area', None)
            self.previous_episode_data['black_polygons'] = getattr(self, 'black_polygons', None)
        
        # Initialize position history for new episode
        # Do this early to avoid issues with find_safe_starting_point
        self.position_history = []
        self.total_degrees_turned = 0.0
        
        # ========================================
        # RESET SIMULATION STATE
        # ========================================
        self.time_rel = 0
        self.end_sim_reason = None
        self.end_sim = False        
        self.time_abs = para.start_time
        self.percentage_of_episode_finished_by_time_passed = 0.0
        
        # Regenerate surface grid if using perlin noise
        if para.surface_grid_creation_type == 'perlin noise':
            self.surface_grid = self.generate_surface_grid_via_perlin_noise()
        
        # ========================================
        # RESET OBSERVATION STATE
        # ========================================
        self.observed_radiation_points = np.empty((0, 3))
        self.observed_radiation_points_freshness = np.empty((0, 3))
        
        # ========================================
        # REGENERATE RADIATION SCENARIOS
        # ========================================
        # ========================================
        # REGENERATE RADIATION SCENARIOS
        # ========================================
        # Regenerate radiation scenarios for each reset
        if para.z_rad_type == 'mov':
            self.radiation_grid_episode_parameters = self.generate_random_radar_params()
            self.geo_coordinates_of_radiation_grid, self.measuring_area_scenario = create_single_radiation_scenario(
                self.radiation_grid_base, para.granularity
            )
            if self.radiation_grid_visualization:
                self.radiation_grid = self.get_time_dependent_radiation_grid(self.time_rel, self.radiation_grid_episode_parameters)
                
        elif para.z_rad_type == 'bfs':
            self.radiation_grid, self.geo_coordinates_of_radiation_grid, self.measuring_area_scenario = create_single_radiation_scenario(
                self.radiation_grid_base, para.granularity
            )
            self.tree_radiation_grid = cKDTree(self.geo_coordinates_of_radiation_grid.reshape(-1, 2))
            
        elif para.z_rad_type == 'jon':
            # Generate and save a random map before selecting a new training image
            self.generate_random_map()
            
            # Load image based on provided image_path or evaluation/training mode
            # IMPORTANT: Always respect provided image_path, regardless of evaluation mode
            if image_path is not None:
                self.IMAGE_PATH = image_path
            elif self.is_evaluation:
                pass  # Keep current evaluation image
            elif self.use_curriculum_learning:
                # Use curriculum-based image selection
                self.IMAGE_PATH = self._get_curriculum_training_image_path()
            else:
                self.IMAGE_PATH = self._get_random_training_image_path()

            # Load and process the new image
            img = self._get_cached_image(self.IMAGE_PATH)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {self.IMAGE_PATH}")
            
            # Store the image for movement restriction checking
            self.current_image_for_movement_restriction = img.copy()
            
            # Extract black polygons from the image
            self.black_polygons, self.black_polygons_boundary, self.black_polygons_dilated, self.black_polygons_dilated_boundary, self.black_polygons_dilated_for_coverage_calc = self.extract_black_polygons_from_png(img)
                            
            # Preprocess the image with morphological operations and blur
            self.precalculated_odl_map = self.preprocess_image_for_odl(img)
                        
            # Create flat coordinate scenario directly from image dimensions
            img_height, img_width = img.shape
            self.measuring_area_scenario = {
                'polygon_center': [img_width / 2, img_height / 2],
                'target_area': Polygon([
                    (0, 0),
                    (img_width-1, 0),
                    (img_width-1, img_height-1),
                    (0, img_height-1),
                    (0, 0)
                ])
            }
            
            # Create multi-scale grids if not using fisheye ablation study
            if not para.ablation_study_fisheye_instead_of_multi_scale_maps:
                # Precompute rotations for all 4 grids (only on first reset)
                if not self._has_reset:
                    self.create_multi_scale_grids()
                    print("Precomputing multi-scale grid rotations...")
                    self.precompute_multi_scale_grid_rotations()
                    print("Multi-scale grid rotations precomputed!")
        
        # ========================================
        # SETUP TARGET AREA AND POSITION
        # ========================================
        # ========================================
        # SETUP TARGET AREA AND POSITION
        # ========================================
        self.polygon_center = self.measuring_area_scenario['polygon_center']
        self.target_area = MultiPolygon()
        
        # Store initial target area for position clipping (used with jon radiation type)
        self.target_area_borders = self.measuring_area_scenario['target_area']
        if isinstance(self.target_area_borders, Polygon):
            self.target_area_borders = MultiPolygon([self.target_area_borders])
        self.initial_target_area_borders = copy.deepcopy(self.target_area_borders)
        area = self.target_area_borders.area
        self.target_area_size_borders = abs(area)
        
        # Calculate available area by subtracting black polygons from target area borders
        if hasattr(self, 'black_polygons_dilated_for_coverage_calc') and self.black_polygons_dilated_for_coverage_calc is not None:
            target_area_minus_black_polygons = self.safe_geometry_difference(self.target_area_borders, self.black_polygons_dilated_for_coverage_calc)
            if target_area_minus_black_polygons is not None:
                self.target_area_size_start_of_episode = abs(target_area_minus_black_polygons.area)
        
        # Calculate target area for curriculum learning using original mower_env dilation
        # This uses dilation_for_coverage_limit_original_mower_env_value_pixels instead of dilation_for_coverage_limit_calculation_m
        if self.use_curriculum_learning and hasattr(self, 'black_polygons') and self.black_polygons is not None:
            # Apply mower_env dilation (obstacle_dilation=9 in pixel space)
            mower_env_dilation_pixels = para.dilation_for_coverage_limit_original_mower_env_value_pixels
            dilated_for_curriculum = []
            
            for poly in self.black_polygons.geoms if hasattr(self.black_polygons, 'geoms') else [self.black_polygons]:
                try:
                    dilated_poly = poly.buffer(mower_env_dilation_pixels)
                    if dilated_poly.is_valid and not dilated_poly.is_empty:
                        if hasattr(dilated_poly, 'geoms'):
                            dilated_for_curriculum.extend(dilated_poly.geoms)
                        else:
                            dilated_for_curriculum.append(dilated_poly)
                except Exception:
                    dilated_for_curriculum.append(poly)
            
            curriculum_dilated_polygons = MultiPolygon(dilated_for_curriculum) if dilated_for_curriculum else MultiPolygon()
            curriculum_area_minus_black = self.safe_geometry_difference(self.target_area_borders, curriculum_dilated_polygons)
            if curriculum_area_minus_black is not None:
                self.target_area_size_for_curriculum = abs(curriculum_area_minus_black.area)
            else:
                self.target_area_size_for_curriculum = self.target_area_size_start_of_episode
        else:
            self.target_area_size_for_curriculum = getattr(self, 'target_area_size_start_of_episode', 0)
        
        # Initialize bearing and observation state
        self.current_bearing = 0
        self.observed_radiation_points = np.empty((0, 3))
        self.observed_radiation_points_freshness = np.empty((0, 3))
        
        # Initialize lidar obstacle maps if using lidar ablation study or
        # lidar-reconstructed wall gliding
        if self.lidar_obstacle_tracking_enabled:
            if para.z_rad_type == 'jon':
                # For flat coordinate environment, use pixel-based maps
                img_height, img_width = self.current_image_for_movement_restriction.shape
                self.known_obstacle_map = np.zeros((img_height, img_width), dtype=np.uint8)
                self.unknown_obstacle_map = np.zeros((img_height, img_width), dtype=np.uint8)
                
                # Initialize known obstacles from black polygons (inverted image)
                # Black areas (0) in image are obstacles
                self.known_obstacle_map = (self.current_image_for_movement_restriction == 0).astype(np.uint8)
                
                # Discretize obstacle boundaries into segments for efficient lidar detection
                # Key optimization: only store ONE point per segment (0.05m length)
                self._discretize_obstacle_boundaries()
            else:
                # For other radiation types, we'd need to adapt this
                # For now, just initialize empty maps
                self.known_obstacle_map = np.zeros((100, 100), dtype=np.uint8)
                self.unknown_obstacle_map = np.zeros((100, 100), dtype=np.uint8)
                self._discretize_obstacle_boundaries()
        
        # Find safe starting point
        self.position = np.array([self.polygon_center[0] - 1, self.polygon_center[1]])  # Temporary position for find_safe_starting_point
        random_point_on_boundary = self.find_safe_starting_point()
        self.anchor_point = np.array([random_point_on_boundary.x, random_point_on_boundary.y])
        distances, indices = self.tree_surface_point_cloud_.query(np.array([self.anchor_point]))
        
        if para.surface_grid_creation_type == 'no_height_map':
            self.height_of_surface_grid_at_start_position = 0
        else:
            self.height_of_surface_grid_at_start_position = self.surface_point_cloud_[indices[0]][2]
        
        # Clear position_history before setting initial position
        # This prevents positions from find_safe_starting_point from being included
        self.position_history = []
        
        # Update position and radiation exposure. Initial placement, not
        # movement: pre-set the position so the swept-area movement physics
        # don't reject the teleport to the start point.
        self.position = np.asarray(self.anchor_point, dtype=float).flatten()[:2].copy()
        self.update_position(self.anchor_point)
        
        # Initialize measurement state attributes
        self.current_radiation_exposure = 0
        self.last_cone_polygon = Polygon()
        self.cone_union_last_cone_convex_hull = Polygon()
        self.target_area_last_step_border_length = 0
        self.target_area_last_step_size = 0
        self.length_new_measured_minus_length_old_measured = 0
        self.percentage_of_new_area_measured_ = 0
        self.position_z_above_ground = 90
        
        # Reset path length metric for jon radiation type
        if para.z_rad_type == 'jon':
            self.length_of_path_in_meters = 0.0

        # ========================================
        # SETUP OBSERVATION GRIDS
        # ========================================
        self.observed_radiation_map = np.zeros((self.gridsize[0], self.gridsize[1]))
        self.cone_polygon = Polygon()
        self.update_radiation_exposure_history_target_area()
        self.position_init = self.position
        self.list_of_observation_grids = []
        self.grids_kdtrees = []
        
        # Update grids based on current bearing using pre-computed values
        self._update_grids_for_bearing(self.current_bearing)
        
        # Enable fast grid mapping and calculate base grid distances (only on first reset)
        if not self._has_reset:
            self._enable_fast_grid_mapping(True)
            self._calculate_base_grid_distances()
        
        # ========================================
        # INITIALIZE REWARD NORMALIZERS
        # ========================================
        self._initialize_reward_normalizers()
        
        # ========================================
        # GET INITIAL OBSERVATION
        # ========================================
        obs = self._get_obs()
        self.observation = obs
        
        # ========================================
        # RESET REWARD VARIABLES
        # ========================================
        self.reward = 0
        self.reward_for_taking_or_ignoring_step = 0
        self.punishment_for_ignored_steps = 0
        self.total_reward_per_episode = 0
        self.average_reward_per_episode = 0
        self.total_actions_per_episode = 0
        
        # ========================================
        # SET TIME AND MAX FLIGHT TIME
        # ========================================
        if para.set_episode_length_based_on_first_episode_length:
            print("Setting max_flight_time based on first episode length")
        else:
            self.set_time_and_max_flight_time()
        
        # ========================================
        # SET LOGGING POSITION DIFFERENCE
        # ========================================
        self.difference_from_anchor_to_current_position_geo = np.array(self.position) - np.array([para.anchor_point[0], para.anchor_point[1]])
            
        # ========================================
        # INCREMENT EVAL EPISODE COUNTER IF NEEDED
        # ========================================
        # Only increment eval_episode_counter when using automatic evaluation image selection
        # (i.e., when no specific image_path was provided)
        if self.is_evaluation and para.z_rad_type == 'jon' and image_path is None:
            self.eval_episode_counter += 1
        
        # ========================================
        # MARK RESET AS COMPLETE
        # ========================================
        self._has_reset = True
        
        return self.observation, {}
    def _update_grids_for_bearing(self, bearing):
        """
        Update list_of_observation_grids and grids_kdtrees based on current bearing.
        Uses pre-computed rotated grids for performance.
        
        For fisheye ablation: loads 1 precomputed grid
        For multi-scale: loads 4 precomputed grids
        """
        # Convert to float and normalize bearing to [0, 360)
        bearing_float = - float(bearing)
        bearing_normalized = bearing_float % 360
        
        # Round to nearest 0.1 degree for lookup
        bearing_rounded = round(bearing_normalized * 10) / 10.0
        bearing_key = round(bearing_rounded, 1)
        
        # Handle edge case where rounding might give 360.0
        if bearing_key >= 360.0:
            bearing_key = 0.0
        
        # Store current bearing key for arithmetic grid mapping
        self._current_bearing_key = bearing_key
        
        self.list_of_observation_grids = []
        self.grids_kdtrees = []
        
        if para.z_rad_type == 'jon':
            # For flat coordinate environment
            if para.ablation_study_fisheye_instead_of_multi_scale_maps:
                # Use single fisheye distorted grid
                self.list_of_observation_grids.append(self.precomputed_rotated_grids[bearing_key])
                self.grids_kdtrees.append(self.precomputed_kdtrees[bearing_key])
            else:
                # Use 4 multi-scale grids
                if hasattr(self, 'precomputed_multi_scale_grids') and bearing_key in self.precomputed_multi_scale_grids:
                    self.list_of_observation_grids = self.precomputed_multi_scale_grids[bearing_key]
                    self.grids_kdtrees = self.precomputed_multi_scale_kdtrees[bearing_key]
                else:
                    print(f"Warning: Multi-scale grids not precomputed for bearing {bearing_key}")
        else:
            # For geodesic coordinate environment ('mov', 'bfs'), use geodesic grids
            # Use pre-computed rotated distorted grid
            self.list_of_observation_grids.append(self.precomputed_rotated_grids[bearing_key])
            self.grids_kdtrees.append(self.precomputed_kdtrees[bearing_key])

    def generate_distorted_grid(self):
        n = para.observed_image_size
        range_m = 5 #para.range_m_distorted_observed_grid
        
        # === TUNABLE FISHEYE PARAMETERS ===
        # Adjust these variables to fine-tune the fisheye distortion
        threshold_radius = 1     # Radius where linear scaling starts (0.0 to 1.0)
        linear_scale_factor = 10.0      # Multiplier for linear scaling intensity
        
        """
        Generate a fisheye distorted grid with equidistant spacing in center and linear increase outside.
        
        The grid has the following characteristis:
        - Constant distances for all points within radius threshold_radius (equidistant center)
        - Linear increasing distances for points farther outside
        
        Tunable Parameters (modify above):
            threshold_radius (float): Radius where linear scaling starts (0.0-1.0, default 0.25)
            linear_scale_factor (float): Intensity of linear scaling (default 2.0)
            equidistant_zone_enabled (bool): Enable constant spacing in center (default True)
            use_smooth_transition (bool): Smooth transition between zones (default False)
            smooth_transition_width (float): Width of transition zone (default 0.05)
        
        Parameters:
            n (int): Number of points along each dimension.
            range_m (float): Range for the grid in meters.
            a (float): Not used in new implementation.
        
        Returns:
            np.ndarray: Distorted grid of shape (n, n, 2) with latitude and longitude values.
        """
        # Use polygon_center if position is not yet set (during initialization)
        if self.position is None:
            center_lon, center_lat = self.measuring_area_scenario['polygon_center']
        else:
            center_lon, center_lat = self.position  # Center of the grid

        # Create normalized grid coordinates from -1 to 1
        coords_1d = np.linspace(-1, 1, n)
        y_grid, x_grid = np.meshgrid(coords_1d, coords_1d, indexing='ij')
        
        # Calculate radial distance from center (0 to sqrt(2) for corners)
        r_normalized = np.sqrt(x_grid**2 + y_grid**2)
        max_radius = np.sqrt(2)  # Corner distance in normalized coordinates
        
        # Create distortion factor
        distortion_factor = np.ones_like(r_normalized)
        
        # Apply fisheye transformation with equidistant center
        beyond_threshold = r_normalized > threshold_radius
    
        # Sharp transition at threshold
        if np.any(beyond_threshold):
            # Linear scaling from threshold to max radius
            linear_scale = 1 + (r_normalized[beyond_threshold] - threshold_radius) / (max_radius - threshold_radius) * linear_scale_factor
            distortion_factor[beyond_threshold] = linear_scale
        # Apply distortion (radial scaling)
        # Avoid division by zero at center
        safe_r = np.where(r_normalized == 0, 1e-10, r_normalized)
        scale_factor = distortion_factor * r_normalized / safe_r
        
        # Apply scaling to coordinates
        x_distorted = x_grid * scale_factor
        y_distorted = y_grid * scale_factor
        
        # Convert distorted normalized coordinates to metric distances
        x_distances = x_distorted * (range_m / 2)
        y_distances = y_distorted * (range_m / 2)
        
        # Convert metric distances to geographic coordinates using geodesic calculations
        # Vectorized approach for better performance
        lat_points = np.zeros_like(y_distances)
        lon_points = np.zeros_like(x_distances)
        
        # Calculate geographic coordinates for each point
        for i in range(n):
            for j in range(n):
                # North-south displacement (y_distances[i,j])
                if y_distances[i, j] != 0:
                    _, lat_temp, _ = self.geod.fwd(center_lon, center_lat, 0 if y_distances[i, j] > 0 else 180, abs(y_distances[i, j]))
                else:
                    lat_temp = center_lat
                
                # East-west displacement (x_distances[i,j]) 
                if x_distances[i, j] != 0:
                    lon_temp, _, _ = self.geod.fwd(center_lon, lat_temp, 90 if x_distances[i, j] > 0 else 270, abs(x_distances[i, j]))
                else:
                    lon_temp = center_lon
                
                lat_points[i, j] = lat_temp
                lon_points[i, j] = lon_temp

        # Stack to form an array of shape (n, n, 2) with each element as a (lon, lat) pair
        grid = np.stack((lon_points, lat_points), axis=-1)
        return grid
    
    def generate_flat_distorted_grid(self):
        """
        Generate a fisheye distorted grid in flat coordinate space for 'jon' radiation type.
        Returns grid in image pixel coordinates instead of geodesic coordinates.
        """
        n = para.observed_image_size
        range_pixels = 40  # Range for the grid in pixels
        
        # === TUNABLE FISHEYE PARAMETERS ===
        threshold_radius = 0.426692    # Radius where scaling starts (0.0 to 1.0)
        linear_scale_factor = 52.174442    # Multiplier for scaling intensity
        distortion_exponent = 3.072314    # Exponent for smooth scaling (1.0 = linear, >1.0 = super-linear)
        
        # Center position in image coordinates
        # Position must be set before calling this method
        if self.position is None:
            raise ValueError("Position must be set before generating flat distorted grid")
        center_x, center_y = self.position  # These are flat coordinates
        
        # Create normalized grid coordinates from -1 to 1
        coords_1d = np.linspace(-1, 1, n)
        y_grid, x_grid = np.meshgrid(coords_1d, coords_1d, indexing='ij')
        
        # Calculate radial distance from center (0 to sqrt(2) for corners)
        r_normalized = np.sqrt(x_grid**2 + y_grid**2)
        max_radius = np.sqrt(2)  # Corner distance in normalized coordinates
        
        # Create distortion factor
        distortion_factor = np.ones_like(r_normalized)
        
        # Apply fisheye transformation with equidistant center
        beyond_threshold = r_normalized > threshold_radius

        # Smooth transition at threshold with exponent
        if np.any(beyond_threshold):
            excess_radius = r_normalized[beyond_threshold] - threshold_radius
            distortion_factor[beyond_threshold] = 1.0 + linear_scale_factor * (excess_radius ** distortion_exponent)
            
        # Apply distortion (radial scaling)
        # Avoid division by zero at center
        safe_r = np.where(r_normalized == 0, 1e-10, r_normalized)
        scale_factor = distortion_factor * r_normalized / safe_r
        
        # Apply scaling to coordinates
        x_distorted = x_grid * scale_factor
        y_distorted = y_grid * scale_factor
        
        # Convert distorted normalized coordinates to pixel distances
        x_distances = x_distorted * (range_pixels / 2)
        y_distances = y_distorted * (range_pixels / 2)
        
        # Convert to absolute image coordinates
        x_points = center_x + x_distances
        y_points = center_y + y_distances
        
        # Stack to form an array of shape (n, n, 2) with each element as a (x, y) pair
        grid = np.stack((x_points, y_points), axis=-1)
        return grid
    
    def generate_flat_grid(self, grid_size_in_pixels):
        """
        Generate a regular flat grid in image pixel coordinates for 'jon' radiation type.
        """
        n = para.observed_image_size
        
        # Center position in image coordinates
        center_x, center_y = self.position  # These are now flat coordinates
        
        # Create regular grid
        half_size = grid_size_in_pixels / 2
        coords_1d = np.linspace(-half_size, half_size, n)
        y_grid, x_grid = np.meshgrid(coords_1d, coords_1d, indexing='ij')
        
        # Convert to absolute image coordinates
        x_points = center_x + x_grid
        y_points = center_y + y_grid
        
        # Stack to form an array of shape (n, n, 2) with each element as a (x, y) pair
        grid = np.stack((x_points, y_points), axis=-1)
        return grid
    
    def create_multi_scale_grids(self):
        """
        Create 4 grids with equal pixel spacing at different scales.
        Uses the same dimensions as mower_env.py: input_size=32, scale_factor=4, num_maps=4 (when using lidar)
        or para.observed_image_size (when using radiation instead of lidar).
        Grid sizes: 32px, 128px, 512px, 2048px (or 1.2m, 4.8m, 19.2m, 76.8m) when using lidar.
        
        Stores the grids as class attributes:
        - self.multi_scale_grids: list of 4 grids, each with shape (n, n, 2)
        - self.multi_scale_grid_sizes_px: list of grid sizes in pixels
        - self.multi_scale_grid_sizes_m: list of grid sizes in meters
        """
        # Get image dimensions and center
        if not hasattr(self, 'current_image_for_movement_restriction') or self.current_image_for_movement_restriction is None:
            raise ValueError("No image available for grid creation")
            
        img = self.current_image_for_movement_restriction
        img_height, img_width = img.shape
        
        # Use helicopter position as center (same as fisheye grid)
        # Position must be set before calling this method
        if self.position is None:
            raise ValueError("Position must be set before creating multi-scale grids")
        center_x, center_y = self.position  # These are flat coordinates
        
        # Store the actual grid center for use in precompute_multi_scale_grid_rotations
        # This is the true center (heli position), not a pixel position
        # Important for even-sized grids (e.g., 32x32) where center is between pixels
        self.multi_scale_grid_center = (center_x, center_y)
        
        # Define 4 different grid sizes in pixels matching mower_env.py multi-scale approach
        # input_size * scale_factor^n where input_size=32 (for lidar) or para.observed_image_size (for radiation), scale_factor=4, n=0,1,2,3
        # Use para.observed_image_size when ablation_study_use_radiation_instead_of_lidar is true to maintain consistency
        input_size = para.observed_image_size
        scale_factor = 4
        meters_per_pixel = self.meters_per_pixel_mower  # 0.0375 m/px
        self.multi_scale_grid_sizes_px = [input_size * (scale_factor ** n) for n in range(4)]
        self.multi_scale_grid_sizes_m = [s * meters_per_pixel for s in self.multi_scale_grid_sizes_px]
        
        # Create grids at each scale with resolution matching input_size
        n = input_size  # Grid resolution (32 for lidar, para.observed_image_size for radiation)
        self.multi_scale_grids = []
        
        for grid_size_px in self.multi_scale_grid_sizes_px:
            # Create regular grid
            half_size = grid_size_px / 2
            coords_1d = np.linspace(-half_size, half_size, n)
            y_grid, x_grid = np.meshgrid(coords_1d, coords_1d, indexing='ij')
            
            # Convert to absolute image coordinates
            x_points = center_x + x_grid
            y_points = center_y + y_grid
            
            # Stack to form an array of shape (n, n, 2) with each element as a (x, y) pair
            grid = np.stack((x_points, y_points), axis=-1)
            self.multi_scale_grids.append(grid)
    
    def precompute_multi_scale_grid_rotations(self):
        """
        Precompute rotated versions of all 4 multi-scale grids for all 0.1-degree increments.
        This should be called once after create_multi_scale_grids() in reset().
        
        Stores results in:
        - self.precomputed_multi_scale_grids: {angle: [grid1, grid2, grid3, grid4]}
        - self.precomputed_multi_scale_kdtrees: {angle: [tree1, tree2, tree3, tree4]}
        - self.precomputed_multi_scale_transforms: {angle: [(cos, sin, pixel_size), ...]} for arithmetic mapping
        """
        if not hasattr(self, 'multi_scale_grids') or not self.multi_scale_grids:
            print("Warning: multi_scale_grids not initialized. Call create_multi_scale_grids() first.")
            return
        
        self.precomputed_multi_scale_grids = {}
        self.precomputed_multi_scale_kdtrees = {}
        self.precomputed_multi_scale_transforms = {}
        
        # Get center point - use the stored actual center (heli position at grid creation time)
        # For even-sized grids (e.g., 32x32), the true center is between pixels,
        # so we can't extract it from a pixel index like [16,16]
        center_x, center_y = self.multi_scale_grid_center
        
        # Get grid shape for other calculations
        base_grid = self.multi_scale_grids[0]
        grid_shape = base_grid.shape[:2]
        
        # Precompute pixel sizes for each grid scale
        n = grid_shape[0]  # Grid resolution (e.g., 32)
        self.multi_scale_pixel_sizes = []
        for grid_size_px in self.multi_scale_grid_sizes_px:
            # Pixel size = total grid extent / (n - 1)
            pixel_size = grid_size_px / (n - 1) if n > 1 else grid_size_px
            self.multi_scale_pixel_sizes.append(pixel_size)
        
        # Pre-calculate for all 0.1-degree increments
        for i in range(3600):  # 360 * 10 = 3600 angles (0.0, 0.1, 0.2, ..., 359.9)
            angle = i * 0.1
            angle_key = round(angle, 1)
            
            rotated_grids = []
            kdtrees = []
            transforms = []
            
            # Precompute rotation values for inverse rotation
            # The grid is rotated by +angle, so to get back to local coords we need to rotate by -angle
            # But since bearing_key = (-bearing) % 360, the angle stored is already negated
            # So we need to use +angle here (not -angle) to get the correct inverse
            angle_rad = np.deg2rad(angle)  # Positive angle for inverse rotation (bearing_key is already negated)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # Rotate all 4 grids
            for grid_idx, grid in enumerate(self.multi_scale_grids):
                grid_coords_flat = grid.reshape(-1, 2)
                rotated_coords = self.rotate_coords_flat(grid_coords_flat, center_x, center_y, angle)
                rotated_grid = rotated_coords.reshape(grid.shape)
                rotated_grids.append(rotated_grid)
                kdtrees.append(cKDTree(rotated_coords))
                # Store transform parameters: (cos, sin, pixel_size, half_size, n, center_x, center_y)
                half_size = self.multi_scale_grid_sizes_px[grid_idx] / 2
                transforms.append((cos_a, sin_a, self.multi_scale_pixel_sizes[grid_idx], half_size, n, center_x, center_y))
            
            self.precomputed_multi_scale_grids[angle_key] = rotated_grids
            self.precomputed_multi_scale_kdtrees[angle_key] = kdtrees
            self.precomputed_multi_scale_transforms[angle_key] = transforms
    
    def _coords_to_grid_indices_arithmetic(self, coords, grid_idx, offset_to_heli):
        """
        Convert world coordinates to grid pixel indices using direct arithmetic calculation.
        Much faster than KD-tree for uniform multi-scale grids.
        
        For multi-scale grids, the grids are uniform (equal pixel spacing) but rotated.
        To map a world point to grid indices:
        1. Subtract grid_center to get coordinates relative to rotation center
        2. Apply inverse rotation to get coordinates in the grid's local (unrotated) frame
        3. Convert local coordinates to pixel indices using uniform spacing
        
        Args:
            coords: (N, 2) array of query coordinates (already adjusted: boundary_coords - offset_to_heli)
            grid_idx: Index of the grid (0-3 for multi-scale)
            offset_to_heli: Offset applied to grid (current_heli - grid_center)
            
        Returns:
            (row_indices, col_indices) arrays for each input point
        """
        if len(coords) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        
        # Get current transform parameters
        bearing_key = self._current_bearing_key
        cos_a, sin_a, pixel_size, half_size, n, center_x, center_y = self.precomputed_multi_scale_transforms[bearing_key][grid_idx]
        
        # Step 1: Subtract grid_center to get coordinates relative to rotation center
        # Input coords = boundary_coords - offset_to_heli = boundary_coords - current_heli + grid_center
        # Subtracting grid_center gives: boundary_coords - current_heli (relative to current heli)
        # But we need relative to grid_center for rotation, so subtract grid_center from query coords
        rel_coords = coords - np.array([center_x, center_y])
        
        # Step 2: Apply inverse rotation to get coordinates in grid's local frame
        # Inverse rotation: [x', y'] = [x*cos + y*sin, -x*sin + y*cos]
        local_x = rel_coords[:, 0] * cos_a + rel_coords[:, 1] * sin_a
        local_y = -rel_coords[:, 0] * sin_a + rel_coords[:, 1] * cos_a
        
        # Step 3: Convert local coordinates to pixel indices
        # Grid local coordinates range from -half_size to +half_size
        # Pixel indices range from 0 to n-1
        # Formula: idx = (local_coord + half_size) / pixel_size
        col_indices = (local_x + half_size) / pixel_size
        row_indices = (local_y + half_size) / pixel_size
        
        # Round and clamp to valid range
        row_indices = np.clip(np.round(row_indices).astype(np.int32), 0, n - 1)
        col_indices = np.clip(np.round(col_indices).astype(np.int32), 0, n - 1)
        
        return row_indices, col_indices
    
    def rotate_coords_flat(self, coords, origin_x, origin_y, angle_deg):
        """
        Rotate coordinates in flat image space (simple 2D rotation).
        """
        coords_np = np.array(coords)
        
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Convert to local coordinates relative to origin
        dx = coords_np[:, 0] - origin_x
        dy = coords_np[:, 1] - origin_y
        
        # Apply rotation matrix
        dx_rot = dx * cos_a - dy * sin_a
        dy_rot = dx * sin_a + dy * cos_a
        
        # Convert back to absolute coordinates
        new_x = origin_x + dx_rot
        new_y = origin_y + dy_rot
        
        return np.column_stack([new_x, new_y])
    
    def rotate_coords_fast(self, coords, origin_lon, origin_lat, angle_deg):
        """
        Fast approximate rotation for small distances using vectorized NumPy operations.
        Much faster than geodesic rotation for local coordinates.
        """
        coords_np = np.array(coords)
        
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Convert to local coordinates (approximate for small distances)
        dx = (coords_np[:, 0] - origin_lon) * np.cos(np.deg2rad(origin_lat))
        dy = coords_np[:, 1] - origin_lat
        
        # Apply rotation matrix
        dx_rot = dx * cos_a - dy * sin_a
        dy_rot = dx * sin_a + dy * cos_a
        
        # Convert back to geographic coordinates
        new_lon = origin_lon + dx_rot / np.cos(np.deg2rad(origin_lat))
        new_lat = origin_lat + dy_rot
        
        return np.column_stack([new_lon, new_lat])


    def get_grid_from_polygons(self,points, polygon):
        x = points[..., 0]
        y = points[..., 1]
        return vectorized.contains(polygon, x, y)
    def set_time_and_max_flight_time(self):
        self.max_flight_time = min(35000*para.base_timestep,max(para.base_timestep*3, para.episode_length_added + para.episode_length_factor * self.base_timestep * self.target_area_size_start_of_episode / self.cone_union_last_cone_convex_hull_max_area))
        # set time of episode to some time in the future depending on how much has been measured in the init measurement state and how much is radiated.
        self.time_rel = 0.0
        self.time_abs = self.start_time
        self.percentage_of_episode_finished_by_time_passed = 0.0
    def _get_obs(self):
        if para.ablation_study_use_frontier_maps:
            if para.ablation_study_use_visit_frequency:
                if para.include_time_in_observation:
                    self.boolean_maps_within_target_area, self.observed_radiation_mapped_to_grid, self.observed_radiation_freshness_mapped_to_grid, self.observed_radiation_point_counts, self.frontier_maps = self.get_observation_as_images()
                else:
                    self.boolean_maps_within_target_area, self.observed_radiation_mapped_to_grid, self.observed_radiation_point_counts, self.frontier_maps = self.get_observation_as_images()
            else:
                if para.include_time_in_observation:
                    self.boolean_maps_within_target_area, self.observed_radiation_mapped_to_grid, self.observed_radiation_freshness_mapped_to_grid, self.frontier_maps = self.get_observation_as_images()
                else:
                    self.boolean_maps_within_target_area, self.observed_radiation_mapped_to_grid, self.frontier_maps = self.get_observation_as_images()
        else:
            if para.ablation_study_use_visit_frequency:
                if para.include_time_in_observation:
                    self.boolean_maps_within_target_area, self.observed_radiation_mapped_to_grid, self.observed_radiation_freshness_mapped_to_grid, self.observed_radiation_point_counts = self.get_observation_as_images()
                else:
                    self.boolean_maps_within_target_area, self.observed_radiation_mapped_to_grid, self.observed_radiation_point_counts = self.get_observation_as_images()
            else:
                if para.include_time_in_observation:
                    self.boolean_maps_within_target_area, self.observed_radiation_mapped_to_grid, self.observed_radiation_freshness_mapped_to_grid = self.get_observation_as_images()
                else:
                    self.boolean_maps_within_target_area, self.observed_radiation_mapped_to_grid = self.get_observation_as_images()
        
        # Compute lidar internally (for obstacle detection) but don't include in observation
        if not para.ablation_study_use_radiation_instead_of_lidar:
            # Compute lidar observation from current lidar points (used internally only)
            if hasattr(self, 'last_lidar_pts') and hasattr(self, 'last_lidar_pts_info'):
                lidar_obs = self._compute_lidar_observation(self.last_lidar_pts, self.position)
            else:
                # If no lidar data yet, return ones (max range)
                lidar_obs = np.ones(self.lidar_rays, dtype=np.float32)
            # Store for internal use but don't include in observation
            self._internal_lidar_obs = lidar_obs
        
        # ALWAYS return Box observation format (no lidar in observation)
        # Stack all observation maps along the channel dimension
        obs_images = []
        for i in range(len(self.boolean_maps_within_target_area)):
            target_area_formatted = (self.boolean_maps_within_target_area[i] * 255).astype(np.uint8)
            
            # Format radiation/wall map depending on ablation study
            if not para.ablation_study_use_radiation_instead_of_lidar:
                # For lidar, the map already contains wall pixels in 0-255 range (255=wall)
                radiation_mapped_formatted = self.observed_radiation_mapped_to_grid[i].astype(np.uint8)
            else:
                # For radiation, normalize by radiation value to avoid
                radiation_mapped_formatted = np.clip(
                            (255 * self.observed_radiation_mapped_to_grid[i] / (para.radiation_value_to_avoid * 1.1)),
                            0, 255
                        ).astype(np.uint8)

            channels = [
                target_area_formatted,
                radiation_mapped_formatted
            ]
            if para.ablation_study_use_visit_frequency:
                channels.append(self.observed_radiation_point_counts[i])  # Already scaled and converted to uint8
            if para.ablation_study_use_frontier_maps:
                channels.append(self.frontier_maps[i])  # Already uint8 with 255 for frontier pixels
            if para.include_time_in_observation:
                channels.append(np.clip(self.observed_radiation_freshness_mapped_to_grid[i], 0, 255).astype(np.uint8))
            obs_images.append(np.stack(channels, axis=-1))
        
        # Concatenate all maps along the channel dimension
        # For fisheye: obs_images has 1 element with shape (observed_image_size, observed_image_size, image_channels)
        # For multi-scale: obs_images has 4 elements, each with shape (observed_image_size, observed_image_size, image_channels)
        # Result should be (observed_image_size, observed_image_size, image_channels * num_maps)
        obs_image = np.concatenate(obs_images, axis=-1)
        
        if para.surface_grid_creation_type != "no_height_map":
            pos_z = self.position_z_above_ground / (self.gridsize[2])
        
        obs = dict(image=obs_image)
        
        # Add lidar observation if fusion is enabled
        if self.fuse_lidar_with_image and not para.ablation_study_use_radiation_instead_of_lidar:
            if hasattr(self, '_internal_lidar_obs'):
                obs['lidar'] = self._internal_lidar_obs
            else:
                # Return max range if no lidar data yet
                obs['lidar'] = np.ones(self.lidar_rays, dtype=np.float32)
        
        if para.use_motion_blur:
            # Use the 10th last image for motion blur
            if not hasattr(self, 'obs_image_history'):
                self.obs_image_history = []
            self.obs_image_history.append(obs_image)
            if len(self.obs_image_history) > para.motion_blur_image_number:
                tenth_last_image = self.obs_image_history[-para.motion_blur_image_number]
                ratio = 0.9
                # Ensure motion blur result is clipped and cast to uint8
                blurred_image = (obs_image * ratio + tenth_last_image * (1 - ratio))
                obs_image = np.clip(blurred_image, 0, 255).astype(np.uint8)
                obs['image'] = obs_image
            # Keep only the last 10 images in history
            self.obs_image_history = self.obs_image_history[-para.motion_blur_image_number:]
        
        # Ensure all observations are uint8 for constrained RL
        if para.training_library == 'omnisafe':
            obs['image'] = obs['image'].astype(np.uint8)
            # IMPORTANT: Assertion to verify uint8 conversion
            assert obs['image'].dtype == np.uint8, (
                "CRITICAL: Observation dtype conversion failed! Expected uint8 for constrained RL "
                "but got different dtype. This will cause issues with GymEnvOmniSafe tensor conversion. "
                f"Current dtype: {obs['image'].dtype}"
            )
        
        return obs
    def check_if_sim_needs_termination(self):
        if self.end_sim_reason == 'Time limit reached':
            self.ter_reason_time_limit_count += 1
        elif self.end_sim_reason == 'Manual termination':
            self.ter_reason_manual_termination_count += 1
    def step(self, action):
        # ========================================
        # CHECK IF RESET HAS BEEN CALLED
        # ========================================
        if not self._has_reset:
            raise RuntimeError(
                "Cannot call env.step() before calling env.reset(). "
                "The environment requires reset() to be called before it can be used."
            )
        action = action[0]
        self.current_episode_step += 1
        self.current_overall_step += 1
        if para.loop_step_until_radiation_above_threshold:
            # Initial step
            self.cost_sum = 0.0
            total_reward = 0.0
            self.observation, reward_step, self.end_sim, self.truncated, self.info = self.one_step(action)
            cost_of_step = 0.0 if self.current_radiation_exposure < para.radiation_value_to_avoid else 1.0
            self.cost_sum += cost_of_step
            total_reward += reward_step
            loop_steps = 1
            # Continue stepping until radiation threshold is reached
            while self.current_radiation_exposure < para.radiation_detection_threshold and loop_steps < para.loop_max_steps:
                self.observation, reward_step, self.end_sim, self.truncated, self.info = self.one_step([0.0],only_calc_obs_if_threshold_exceeded=True)
                cost_of_step = 0.0 if self.current_radiation_exposure < para.radiation_value_to_avoid else 1.0
                total_reward += reward_step
                self.cost_sum += cost_of_step
                loop_steps += 1
                if self.end_sim:
                    break
            # Set the final accumulated reward
            self.reward = total_reward/(loop_steps)
        else:
            self.observation, self.reward, self.end_sim, self.truncated, self.info = self.one_step(action)
        terminated = self.end_sim            
        if para.split_episodes_into_sub_episodes and not self.is_evaluation:
            if self.current_episode_step % para.max_episode_length_before_split == 0:
                terminated = True
                self.end_sim_reason = "split_episodes_into_sub_episodes"
        return self.observation, self.reward, terminated, False, {'final_observation': self.observation}
    def one_step(self, action, only_calc_obs_if_threshold_exceeded = False):
        self.current_radiation_exposure_previous = self.current_radiation_exposure
        #try:
        self.sim_step(action)
        #except Exception as e:
        #    import traceback
        #    print(f"\n{'='*80}")
        #    print(f"ERROR occurred during simulation step")
        #    print(f"{'='*80}")
        #    print(f"Episode: {self.episode_count}, Step: {self.current_episode_step}")
        #    print(f"Action: {action}")
        #    print(f"Position: {self.position}")
        #    print(f"Exception type: {type(e).__name__}")
        #    print(f"Exception message: {e}")
        #    print(f"\nFull traceback:")
        #    print(traceback.format_exc())
        #    print(f"{'='*80}\n")
        #    self.end_sim = True
        if self.current_radiation_exposure >= para.radiation_value_to_avoid:
            self.count_radiation_exposure_exceeded += 1
        self.difference_from_anchor_to_current_position_geo = np.array(self.position) - np.array([para.anchor_point[0], para.anchor_point[1]])
        self.check_if_sim_needs_termination()
        
        # Check for goal coverage termination
        if 1 - self.percentage_of_target_area_left > self.goal_coverage_percentage_currently:
            self.end_sim = True
            self.end_sim_reason = 'Goal coverage reached'
        
        # Check for curriculum learning advancement (using separate dilation for curriculum calculation)
        if self.use_curriculum_learning and not self.is_evaluation:
            curriculum_coverage = self._calculate_coverage_for_curriculum()
            if curriculum_coverage >= self.curriculum_goal_coverage:
                self._check_curriculum_advancement()
        
        # Check for early termination episode step limit
        if (hasattr(para, 'early_termination_episode_step_limit') and 
            self.current_episode_step >= self.early_termination_episode_step_limit_currently):
            self.end_sim = True
            self.end_sim_reason = 'Early termination step limit reached'
            
        # Preserve episode statistics for logging (before they get reset)
        if self.end_sim:
            self.last_episode_collisions = self.num_episode_collisions
            self.last_episode_total_reward = self.total_reward_per_episode
            self.last_episode_steps = self.current_episode_step
        reward_for_this_step = self._calculate_reward(action)
        
        # Incremental goal percentage update logic
        # Add current reward to 10k history
        self.reward_history_10k.append(reward_for_this_step)
        self.reward_10k_sum += reward_for_this_step
        
        # Maintain only last 10000 steps
        if len(self.reward_history_10k) > 10000:
            removed_reward = self.reward_history_10k.pop(0)
            self.reward_10k_sum -= removed_reward
        
        # Check for goal percentage progression (only if we have enough history and not in evaluation)
        if (len(self.reward_history_10k) >= 10000 and not self.is_evaluation and
            hasattr(para, 'constant_goal_coverage') and not para.constant_goal_coverage):
            
            # Calculate average reward over last 10000 steps
            avg_reward_10k = self.reward_10k_sum / len(self.reward_history_10k)
            
            # Check if avg reward > 0.9 - current reward (progression condition)
            progression_threshold = para.progression_threshold_without_reward_for_time_passed + para.reward_for_time_passed
            if avg_reward_10k > progression_threshold:
                # Progress goal percentage by 1% (increment progression step)
                
                # Reset the reward history for next progression cycle
                self.reward_history_10k = []
                self.reward_10k_sum = 0.0
                
                # Update current goal coverage percentage
                self.goal_coverage_percentage_currently += 0.01
                self.goal_coverage_percentage_currently = min(0.99, self.goal_coverage_percentage_currently)

                print(f"New goal: {self.goal_coverage_percentage_currently}, "
                      f"Avg reward 10k: {avg_reward_10k}, Current reward: {reward_for_this_step}")

        # Early termination episode step limit progression (similar to goal coverage progression)
        if (len(self.reward_history_10k) >= 10000 and not self.is_evaluation and
            hasattr(para, 'early_termination_episode_step_limit')):
            
            # Calculate average reward over last 10000 steps (reuse from above or calculate fresh)
            avg_reward_10k = self.reward_10k_sum / len(self.reward_history_10k)
            
            # Check if avg reward > progression threshold (same condition as goal coverage)
            progression_threshold = para.progression_threshold_without_reward_for_time_passed + para.reward_for_time_passed
            if avg_reward_10k > progression_threshold:
                # Increase episode step limit by 100 steps
                self.early_termination_episode_step_limit_currently += 100
                self.reward_history_10k = []
                self.reward_10k_sum = 0.0
                print(f"Early termination step limit increased to: {self.early_termination_episode_step_limit_currently}, "
                      f"Avg reward 10k: {avg_reward_10k}")

        self.rewards.append(reward_for_this_step)
        self.total_actions_per_episode += 1
        self.total_reward_per_episode += reward_for_this_step
        self.average_reward_per_episode = self.total_reward_per_episode/self.total_actions_per_episode
        if only_calc_obs_if_threshold_exceeded:
            if para.radiation_detection_threshold < self.current_radiation_exposure:
                self.observation = self._get_obs()
        else:
            self.observation = self._get_obs()
        if self.end_sim:
            self.episode_count += 1
            episode_summary = {
                'total_reward': self.total_reward_per_episode,
                'mean_reward': self.total_reward_per_episode/self.total_actions_per_episode if self.total_actions_per_episode != 0 else 0,
                'total_actions': self.total_actions_per_episode,
                'radiation_exceeded': self.count_radiation_exposure_exceeded,
                'target_area_left': self.percentage_of_target_area_left,
                'mean_little_action': self.rew_little_action_per_episode/self.total_actions_per_episode if self.total_actions_per_episode != 0 else 0,
                'mean_tv': self.rew_incremental_tv_per_episode/self.total_actions_per_episode if self.total_actions_per_episode != 0 else 0,
            }
            self.episode_history.append(episode_summary)
            # Keep only the last 100 episodes
            if len(self.episode_history) > 100:
                self.episode_history.pop(0)
            n = len(self.episode_history)
            mean_total_reward = sum(e['total_reward'] for e in self.episode_history) / n
            mean_mean_reward = sum(e['mean_reward'] for e in self.episode_history) / n
            mean_total_actions = sum(e['total_actions'] for e in self.episode_history) / n
            mean_radiation_exceeded = sum(e['radiation_exceeded'] for e in self.episode_history) / n
            mean_target_area_left = sum(e['target_area_left'] for e in self.episode_history) / n
            mean_mean_little_action = sum(e['mean_little_action'] for e in self.episode_history) / n
            mean_mean_tv = sum(e['mean_tv'] for e in self.episode_history) / n

            self.info['info_avg_last_100_episodes'] = {
                'total reward': mean_total_reward,
                'avg reward': mean_mean_reward,
                'total actions': mean_total_actions,
                'radiation exceeded': mean_radiation_exceeded,
                'target area left': mean_target_area_left,
                'avg little action': mean_mean_little_action,
                'avg tv': mean_mean_tv,
            }
            self.mean_target_area_left = mean_target_area_left
            self.mean_radiation_exceeded = mean_radiation_exceeded
            
            if self.logging_enabled:
                with open(self.log_file_path, "w") as f:
                    json.dump(self.info['info_avg_last_100_episodes'], f)
                    f.write("\n")

        return self.observation, reward_for_this_step, self.end_sim, False, self.info
    def _calculate_reward(self, action):
        if para.training_library == 'sb3' and self.current_radiation_exposure > para.radiation_value_to_avoid:
            self.rew_radiation = para.constant_radiation_reward # * self.cumulative_radiation_exposure_in_between_actions_normalized
            reward = self.rew_radiation
        else:
            if para.include_measurement_reward:            
                self.scaling_factor_for_percentage_of_new_measured_points = self.scale_measurement_reward_weight_by_height(heli_height=self.position_z_above_ground)
                
                # Apply area-based scaling if enabled
                if para.scale_measurement_reward:
                    # Linear interpolation between start_value (when 100% area left) and end_value (when 0% area left)
                    area_scaling_factor = para.scale_measurement_reward_start_value + (para.scale_measurement_reward_end_value - para.scale_measurement_reward_start_value) * (1 - self.percentage_of_target_area_left)
                    self.scaling_factor_for_percentage_of_new_measured_points *= area_scaling_factor
                
                self.rew_measurement_via_percentage_of_new_measured_points_scaled_by_height =  self.percentage_of_new_area_measured_ * self.scaling_factor_for_percentage_of_new_measured_points
                if para.z_rad_type == 'jon':
                    reward = self.rew_measurement_via_percentage_of_new_measured_points_scaled_by_height
                else:
                    if self.current_radiation_exposure < para.radiation_value_to_avoid:
                        reward = self.rew_measurement_via_percentage_of_new_measured_points_scaled_by_height
                    else:
                        reward = 0.0
            else:
                reward = 0.0
            if para.include_tv_reward:
                if para.use_tv_reward_decay:
                    self.rew_incremental_tv = max(0.0, 1 - self.current_overall_step / para.weight_incremental_tv_reward_zero_after_steps) * self.incremental_tv_reward_normalizer * (self.length_new_measured_minus_length_old_measured)
                else:
                    self.rew_incremental_tv = self.incremental_tv_reward_normalizer * (self.length_new_measured_minus_length_old_measured)

                reward += self.rew_incremental_tv
                self.rew_incremental_tv_per_episode += self.rew_incremental_tv
            # Action is now a scalar (single value), extract it properly
            action_value = action
            self.rew_little_action_summand = para.reward_for_little_action_weight_of_summand * abs(action_value)
            reward -= self.rew_little_action_summand
            self.rew_little_action_per_episode -= self.rew_little_action_summand
        reward += para.reward_for_time_passed
        if para.multiply_measurement_reward_by_inverse_time_left_included:
            reward *= (1 + para.multiply_measurement_reward_by_inverse_time_left_multiplier * (1 - self.percentage_of_target_area_left))
        return reward
    def scale_measurement_reward_weight_by_height(self, heli_height):
        heights = [-100, 0, 80, 85, 90, 95, 100, 349, 1000]
        weights = [0.01, 0.01, para.rew_weight_height_interpolation , 1, 1, 1, para.rew_weight_height_interpolation, 0.01, 0.01]
        return np.interp(heli_height, heights, weights)
    def compute_grid_averages_vectorized_no_freshness_original(self, radiation_points, grid, tree=None):
        """
        Original method that returns only radiation averages (for backward compatibility).
        Map each radiation_point to its closest grid point and compute the average value for each grid cell.
        Optimized version with reduced memory allocations and faster operations.
        
        Key optimizations:
        - Reduced memory allocation by avoiding unnecessary copies
        - Using smaller data types for better cache performance  
        - Vectorized operations without explicit masking
        - Direct reshaping without intermediate arrays
        - Fast grid mapping using precomputed inverse transformation
        """
        # Early return for empty input
        if len(radiation_points) == 0:
            return np.zeros(grid.shape[:2], dtype=np.float64)
        
        # Get grid dimensions for direct operations
        grid_shape = grid.shape[:2]
        n_cells = grid_shape[0] * grid_shape[1]
        
        # Use KDTree to map points to grid indices
        _, idxs = tree.query(radiation_points[:, :2])
        
        # Use smaller data types for better cache performance and memory usage
        # uint16 sufficient for counts (max 65535), float32 for intermediate calculations
        sums = np.zeros(n_cells, dtype=np.float32)
        counts = np.zeros(n_cells, dtype=np.uint16)
        
        # Single pass accumulation - convert radiation values once
        radiation_values = radiation_points[:, 2].astype(np.float32, copy=False)
        np.add.at(sums, idxs, radiation_values)
        np.add.at(counts, idxs, 1)
        
        # Vectorized division with automatic zero-handling (most efficient approach)
        averages = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        
        # Direct reshape and convert back to expected precision
        return averages.reshape(grid_shape).astype(np.float64, copy=False)

    def compute_grid_averages_vectorized_no_freshness(self, radiation_points, grid, tree=None):
        """
        Map each radiation_point to its closest grid point and compute the average value for each grid cell.
        Optimized version with reduced memory allocations and faster operations.
        
        Key optimizations:
        - Reduced memory allocation by avoiding unnecessary copies
        - Using smaller data types for better cache performance  
        - Vectorized operations without explicit masking
        - Direct reshaping without intermediate arrays
        - Fast grid mapping using precomputed inverse transformation
        """
        # Early return for empty input
        if len(radiation_points) == 0:
            grid_shape = grid.shape[:2]
            return np.zeros(grid_shape, dtype=np.float64), np.zeros(grid_shape, dtype=np.float64)
        
        # Get grid dimensions for direct operations
        grid_shape = grid.shape[:2]
        n_cells = grid_shape[0] * grid_shape[1]
        
        # Use KDTree to map points to grid indices
        _, idxs = tree.query(radiation_points[:, :2])
        
        # Use smaller data types for better cache performance and memory usage
        # uint16 sufficient for counts (max 65535), float32 for intermediate calculations
        sums = np.zeros(n_cells, dtype=np.float32)
        counts = np.zeros(n_cells, dtype=np.uint16)
        
        # Single pass accumulation - convert radiation values once
        radiation_values = radiation_points[:, 2].astype(np.float32, copy=False)
        np.add.at(sums, idxs, radiation_values)
        np.add.at(counts, idxs, 1)
        
        # Vectorized division with automatic zero-handling (most efficient approach)
        averages = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        
        # Direct reshape and convert back to expected precision
        averages_reshaped = averages.reshape(grid_shape).astype(np.float64, copy=False)
        counts_reshaped = counts.reshape(grid_shape).astype(np.float64, copy=False)
        
        # Apply distance-based scaling to counts_reshaped
        # Determine which distance matrix to use based on grid structure
        if hasattr(self, 'base_grid_distances_list'):
            # Multi-scale grids: find which grid index this is by matching the tree
            grid_idx = 0
            for i, kdtree in enumerate(self.grids_kdtrees):
                if kdtree is tree:
                    grid_idx = i
                    break
            distance_matrix = self.base_grid_distances_list[grid_idx]
        elif hasattr(self, 'base_grid_distances'):
            # Single grid
            distance_matrix = self.base_grid_distances
        else:
            # No distance matrix available, return without scaling
            return averages_reshaped, counts_reshaped
        
        # Apply scaling: multiply counts by 1/distance
        # Avoid division by zero (though distances should never be zero in practice)
        safe_distances = np.maximum(distance_matrix, 1e-10)
        counts_reshaped = counts_reshaped / safe_distances
        
        return averages_reshaped, counts_reshaped

    def _fast_grid_mapping(self, points, grid, grid_shape):
        """
        Fast alternative to KDTree query for regular grids.
        Maps points to grid indices using direct coordinate transformation.
        
        This is much faster than KDTree for structured grids because:
        1. No tree traversal required
        2. Vectorized operations
        3. Direct index calculation
        4. Better cache locality
        
        Args:
            points: (N, 2) array of point coordinates
            grid: (H, W, 2) grid coordinates  
            grid_shape: (H, W) shape tuple
            
        Returns:
            Array of grid indices for each point
        """
        # Get grid bounds for normalization
        grid_flat = grid.reshape(-1, 2)
        
        # Method 1: Direct coordinate mapping (fastest for regular grids)
        if hasattr(self, '_grid_bounds_cache'):
            # Use cached bounds for speed
            min_coords, max_coords = self._grid_bounds_cache
        else:
            # Compute and cache grid bounds
            min_coords = grid_flat.min(axis=0)
            max_coords = grid_flat.max(axis=0)
            self._grid_bounds_cache = (min_coords, max_coords)
        
        # Normalize point coordinates to grid space [0, grid_shape-1]
        coord_ranges = max_coords - min_coords
        # Avoid division by zero for constant coordinates
        coord_ranges = np.where(coord_ranges == 0, 1, coord_ranges)
        
        normalized_coords = (points - min_coords) / coord_ranges
        
        # Map to grid indices
        grid_indices = normalized_coords * (np.array(grid_shape) - 1)
        
        # Round to nearest grid point and clamp to valid range
        grid_indices = np.round(grid_indices).astype(np.int32)
        grid_indices[:, 0] = np.clip(grid_indices[:, 0], 0, grid_shape[0] - 1)
        grid_indices[:, 1] = np.clip(grid_indices[:, 1], 0, grid_shape[1] - 1)
        
        # Convert 2D indices to flat indices
        flat_indices = grid_indices[:, 0] * grid_shape[1] + grid_indices[:, 1]
        
        return flat_indices

    def _enable_fast_grid_mapping(self, enable=True):
        """
        Enable/disable fast grid mapping optimization.
        Call this after grid initialization to use the faster method.
        """
        self._use_fast_grid_mapping = enable
        if enable:
            # Clear any cached bounds to ensure fresh calculation
            if hasattr(self, '_grid_bounds_cache'):
                delattr(self, '_grid_bounds_cache')


    def _calculate_base_grid_distances(self):
        """
        Calculate distances from each grid point to its closest neighbor in the base grid(s).
        Store normalized distances where center point(s) have distance 1.
        For multi-scale grids, calculate distance matrices for all 4 grids.
        """
        from scipy.spatial.distance import cdist
        
        # Check if using multi-scale grids
        if (para.z_rad_type == 'jon' and 
            not para.ablation_study_fisheye_instead_of_multi_scale_maps and 
            hasattr(self, 'multi_scale_grids')):
            # Multi-scale: calculate distances for all 4 grids
            self.base_grid_distances_list = []
            self.base_grid_distances_raw_list = []
            
            for grid in self.multi_scale_grids:
                # Flatten grid to get all point coordinates
                grid_points = grid.reshape(-1, 2)  # Shape: (N*N, 2)
                grid_shape = grid.shape[:2]
                
                # Calculate distances between all pairs of points
                distances_matrix = cdist(grid_points, grid_points)
                
                # For each point, find distance to closest other point (exclude self)
                np.fill_diagonal(distances_matrix, np.inf)  # Exclude self-distances
                min_distances = np.min(distances_matrix, axis=1)
                
                # Reshape back to grid format
                distances_grid = min_distances.reshape(grid_shape)
                
                # Find center point(s) for normalization
                center_row = grid_shape[0] // 2
                center_col = grid_shape[1] // 2
                
                if grid_shape[0] % 2 == 1 and grid_shape[1] % 2 == 1:
                    # Odd grid - single center point
                    center_distance = distances_grid[center_row, center_col]
                else:
                    # Even grid - average of 4 center points
                    if grid_shape[0] % 2 == 0 and grid_shape[1] % 2 == 0:
                        # Both dimensions even - 4 center points
                        center_distances = [
                            distances_grid[center_row-1, center_col-1],
                            distances_grid[center_row-1, center_col],
                            distances_grid[center_row, center_col-1],
                            distances_grid[center_row, center_col]
                        ]
                        center_distance = np.mean(center_distances)
                    else:
                        # One dimension even, one odd - 2 center points
                        if grid_shape[0] % 2 == 0:  # Rows even, cols odd
                            center_distances = [
                                distances_grid[center_row-1, center_col],
                                distances_grid[center_row, center_col]
                            ]
                        else:  # Rows odd, cols even
                            center_distances = [
                                distances_grid[center_row, center_col-1],
                                distances_grid[center_row, center_col]
                            ]
                        center_distance = np.mean(center_distances)
                
                # Normalize distances so center has distance 1
                self.base_grid_distances_raw_list.append(distances_grid.copy())
                self.base_grid_distances_list.append(distances_grid / center_distance)
        else:
            # Single grid (fisheye or non-jon radiation types)
            # Get the base grid (use first grid as reference)
            if para.z_rad_type == 'jon':
                # For flat distorted grid
                base_grid = self.generate_flat_distorted_grid()
            else:
                # For geodesic distorted grid
                base_grid = self.generate_distorted_grid()
            
            # Flatten grid to get all point coordinates
            grid_points = base_grid.reshape(-1, 2)  # Shape: (N*N, 2)
            grid_shape = base_grid.shape[:2]
            
            # Calculate distances between all pairs of points
            distances_matrix = cdist(grid_points, grid_points)
            
            # For each point, find distance to closest other point (exclude self)
            np.fill_diagonal(distances_matrix, np.inf)  # Exclude self-distances
            min_distances = np.min(distances_matrix, axis=1)
            
            # Reshape back to grid format
            distances_grid = min_distances.reshape(grid_shape)
            
            # Find center point(s) for normalization
            center_row = grid_shape[0] // 2
            center_col = grid_shape[1] // 2
            
            if grid_shape[0] % 2 == 1 and grid_shape[1] % 2 == 1:
                # Odd grid - single center point
                center_distance = distances_grid[center_row, center_col]
            else:
                # Even grid - average of 4 center points
                if grid_shape[0] % 2 == 0 and grid_shape[1] % 2 == 0:
                    # Both dimensions even - 4 center points
                    center_distances = [
                        distances_grid[center_row-1, center_col-1],
                        distances_grid[center_row-1, center_col],
                        distances_grid[center_row, center_col-1],
                        distances_grid[center_row, center_col]
                    ]
                    center_distance = np.mean(center_distances)
                else:
                    # One dimension even, one odd - 2 center points
                    if grid_shape[0] % 2 == 0:  # Rows even, cols odd
                        center_distances = [
                            distances_grid[center_row-1, center_col],
                            distances_grid[center_row, center_col]
                        ]
                    else:  # Rows odd, cols even
                        center_distances = [
                            distances_grid[center_row, center_col-1],
                            distances_grid[center_row, center_col]
                        ]
                    center_distance = np.mean(center_distances)
            
            # Normalize distances so center has distance 1
            self.base_grid_distances_raw = distances_grid.copy()
            self.base_grid_distances = distances_grid / center_distance

    def compute_grid_averages_vectorized_original(self, visited_points_geo_coords, radiation_values, freshness_values, grid, tree=None):
        """
        Original method that returns only radiation and freshness averages (for backward compatibility).
        Map each visited point to its closest grid point and compute the average radiation and freshness value for each grid cell.
        Args:
            visited_points_geo_coords: np.ndarray of shape (n, 2) (lon, lat)
            radiation_values: np.ndarray of shape (n,)
            freshness_values: np.ndarray of shape (n,)
            grid: np.ndarray of shape (N, N, 2)
            tree: KDTree for grid
        Returns:
            Tuple of two images: (averages, averages_of_freshness), both shape (N, N)
        """
        grid_flat = grid.reshape(-1, 2)
        _, idxs = tree.query(visited_points_geo_coords)
        # For each grid cell, find the freshest (max) freshness value and corresponding radiation
        n_cells = grid_flat.shape[0]
        # Initialize with -inf for freshness, 0 for radiation
        freshest_freshness = np.full(n_cells, -np.inf, dtype=np.float64)
        corresponding_radiation = np.zeros(n_cells, dtype=np.float64)
        # Use np.maximum.at to get the freshest value per cell
        np.maximum.at(freshest_freshness, idxs, freshness_values)
        # For each cell, we want the radiation value corresponding to the freshest freshness value
        # To do this efficiently, find for each cell the indices of points that map to it and have the freshest value
        # Step 1: For each point, check if its freshness is the freshest for its cell
        is_freshest = (freshness_values == freshest_freshness[idxs])
        # Step 2: For each cell, pick the first (or any) radiation value where is_freshest is True
        # We'll use np.flatnonzero and np.unique to do this vectorized
        # Get indices of all points that are the freshest for their cell
        freshest_idxs = np.flatnonzero(is_freshest)
        # For each cell, keep only the first occurrence
        # This gives us the mapping: cell_idx -> index in freshest_idxs
        unique_cells, first_indices = np.unique(idxs[freshest_idxs], return_index=True)
        # Set the corresponding radiation for these cells
        corresponding_radiation[unique_cells] = radiation_values[freshest_idxs[first_indices]]
        # For cells with no points, keep 0 (or could be np.nan)
        # Set freshness to 0 for cells with no points (where freshest_freshness is still -inf)
        mask = freshest_freshness > -np.inf
        freshest_freshness[~mask] = 0.0
        corresponding_radiation[~mask] = 0.0
        return corresponding_radiation.reshape(grid.shape[:2]), freshest_freshness.reshape(grid.shape[:2])

    def compute_grid_averages_vectorized(self, visited_points_geo_coords, radiation_values, freshness_values, grid, tree=None):
        """
        Map each visited point to its closest grid point and compute the average radiation and freshness value for each grid cell.
        Args:
            visited_points_geo_coords: np.ndarray of shape (n, 2) (lon, lat)
            radiation_values: np.ndarray of shape (n,)
            freshness_values: np.ndarray of shape (n,)
            grid: np.ndarray of shape (N, N, 2)
            tree: KDTree for grid
        Returns:
            Tuple of three images: (averages, averages_of_freshness, point_counts), all shape (N, N)
        """
        grid_flat = grid.reshape(-1, 2)
        _, idxs = tree.query(visited_points_geo_coords)
        # For each grid cell, find the freshest (max) freshness value and corresponding radiation
        n_cells = grid_flat.shape[0]
        # Initialize with -inf for freshness, 0 for radiation
        freshest_freshness = np.full(n_cells, -np.inf, dtype=np.float64)
        corresponding_radiation = np.zeros(n_cells, dtype=np.float64)
        point_counts = np.zeros(n_cells, dtype=np.uint16)
        
        # Count points per cell
        np.add.at(point_counts, idxs, 1)
        
        # Use np.maximum.at to get the freshest value per cell
        np.maximum.at(freshest_freshness, idxs, freshness_values)
        # For each cell, we want the radiation value corresponding to the freshest freshness value
        # To do this efficiently, find for each cell the indices of points that map to it and have the freshest value
        # Step 1: For each point, check if its freshness is the freshest for its cell
        is_freshest = (freshness_values == freshest_freshness[idxs])
        # Step 2: For each cell, pick the first (or any) radiation value where is_freshest is True
        # We'll use np.flatnonzero and np.unique to do this vectorized
        # Get indices of all points that are the freshest for their cell
        freshest_idxs = np.flatnonzero(is_freshest)
        # For each cell, keep only the first occurrence
        # This gives us the mapping: cell_idx -> index in freshest_idxs
        unique_cells, first_indices = np.unique(idxs[freshest_idxs], return_index=True)
        # Set the corresponding radiation for these cells
        corresponding_radiation[unique_cells] = radiation_values[freshest_idxs[first_indices]]
        # For cells with no points, keep 0 (or could be np.nan)
        # Set freshness to 0 for cells with no points (where freshest_freshness is still -inf)
        mask = freshest_freshness > -np.inf
        freshest_freshness[~mask] = 0.0
        corresponding_radiation[~mask] = 0.0
        return (corresponding_radiation.reshape(grid.shape[:2]), 
                freshest_freshness.reshape(grid.shape[:2]),
                point_counts.reshape(grid.shape[:2]).astype(np.float64))

    def _log_detailed_time(self, operation_name, duration, method_context=None):
        """Helper method to log detailed timing information for profiling."""
        import inspect
        
        # Determine which method called this by looking at the call stack
        if method_context is None:
            frame = inspect.currentframe().f_back
            method_context = frame.f_code.co_name
        
        # Log to the appropriate dictionary based on the calling method
        if method_context == 'get_observation_as_images':
            times_dict = getattr(self, 'observation_detailed_times', {})
            if operation_name not in times_dict:
                times_dict[operation_name] = []
            times_dict[operation_name].append(duration)
        elif method_context == 'update_radiation_exposure_history_target_area':
            times_dict = getattr(self, 'update_radiation_detailed_times', {})
            if operation_name not in times_dict:
                times_dict[operation_name] = []
            times_dict[operation_name].append(duration)

    def _create_cross_blob_mask(self, blob_radius=1):
        """Create a cross-shaped blob mask with cardinal and diagonal directions.
        Uses caching to avoid recomputation."""
        # Check for cached mask
        cache_attr = f'_cached_cross_blob_mask_{blob_radius}'
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)
        
        blob_size = 2 * blob_radius + 1  # 5x5 grid
        center = blob_radius  # Center position (2,2 in 5x5 grid)
        mask = np.zeros((blob_size, blob_size), dtype=bool)
        
        # Center pixel
        mask[center, center] = True
        
        # Cardinal directions (2 pixels each)
        mask[center-1:center, center] = True  # Up
        mask[center+1:center+2, center] = True  # Down
        mask[center, center-1:center] = True  # Left
        mask[center, center+1:center+2] = True  # Right
        
        # Diagonal directions (1 pixel each)
        mask[center-1, center-1] = True  # Top-left
        mask[center-1, center+1] = True  # Top-right
        mask[center+1, center-1] = True  # Bottom-left
        mask[center+1, center+1] = True  # Bottom-right
        
        # Cache the mask
        setattr(self, cache_attr, mask)
        return mask

    def _apply_blob_to_radiation_grid(self, rad_point, radiation_grid, offset_to_heli, grid_with_offset, is_most_recent_point=True, grid_idx=0):
        """Apply cross-shaped blob to the radiation grid for a single point."""
        # Convert radiation point coordinates to grid indices
        point_coords_adjusted = rad_point[:2] - offset_to_heli
        
        # Use arithmetic calculation for multi-scale grids (much faster than KD-tree)
        use_arithmetic = (para.z_rad_type == 'jon' and 
                        not para.ablation_study_fisheye_instead_of_multi_scale_maps and
                        hasattr(self, 'precomputed_multi_scale_transforms'))
        
        grid_shape = grid_with_offset.shape[:2]
        
        if use_arithmetic:
            # Direct arithmetic calculation (no KD-tree)
            row_indices, col_indices = self._coords_to_grid_indices_arithmetic(
                point_coords_adjusted.reshape(1, -1), grid_idx, offset_to_heli
            )
            row_idx = row_indices[0]
            col_idx = col_indices[0]
        else:
            # Fallback to KD-tree for fisheye or other grid types
            _, grid_idx_flat = self.grids_kdtrees[grid_idx].query(point_coords_adjusted.reshape(1, -1))
            grid_idx_flat = grid_idx_flat[0]
            row_idx = grid_idx_flat // grid_shape[1]
            col_idx = grid_idx_flat % grid_shape[1]
        
        blob_radius = 1
        mask = self._create_cross_blob_mask(blob_radius)
        
        # Calculate the blob bounds in the grid
        row_start = max(0, row_idx - blob_radius)
        row_end = min(grid_shape[0], row_idx + blob_radius + 1)
        col_start = max(0, col_idx - blob_radius)
        col_end = min(grid_shape[1], col_idx + blob_radius + 1)
        
        # Adjust mask for boundary conditions
        mask_row_start = max(0, blob_radius - row_idx)
        mask_row_end = mask_row_start + (row_end - row_start)
        mask_col_start = max(0, blob_radius - col_idx)
        mask_col_end = mask_col_start + (col_end - col_start)
        
        radiation_value = rad_point[2] + 1  # Add 1 as done elsewhere in the code
        
        if row_end > row_start and col_end > col_start:
            blob_region = radiation_grid[row_start:row_end, col_start:col_end]
            mask_region = mask[mask_row_start:mask_row_end, mask_col_start:mask_col_end]
            
            # Only overwrite pixels that are 0 (no existing radiation data)
            zero_pixels = blob_region == 0
            combined_mask = mask_region & zero_pixels
            
            if is_most_recent_point:
                # Most recent point - only overwrite zero pixels
                blob_region[combined_mask] = radiation_value
            else:
                # Second-to-last point - use maximum to avoid overwriting higher values
                blob_region[combined_mask] = np.maximum(blob_region[combined_mask], radiation_value)

    def _discretize_obstacle_boundaries(self):
        """
        Discretize obstacle boundaries into segments of fixed length (0.05m).
        For each segment, store only the center point. This dramatically reduces
        the number of points tracked compared to storing every lidar hit.
        
        This method extracts all obstacle boundaries from known_obstacle_map,
        discretizes them into segments, and initializes tracking arrays.
        """
        if not hasattr(self, 'known_obstacle_map') or self.known_obstacle_map is None:
            return
        
        # Extract obstacle boundaries using edge detection
        import cv2
        edges = cv2.Canny(self.known_obstacle_map * 255, 50, 150)
        
        # Find contours of obstacles
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # Convert segment length from meters to pixels
        segment_length_pixels = self.segment_length_meters / self.meters_per_pixel_mower
        
        # Collect all segment centers
        all_segments = []
        
        for contour in contours:
            if len(contour) < 2:
                continue
            
            # Extract contour points as [x, y] coordinates
            contour_points = contour.squeeze()
            if contour_points.ndim == 1:
                contour_points = contour_points.reshape(1, -1)
            
            # Calculate cumulative distance along contour
            diffs = np.diff(contour_points, axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
            total_length = cumulative_dist[-1]
            
            if total_length < segment_length_pixels:
                # Contour too short, just use its center
                center = np.mean(contour_points, axis=0)
                all_segments.append(center)
                continue
            
            # Generate segment positions along the contour
            num_segments = max(1, int(np.ceil(total_length / segment_length_pixels)))
            segment_positions = np.linspace(0, total_length, num_segments)
            
            # Interpolate to find segment centers
            from scipy.interpolate import interp1d
            interp_x = interp1d(cumulative_dist, contour_points[:, 0], kind='linear', fill_value='extrapolate')
            interp_y = interp1d(cumulative_dist, contour_points[:, 1], kind='linear', fill_value='extrapolate')
            
            for pos in segment_positions:
                x = interp_x(pos)
                y = interp_y(pos)
                all_segments.append([x, y])
        
        # Convert to numpy array
        if len(all_segments) > 0:
            self.obstacle_segments = np.array(all_segments, dtype=np.float32)
            self.obstacle_segments_discovered = np.zeros(len(all_segments), dtype=bool)
        else:
            # No obstacles found
            self.obstacle_segments = np.zeros((0, 2), dtype=np.float32)
            self.obstacle_segments_discovered = np.zeros(0, dtype=bool)
        
        # Build KDTree for fast nearest-neighbor lookups
        if len(self.obstacle_segments) > 0:
            from scipy.spatial import cKDTree
            self.obstacle_segments_kdtree = cKDTree(self.obstacle_segments)
        else:
            self.obstacle_segments_kdtree = None

    def _compute_lidar_pts_flat(self, position, heading_deg):
        """
        Compute lidar point cloud for flat coordinate system (jon radiation type).
        VECTORIZED VERSION for significant performance improvement.
        
        Parameters:
        -----------
        position : array [x, y] in flat pixel coordinates
        heading_deg : heading in degrees (0 = North, 90 = East, etc.)
        
        Returns:
        --------
        lidar_pts : array of detected 2D points in flat pixel coordinates
        pts_info : detection info
            0 = max range (no detection)
            1 = known obstacle
            2 = unknown obstacle  
            3 = out of bounds
        """
        t_start = time.perf_counter()
        
        heading_deg = (heading_deg) % 360  # Normalize heading to [0, 360)
        
        # Convert lidar range from meters to pixels
        lidar_range_pixels = self.lidar_range / self.meters_per_pixel_mower
        samples = int(lidar_range_pixels)  # Number of samples per ray
        
        # Get map dimensions
        map_height, map_width = self.known_obstacle_map.shape
        
        t1 = time.perf_counter()
        # Pre-compute all ray angles (vectorized)
        angle_offsets = np.linspace(-self.lidar_fov/2, self.lidar_fov/2, num=self.lidar_rays)
        nav_angles_deg = heading_deg + angle_offsets + 180
        math_angles_deg = 90 - nav_angles_deg
        ang_rads = np.radians(math_angles_deg)
        
        # Compute all search vectors at once (shape: [lidar_rays, 2])
        search_vecs = np.stack([np.cos(ang_rads), -np.sin(ang_rads)], axis=1)
        t2 = time.perf_counter()
        self._log_detailed_time('lidar_compute_angle_setup', t2 - t1)
        
        t1 = time.perf_counter()
        # Fully vectorized ray casting over all rays and samples at once.
        # Shapes: [rays, samples] grids; first hit along each ray found via argmax.
        num_rays = self.lidar_rays
        position = np.asarray(position, dtype=np.float64).flatten()[:2]
        sample_indices = np.arange(1, samples + 1, dtype=np.float64)
        # positions[r, s] = position + (s+1) * search_vecs[r]
        positions = position[None, None, :] + sample_indices[None, :, None] * search_vecs[:, None, :]
        j_grid = positions[..., 0].astype(np.int32)  # x coordinates
        i_grid = positions[..., 1].astype(np.int32)  # y coordinates

        valid = (i_grid >= 0) & (i_grid < map_height) & (j_grid >= 0) & (j_grid < map_width)
        invalid = ~valid
        has_invalid = invalid.any(axis=1)
        # Number of leading valid samples per ray (== first invalid index, or all samples)
        ray_len = np.where(has_invalid, invalid.argmax(axis=1), samples)
        in_range = np.arange(samples)[None, :] < ray_len[:, None]

        i_clip = np.clip(i_grid, 0, map_height - 1)
        j_clip = np.clip(j_grid, 0, map_width - 1)
        known_hit = (self.known_obstacle_map[i_clip, j_clip] > 0) & in_range
        unknown_hit = (self.unknown_obstacle_map[i_clip, j_clip] > 0) & in_range
        has_known = known_hit.any(axis=1)
        known_idx = known_hit.argmax(axis=1)
        has_unknown = unknown_hit.any(axis=1)
        unknown_idx = unknown_hit.argmax(axis=1)

        lidar_pts = np.zeros((num_rays, 2), dtype=np.int32)
        pts_info = np.zeros(num_rays, dtype=np.int32)
        ray_ids = np.arange(num_rays)

        oob = ray_len == 0  # first sample already out of bounds
        pts_info[oob] = 3
        lidar_pts[oob] = (position[None, :] + search_vecs[oob]).astype(np.int32)

        # Known obstacles take priority over unknown along the whole (trimmed) ray,
        # matching the original per-ray implementation.
        m_known = ~oob & has_known
        pts_info[m_known] = 1
        lidar_pts[m_known, 0] = j_grid[ray_ids[m_known], known_idx[m_known]]
        lidar_pts[m_known, 1] = i_grid[ray_ids[m_known], known_idx[m_known]]

        m_unknown = ~oob & ~has_known & has_unknown
        pts_info[m_unknown] = 2
        lidar_pts[m_unknown, 0] = j_grid[ray_ids[m_unknown], unknown_idx[m_unknown]]
        lidar_pts[m_unknown, 1] = i_grid[ray_ids[m_unknown], unknown_idx[m_unknown]]

        m_free = ~oob & ~has_known & ~has_unknown
        lidar_pts[m_free] = (position[None, :] + samples * search_vecs[m_free]).astype(np.int32)
        t2 = time.perf_counter()
        self._log_detailed_time('lidar_ray_casting_loop', t2 - t1)
        
        t_end = time.perf_counter()
        self._log_detailed_time('lidar_compute_total', t_end - t_start)
        
        return lidar_pts, pts_info
    
    def _update_obstacles_from_lidar_pts(self, lidar_pts, pts_info):
        """
        Update known obstacle map based on detected obstacles.
        Instead of storing every hit point, mark discretized segments as discovered.
        This dramatically reduces memory and computation time.
        """
        t_start = time.perf_counter()
        
        if self.obstacle_segments_kdtree is None or len(self.obstacle_segments) == 0:
            return
        
        map_height, map_width = self.known_obstacle_map.shape
        
        t1 = time.perf_counter()
        # Collect all detected obstacle points (vectorized over rays)
        hit_mask = (pts_info == 1) | (pts_info == 2)
        j_hits = lidar_pts[hit_mask, 0]
        i_hits = lidar_pts[hit_mask, 1]
        in_bounds = (i_hits >= 0) & (i_hits < map_height) & (j_hits >= 0) & (j_hits < map_width)
        j_hits = j_hits[in_bounds]
        i_hits = i_hits[in_bounds]
        # Newly discovered (previously unknown) obstacle pixels become known
        unknown_sel = pts_info[hit_mask][in_bounds] == 2
        self.known_obstacle_map[i_hits[unknown_sel], j_hits[unknown_sel]] = 1

        # Query KDTree to find nearest segments for all detected points at once
        if len(j_hits) > 0:
            detected_points = np.stack([j_hits, i_hits], axis=1).astype(np.float32)

            # Find nearest segment for each detected point (within reasonable distance)
            distances, indices = self.obstacle_segments_kdtree.query(detected_points, k=1)
            
            # Only mark segments that are close enough (within segment_length_pixels)
            segment_length_pixels = self.segment_length_meters / self.meters_per_pixel_mower
            valid_mask = distances < segment_length_pixels
            valid_indices = indices[valid_mask]
            
            # Mark these segments as discovered (use numpy's efficient indexing)
            if len(valid_indices) > 0:
                self.obstacle_segments_discovered[valid_indices] = True
        
        t2 = time.perf_counter()
        self._log_detailed_time('lidar_update_obstacles_loop', t2 - t1)
        
        t_end = time.perf_counter()
        self._log_detailed_time('lidar_update_obstacles_total', t_end - t_start)
    
    def _compute_lidar_observation(self, lidar_pts, position):
        """
        Compute normalized lidar distances from lidar point cloud.
        Similar to mower environment's lidar observation computation.
        
        Parameters:
        -----------
        lidar_pts : array of 2D points [x, y] in flat pixel coordinates
        position : current position [x, y] in flat pixel coordinates
        
        Returns:
        --------
        lidar_obs : array of lidar distances, normalized to [0, 1]
        """
        offsets = np.asarray(lidar_pts, dtype=np.float64) - np.asarray(position, dtype=np.float64)
        dist_meters = np.sqrt((offsets ** 2).sum(axis=1)) * self.meters_per_pixel_mower
        if self.lidar_noise > 0:
            dist_meters = np.random.normal(loc=dist_meters, scale=self.lidar_noise)
        return np.clip(dist_meters / self.lidar_range, 0, 1).astype(np.float32)

    def get_observation_as_images(self):
        
        start_time = time.perf_counter()
        
        # Initialize granular timing dictionary
        if not hasattr(self, 'observation_detailed_times'):
            self.observation_detailed_times = {}
        
        boolean_maps_within_target_area = []
        observed_radiation_mapped_to_grid = []
        if para.ablation_study_use_visit_frequency:
            observed_radiation_point_counts = []
        if para.ablation_study_use_frontier_maps:
            frontier_maps = []
            # Pre-compute boundary and coordinates once for all grids (major optimization)
            # Only time and compute if cache is not populated
            if not hasattr(self, '_cached_target_area_boundary') or self._cached_target_area_boundary is None:
                t_boundary_start = time.perf_counter()
                self._cached_target_area_boundary = self.safe_boundary(self.target_area)
                # Also cache the extracted coordinates
                if self._cached_target_area_boundary is not None and not self._cached_target_area_boundary.is_empty:
                    if hasattr(self._cached_target_area_boundary, 'geoms'):
                        self._cached_target_area_boundary_coords = np.vstack([np.array(line.coords) for line in self._cached_target_area_boundary.geoms])
                    else:
                        self._cached_target_area_boundary_coords = np.array(self._cached_target_area_boundary.coords)
                    
                    # Pre-filter boundary coords by obstacle mask once (black_polygons don't change during episode)
                    if (self._cached_target_area_boundary_coords is not None and 
                        len(self._cached_target_area_boundary_coords) > 0 and
                        hasattr(self, 'black_polygons_dilated') and 
                        self.black_polygons_dilated is not None and 
                        not self.black_polygons_dilated.is_empty):
                        not_near_obstacle_mask = ~vectorized.contains(
                            self.black_polygons_dilated, 
                            self._cached_target_area_boundary_coords[:, 0], 
                            self._cached_target_area_boundary_coords[:, 1]
                        )
                        self._cached_boundary_coords_filtered_obstacles = self._cached_target_area_boundary_coords[not_near_obstacle_mask]
                    else:
                        self._cached_boundary_coords_filtered_obstacles = self._cached_target_area_boundary_coords
                    
                    # Pre-compute dense boundary coords for grid 0 (2x denser with midpoints)
                    if self._cached_boundary_coords_filtered_obstacles is not None and len(self._cached_boundary_coords_filtered_obstacles) > 1:
                        boundary_base = self._cached_boundary_coords_filtered_obstacles
                        midpoints = (boundary_base[:-1] + boundary_base[1:]) / 2.0
                        dense_coords = np.empty((len(boundary_base) + len(midpoints), 2), dtype=boundary_base.dtype)
                        dense_coords[0::2][:len(boundary_base)] = boundary_base
                        dense_coords[1::2] = midpoints
                        self._cached_boundary_coords_dense = dense_coords
                    else:
                        self._cached_boundary_coords_dense = self._cached_boundary_coords_filtered_obstacles
                else:
                    self._cached_target_area_boundary_coords = None
                    self._cached_boundary_coords_filtered_obstacles = None
                    self._cached_boundary_coords_dense = None
                t_boundary_end = time.perf_counter()
                self._log_detailed_time('frontier_boundary_precompute', t_boundary_end - t_boundary_start)
        if para.include_time_in_observation:
            self.observed_radiation_freshness_mapped_to_grid = []
        
        # Determine how many grids to process
        num_grids = len(self.list_of_observation_grids)
        
        # Loop through all grids (1 for fisheye, 4 for multi-scale)
        for grid_idx in range(num_grids):
            # Step 1: Calculate offset so that grid center aligns with current helicopter position
            t1 = time.perf_counter()
            current_heli_position = self.position
            
            # Get the current grid (rotated based on bearing)
            current_grid = self.list_of_observation_grids[grid_idx]
            
            # Calculate grid center dynamically based on actual grid dimensions
            # Account for both odd and even grid sizes
            grid_shape = current_grid.shape[:2]  # (rows, cols)
            grid_center_row = (grid_shape[0] - 1) / 2.0  # Use float for accurate center
            grid_center_col = (grid_shape[1] - 1) / 2.0
            
            # For even grids, this gives us the point between the two middle cells
            # For odd grids, this gives us the exact middle cell
            # Interpolate to get the exact center coordinates
            row_floor = int(np.floor(grid_center_row))
            row_ceil = int(np.ceil(grid_center_row))
            col_floor = int(np.floor(grid_center_col))
            col_ceil = int(np.ceil(grid_center_col))
            
            if row_floor == row_ceil and col_floor == col_ceil:
                # Odd dimensions - exact center cell exists
                grid_center = current_grid[row_floor, col_floor]
            else:
                # Even dimensions - interpolate between cells
                row_weight = grid_center_row - row_floor
                col_weight = grid_center_col - col_floor
                
                # Bilinear interpolation
                grid_center = (
                    (1 - row_weight) * (1 - col_weight) * current_grid[row_floor, col_floor] +
                    (1 - row_weight) * col_weight * current_grid[row_floor, col_ceil] +
                    row_weight * (1 - col_weight) * current_grid[row_ceil, col_floor] +
                    row_weight * col_weight * current_grid[row_ceil, col_ceil]
                )
            
            # Calculate offset to move grid center to helicopter position
            offset_to_heli = current_heli_position - grid_center
            t2 = time.perf_counter()
            self._log_detailed_time(f'grid_{grid_idx}_setup_offset', t2 - t1)
            
            # Apply offset to position grid center at helicopter position
            t1 = time.perf_counter()
            grid_with_offset = current_grid + offset_to_heli
            
            # Use precomputed half_size for fast bounds calculation (avoids min/max over whole grid)
            use_fast_bounds = (para.z_rad_type == 'jon' and 
                              not para.ablation_study_fisheye_instead_of_multi_scale_maps and
                              hasattr(self, 'precomputed_multi_scale_transforms'))
            
            if use_fast_bounds:
                bearing_key = self._current_bearing_key
                _, _, _, half_size, _, _, _ = self.precomputed_multi_scale_transforms[bearing_key][grid_idx]
                # For a rotated square grid, the bounding box is slightly larger than the grid extent
                # But for point filtering, we can use the grid extent + margin
                margin = half_size * 0.05  # Small margin for numerical precision
                grid_min_lon = current_heli_position[0] - half_size - margin
                grid_max_lon = current_heli_position[0] + half_size + margin
                grid_min_lat = current_heli_position[1] - half_size - margin
                grid_max_lat = current_heli_position[1] + half_size + margin
            else:
                grid_min_lon = np.min(grid_with_offset[..., 0])
                grid_max_lon = np.max(grid_with_offset[..., 0])
                grid_min_lat = np.min(grid_with_offset[..., 1])
                grid_max_lat = np.max(grid_with_offset[..., 1])
            t2 = time.perf_counter()
            self._log_detailed_time(f'grid_{grid_idx}_bounds_calculation', t2 - t1)

            # Filter radiation points to those within grid bounds
            t1 = time.perf_counter()
            radiation_points_full = self.observed_radiation_points  # shape: (n, 3): lon, lat, radiation_value
            if len(radiation_points_full) > 0:
                within_bounds_mask = (
                    (radiation_points_full[:, 0] >= grid_min_lon) & 
                    (radiation_points_full[:, 0] <= grid_max_lon) &
                    (radiation_points_full[:, 1] >= grid_min_lat) & 
                    (radiation_points_full[:, 1] <= grid_max_lat)
                )
                radiation_points = radiation_points_full[within_bounds_mask]
                # Also filter freshness values using the same mask
                if len(self.observed_radiation_points_freshness) > 0:
                    radiation_points_freshness = self.observed_radiation_points_freshness[within_bounds_mask]
                else:
                    radiation_points_freshness = self.observed_radiation_points_freshness
            else:
                radiation_points = radiation_points_full
                radiation_points_freshness = self.observed_radiation_points_freshness
            
            # Extract radiation coordinates without rotation
            radiation_coords = radiation_points[:, 0:2]  # Extract lon, lat only
            radiation_coords_adjusted = radiation_coords - offset_to_heli
            t2 = time.perf_counter()
            self._log_detailed_time(f'grid_{grid_idx}_radiation_points_filtering', t2 - t1)
            
            t1 = time.perf_counter()
            # Create three-level boolean map:
            # 1.0 for remeasured areas, 0.5 for target_area, 0.0 for outside
            #remeasured_mask = self.get_grid_from_polygons(grid_with_offset, self.target_area_remeasured_1_times)
            target_area_mask = self.get_grid_from_polygons(grid_with_offset, self.target_area)
            
            # Start with zeros
            #three_level_map = np.zeros_like(target_area_mask, dtype=np.float32)
            ## Set target_area points to 0.5
            #three_level_map[target_area_mask] = 0.5
            ## Override with 1.0 for remeasured points
            #three_level_map[remeasured_mask] = 1.0
            
            boolean_maps_within_target_area.append(target_area_mask)
            t2 = time.perf_counter()
            self._log_detailed_time(f'grid_{grid_idx}_target_area_grid_mapping', t2 - t1)
            
            # Compute frontier map if enabled
            if para.ablation_study_use_frontier_maps:
                t1 = time.perf_counter()
                
                # Substep 1: Use pre-cached boundary (computed before the loop)
                t_sub1 = time.perf_counter()
                target_area_boundary = self._cached_target_area_boundary
                t_sub2 = time.perf_counter()
                self._log_detailed_time(f'grid_{grid_idx}_frontier_boundary_extraction', t_sub2 - t_sub1)
                
                # Create frontier map by mapping boundary points to grid
                frontier_img = np.zeros(grid_with_offset.shape[:2], dtype=np.uint8)
                
                if target_area_boundary is not None and not target_area_boundary.is_empty:
                    # Substep 2: Use pre-cached boundary coordinates (already filtered by obstacles)
                    t_sub1 = time.perf_counter()
                    # Use pre-cached dense boundary coords for grid 0, regular for others
                    if grid_idx == 0:
                        if self._cached_boundary_coords_dense is not None:
                            boundary_coords = self._cached_boundary_coords_dense
                        else:
                            boundary_coords = np.array([]).reshape(0, 2)
                    else:
                        if self._cached_boundary_coords_filtered_obstacles is not None:
                            boundary_coords = self._cached_boundary_coords_filtered_obstacles
                        else:
                            boundary_coords = np.array([]).reshape(0, 2)
                    t_sub2 = time.perf_counter()
                    self._log_detailed_time(f'grid_{grid_idx}_frontier_coords_extraction', t_sub2 - t_sub1)
                    
                    # Substep 3: Obstacle filter now done in precompute step (just log timing for comparison)
                    t_sub1 = time.perf_counter()
                    # No-op: obstacle filtering is now done once during precompute
                    t_sub2 = time.perf_counter()
                    self._log_detailed_time(f'grid_{grid_idx}_frontier_obstacle_filter', t_sub2 - t_sub1)
                    
                    # Substep 4: Filter boundary points to those within grid bounds and map to grid
                    t_sub1 = time.perf_counter()
                    if len(boundary_coords) > 0:
                        within_bounds_mask = (
                            (boundary_coords[:, 0] >= grid_min_lon) & 
                            (boundary_coords[:, 0] <= grid_max_lon) &
                            (boundary_coords[:, 1] >= grid_min_lat) & 
                            (boundary_coords[:, 1] <= grid_max_lat)
                        )
                        boundary_coords_filtered = boundary_coords[within_bounds_mask]
                        
                        if len(boundary_coords_filtered) > 0:
                            # Adjust coordinates for current grid offset
                            boundary_coords_adjusted = boundary_coords_filtered - offset_to_heli
                            
                            # Use arithmetic grid mapping for multi-scale grids (much faster than KD-tree)
                            use_arithmetic = (para.z_rad_type == 'jon' and 
                                            not para.ablation_study_fisheye_instead_of_multi_scale_maps and
                                            hasattr(self, 'precomputed_multi_scale_transforms'))
                            
                            if use_arithmetic:
                                # Direct arithmetic calculation (no KD-tree)
                                row_indices, col_indices = self._coords_to_grid_indices_arithmetic(
                                    boundary_coords_adjusted, grid_idx, offset_to_heli
                                )
                            else:
                                # Fallback to KD-tree for fisheye or other grid types
                                _, grid_indices = self.grids_kdtrees[grid_idx].query(boundary_coords_adjusted)
                                grid_shape = grid_with_offset.shape[:2]
                                row_indices = grid_indices // grid_shape[1]
                                col_indices = grid_indices % grid_shape[1]
                            
                            # Mark frontier pixels (use value 255 for frontier)
                            frontier_img[row_indices, col_indices] = 255
                    t_sub2 = time.perf_counter()
                    self._log_detailed_time(f'grid_{grid_idx}_frontier_grid_mapping', t_sub2 - t_sub1)
                
                frontier_maps.append(frontier_img)
                t2 = time.perf_counter()
                self._log_detailed_time(f'grid_{grid_idx}_frontier_mapping', t2 - t1)

            # Choose between lidar wall mapping and radiation mapping based on ablation study parameter
            if not para.ablation_study_use_radiation_instead_of_lidar:
                # Use lidar wall mapping instead of radiation mapping
                t1 = time.perf_counter()
                
                # Get only discovered segments
                if hasattr(self, 'obstacle_segments_discovered') and np.any(self.obstacle_segments_discovered):
                    discovered_segments = self.obstacle_segments[self.obstacle_segments_discovered]
                    
                    # Filter to segments within grid bounds
                    within_bounds_mask = (
                        (discovered_segments[:, 0] >= grid_min_lon) & 
                        (discovered_segments[:, 0] <= grid_max_lon) &
                        (discovered_segments[:, 1] >= grid_min_lat) & 
                        (discovered_segments[:, 1] <= grid_max_lat)
                    )
                    lidar_wall_points = discovered_segments[within_bounds_mask]
                else:
                    lidar_wall_points = np.zeros((0, 2), dtype=np.float32)
                
                # Create wall map by mapping lidar points to grid
                wall_img = np.zeros(grid_with_offset.shape[:2], dtype=np.uint8)
                
                if len(lidar_wall_points) > 0:
                    # Adjust coordinates for current grid offset
                    lidar_coords_adjusted = lidar_wall_points - offset_to_heli
                    
                    # Use arithmetic grid mapping for multi-scale grids (much faster than KD-tree)
                    use_arithmetic = (para.z_rad_type == 'jon' and 
                                    not para.ablation_study_fisheye_instead_of_multi_scale_maps and
                                    hasattr(self, 'precomputed_multi_scale_transforms'))
                    
                    if use_arithmetic:
                        # Direct arithmetic calculation (no KD-tree)
                        row_indices, col_indices = self._coords_to_grid_indices_arithmetic(
                            lidar_coords_adjusted, grid_idx, offset_to_heli
                        )
                    else:
                        # Fallback to KD-tree for fisheye or other grid types
                        _, grid_indices = self.grids_kdtrees[grid_idx].query(lidar_coords_adjusted)
                        grid_shape = grid_with_offset.shape[:2]
                        row_indices = grid_indices // grid_shape[1]
                        col_indices = grid_indices % grid_shape[1]
                    
                    # Mark wall pixels (use value 255 for detected walls)
                    wall_img[row_indices, col_indices] = 255
                
                observed_radiation_mapped_to_grid.append(wall_img)
                
                # If footprint with visit frequency is enabled, create empty point counts
                # (not applicable for lidar, but needed for consistent return structure)
                if para.ablation_study_use_visit_frequency:
                    empty_counts = np.zeros(grid_with_offset.shape[:2], dtype=np.uint8)
                    observed_radiation_point_counts.append(empty_counts)
                
                t2 = time.perf_counter()
                self._log_detailed_time(f'grid_{grid_idx}_lidar_wall_mapping', t2 - t1)
                
            elif para.include_time_in_observation:
                t1 = time.perf_counter()
                rad_vals = radiation_points[:, 2] + 1
                fresh_vals = radiation_points_freshness[:, 2]  # shape: (n, 3): lon, lat, freshness_value
                
                if para.ablation_study_use_visit_frequency:
                    rad_img, fresh_img, point_counts = self.compute_grid_averages_vectorized(
                        radiation_coords_adjusted,
                        rad_vals,
                        fresh_vals,
                        grid_with_offset,
                        tree=self.grids_kdtrees[grid_idx]
                    )
                    # Scale point counts according to specification: 0->0, 1->255/20, 2->255*2/20, etc., clip at 20
                    scaled_counts = np.clip(point_counts * 255 / 2, 0, 255).astype(np.uint8)
                    observed_radiation_point_counts.append(scaled_counts)
                else:
                    # Use original method that returns only 2 values
                    rad_img, fresh_img = self.compute_grid_averages_vectorized_original(
                        radiation_coords_adjusted,
                        rad_vals,
                        fresh_vals,
                        grid_with_offset,
                        tree=self.grids_kdtrees[grid_idx]
                    )
                
                self.observed_radiation_freshness_mapped_to_grid.append(fresh_img)
                observed_radiation_mapped_to_grid.append(rad_img)
                
                t2 = time.perf_counter()
                self._log_detailed_time(f'grid_{grid_idx}_grid_averages_with_freshness', t2 - t1)
            else:
                t1 = time.perf_counter()
                # Use adjusted coordinates without rotation
                radiation_points_adjusted = np.column_stack([
                    radiation_coords_adjusted, 
                    radiation_points[:, 2]
                ])
                
                if para.ablation_study_use_visit_frequency:
                    rad_img, point_counts = self.compute_grid_averages_vectorized_no_freshness(
                        radiation_points_adjusted,
                        grid_with_offset,
                        tree=self.grids_kdtrees[grid_idx]
                    )
                    # Scale point counts according to specification: 0->0, 1->255/20, 2->255*2/20, etc., clip at 20
                    scaled_counts = np.clip(point_counts * 255 / 2, 0, 255).astype(np.uint8)
                    observed_radiation_point_counts.append(scaled_counts)
                else:
                    # Use original method that returns only radiation averages
                    rad_img = self.compute_grid_averages_vectorized_no_freshness_original(
                        radiation_points_adjusted,
                        grid_with_offset,
                        tree=self.grids_kdtrees[grid_idx]
                    )
                
                observed_radiation_mapped_to_grid.append(rad_img)
                
                t2 = time.perf_counter()
                self._log_detailed_time(f'grid_{grid_idx}_grid_averages_no_freshness', t2 - t1)
            
            # Only apply Gaussian blob to first image when using multi-scale maps
            # or to all images when using fisheye
            should_apply_blob = (para.ablation_study_fisheye_instead_of_multi_scale_maps or grid_idx == 0)
            
            if para.include_gaussian_blob_around_first_two_radiation_points_in_observation_scaled_by_radiation_value and should_apply_blob:
                t1 = time.perf_counter()
                # Add blob around the first two radiation points (current and previous helicopter positions)
                # Process the last two radiation points (or just one if that's all we have)
                if len(radiation_points) >= 1:
                    points_to_process = radiation_points[-2:] if len(radiation_points) >= 2 else [radiation_points[-1]]
                    
                    # Process points in order: second-to-last first, then most recent
                    for i, rad_point in enumerate(points_to_process):
                        is_most_recent = (i == len(points_to_process) - 1)
                        self._apply_blob_to_radiation_grid(
                            rad_point, observed_radiation_mapped_to_grid[-1], 
                            offset_to_heli, grid_with_offset, is_most_recent, grid_idx
                        )
                t2 = time.perf_counter()
                self._log_detailed_time(f'grid_{grid_idx}_gaussian_blob_application', t2 - t1)
        
        # Record timing for get_observation_as_images
        end_time = time.perf_counter()
        if not hasattr(self, 'get_observation_as_images_times'):
            self.get_observation_as_images_times = []
        self.get_observation_as_images_times.append(end_time - start_time)
        
        if para.ablation_study_use_frontier_maps:
            if para.ablation_study_use_visit_frequency:
                if para.include_time_in_observation:
                    return boolean_maps_within_target_area, observed_radiation_mapped_to_grid, self.observed_radiation_freshness_mapped_to_grid, observed_radiation_point_counts, frontier_maps
                else:
                    return boolean_maps_within_target_area, observed_radiation_mapped_to_grid, observed_radiation_point_counts, frontier_maps
            else:
                if para.include_time_in_observation:
                    return boolean_maps_within_target_area, observed_radiation_mapped_to_grid, self.observed_radiation_freshness_mapped_to_grid, frontier_maps
                else:
                    return boolean_maps_within_target_area, observed_radiation_mapped_to_grid, frontier_maps
        else:
            if para.ablation_study_use_visit_frequency:
                if para.include_time_in_observation:
                    return boolean_maps_within_target_area, observed_radiation_mapped_to_grid, self.observed_radiation_freshness_mapped_to_grid, observed_radiation_point_counts
                else:
                    return boolean_maps_within_target_area, observed_radiation_mapped_to_grid, observed_radiation_point_counts
            else:
                if para.include_time_in_observation:
                    return boolean_maps_within_target_area, observed_radiation_mapped_to_grid, self.observed_radiation_freshness_mapped_to_grid
                else:
                    return boolean_maps_within_target_area, observed_radiation_mapped_to_grid
    def set_seed(self, seed: int):
        """
        Set random seeds for various RNGs used by the environment to increase
        determinism between runs.
        """
        random.seed(seed)
        try:
            import numpy as _np
            _np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch as _torch
            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
        except Exception:
            # torch might not be installed in some lightweight test environments
            pass

    def close(self):
        pass

    def render(self, save_path=None):
        """
        Render the current state of the environment and save as PNG.
        
        Args:
            save_path (str, optional): Custom save path for the PNG file.
                                     If None, saves to misc/logs/
        """
        # Use previous episode data if available
        episode_data = getattr(self, 'previous_episode_data', {})
        position_history = episode_data.get('position_history', [])
        image_path = episode_data.get('IMAGE_PATH', getattr(self, 'IMAGE_PATH', None))
        cached_image = episode_data.get('cached_image', None)
        target_area_size = episode_data.get('target_area_size', 0)
        path_length = episode_data.get('length_of_path_in_meters', 0)
        total_degrees_turned = episode_data.get('total_degrees_turned', 0.0)
        start_position = episode_data.get('position_init', None)
        end_position = episode_data.get('position', None)
        percentage_of_target_area_left = episode_data.get('percentage_of_target_area_left', 1.0)
        # Early exit if no data to render
        if not position_history or image_path is None:
            print("Warning: No position history or image path available for rendering")
            return
        
        try:
            img = episode_data['current_image_for_movement_restriction']
            
            # Get image dimensions
            img_height, img_width = img.shape
            
            # Use the maximum dimension to ensure square aspect ratio
            max_dim = max(img_height, img_width)
            
            # Create figure and axis with white background
            # Use fixed size for consistent output
            fig, ax = plt.subplots(figsize=(12, 12), dpi=225)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # Adjust subplot to minimize margins except top (for title)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
            
            # Set axis limits to square dimensions (always same size)
            ax.set_xlim(0, max_dim)
            ax.set_ylim(0, max_dim)
            
            # Force equal aspect ratio (1:1) to prevent distortion
            ax.set_aspect('equal', adjustable='box')
            
            # Create purple background within the image bounds (centered in square)
            from matplotlib.patches import Rectangle
            # Center the actual content if dimensions differ
            x_offset = (max_dim - img_width) / 2
            y_offset = (max_dim - img_height) / 2
            purple_bg = Rectangle((x_offset, y_offset), img_width, img_height, 
                                 facecolor='#4f4d7a', edgecolor='none', zorder=0)
            ax.add_patch(purple_bg)
            
            # Plot black polygons (obstacles)
            black_polygons = episode_data.get('black_polygons', 
                                             self.black_polygons if hasattr(self, 'black_polygons') else None)
            if black_polygons is not None:
                from matplotlib.path import Path
                from matplotlib.patches import PathPatch
                from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
                
                # Handle different black_polygons types
                polygons_to_render = []
                
                if isinstance(black_polygons, ShapelyPolygon):
                    polygons_to_render.append(black_polygons)
                elif isinstance(black_polygons, MultiPolygon):
                    polygons_to_render.extend(black_polygons.geoms)
                elif isinstance(black_polygons, list):
                    for item in black_polygons:
                        if isinstance(item, (ShapelyPolygon, MultiPolygon)):
                            polygons_to_render.append(item)
                
                # Render each polygon with proper hole support using PathPatch
                for polygon in polygons_to_render:
                    if isinstance(polygon, ShapelyPolygon):
                        # Get exterior coordinates
                        exterior_coords = list(polygon.exterior.coords)
                        flipped_exterior = [(x + x_offset, img_height - 1 - y + y_offset) for x, y in exterior_coords]
                        
                        # Build path with exterior and holes
                        vertices = [flipped_exterior[0]]
                        codes = [Path.MOVETO]
                        
                        for coord in flipped_exterior[1:]:
                            vertices.append(coord)
                            codes.append(Path.LINETO)
                        codes[-1] = Path.CLOSEPOLY
                        
                        # Add holes (interiors)
                        for interior in polygon.interiors:
                            hole_coords = list(interior.coords)
                            flipped_hole = [(x + x_offset, img_height - 1 - y + y_offset) for x, y in hole_coords]
                            
                            vertices.append(flipped_hole[0])
                            codes.append(Path.MOVETO)
                            
                            for coord in flipped_hole[1:]:
                                vertices.append(coord)
                                codes.append(Path.LINETO)
                            codes[-1] = Path.CLOSEPOLY
                        
                        # Create path and patch
                        path = Path(vertices, codes)
                        patch = PathPatch(path, facecolor='black', edgecolor='none', zorder=1)
                        ax.add_patch(patch)
            
            # Plot target_area polygons with color (20, 225, 230, 100) and white border
            target_area = episode_data.get('target_area', self.target_area if hasattr(self, 'target_area') else None)
            if target_area is not None:
                from matplotlib.path import Path
                from matplotlib.patches import PathPatch
                from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
                
                # Handle different target_area types
                polygons_to_render = []
                
                if isinstance(target_area, ShapelyPolygon):
                    polygons_to_render.append(target_area)
                elif isinstance(target_area, MultiPolygon):
                    polygons_to_render.extend(target_area.geoms)
                elif isinstance(target_area, list):
                    for item in target_area:
                        if isinstance(item, (ShapelyPolygon, MultiPolygon)):
                            polygons_to_render.append(item)
                
                # Render each polygon with proper hole support using PathPatch
                for polygon in polygons_to_render:
                    if isinstance(polygon, ShapelyPolygon):
                        # Get exterior coordinates
                        exterior_coords = list(polygon.exterior.coords)
                        flipped_exterior = [(x + x_offset, img_height - 1 - y + y_offset) for x, y in exterior_coords]
                        
                        # Build path with exterior and holes
                        vertices = [flipped_exterior[0]]
                        codes = [Path.MOVETO]
                        
                        for coord in flipped_exterior[1:]:
                            vertices.append(coord)
                            codes.append(Path.LINETO)
                        codes[-1] = Path.CLOSEPOLY
                        
                        # Add holes (interiors)
                        for interior in polygon.interiors:
                            hole_coords = list(interior.coords)
                            flipped_hole = [(x + x_offset, img_height - 1 - y + y_offset) for x, y in hole_coords]
                            
                            vertices.append(flipped_hole[0])
                            codes.append(Path.MOVETO)
                            
                            for coord in flipped_hole[1:]:
                                vertices.append(coord)
                                codes.append(Path.LINETO)
                            codes[-1] = Path.CLOSEPOLY
                        
                        # Create path and patch with target area color and white border
                        path = Path(vertices, codes)
                        patch = PathPatch(path, 
                                        facecolor=(20/255, 225/255, 230/255, 100/255),
                                        edgecolor='white',
                                        linewidth=0.02,
                                        zorder=2)
                        ax.add_patch(patch)
            
            # Plot helicopter path (red line)
            # Flip y-coordinates to match the flipped image and apply offset
            if len(position_history) > 1:
                path_x = [pos[0] + x_offset for pos in position_history]
                path_y = [img_height - 1 - pos[1] + y_offset for pos in position_history]
                ax.plot(path_x, path_y, color='#ff0000', linewidth=2, alpha=0.8, label='Helicopter Path')
            
            # Remove axes, ticks, and labels to clean up the visualization
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            # Remove frame/border around the plot
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Set title with coverage and path length information
            coverage_percentage = (1 - percentage_of_target_area_left) * 100
            title = f"Coverage: {coverage_percentage:.1f}% | Length: {path_length:.1f}m"
            ax.set_title(title, fontsize=36, fontweight='bold', pad=10, color='black')
            
            # Prepare save path
            if save_path is None:
                # Default save location
                save_dir = os.path.join(os.path.dirname(__file__), "misc", "logs")
                os.makedirs(save_dir, exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"environment_render_{timestamp}.png"
                save_path = os.path.join(save_dir, filename)
            
            # Save the figure with white background and fixed dimensions
            plt.savefig(save_path, dpi=225, facecolor='white', pad_inches=0)
            plt.close(fig)
            
            print(f"Rendered environment state saved to: {save_path}")
            
        except Exception as e:
            print(f"Error rendering environment: {e}")
            import traceback
            traceback.print_exc()

    def get_state(self):
        """
        Returns a dict containing all state needed for _get_obs().
        """
        state = {
            # GymnasiumEnv state
            "difference_from_anchor_to_current_position_geo": np.copy(self.difference_from_anchor_to_current_position_geo),
            "list_of_observation_grids": [np.copy(grid) for grid in self.list_of_observation_grids],
            # Sim state (merged env and heli)
            "time_abs": self.time_abs,
            "surface_grid": np.copy(self.surface_grid),
            # Merged heli state
            "position": np.copy(self.position),
            "position_z_above_ground": self.position_z_above_ground,
            "current_bearing": self.current_bearing,
            "observed_radiation_points": np.copy(self.observed_radiation_points),
            "target_area": copy.deepcopy(self.target_area),
            "initial_target_area": copy.deepcopy(self.initial_target_area_borders),
            "polygon_center": self.polygon_center,
        }
        return state

    def set_state(self, state):
        """
        Restores all state needed for _get_obs() from a dict.
        """
        self.difference_from_anchor_to_current_position_geo = np.copy(state["difference_from_anchor_to_current_position_geo"])
        self.list_of_observation_grids = [np.copy(grid) for grid in state["list_of_observation_grids"]]
        self.time_abs = state["time_abs"]
        self.surface_grid = np.copy(state["surface_grid"])
        # Merged heli state
        self.position = np.copy(state["position"])
        self.position_z_above_ground = state["position_z_above_ground"]
        self.current_bearing = state["current_bearing"]
        self.observed_radiation_points = np.copy(state["observed_radiation_points"])
        self.target_area = state["target_area"]
        self.initial_target_area_borders = state["initial_target_area_borders"]
        self.polygon_center = state["polygon_center"]

    # Methods merged from sim class
    def get_time_dependent_radiation_grid(self, t, radiation_grid_episode_parameters):
        """
        Vectorized computation of the time-dependent radiation grid, matching the random seed logic of multi_scale_system_point.
        Ensures that the random values for each (i, j) are identical to those used in single-point calls.
        """
        border = self.geo_coordinates_of_radiation_grid.shape[:2]
        size_x, size_y = border

        # Prepare i, j grid
        I, J = np.meshgrid(np.arange(size_x), np.arange(size_y), indexing='ij')
        # Unpack parameters
        p = radiation_grid_episode_parameters
        base_phase_offset = np.array(p['phase_offset'])
        base_speed_multiplier = np.array(p['speed_multiplier'])
        base_intensity_multiplier = np.array(p['intensity_multiplier'])
        frequency_multiplier = np.array(p['frequency_multiplier'])
        noise_level = p['noise_level']
        movement_x = np.array(p['movement_x'])
        movement_y = np.array(p['movement_y'])
        # Vectorized random phase: must match the per-point seed logic
        # Large scale
        large_scale = (
            base_intensity_multiplier[0] * np.sin(
                (I * movement_x[0] + J * movement_x[1] + t * 1.5 * base_speed_multiplier[0] + base_phase_offset[0]) * 0.01 * frequency_multiplier[0]
            ) * np.cos(
                (I * movement_y[0] + J * movement_y[1] + t * 1.2 * base_speed_multiplier[0]) * 0.015 * frequency_multiplier[0]
            )
        )
        # Meso scale
        meso_scale = (
            0.6 * base_intensity_multiplier[1] * np.sin(
                (I * movement_x[2] + J * movement_x[3] + t * 3 * base_speed_multiplier[1] + base_phase_offset[1]) * 0.04 * frequency_multiplier[1]
            ) * np.cos(
                (I * movement_y[2] + J * movement_y[3] + t * 2.5 * base_speed_multiplier[1]) * 0.05 * frequency_multiplier[1]
            )
        )
        # Micro scale
        micro_scale = (

            0.3 * base_intensity_multiplier[2] * np.sin(
                (I * J * movement_x[4] + (I + J) * movement_y[4] + t * 5 * base_speed_multiplier[2] + base_phase_offset[2]) * 0.001 * frequency_multiplier[2]
            )
        )
        # Noise (deterministic, not random per point)
        noise = noise_level
        combined = large_scale + meso_scale + micro_scale + noise
        combined = np.maximum(0, combined)
        # Vignette effect (vectorized)
        return 40 * combined

    def generate_surface_grid_200m_const_height(self):
        height_map = np.ones(self.gridsize[:2])*200
        return height_map

    def generate_surface_grid_via_perlin_noise(self):
        # Placeholder implementation - returns flat surface for now
        return np.ones(self.gridsize[:2])*200

    def sim_step(self, action_vector): # Update sim state
        action_vector = self.map_gymenv_action_to_sim_action(action_vector)
        self.time_rel_before_executing_action = self.time_rel
        
        # Original env step functionality
        self.time_rel += self.base_timestep
        self.time_abs += self.base_timestep
        if para.z_rad_type == 'mov' and self.radiation_grid_visualization:
            self.radiation_grid = self.get_time_dependent_radiation_grid(self.time_rel, self.radiation_grid_episode_parameters)
        self.percentage_of_episode_finished_by_time_passed = self.time_rel / self.max_flight_time
        
        # Merged step functionality from heli class (if action_vector is provided)
        if action_vector is not None:
            old_bearing = self.current_bearing
            self.current_bearing = (self.current_bearing + action_vector[0]) % 360
            
            # Track total degrees turned
            degrees_change = abs(action_vector[0])
            self.total_degrees_turned += degrees_change
            
            # Update grids if bearing changed
            self._update_grids_for_bearing(self.current_bearing)
            
            if para.z_rad_type == 'jon':
                # For flat coordinate system, use simple trigonometric movement
                bearing_rad = np.deg2rad(self.current_bearing)
                dx = self.distance_to_move_per_step * np.sin(bearing_rad)
                dy = self.distance_to_move_per_step * np.cos(bearing_rad)
                # FIX: Ensure position is 1D before arithmetic operations
                current_pos = self.position
                new_x = current_pos[0] + dx
                new_y = current_pos[1] + dy
                self.update_position(np.array([new_x, new_y]))
            else:
                # For geodesic coordinate system ('mov', 'bfs'), use geodesic calculations
                destination = distance(meters=self.distance_to_move_per_step).destination(GeopyPoint(self.position[1], self.position[0]), bearing=self.current_bearing)
                self.update_position(np.array([destination.longitude, destination.latitude]))
            
            # Compute lidar observations if using lidar ablation study or
            # lidar-reconstructed wall gliding
            if self.lidar_obstacle_tracking_enabled and para.z_rad_type == 'jon':
                # Compute lidar point cloud
                lidar_pts, pts_info = self._compute_lidar_pts_flat(self.position, self.current_bearing)
                # Store for use in observation
                self.last_lidar_pts = lidar_pts
                self.last_lidar_pts_info = pts_info
                # Update obstacle maps based on lidar detections
                self._update_obstacles_from_lidar_pts(lidar_pts, pts_info)
            
            self.update_radiation_exposure_history_target_area()
        
        if self.percentage_of_episode_finished_by_time_passed >= 1:
            self.end_sim = True
            self.end_sim_reason = 'Time limit reached'

    def map_gymenv_action_to_sim_action(self, action_vector):
        """
        Maps the gym env action vector to the sim action vector.
        Action is always a single scalar for turning angle (no height change).
        
        :param action_vector: The gym env action vector (single scalar).
        """
        # Extract scalar value if it's an array
        action_value = action_vector[0] if isinstance(action_vector, (list, np.ndarray)) else action_vector
        # Convert to turning angle
        action_vector = np.array([action_value * para.largest_horizontal_turning_angle_deg])
        return action_vector
    
    # Merged methods from heli class
    def calculate_angle(self, A, B, current_direction):
        if (current_direction != np.array([0, 0])).any():
            # Extract longitude and latitude values
            lon_A, lat_A = A
            lon_B, lat_B = B[1], B[0]  # B is a Shapely Point

            # Create vectors
            vector_AB = np.array([lon_B - lon_A, lat_B - lat_A])
            vector_current = current_direction

            # Calculate the cosine of the angle
            cosine_angle = np.dot(vector_AB, vector_current) / (np.linalg.norm(vector_AB) * np.linalg.norm(vector_current))
            
            return np.clip(cosine_angle, -1, 1)
        else:
            return 0
            
    def preprocess_image_for_odl(self, img):
        """
        Preprocess image for ODL calculation by creating a "bouquet" (dilated area) around black shapes.
        Keeps black areas black and adds dilated regions around them.
        
        Parameters:
        - img: grayscale image (0=black, 255=white)
        
        Returns:
        - processed image with black shapes having dilated "bouquet" around them
        """
        # Create binary mask where black pixels (radiation sources) are True
        black_mask = (img == 0)
        
        # Apply morphological dilation with 5 pixel radius (circular kernel)
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 5 pixel radius = 11x11 kernel
        dilated_mask = cv2.dilate(black_mask.astype(np.uint8), dilation_kernel, iterations=1)
        
        # Define structuring element for Gaussian blur (circular kernel)
        kernel_radius_gaussian_blur = 17
        # Apply Gaussian blur to smooth the dilated mask for gradual transitions
        blurred_mask = cv2.GaussianBlur(dilated_mask.astype(np.float32), (kernel_radius_gaussian_blur, kernel_radius_gaussian_blur), sigmaX=0.3*kernel_radius_gaussian_blur, sigmaY=0.3*kernel_radius_gaussian_blur)
        
        # Create output image: start with original image
        processed_img = img.copy().astype(np.float32)
        
        # Apply the blurred mask: blend between original and black based on mask intensity
        # Where blurred_mask is 1.0, make it black (0); where it's 0.0, keep original
        processed_img = processed_img * (1.0 - blurred_mask) + 0.0 * blurred_mask
        
        # Convert back to uint8
        processed_img = processed_img.astype(np.uint8)
        # invert
        processed_img = 255 - processed_img
        return para.improved_odl_scaling_factor * processed_img


    def extract_black_polygons_from_png(self, img):
        """
        Extract black polygons from a PNG image and convert them to shapely MultiPolygon.
        Uses OpenCV's hierarchical contour detection to properly handle nested polygons.
        
        Parameters:
        - img: grayscale image (0=black, 255=white)
        
        Returns:
        - Tuple of (MultiPolygon, MultiLineString, MultiPolygon, MultiLineString, MultiPolygon): 
          original polygons, original boundaries, dilated polygons, dilated boundaries, dilated for coverage calc polygons
        """
        # Create binary mask where black pixels are True
        black_mask = (img == 0).astype(np.uint8)
        
        # Find contours with hierarchy to handle nested polygons (holes)
        contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # No black regions found
            empty_multi = MultiPolygon()
            return empty_multi, empty_multi.boundary, empty_multi, empty_multi.boundary, empty_multi
        
        polygons = []
        
        # Process contours using hierarchy to identify outer contours and holes
        for i, contour in enumerate(contours):
            if len(contour) < 3:
                continue
                
            # Extract coordinates from contour
            coords = [(point[0][0], point[0][1]) for point in contour]
            if len(coords) < 3:
                continue
            
            # Check if this is an outer contour (parent is -1) or a hole (parent exists)
            if hierarchy is not None and hierarchy[0][i][3] == -1:
                # This is an outer contour, find its holes
                holes = []
                
                # Find child contours (holes) for this outer contour
                child_idx = hierarchy[0][i][2]  # First child
                while child_idx != -1:
                    if child_idx < len(contours) and len(contours[child_idx]) >= 3:
                        hole_coords = [(point[0][0], point[0][1]) for point in contours[child_idx]]
                        if len(hole_coords) >= 3:
                            holes.append(hole_coords)
                    # Move to next sibling
                    child_idx = hierarchy[0][child_idx][0] if child_idx < len(hierarchy[0]) else -1
                
                # Create polygon with holes
                try:
                    if holes:
                        polygon = Polygon(coords, holes=holes)
                    else:
                        polygon = Polygon(coords)
                    
                    if polygon.is_valid and not polygon.is_empty:
                        polygons.append(polygon)
                except Exception:
                    # If polygon creation fails, try without holes
                    try:
                        polygon = Polygon(coords)
                        if polygon.is_valid and not polygon.is_empty:
                            polygons.append(polygon)
                    except Exception:
                        continue
        
        # Create original MultiPolygon
        original_multi_polygon = MultiPolygon(polygons) if polygons else MultiPolygon()
        
        # Apply dilation to all polygons
        dilation_pixels = para.dilation_of_black_polygons_in_m / self.meters_per_pixel_mower
        dilated_polygons = []
        
        for poly in polygons:
            try:
                dilated_poly = poly.buffer(dilation_pixels)
                if dilated_poly.is_valid and not dilated_poly.is_empty:
                    if hasattr(dilated_poly, 'geoms'):
                        # Handle MultiPolygon result from buffer
                        dilated_polygons.extend(dilated_poly.geoms)
                    else:
                        dilated_polygons.append(dilated_poly)
            except Exception:
                # Keep original polygon if dilation fails
                dilated_polygons.append(poly)
        
        dilated_multi_polygon = MultiPolygon(dilated_polygons) if dilated_polygons else MultiPolygon()
        
        # Apply coverage calculation dilation (mower radius based, similar to mower_env.py)
        coverage_dilation_pixels = para.dilation_for_coverage_limit_calculation_m * 0.15 / self.meters_per_pixel_mower  # Mower radius in pixels
        dilated_for_coverage_polygons = []
        
        for poly in polygons:
            try:
                dilated_poly = poly.buffer(coverage_dilation_pixels)
                if dilated_poly.is_valid and not dilated_poly.is_empty:
                    if hasattr(dilated_poly, 'geoms'):
                        # Handle MultiPolygon result from buffer
                        dilated_for_coverage_polygons.extend(dilated_poly.geoms)
                    else:
                        dilated_for_coverage_polygons.append(dilated_poly)
            except Exception:
                # Keep original polygon if dilation fails
                dilated_for_coverage_polygons.append(poly)
        
        dilated_for_coverage_multi_polygon = MultiPolygon(dilated_for_coverage_polygons) if dilated_for_coverage_polygons else MultiPolygon()
        return original_multi_polygon, original_multi_polygon.boundary, dilated_multi_polygon, dilated_multi_polygon.boundary, dilated_for_coverage_multi_polygon

    def calc_odl_jon_improved(self, x, y):
        """
        Calculate ODL value using bilinear interpolation between 4 closest points.
        
        Parameters:
        - x, y: continuous pixel coordinates in the original image
        
        Returns:
        - ODL value in mSv/h
        """
        height, width = self.precalculated_odl_map.shape
        
        # Clamp coordinates to valid range
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        
        # Get the 4 surrounding pixel coordinates
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, width - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, height - 1)
        
        # Get fractional parts for interpolation weights
        fx = x - x0
        fy = y - y0
        
        # Get the 4 corner values
        q00 = self.precalculated_odl_map[y0, x0]  # top-left
        q01 = self.precalculated_odl_map[y1, x0]  # bottom-left
        q10 = self.precalculated_odl_map[y0, x1]  # top-right
        q11 = self.precalculated_odl_map[y1, x1]  # bottom-right
        
        # Bilinear interpolation
        # First interpolate in x-direction
        r0 = q00 * (1 - fx) + q10 * fx  # top edge
        r1 = q01 * (1 - fx) + q11 * fx  # bottom edge
        
        # Then interpolate in y-direction
        interpolated_value = r0 * (1 - fy) + r1 * fy
        
        return interpolated_value
            
    def calculate_odl(self):
        if para.z_rad_type == 'mov':
            x, y = self.position_as_xy
            radiation_exposure = self.multi_scale_system_point(x, y, self.time_rel, self.radiation_grid_episode_parameters)
        elif para.z_rad_type == 'bfs':
            index_of_closest_radiation_grid_coordinate = self.tree_radiation_grid.query(self.position)[1]
            i, j = np.unravel_index(index_of_closest_radiation_grid_coordinate, self.geo_coordinates_of_radiation_grid.shape[:2])
            radiation_exposure = self.radiation_grid[i, j]
        elif para.z_rad_type == 'jon':
            # Use position_as_xy which now uses correct granularity for cpp maps
            x, y = self.position_as_xy
            # Use improved ODL calculation with precalculated values
            radiation_exposure = self.calc_odl_jon_improved(x, y)
        return radiation_exposure  # in mSv/h
    def multi_scale_system_point(self, x, y, t, instance_params):
        base_phase_offset = instance_params['phase_offset']
        base_speed_multiplier = instance_params['speed_multiplier']
        base_intensity_multiplier = instance_params['intensity_multiplier']
        frequency_multiplier = instance_params['frequency_multiplier']
        noise_level = instance_params['noise_level']
        movement_x = instance_params['movement_x']
        movement_y = instance_params['movement_y']
        # Note: to maintain same behavior as original where i=y_coord and j=x_coord,
        # we need to use y where we used i, and x where we used j
        large_scale = base_intensity_multiplier[0] * np.sin(
            (y*movement_x[0] + x*movement_x[1] + t*1.5*base_speed_multiplier[0] + base_phase_offset[0] )*0.01*frequency_multiplier[0]
        ) * np.cos(
            (y*movement_y[0] + x*movement_y[1] + t*1.2*base_speed_multiplier[0])*0.015*frequency_multiplier[0]
        )
        meso_scale = 0.6 * base_intensity_multiplier[1] * np.sin(
            (y*movement_x[2] + x*movement_x[3] + t*3*base_speed_multiplier[1] + base_phase_offset[1] )*0.04*frequency_multiplier[1]
        ) * np.cos(
            (y*movement_y[2] + x*movement_y[3] + t*2.5*base_speed_multiplier[1])*0.05*frequency_multiplier[1]
        )
        micro_scale = (
            0.3 * base_intensity_multiplier[2] * np.sin(
                (y*x*movement_x[4] + (y+x)*movement_y[4] + t*5*base_speed_multiplier[2] + base_phase_offset[2] )*0.001*frequency_multiplier[2]
            )
        )
        noise = noise_level #* np.random.normal(0, 1)
        combined = large_scale + meso_scale + micro_scale + noise
        combined = max(0, combined)
        return 40 *combined

    def generate_random_radar_params(self):
        return {
            'phase_offset': np.random.uniform(0, 2*np.pi, 3),
            'speed_multiplier': np.random.uniform(0.02, 0.02, 3),  # slower movement
            'intensity_multiplier': np.random.uniform(1, 1, 3),
            'frequency_multiplier': np.random.uniform(3, 3, 3),  # higher freq = smaller clouds
            'noise_level': 0.0, #np.random.uniform(0.01, 0.04),  # less noise
            'movement_x': np.random.uniform(-0.5, 0.5, 5),  # less variance
            'movement_y': np.random.uniform(-0.5, 0.5, 5)   # less variance
        }    
    def random_number_between_x_y_with_close_to_y_more_likely(self, x, y):
        # Generate a uniform random number in [0, 1]
        u = np.random.uniform(0, 0.03)
        # Transform it using the inverse CDF
        z = ((u * (y**3 - x**3)) + x**3)**(1/3)
        return z

    def idw_interpolation(self,points, values, target, p=2):
        distances = np.linalg.norm(points - target, axis=1)
        weights = 1 / (distances ** p)
        return np.sum(weights * values) / np.sum(weights)
    
    def safe_geometry_union(self, geom1, geom2, buffer_size=1e-10):
        """
        Safely perform geometry union operation with topology validation and buffering.
        
        Args:
            geom1: First geometry
            geom2: Second geometry
            buffer_size: Small buffer to fix topology issues
            
        Returns:
            Geometry result of geom1.union(geom2) or geom1 if operation fails
        """
        # Early exit if either geometry is None
        if geom1 is None and geom2 is None:
            return MultiPolygon()
        if geom1 is None:
            return geom2
        if geom2 is None:
            return geom1
            
        # Early exit if one geometry is empty
        if geom1.is_empty:
            return geom2
        if geom2.is_empty:
            return geom1
        try:
            # First try the operation as-is
            result = geom1.union(geom2)
            if result.is_valid:
                return result
        except Exception:
            pass
            
        # If that fails, try buffering both geometries slightly to fix topology issues
        try:
            # Buffer both geometries with a tiny amount to fix topology
            geom1_buffered = geom1.buffer(buffer_size)
            geom2_buffered = geom2.buffer(buffer_size)
            
            if geom1_buffered.is_valid and geom2_buffered.is_valid:
                result = geom1_buffered.union(geom2_buffered)
                if result.is_valid:
                    return result
        except Exception:
            pass
            
        # If buffering fails, try making geometries valid
        try:
            geom1_valid = make_valid(geom1)
            geom2_valid = make_valid(geom2)
            
            if geom1_valid.is_valid and geom2_valid.is_valid:
                result = geom1_valid.union(geom2_valid)
                if result.is_valid:
                    return result
        except Exception:
            pass
            
        # Last resort: return the larger geometry or the first one
        print(f"Warning: Geometry union operation failed, returning first geometry")
        return geom1

    def safe_geometry_intersection(self, geom1, geom2, buffer_size=1e-10):
        """
        Safely perform geometry intersection operation with topology validation and buffering.
        
        Args:
            geom1: First geometry
            geom2: Second geometry
            buffer_size: Small buffer to fix topology issues
            
        Returns:
            Geometry result of geom1.intersection(geom2) or empty geometry if operation fails
        """
        # Early exit if either geometry is None
        if geom1 is None or geom2 is None:
            return MultiPolygon()
            
        # Early exit if either geometry is empty
        if geom1.is_empty or geom2.is_empty:
            return MultiPolygon()
            
        try:
            # First try the operation as-is
            result = geom1.intersection(geom2)
            if result.is_valid:
                return result
        except Exception:
            pass
            
        # If that fails, try buffering both geometries slightly to fix topology issues
        try:
            # Buffer both geometries with a tiny amount to fix topology
            geom1_buffered = geom1.buffer(buffer_size)
            geom2_buffered = geom2.buffer(buffer_size)
            
            if geom1_buffered.is_valid and geom2_buffered.is_valid:
                result = geom1_buffered.intersection(geom2_buffered)
                if result.is_valid:
                    return result
        except Exception:
            pass
            
        # If buffering fails, try making geometries valid
        try:
            geom1_valid = make_valid(geom1)
            geom2_valid = make_valid(geom2)
            
            if geom1_valid.is_valid and geom2_valid.is_valid:
                result = geom1_valid.intersection(geom2_valid)
                if result.is_valid:
                    return result
        except Exception:
            pass
            
        # Last resort: return empty geometry
        print(f"Warning: Geometry intersection operation failed, returning empty geometry")
        return MultiPolygon()

    def safe_geometry_difference(self, geom1, geom2, buffer_size=1e-10):
        """
        Safely perform geometry difference operation with topology validation and buffering.
        
        Args:
            geom1: First geometry (minuend)
            geom2: Second geometry (subtrahend)
            buffer_size: Small buffer to fix topology issues
            
        Returns:
            Geometry result of geom1.difference(geom2) or geom1 if operation fails
        """
        # Early exit if either geometry is None
        if geom1 is None:
            return MultiPolygon()
        if geom2 is None:
            return geom1
            
        # Early exit if either geometry is empty
        if geom1.is_empty or geom2.is_empty:
            return geom1
            
        try:
            # First try the operation as-is
            result = geom1.difference(geom2)
            if result.is_valid:
                return result
        except Exception:
            pass
            
        # If that fails, try buffering both geometries slightly to fix topology issues
        try:
            # Buffer both geometries with a tiny amount to fix topology
            geom1_buffered = geom1.buffer(buffer_size)
            geom2_buffered = geom2.buffer(buffer_size)
            
            if geom1_buffered.is_valid and geom2_buffered.is_valid:
                result = geom1_buffered.difference(geom2_buffered)
                if result.is_valid:
                    return result
        except Exception:
            pass
            
        # If buffering fails, try making geometries valid
        try:
            geom1_valid = make_valid(geom1)
            geom2_valid = make_valid(geom2)
            
            if geom1_valid.is_valid and geom2_valid.is_valid:
                result = geom1_valid.difference(geom2_valid)
                if result.is_valid:
                    return result
        except Exception:
            pass
            
        # Last resort: return the original geometry
        print(f"Warning: Geometry difference operation failed, returning original geometry")
        return geom1

    def safe_line_intersection(self, line1, line2, buffer_size=1e-10):
        """
        Safely perform line geometry intersection operation with topology validation.
        
        Args:
            line1: First line geometry
            line2: Second line geometry
            buffer_size: Small buffer to fix topology issues
            
        Returns:
            Geometry result of line1.intersection(line2) or empty LineString if operation fails
        """
        # Early exit if either geometry is None or empty
        if line1 is None or line2 is None or line1.is_empty or line2.is_empty:
            return LineString()
            
        try:
            # First try the operation as-is
            result = line1.intersection(line2)
            return result
        except Exception as e:
            print(f"Warning: Line intersection operation failed: {e}, returning empty LineString")
            return LineString()

    def safe_boundary(self, geom):
        """
        Safely get the boundary of a geometry, handling GeometryCollection cases.
        For GeometryCollections, extracts polygonal parts first before getting boundary.
        
        Args:
            geom: Input geometry (any Shapely geometry type)
            
        Returns:
            LineString/MultiLineString boundary or empty LineString if operation fails
        """
        # Early exit if geometry is None or empty
        if geom is None or geom.is_empty:
            return LineString()
            
        try:
            # Check if it's a GeometryCollection
            if hasattr(geom, 'geom_type') and geom.geom_type == 'GeometryCollection':
                # Extract only polygonal parts from the collection
                poly_geom = self.polygons_only_vec(geom)
                if poly_geom is None or poly_geom.is_empty:
                    return LineString()
                return poly_geom.boundary
            else:
                # Direct boundary for non-collections
                return geom.boundary
        except Exception as e:
            print(f"Warning: Boundary operation failed: {e}, returning empty LineString")
            return LineString()

    def fast_geometry_difference(self, small_geom, large_geom):
        """
        Optimized geometry difference operation for small geometry minus large geometry.
        Uses spatial indexing and bounding box filtering to improve performance.
        
        Args:
            small_geom: Small geometry (e.g., cone union)
            large_geom: Large geometry (e.g., target_area) 
        
        Returns:
            Geometry result of small_geom.difference(large_geom)
        """
        # Early exit if either geometry is None
        if small_geom is None:
            return MultiPolygon()
        if large_geom is None:
            return small_geom
            
        # Early exit if either geometry is empty
        if small_geom.is_empty or large_geom.is_empty:
            return small_geom
            
        # Get bounding boxes for quick intersection test
        small_bounds = small_geom.bounds  # (minx, miny, maxx, maxy)
        large_bounds = large_geom.bounds
        
        # Check if bounding boxes don't intersect - if so, no difference needed
        if (small_bounds[2] < large_bounds[0] or  # small maxX < large minX
            small_bounds[0] > large_bounds[2] or  # small minX > large maxX  
            small_bounds[3] < large_bounds[1] or  # small maxY < large minY
            small_bounds[1] > large_bounds[3]):   # small minY > large maxY
            return small_geom  # No intersection, return original
        
        # For MultiPolygon, use spatial indexing to find only intersecting parts
        if hasattr(large_geom, 'geoms') and len(large_geom.geoms) > 1:
            try:
                # Create spatial index for large geometry parts
                geoms_list = list(large_geom.geoms)
                if len(geoms_list) > 10:  # Only use spatial index for substantial polygons
                    tree = STRtree(geoms_list)
                    # Query for potential intersections with small geometry
                    potential_intersections = tree.query(small_geom)
                    
                    if not potential_intersections:
                        return small_geom  # No intersections found
                    
                    # Build reduced geometry from only intersecting parts
                    intersecting_parts = []
                    for geom in potential_intersections:
                        if small_geom.intersects(geom):
                            intersecting_parts.append(geom)
                    
                    if not intersecting_parts:
                        return small_geom
                    
                    # Create reduced geometry for difference operation
                    if len(intersecting_parts) == 1:
                        reduced_large_geom = intersecting_parts[0]
                    else:
                        reduced_large_geom = MultiPolygon(intersecting_parts)
                    
                    # Perform difference with reduced geometry using direct shapely operation
                    return small_geom.difference(reduced_large_geom)
            except Exception:
                # Fallback to regular difference if spatial indexing fails
                pass
        
        # Fallback to direct shapely difference operation
        return small_geom.difference(large_geom)
    
    def fast_geometry_intersection(self, geom1, geom2):
        """
        Optimized geometry intersection operation with bounding box filtering.
        Uses spatial indexing and bounding box filtering to improve performance.
        
        Args:
            geom1: First geometry
            geom2: Second geometry
        
        Returns:
            Geometry result of geom1.intersection(geom2)
        """
        # Early exit if either geometry is None
        if geom1 is None or geom2 is None:
            return MultiPolygon()
            
        # Early exit if either geometry is empty
        if geom1.is_empty or geom2.is_empty:
            return MultiPolygon()
            
        # Get bounding boxes for quick intersection test
        geom1_bounds = geom1.bounds  # (minx, miny, maxx, maxy)
        geom2_bounds = geom2.bounds
        
        # Check if bounding boxes don't intersect - if so, no intersection
        if (geom1_bounds[2] < geom2_bounds[0] or  # geom1 maxX < geom2 minX
            geom1_bounds[0] > geom2_bounds[2] or  # geom1 minX > geom2 maxX  
            geom1_bounds[3] < geom2_bounds[1] or  # geom1 maxY < geom2 minY
            geom1_bounds[1] > geom2_bounds[3]):   # geom1 minY > geom2 maxY
            return MultiPolygon()  # No intersection
        
        # For MultiPolygon, use spatial indexing to find only intersecting parts
        if hasattr(geom2, 'geoms') and len(geom2.geoms) > 1:
            try:
                # Create spatial index for geom2 parts
                geoms_list = list(geom2.geoms)
                if len(geoms_list) > 10:  # Only use spatial index for substantial polygons
                    tree = STRtree(geoms_list)
                    # Query for potential intersections with geom1
                    potential_intersections = tree.query(geom1)
                    
                    if not potential_intersections:
                        return MultiPolygon()  # No intersections found
                    
                    # Build list of actual intersections
                    intersection_results = []
                    for geom in potential_intersections:
                        if geom1.intersects(geom):
                            try:
                                intersection = geom1.intersection(geom)
                                if not intersection.is_empty:
                                    intersection_results.append(intersection)
                            except Exception:
                                continue
                    
                    if not intersection_results:
                        return MultiPolygon()
                    
                    # Combine all intersection results
                    if len(intersection_results) == 1:
                        return intersection_results[0]
                    else:
                        # Filter to only polygons
                        polygons = []
                        for result in intersection_results:
                            if hasattr(result, 'geom_type'):
                                if result.geom_type == 'Polygon':
                                    polygons.append(result)
                                elif result.geom_type == 'MultiPolygon':
                                    polygons.extend(result.geoms)
                        return MultiPolygon(polygons) if polygons else MultiPolygon()
            except Exception:
                # Fallback to regular intersection if spatial indexing fails
                pass
        
        # Fallback to direct shapely intersection operation
        return geom1.intersection(geom2)
    
    def polygons_only_vec(self, geom):
        """
        Vectorized function to extract only polygonal types from a geometry and return as MultiPolygon.
        Handles complex geometries including collections and ensures consistent MultiPolygon output.
        
        Args:
            geom: Input geometry (any Shapely geometry type)
            
        Returns:
            MultiPolygon containing only the polygonal parts of the input geometry
        """
        
        if geom is None or geom.is_empty:
            return MultiPolygon([])

        # 1) Explode the geometry/collection into parts (vectorized, C-level)
        parts = get_parts(geom)              # -> numpy array of geometries

        # 2) Keep only polygonal types (type_id: 3=Polygon, 6=MultiPolygon)
        type_ids = get_type_id(parts)
        mask_poly_like = np.isin(type_ids, [3, 6])  # 3=Polygon, 6=MultiPolygon
        poly_like = parts[mask_poly_like]

        if len(poly_like) == 0:
            # Return an empty MultiPolygon
            return MultiPolygon([])

        # 3) Explode MultiPolygons to Polygons (still vectorized)
        poly_parts = get_parts(poly_like)    # -> Polygons if any were MultiPolygons
        poly_type_ids = get_type_id(poly_parts)
        only_polys = poly_parts[poly_type_ids == 3]  # 3=Polygon
        
        if len(only_polys) == 0:
            return MultiPolygon([])

        # 4) Dissolve touching pieces (fast reduction) and ensure MultiPolygon output
        result = union_all(only_polys)
        
        # Ensure result is always a MultiPolygon
        if hasattr(result, 'geom_type'):
            if result.geom_type == 'Polygon':
                return MultiPolygon([result])
            elif result.geom_type == 'MultiPolygon':
                return result
        
        # Fallback for unexpected geometry types
        return MultiPolygon([])

    def simplify_polygon_if_needed(self, step_counter, simplification_interval, polygon, area_loss_threshold=0.95):
        """
        Periodically simplify target area geometry to prevent performance degradation.
        
        Args:
            step_counter: Current step/iteration counter
            simplification_interval: How often to perform simplification
        """
        if step_counter % simplification_interval == 0 and not polygon.is_empty:
            try:
                # Calculate appropriate tolerance based on area size
                area = polygon.area
                tolerance = max(1e-8, area * 1e-6)  # Dynamic tolerance
                
                simplified = polygon.simplify(tolerance, preserve_topology=True)
                
                # Only use simplified version if it's valid and area loss is minimal
                if simplified.is_valid and not simplified.is_empty:
                    area_ratio = simplified.area / area if area > 0 else 1
                    if area_ratio > area_loss_threshold:  # Less than 5% area loss
                        polygon = simplified
            except Exception:
                # If simplification fails, continue with original geometry
                pass
    
    def update_radiation_exposure_history_target_area(self, odl_from_live_data=None, live_mode=False):
        self.target_area_last_step_size = self.target_area_size

        if live_mode:
            self.current_radiation_exposure = odl_from_live_data
        else:
            if para.how_to_calculate_odl == 'via simulated radiation grid':
                self.current_radiation_exposure = self.calculate_odl()
            elif para.how_to_calculate_odl == 'random_test_values':
                self.current_radiation_exposure = np.random.uniform(0, 1)

        if self.current_radiation_exposure >= para.radiation_value_cutoff:
            self.observed_radiation_points = np.vstack((
                self.observed_radiation_points,
                np.array([self.position[0], self.position[1], self.current_radiation_exposure]).reshape(1, -1)
            ))

        n = self.observed_radiation_points.shape[0]
        if para.include_time_in_observation:
            freshness = np.arange(0, n)
            freshness = np.clip(255 - freshness, 0, 255)[::-1]
            self.observed_radiation_points_freshness = np.hstack((
                self.observed_radiation_points[:, :2],
                freshness.reshape(-1, 1)
            ))

        # Calculate circle geometry using pre-created base polygon and translation
        self.last_cone_polygon = getattr(self, "cone_polygon", None)
        self.cone_polygon = self._get_cone_polygon_at_position(self.position)

        # Create circle union and get its convex hull using safe operations
        cone_union = self.cone_polygon if self.last_cone_polygon is None else self.safe_geometry_union(self.cone_polygon, self.last_cone_polygon)
        self.cone_union_last_cone_convex_hull = cone_union.convex_hull
        
        # Calculate key intersections using safe operations (reuse black polygon intersection)
        cone_black_intersection = self.safe_geometry_intersection(self.cone_union_last_cone_convex_hull, self.black_polygons_dilated)
        self.cone_union_last_cone_convex_hull_intersection_target_area = self.safe_geometry_intersection(self.cone_union_last_cone_convex_hull, self.target_area)
        self.cone_union_last_cone_convex_hull_intersection_black_polygons = cone_black_intersection
        
        # Track areas measured multiple times
        # Calculate the newly measured area (excluding last cone to avoid double counting)
        cone_union_last_cone_convex_hull_intersection_target_area_without_last_cone = self.safe_geometry_difference(
            self.cone_union_last_cone_convex_hull_intersection_target_area, 
            self.last_cone_polygon
        ) if self.last_cone_polygon is not None else self.cone_union_last_cone_convex_hull_intersection_target_area
        
        # Ensure we only keep polygons/multipolygons
        cone_union_last_cone_convex_hull_intersection_target_area_without_last_cone = self.polygons_only_vec(
            cone_union_last_cone_convex_hull_intersection_target_area_without_last_cone
        )
        
        # Store the state BEFORE updating for correct tracking of multiple measurements
        #previous_target_area_remeasured_1_times = self.target_area_remeasured_1_times
        #previous_target_area_remeasured_2_times = self.target_area_remeasured_2_times
        
        #if cone_union_last_cone_convex_hull_intersection_target_area_without_last_cone is not None and not cone_union_last_cone_convex_hull_intersection_target_area_without_last_cone.is_empty:
            # Append to target_area_remeasured_1_times (first measurement)
            #self.target_area_remeasured_1_times = self.safe_geometry_union(
            #    self.target_area_remeasured_1_times,
            #    cone_union_last_cone_convex_hull_intersection_target_area_without_last_cone
            #)
            #self.target_area_remeasured_1_times = self.polygons_only_vec(self.target_area_remeasured_1_times)
            
            #if self.target_area_remeasured_1_times is not None and not self.target_area_remeasured_1_times.is_empty:
            #    # Calculate target_area_remeasured_2_times: areas that were ALREADY in measured_1_times AND are now covered again
            #    self.cone_union_last_cone_convex_hull_without_last_cone = self.safe_geometry_difference(self.cone_union_last_cone_convex_hull, self.last_cone_polygon) if self.last_cone_polygon is not None else MultiPolygon()
            #    #cone_union_without_last_cone = self.polygons_only_vec(self.cone_union_last_cone_convex_hull_without_last_cone)
            #    newly_remeasured_2_times = self.safe_geometry_intersection(
            #        previous_target_area_remeasured_1_times,  # Use PREVIOUS state, not current
            #        self.cone_union_last_cone_convex_hull_without_last_cone
            #    )
            #    #newly_remeasured_2_times = self.polygons_only_vec(newly_remeasured_2_times)
#
            #    self.target_area_remeasured_2_times = self.safe_geometry_union(
            #        self.target_area_remeasured_2_times,
            #        newly_remeasured_2_times
            #    )
            #    #self.target_area_remeasured_2_times = self.polygons_only_vec(self.target_area_remeasured_2_times)

        # Calculate new measured area efficiently using safe operations
        temp_diff = self.safe_geometry_difference(self.cone_union_last_cone_convex_hull, self.cone_union_last_cone_convex_hull_intersection_target_area)
        self.new_measured_area = self.safe_geometry_difference(temp_diff, cone_black_intersection)
        self.new_measured_area_with_black_polygons = temp_diff

        # Update target area using safe operations
        self.target_area = self.safe_geometry_union(self.target_area, self.cone_union_last_cone_convex_hull)
        
        # Prepare target_area for faster point-in-polygon queries (shapely optimization)
        if self.target_area is not None and not self.target_area.is_empty:
            prepare(self.target_area)
        
        # Invalidate cached boundary since target_area was updated
        self._cached_target_area_boundary = None
        self._cached_target_area_boundary_coords = None
        self._cached_boundary_coords_filtered_obstacles = None
        self._cached_boundary_coords_dense = None

        # Calculate boundaries and measurements using safe operations
        # Count newly measured area only within the same region the coverage
        # denominator uses (map borders minus dilated obstacles); cone overhang
        # beyond the outer wall or over obstacles must not count as coverage.
        new_measured_clipped = self.safe_geometry_intersection(
            self.new_measured_area_with_black_polygons, self.initial_target_area_borders) \
            if self.new_measured_area_with_black_polygons is not None else None
        if new_measured_clipped is not None and getattr(self, 'black_polygons_dilated_for_coverage_calc', None) is not None:
            new_measured_clipped = self.safe_geometry_difference(
                new_measured_clipped, self.black_polygons_dilated_for_coverage_calc)
        self.new_measured_area_size = abs(new_measured_clipped.area) if new_measured_clipped is not None else 0.0
        new_measured_area_minus_black = self.safe_geometry_difference(self.new_measured_area, self.black_polygons_dilated)
        self.new_measured_area_boundary = self.safe_boundary(new_measured_area_minus_black) if new_measured_area_minus_black is not None and not new_measured_area_minus_black.is_empty else None
        cone_black_boundary = self.safe_boundary(cone_black_intersection) if cone_black_intersection is not None and not cone_black_intersection.is_empty else None
        self.cone_union_last_cone_convex_hull_intersection_target_area_boundary = self.safe_boundary(self.cone_union_last_cone_convex_hull_intersection_target_area) if self.cone_union_last_cone_convex_hull_intersection_target_area is not None and not self.cone_union_last_cone_convex_hull_intersection_target_area.is_empty else None

        self.new_measured_area_boundary_intersection_black_polygon_boundary = self.safe_line_intersection(self.new_measured_area_boundary, cone_black_boundary) if self.new_measured_area_boundary is not None and cone_black_boundary is not None else None
        target_area_minus_black = self.safe_geometry_difference(self.cone_union_last_cone_convex_hull_intersection_target_area, self.black_polygons_dilated)
        self.new_measured_area_boundary_intersection_target_area_boundary = self.safe_geometry_intersection(new_measured_area_minus_black, target_area_minus_black) if new_measured_area_minus_black is not None and target_area_minus_black is not None else None

        # Calculate length metrics with null checks
        main_length = self.new_measured_area_boundary.length if self.new_measured_area_boundary is not None else 0.0
        black_length = self.new_measured_area_boundary_intersection_black_polygon_boundary.length if self.new_measured_area_boundary_intersection_black_polygon_boundary is not None else 0.0
        target_length = self.new_measured_area_boundary_intersection_target_area_boundary.length if self.new_measured_area_boundary_intersection_target_area_boundary is not None else 0.0

        if para.ablation_study_use_optv:
            self.length_new_measured_minus_length_old_measured = main_length - 2 * (black_length + target_length)
        else:
            self.length_new_measured_minus_length_old_measured = main_length - 2 * target_length
        self.target_area_size += self.new_measured_area_size
        self.percentage_of_target_area_left = (
            (self.target_area_size_start_of_episode - self.target_area_size) /
            self.target_area_size_start_of_episode
            if self.target_area_size_start_of_episode != 0.0 else 0.0
        )
        self.percentage_of_new_area_measured_ = (
            (self.target_area_size - self.target_area_last_step_size) /
            self.cone_union_last_cone_convex_hull_max_area
        )
        #print("base timestel 10 again pls. is 50 anbd buffer_between_agent_and_no_fly_zone_in_meters: 0.15, dilation_of_black_polygons_in_m: 0.021")
        #self.save_geometry_visualization( # adjust such that no dotted lines are shown. only solid lines. adjust such that resolution 5x
        #    self.new_measured_area_boundary # should be in background and a lot thicker
        #    , self.new_measured_area_boundary_intersection_black_polygon_boundary # should be be in the foreground and thin
        #    , self.new_measured_area_boundary_intersection_target_area_boundary # should be in the middle and medium thickness
        #    , self.target_area # filled polygon in background with color #00E1E6
        #)
        self.simplify_polygon_if_needed(self.current_episode_step, simplification_interval=30, polygon=self.target_area)
        #self.simplify_polygon_if_needed(self.current_episode_step, simplification_interval=30, polygon=self.target_area_remeasured_1_times, area_loss_threshold=0.80)
    def save_geometry_visualization(self, boundary_intersection_cone, new_area_boundary, third_geometry, filled_background_polygon=None):
        """
        Save a PNG visualization of key geometries using matplotlib.

        Plots:
        - FOV/background: self.initial_target_area_borders (Polygon/MultiPolygon)
        - Filled background polygon: filled_background_polygon (Polygon/MultiPolygon) - filled with #00E1E6
        - New area boundary: new_area_boundary (LineString/MultiLineString)
        - Boundary intersection with cone: boundary_intersection_cone (LineString/MultiLineString)
        - Third geometry: third_geometry (LineString/MultiLineString) - visualized in blue

        The output PNG is stored under misc/logs with a filename containing the
        current episode step (when available) and a timestamp for uniqueness.
        """

        # Helper functions to draw shapely geometries on matplotlib axes
        def _draw_polygon_or_multipolygon(ax, geom, facecolor="none", edgecolor="#888", linewidth=0.2, alpha=1.0, filled=False):
            if geom is None or getattr(geom, "is_empty", True):
                return
            try:
                if hasattr(geom, "geoms"):  # MultiPolygon or GeometryCollection
                    for g in geom.geoms:
                        _draw_polygon_or_multipolygon(ax, g, facecolor, edgecolor, linewidth, alpha, filled)
                elif getattr(geom, "geom_type", "") == "Polygon":
                    if filled:
                        # Draw filled polygon using fill
                        from matplotlib.patches import Polygon as MplPolygon
                        # Exterior
                        poly_patch = MplPolygon(list(geom.exterior.coords), 
                                               facecolor=facecolor, 
                                               edgecolor=edgecolor, 
                                               linewidth=linewidth, 
                                               alpha=alpha)
                        ax.add_patch(poly_patch)
                        # Draw holes (interiors) as white filled polygons
                        for interior in geom.interiors:
                            hole_patch = MplPolygon(list(interior.coords), 
                                                   facecolor="white", 
                                                   edgecolor=edgecolor, 
                                                   linewidth=max(0.5, 0.8*linewidth), 
                                                   alpha=alpha)
                            ax.add_patch(hole_patch)
                    else:
                        # Draw outline only
                        x, y = geom.exterior.xy
                        ax.plot(x, y, color=edgecolor, linewidth=linewidth, alpha=alpha)
                        # Draw holes (interiors) if any
                        for interior in geom.interiors:
                            xi, yi = interior.xy
                            ax.plot(xi, yi, color=edgecolor, linewidth=max(0.5, 0.8*linewidth), alpha=alpha, linestyle="-")
            except Exception as e:
                print(f"save_geometry_visualization: Error drawing polygon: {e}")

        def _draw_lines(ax, geom, color="#d62728", linewidth=0.5, alpha=1.0):
            if geom is None or getattr(geom, "is_empty", True):
                return
            try:
                gtype = getattr(geom, "geom_type", "")
                if gtype == "LineString" or gtype == "LinearRing":
                    x, y = geom.xy
                    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)
                elif gtype == "MultiLineString":
                    for g in geom.geoms:
                        _draw_lines(ax, g, color, linewidth, alpha)
                elif hasattr(geom, "geoms"):
                    for g in geom.geoms:
                        _draw_lines(ax, g, color, linewidth, alpha)
                elif hasattr(geom, "boundary"):
                    _draw_lines(ax, geom.boundary, color, linewidth, alpha)
            except Exception as e:
                print(f"save_geometry_visualization: Error drawing lines: {e}")

        # Prepare figure/axes (3x smaller resolution: 1250/3 ≈ 417)
        fig, ax = plt.subplots(figsize=(8, 8), dpi=417)
        ax.set_aspect("equal", adjustable="box")

        # Set bounds to 3m x 3m around current position (in simulation coordinates)
        current_pos = getattr(self, "position", None)
        if current_pos is not None and len(current_pos) >= 2:
            # 3m x 3m window centered on current position
            half_width = 10.5  # 1.5m on each side = 3m total width
            half_height = 10.5  # 1.5m on each side = 3m total height
            bounds = (
                current_pos[0] - half_width,  # minx
                current_pos[1] - half_height,  # miny
                current_pos[0] + half_width,  # maxx
                current_pos[1] + half_height  # maxy
            )
        else:
            # Fallback if position not available
            bounds = (0, 0, 3, 3)
        
        # Plot FOV (initial target area borders) - optional background
        fov_geom = getattr(self, "initial_target_area_borders", None)
        if fov_geom is not None and not fov_geom.is_empty:
            _draw_polygon_or_multipolygon(ax, fov_geom, edgecolor="#1f77b4", linewidth=0.1, alpha=0.9)

        # Plot filled background polygon (new_measured_area_with_black_polygons) - Layer 0 (bottom)
        if filled_background_polygon is not None and not filled_background_polygon.is_empty:
            _draw_polygon_or_multipolygon(ax, filled_background_polygon, 
                                         facecolor="#00E1E6", 
                                         edgecolor="#00E1E6", 
                                         linewidth=0.5, 
                                         alpha=0.6, 
                                         filled=True)

        # Plot the three line-based overlays
        # Layer 1 (background): boundary_intersection_cone (new_measured_area_boundary) - thickest
        _draw_lines(ax, boundary_intersection_cone, color="#2c2ca0", linewidth=5.5, alpha=0.999)  # green, thick
        # Layer 2 (middle): third_geometry (new_measured_area_boundary_intersection_target_area_boundary) - medium
        _draw_lines(ax, third_geometry, color="#ffd900", linewidth=1.5, alpha=1.0)  # blue, medium
        # Layer 3 (foreground): new_area_boundary (new_measured_area_boundary_intersection_black_polygon_boundary) - thin
        _draw_lines(ax, new_area_boundary, color="#ff0000", linewidth=2.5, alpha=1.0)  # red, thin

        # Optionally plot black polygons (obstacles) lightly for context if available
        black_polys = getattr(self, "black_polygons", None)
        if black_polys is not None and not black_polys.is_empty:
            _draw_polygon_or_multipolygon(ax, black_polys, edgecolor="#444444", linewidth=0.8, alpha=0.5)

        # Set axes limits to exactly 1m x 1m (no padding for precise dimensions)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

        # Titles and labels
        step = getattr(self, "current_episode_step", None)
        title = "Geometry Visualization"
        if step is not None:
            title += f" - step {step}"
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Prepare output path
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        step_part = f"step_{step}_" if step is not None else ""
        out_dir = os.path.join(os.path.dirname(__file__))
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"save_geometry_visualization: Failed to create output dir '{out_dir}': {e}")
            # Fallback to current directory
            out_dir = os.getcwd()

        filename = f"geometry_vis.png"
        out_path = os.path.join(out_dir, filename)
        try:
            fig.tight_layout()
            fig.savefig(out_path)
            print(f"Saved geometry visualization to: {out_path}")
        except Exception as e:
            print(f"save_geometry_visualization: Failed to save figure: {e}")
        finally:
            plt.close(fig)

    def get_exterior_and_interior_perimeter_only(self):
        """Get both exterior and interior perimeters (total perimeter including holes)"""
        if self.target_area.is_empty:
            return 0.0
            
        # Handle both Polygon and MultiPolygon cases
        if hasattr(self.target_area, 'geoms'):
            # MultiPolygon case
            polygons = self.target_area.geoms
        else:
            # Single Polygon case
            polygons = [self.target_area]
            
        return sum(
            abs(poly.exterior.length) + 
            sum(abs(interior.length) for interior in poly.interiors)
            for poly in polygons 
            if hasattr(poly, 'exterior')
        )
    
    def create_border(self, array, border_width):
        array = array.astype(bool)
        structure = np.ones((3, 3), dtype=bool)
        dilated_array = binary_dilation(array, structure=structure, iterations=border_width)
        border = dilated_array & ~array
        return border    

    def get_static_point_90m_above_surface(self):
        while True:
            # Select random x, y, z coordinates within the grid size
            x = 0
            y = 0
            z = self.surface_grid[x, y] + 90
            return np.array([x, y, z])
    
    def get_random_2d_point_in_one_of_the_4_corners(self):
        # Define the possible corner coordinates adjusted 5 grid steps towards the center
        grid_steps = 8
        corners = [
            (grid_steps, grid_steps),
            (grid_steps, self.surface_grid.shape[1] - grid_steps -1),
            (self.surface_grid.shape[0] - grid_steps - 1, grid_steps),
            (self.surface_grid.shape[0] - grid_steps - 1, self.surface_grid.shape[1] - grid_steps - 1 )
        ]
        
        # Randomly select one of the corners
        x, y = corners[np.random.randint(0, 4)]

        # Choose z to be the maximum-min height divided by 2
        return np.array([x, y, 348])
    
    def get_random_point_above_surface(self):
        while True:
            # Select random x, y, z coordinates within the grid size
            x = np.random.randint(0, self.gridsize[0])
            y = np.random.randint(0, self.gridsize[1])
            z = np.random.randint(0, self.gridsize[2])

            # Check if the z coordinate is above the surface
            if z > self.surface_grid[x, y] and z < self.gridsize[2]:
                return np.array([x, y, z])    

    def clip_position_to_initial_target_area(self, position):
        """
        Clips the given position to the initial target area polygon.
        If the position is outside the initial target area, returns the closest point on the boundary.
        
        Args:
            position: numpy array [x, y] (flat coordinates)
            
        Returns:
            numpy array [x, y] representing the clipped position
        """
        point = ShapelyPoint(position[0], position[1])
        
        # Check if point is within any polygon in the MultiPolygon
        if self.initial_target_area_borders.contains(point):
            return position
        
        # Find the closest point on the boundary
        min_distance = float('inf')
        closest_point = None
        
        for poly in self.initial_target_area_borders.geoms:
            # Get the closest point on this polygon's boundary
            boundary_point = poly.boundary.interpolate(poly.boundary.project(point))
            distance = point.distance(boundary_point)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = boundary_point
        
        if closest_point is not None:
            return np.array([closest_point.x, closest_point.y])
        else:
            # Fallback: return original position if no closest point found
            return position

    def is_measuring_cone_clear_of_black_polygons(self, position):
        """
        Check if the measuring circle from the given position has no intersection with black polygons.
        This uses the same geometry as the measurement system to ensure consistency.
        
        Args:
            position: numpy array [x, y] (flat coordinates)
            
        Returns:
            bool: True if the measuring circle is completely clear of black polygons, False otherwise
        """
        # Early exit if no black polygons exist
        if not hasattr(self, 'black_polygons') or self.black_polygons is None or self.black_polygons.is_empty:
            return True
            
        # Create circle polygon using pre-created base polygon and translation
        try:
            cone_polygon_epsilon = self._get_cone_polygon_at_position(position, use_epsilon=True)
            
            # Ensure the polygon is valid, if not try to make it valid
            if not cone_polygon_epsilon.is_valid:
                cone_polygon_epsilon = make_valid(cone_polygon_epsilon)
                if not cone_polygon_epsilon.is_valid:
                    # If we can't create a valid polygon, assume it's not clear
                    return False
                    
        except Exception as e:
            # If we can't create the circle polygon, assume it's not clear
            return False
        
        # Check for intersection with black polygons using fast geometry intersection
        intersection = cone_polygon_epsilon.intersection(self.black_polygons)

        return intersection is None or intersection.is_empty

    def is_swept_area_clear_of_black_polygons(self, start_position, end_position):
        """
        Check that the agent's full body (disk of base_cone_radius) stays clear of
        black polygons along the straight move from start to end. Endpoint disk
        checks alone let the body sweep across thin walls or obstacle corners.
        """
        if not hasattr(self, 'black_polygons') or self.black_polygons is None or self.black_polygons.is_empty:
            return True
        start = np.asarray(start_position, dtype=float).flatten()[:2]
        end = np.asarray(end_position, dtype=float).flatten()[:2]
        if not (np.isfinite(start).all() and np.isfinite(end).all()):
            return False
        if np.linalg.norm(end - start) < 1e-12:
            return self.is_measuring_cone_clear_of_black_polygons(end)
        try:
            capsule = LineString([tuple(start), tuple(end)]).buffer(self.base_cone_radius - 1e-9)
            if not capsule.is_valid:
                capsule = make_valid(capsule)
        except Exception:
            return False
        return not capsule.intersects(self.black_polygons)

    def _min_distance_from_segment_to_points(self, start, end, points):
        """Min distance from the segment [start, end] to a set of 2D points
        (pass start == end for a single-position check)."""
        start = np.asarray(start, dtype=float).flatten()[:2]
        end = np.asarray(end, dtype=float).flatten()[:2]
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return np.inf
        d = end - start
        seg_len_sq = float(d @ d)
        if seg_len_sq < 1e-24:
            diffs = pts - end
        else:
            t = np.clip((pts - start) @ d / seg_len_sq, 0.0, 1.0)
            diffs = pts - (start[None, :] + t[:, None] * d[None, :])
        return float(np.sqrt((diffs * diffs).sum(axis=1).min()))

    def find_closest_position_with_clear_cone(self, position):
        """
        Find the closest position where the measuring circle is clear of black polygons.
        
        Args:
            position: numpy array [x, y] (flat coordinates)
            
        Returns:
            numpy array [x, y] representing the closest valid position
        """
        # Early exit if no black polygons exist - current position is valid
        if not hasattr(self, 'black_polygons') or self.black_polygons is None or self.black_polygons.is_empty:
            return position
            
        # If current position is already valid, return it
        if self.is_measuring_cone_clear_of_black_polygons(position):
            return position
            
        # Search in expanding circles around the original position
        max_search_radius = 50  # Maximum search radius in pixels
        
        for radius in range(1, max_search_radius + 1):
            # Check positions in a circle around the original position
            for angle in np.linspace(0, 2*np.pi, max(8, radius*2), endpoint=False):
                # Extract scalar values to avoid array operations
                pos_x = position[0].item() if hasattr(position[0], 'item') else position[0]
                pos_y = position[1].item() if hasattr(position[1], 'item') else position[1]
                test_x = pos_x + radius * np.cos(angle)
                test_y = pos_y + radius * np.sin(angle)
                test_position = np.array([test_x, test_y])
                
                if self.is_measuring_cone_clear_of_black_polygons(test_position):
                    return test_position
        
        # If no valid position found, return original position as fallback
        return position

    def find_boundary_position_on_trajectory(self, start_position, target_position):
        """
        Find the position along the trajectory from start_position to target_position
        where the measuring cone is just clear of black polygons.
        
        Args:
            start_position: numpy array [x, y] (current position)
            target_position: numpy array [x, y] (desired position)
            
        Returns:
            numpy array [x, y] representing the furthest valid position along trajectory
        """
        # Early exit if no black polygons exist - all positions are valid
        if not hasattr(self, 'black_polygons') or self.black_polygons is None or self.black_polygons.is_empty:
            return target_position
            
        # If the start position is already invalid, return it (no movement)
        if not self.is_measuring_cone_clear_of_black_polygons(start_position):
            return start_position
            
        # If target position is valid, return it
        if self.is_measuring_cone_clear_of_black_polygons(target_position):
            return target_position
            
        # Binary search along the trajectory to find the boundary position
        num_samples = 50  # Number of points to check along trajectory
        valid_position = start_position.copy()
        
        # Use binary search for efficiency
        low = 0.0  # Parameter t where position = start_position + t * (target_position - start_position)
        high = 1.0
        
        for _ in range(20):  # Maximum 20 binary search iterations
            mid = (low + high) / 2.0
            test_position = start_position + mid * (target_position - start_position)
            
            if self.is_measuring_cone_clear_of_black_polygons(test_position):
                valid_position = test_position.copy()
                low = mid  # This position is valid, try going further
            else:
                high = mid  # This position is invalid, step back
                
        return valid_position
    
    def find_alternative_bearing_to_avoid_black_polygons(self, original_bearing, max_adjustment=180):
        """
        Find an alternative bearing based on the first intersection between the
        attempted movement line (last position -> intended new position) and the
        black polygon boundary. The tangent bearing at that intersection is tried
        first; the original incremental scan is used as a fallback.
        
        Args:
            original_bearing: The original bearing that causes intersection
            max_adjustment: Maximum degrees to adjust (default 180, full semicircle)
            
        Returns:
            tuple: (adjusted_bearing, success) where success indicates if a valid bearing was found
        """

        def _bearing_difference(b1: float, b2: float) -> float:
            return abs(((b1 - b2 + 180.0) % 360.0) - 180.0)

        def _attempt_boundary_bearing(obstacles_geometry):
            if obstacles_geometry is None or obstacles_geometry.is_empty:
                return None
            current_pos = np.array(self.position, dtype=float)
            last_pos = np.array(getattr(self, 'last_position', current_pos), dtype=float)

            if current_pos.shape[0] != 2:
                current_pos = current_pos.flatten()[:2]
            if last_pos.shape[0] != 2:
                last_pos = last_pos.flatten()[:2]

            bearing_rad = np.deg2rad(original_bearing)
            intended_dx = self.distance_to_move_per_step * np.sin(bearing_rad)
            intended_dy = self.distance_to_move_per_step * np.cos(bearing_rad)
            intended_position = np.array([current_pos[0] + intended_dx, current_pos[1] + intended_dy], dtype=float)

            if para.z_rad_type == 'jon':
                intended_position = self.clip_position_to_initial_target_area(intended_position)

            if np.allclose(last_pos, intended_position):
                return None

            try:
                movement_line = LineString([tuple(last_pos), tuple(intended_position)])
            except Exception:
                return None

            boundary = getattr(obstacles_geometry, 'boundary', None)
            if boundary is None or boundary.is_empty:
                return None

            try:
                intersection = movement_line.intersection(boundary)
            except Exception:
                return None

            intersection_points = []

            def _collect_points(geom):
                if geom is None or geom.is_empty:
                    return
                gtype = geom.geom_type
                if gtype == 'Point':
                    intersection_points.append(geom)
                elif gtype == 'MultiPoint':
                    intersection_points.extend(list(geom.geoms))
                elif gtype in ('LineString', 'LinearRing'):
                    coords = list(geom.coords)
                    if coords:
                        intersection_points.append(GeopyPoint(coords[0]))
                        intersection_points.append(GeopyPoint(coords[-1]))
                elif gtype == 'MultiLineString':
                    for sub_geom in geom.geoms:
                        _collect_points(sub_geom)
                elif gtype == 'GeometryCollection':
                    for sub_geom in geom.geoms:
                        _collect_points(sub_geom)

            _collect_points(intersection)

            if not intersection_points:
                return None

            unique_points = []
            seen = set()
            for pt in intersection_points:
                key = (round(pt.x, 6), round(pt.y, 6))
                if key not in seen:
                    seen.add(key)
                    unique_points.append(pt)

            if not unique_points:
                return None

            unique_points.sort(key=lambda p: movement_line.project(p))
            first_point = unique_points[0]

            def _find_polygon(geom):
                if geom is None or geom.is_empty:
                    return None
                gtype = geom.geom_type
                if gtype == 'Polygon':
                    if geom.boundary.distance(first_point) < 1e-9 or geom.contains(first_point):
                        return geom
                    return None
                if gtype == 'MultiPolygon':
                    for sub_geom in geom.geoms:
                        found = _find_polygon(sub_geom)
                        if found is not None:
                            return found
                    return None
                if gtype == 'GeometryCollection':
                    for sub_geom in geom.geoms:
                        found = _find_polygon(sub_geom)
                        if found is not None:
                            return found
                    return None
                return None

            polygon = _find_polygon(obstacles_geometry)
            if polygon is None:
                return None

            ring_sequences = [polygon.exterior.coords]
            ring_sequences.extend(interior.coords for interior in polygon.interiors)

            best_segment = None
            best_distance = float('inf')

            for coords in ring_sequences:
                coords_list = list(coords)
                if len(coords_list) < 2:
                    continue
                for idx in range(len(coords_list) - 1):
                    try:
                        segment = LineString([coords_list[idx], coords_list[idx + 1]])
                        dist = segment.distance(first_point)
                        length = segment.length
                    except Exception:
                        segment = None
                        dist = np.linalg.norm(np.array(coords_list[idx]) - np.array([first_point.x, first_point.y]))
                        length = np.linalg.norm(np.array(coords_list[idx + 1]) - np.array(coords_list[idx]))
                    if length == 0:
                        continue
                    if dist < best_distance:
                        best_distance = dist
                        best_segment = (np.array(coords_list[idx], dtype=float), np.array(coords_list[idx + 1], dtype=float))

            if best_segment is None:
                return None

            p0, p1 = best_segment
            seg_dx = p1[0] - p0[0]
            seg_dy = p1[1] - p0[1]
            if np.isclose(seg_dx, 0.0) and np.isclose(seg_dy, 0.0):
                return None

            base_bearing = (np.rad2deg(np.arctan2(seg_dx, seg_dy)) + 360.0) % 360.0
            candidate_bearings = sorted(
                [base_bearing, (base_bearing + 180.0) % 360.0],
                key=lambda b: _bearing_difference(b, original_bearing)
            )

            for candidate in candidate_bearings:
                cand_rad = np.deg2rad(candidate)
                cand_dx = self.distance_to_move_per_step * np.sin(cand_rad)
                cand_dy = self.distance_to_move_per_step * np.cos(cand_rad)
                test_position = np.array([current_pos[0] + cand_dx, current_pos[1] + cand_dy], dtype=float)
                if para.z_rad_type == 'jon':
                    test_position = self.clip_position_to_initial_target_area(test_position)
                if self.is_measuring_cone_clear_of_black_polygons(test_position):
                    return candidate

            return None

        obstacles = getattr(self, 'black_polygons_dilated', None)
        if obstacles is None or obstacles.is_empty:
            obstacles = getattr(self, 'black_polygons', None)

        boundary_bearing = _attempt_boundary_bearing(obstacles)
        if boundary_bearing is not None:
            return boundary_bearing, True

        # Fallback: incremental search over alternative bearings
        current_pos = np.array(self.position, dtype=float)
        for adjustment in np.arange(0.5, max_adjustment + 0.5, 0.5):
            for sign in [1, -1]:
                test_bearing = (original_bearing + sign * adjustment) % 360

                bearing_rad = np.deg2rad(test_bearing)
                dx = self.distance_to_move_per_step * np.sin(bearing_rad)
                dy = self.distance_to_move_per_step * np.cos(bearing_rad)
                test_position = np.array([current_pos[0] + dx, current_pos[1] + dy])

                if para.z_rad_type == 'jon':
                    test_position = self.clip_position_to_initial_target_area(test_position)

                if self.is_measuring_cone_clear_of_black_polygons(test_position):
                    print("fallback")
                    return test_bearing, True

        return original_bearing, False

    def update_position(self, new_position):
        self.last_position = self.position.copy()
        # Clip position to the map borders: beyond the outer wall there are no
        # black polygons, so the cone-clear check alone cannot keep the agent
        # inside the map.
        if para.z_rad_type == 'jon' and getattr(self, 'initial_target_area_borders', None) is not None:
            new_position = self.clip_position_to_initial_target_area(
                np.asarray(new_position, dtype=float).flatten()[:2])
        # Apply black polygon avoidance restriction if enabled
        if para.restrict_movement_to_white_pixels and para.z_rad_type == 'jon':
            collision_detected = not self.is_measuring_cone_clear_of_black_polygons(new_position)
            if not collision_detected:
                # Even with a clear destination disk, the agent's body can sweep
                # across an obstacle corner or thin wall while moving there.
                collision_detected = not self.is_swept_area_clear_of_black_polygons(
                    self.last_position, new_position)
            if collision_detected:
                if para.glide_on_collision:
                    # Increment collision counter when measuring cone intersects with black polygons
                    self.num_episode_collisions += 1
                    adjusted_position = None
                    try:
                        candidate_position = np.array(new_position, dtype=float).flatten()[:2]

                        buffer_meters = getattr(para, 'buffer_between_agent_and_no_fly_zone_in_meters', 0.0)
                        meters_per_pixel = getattr(self, 'meters_per_pixel_mower', None)
                        if meters_per_pixel is None or meters_per_pixel <= 0:
                            meters_per_pixel = 1.0
                        buffer_pixels = buffer_meters / meters_per_pixel

                        obstacle_point = None
                        if para.glide_use_reconstructed_lidar_borders:
                            # Glide along obstacle borders reconstructed from lidar
                            # points: only border segments already discovered by
                            # lidar can be glided along. Undiscovered walls stop
                            # the agent (fallback to last position below).
                            if (getattr(self, 'obstacle_segments_discovered', None) is not None
                                    and np.any(self.obstacle_segments_discovered)):
                                discovered_points = self.obstacle_segments[self.obstacle_segments_discovered]
                                deltas = discovered_points - candidate_position
                                nearest_idx = int(np.argmin((deltas * deltas).sum(axis=1)))
                                nearest_point = discovered_points[nearest_idx].astype(float)
                                segment_length_pixels = self.segment_length_meters / meters_per_pixel
                                # Only glide if the nearest discovered border point is
                                # plausibly the wall being hit; a far-away point would
                                # teleport the agent to a different wall.
                                max_glide_distance = self.base_cone_radius + buffer_pixels + 2.0 * segment_length_pixels
                                if np.linalg.norm(candidate_position - nearest_point) <= max_glide_distance:
                                    obstacle_point = nearest_point
                        else:
                            obstacles = self.black_polygons
                            if obstacles is not None and not obstacles.is_empty:
                                new_point = ShapelyPoint(candidate_position[0], candidate_position[1])
                                closest_on_obstacle, _ = nearest_points(obstacles, new_point)
                                obstacle_point = np.array([closest_on_obstacle.x, closest_on_obstacle.y], dtype=float)

                        if obstacle_point is not None:
                            direction = candidate_position - obstacle_point
                            norm = np.linalg.norm(direction)

                            if norm > 0:
                                unit_direction = direction / norm
                                last_pos = np.asarray(self.last_position, dtype=float).flatten()[:2]
                                if para.glide_use_reconstructed_lidar_borders:
                                    # Lidar mode: the landing DIRECTION and the obstacle
                                    # point come from ONLY the perceived (lidar-discovered)
                                    # border points — ground truth never guides *where* to
                                    # aim. We aim flush against the perceived wall (no
                                    # discretization slack). Ground truth enters solely as
                                    # the collision oracle: a candidate that would make the
                                    # covered area overlap an obstacle polygon is illegal
                                    # (physics, not perception), so we step the landing
                                    # outward by border-sample increments until the covered
                                    # area is clear. This guarantees the covered area never
                                    # overlaps the obstacle polygons while keeping the
                                    # landing as flush as the geometry physically allows.
                                    segment_length_pixels = self.segment_length_meters / meters_per_pixel
                                    required_clearance = max(buffer_pixels, self.base_cone_radius)
                                    committed = None
                                    for extra_margin in [0.0, segment_length_pixels,
                                                         2.0 * segment_length_pixels,
                                                         4.0 * segment_length_pixels]:
                                        candidate_adjusted = obstacle_point + unit_direction * (required_clearance + extra_margin)
                                        # landing disk clear of all perceived border points
                                        if self._min_distance_from_segment_to_points(
                                                candidate_adjusted, candidate_adjusted,
                                                discovered_points) < required_clearance - 1e-6:
                                            continue
                                        # swept body clear of all perceived border points
                                        if self._min_distance_from_segment_to_points(
                                                last_pos, candidate_adjusted,
                                                discovered_points) < self.base_cone_radius - 1e-6:
                                            continue
                                        # covered area + swept body must not overlap the
                                        # true obstacle polygons (hard collision constraint)
                                        if not (self.is_measuring_cone_clear_of_black_polygons(candidate_adjusted)
                                                and self.is_swept_area_clear_of_black_polygons(last_pos, candidate_adjusted)):
                                            continue
                                        committed = candidate_adjusted
                                        break
                                    if committed is not None:
                                        adjusted_position = committed
                                else:
                                    # Ground-truth mode (upper bound / debugging): the
                                    # true obstacle geometry may guide the landing.
                                    candidate_adjusted = obstacle_point + unit_direction * max(buffer_pixels, 0.0)
                                    if self.is_measuring_cone_clear_of_black_polygons(candidate_adjusted) \
                                            and self.is_swept_area_clear_of_black_polygons(last_pos, candidate_adjusted):
                                        adjusted_position = candidate_adjusted

                        if adjusted_position is None:
                            adjusted_position = np.array(self.last_position, dtype=float)

                        new_position = adjusted_position
                    except Exception as e:
                        print(f"Warning: Failed to adjust position away from black polygons: {e}")
                        new_position = self.last_position
                else:
                    new_position = self.last_position

        # FIX: Ensure position is always 1D array with shape (2,)
        if isinstance(new_position, np.ndarray) and new_position.shape != (2,):
            new_position = new_position.flatten()[:2]
        
        # Stuck detection logic - check if position has changed significantly
        position_changed = True
        if hasattr(self, 'position'):
            # Calculate distance moved (in meters)
            distance_moved_pixels = np.linalg.norm(np.array(new_position) - np.array(self.position))
            distance_moved_meters = distance_moved_pixels * self.meters_per_pixel_mower
            # Consider agent stuck if it moved less than 1cm (0.01 meters)
            if distance_moved_meters < 0.01:
                position_changed = False
        
        # Update stuck_steps counter
        if not position_changed:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
            
        # Flip the agent if stuck for too many steps
        if self.flip_when_stuck and self.stuck_steps >= self.max_stuck_steps:
            self.stuck_steps = 0
            # Flip the bearing by 180 degrees
            self.current_bearing = (self.current_bearing + 180) % 360
            
        self.position = new_position
        
        # Track position history for rendering
        if not hasattr(self, 'position_history'):
            self.position_history = []
        self.position_history.append(self.position.copy())
        
        # Calculate and accumulate path length for jon radiation type
        if para.z_rad_type == 'jon' and self.current_episode_step != 0:
            # Calculate distance in pixels
            distance_in_pixels = np.linalg.norm(np.array(self.position) - np.array(self.last_position))
            distance_in_meters = distance_in_pixels * self.meters_per_pixel_mower
            # Convert to meters using meters_per_pixel_mower
            # Accumulate to total path length
            self.length_of_path_in_meters += distance_in_meters
        # Use correct granularity based on radiation type
        if para.z_rad_type == 'jon':
            self.position_as_xy = self.position
        else:
            self.position_as_xy = compute_position_from_geo_coordinate(self.position, self.granularity)
        if para.surface_grid_creation_type == 'no_height_map':
            self.position_z_above_ground = 90
        else:
            distances, indices = self.tree_surface_point_cloud_.query(np.array([self.position]))
            self.height_of_surface_grid_at_position = self.surface_point_cloud_[indices[0]][2]
            print("Height change not implemented. define self.position_z_above_ground")
    
    def rotate_vector_around_z_axis_and_set_height_angle(self, vector, horizontal_angle, vertical_change_in_meters_per_timestep):
        # Convert degrees to radians
        horizontal_angle_rad = np.radians(horizontal_angle)

        vector[2] = vertical_change_in_meters_per_timestep
        
        # Rotation matrix around the z-axis
        Rz = np.array([
            [np.cos(horizontal_angle_rad), -np.sin(horizontal_angle_rad), 0],
            [np.sin(horizontal_angle_rad), np.cos(horizontal_angle_rad), 0],
            [0, 0, 1]
        ])
        
        # Apply the z-axis rotation
        rotated_vector = Rz @ vector
        return rotated_vector

    def find_safe_starting_point(self):
        """Find a random point on the target area boundary with radiation exposure below threshold using calculate_odl_via_precalculated_grids."""
        attempts = 0
        while attempts < 1000:
            if para.z_rad_type == 'mov' or para.z_rad_type == 'bfs':
                if para.z_rad_type == 'mov':
                    self.radiation_grid_episode_parameters = self.generate_random_radar_params()
                    self.geo_coordinates_of_radiation_grid, self.measuring_area_scenario = create_single_radiation_scenario(
                        self.radiation_grid_base, para.granularity
                    )
                    if self.radiation_grid_visualization:
                        self.radiation_grid = self.get_time_dependent_radiation_grid(self.time_rel, self.radiation_grid_episode_parameters)
                elif para.z_rad_type == 'bfs':
                    self.radiation_grid, self.geo_coordinates_of_radiation_grid, self.measuring_area_scenario = create_single_radiation_scenario(
                        self.radiation_grid_base, para.granularity
                    )
                    self.tree_radiation_grid = cKDTree(self.geo_coordinates_of_radiation_grid.reshape(-1, 2))
                self.polygon_center = self.measuring_area_scenario['polygon_center']
                self.target_area = MultiPolygon()
        
                self.target_area_borders = self.measuring_area_scenario['target_area']
                if isinstance(self.target_area_borders, Polygon):
                    self.target_area_borders = MultiPolygon([self.target_area_borders])
                self.initial_target_area_borders = copy.deepcopy(self.target_area_borders)
                area = self.target_area_borders.area
                self.target_area_size_borders = abs(area)
                                
                random_point_on_boundary = self.target_area.boundary.interpolate(random.uniform(0, self.target_area.boundary.length))
                lon, lat = random_point_on_boundary.x, random_point_on_boundary.y
            elif para.z_rad_type == 'jon':
                # Take a random white point from the image get its coordinates, calc the corresponding geo coordinates and save in lon, lat
                img = self._get_cached_image(self.IMAGE_PATH)
                
                # Store the image for movement restriction checking
                self.current_image_for_movement_restriction = img.copy()
                
                # Extract black polygons from the image
                self.black_polygons, self.black_polygons_boundary, self.black_polygons_dilated, self.black_polygons_dilated_boundary, self.black_polygons_dilated_for_coverage_calc = self.extract_black_polygons_from_png(img)
                
                # Preprocess the image with morphological operations and blur
                self.precalculated_odl_map = self.preprocess_image_for_odl(img)
                
                # Calculate and save percentage of white pixels (using original image)
                total_pixels = img.shape[0] * img.shape[1]
                white_pixels = np.sum(img == 255)


                white_points = np.where(img == 255)
                # Select a random white point
                random_idx = np.random.randint(0, len(white_points[0]))
                y, x = white_points[0][random_idx], white_points[1][random_idx]
                
                # Create flat coordinate scenario directly from image dimensions  
                img_height, img_width = img.shape
                self.measuring_area_scenario = {
                    'polygon_center': [img_width / 2, img_height / 2],
                    'target_area': Polygon([
                        (0, 0),
                        (img_width-1, 0),
                        (img_width-1, img_height-1),
                        (0, img_height-1),
                        (0, 0)
                    ])
                }
                
                # Use flat image coordinates directly
                lon, lat = x, y
                self.polygon_center = self.measuring_area_scenario['polygon_center']
                self.target_area = MultiPolygon()
                self.target_area_borders = self.measuring_area_scenario['target_area']
                if isinstance(self.target_area_borders, Polygon):
                    self.target_area_borders = MultiPolygon([self.target_area_borders])
                self.initial_target_area_borders = copy.deepcopy(self.target_area_borders)
                area = self.target_area_borders.area
                self.target_area_size_borders = abs(area)
                
                # Calculate available area by subtracting black polygons from target area borders
                if hasattr(self, 'black_polygons_dilated_for_coverage_calc') and self.black_polygons_dilated_for_coverage_calc is not None:
                    target_area_minus_black_polygons = self.safe_geometry_difference(self.target_area_borders, self.black_polygons_dilated_for_coverage_calc)
                    self.target_area_size_start_of_episode = abs(target_area_minus_black_polygons.area)


            candidate = np.array([lon, lat], dtype=float)
            if (para.restrict_movement_to_white_pixels and para.z_rad_type == 'jon'
                    and not self.is_measuring_cone_clear_of_black_polygons(candidate)):
                attempts += 1
                continue
            # Initial placement, not movement: pre-set the position so the
            # movement physics (swept-area collision / glide) don't reject
            # the cross-map teleport to the candidate start point.
            self.position = candidate.copy()
            self.update_position(candidate)

            # Always use the latest parameters from the environment
            radiation_exposure = self.calculate_odl()
            
            if radiation_exposure < 5:
                return ShapelyPoint(lon, lat)

                
            attempts += 1

        print("Warning: Could not find a safe starting point after 1000 attempts. returning polygon center.")
        return ShapelyPoint(self.polygon_center[0], self.polygon_center[1])

    def generate_random_map(
        self,
        meters_per_pixel=0.0375,
        mower_radius=0.15,
        obstacle_radius=0.25,
        min_size_p=256,
        max_size_p=400,
        p_use_floor_plans=0.7,
        p_use_known_obstacles=0.7,
        p_use_unknown_obstacles=0.7,
        max_known_obstacles=100,
        max_unknown_obstacles=100,
        line_type=cv2.LINE_8,
        rng_seed=None,
    ):
        """
        Generate a random map with obstacles and floor plans.
        Uses size-based probabilities from training images where smaller sizes are more likely.
        """
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        # Get size from a default training image
        default_image_path = os.path.join(os.path.dirname(__file__), "misc", "radiation_data", 
                                        "bw_jon_images_from_paper", "train_0_1.png")
        
        # Load image and read its size directly
        if para.use_size_based_probability:
            # Use consistent ordering between choices and probabilities
            image_sizes_list = [self.train_images_sizes[path] for path in self.train_image_paths_list]
            # Force exact normalization to avoid floating point precision issues
            probs_array = np.array(self.train_image_probs_list)
            probs_array = probs_array / probs_array.sum()
            
            target_size = int(np.sqrt(np.random.choice(
                image_sizes_list, p=probs_array)
            ))
        else:
            target_size = int(np.sqrt(self.train_images_sizes[default_image_path]))  # Convert from total pixels to side length

        # Use the target size for both min and max
        min_size_p = max_size_p = target_size

        pixels_per_meter = 1.0 / meters_per_pixel
        size_p = target_size
        size_m = size_p / pixels_per_meter

        known_obstacle_map   = np.zeros((size_p, size_p), dtype=float)
        unknown_obstacle_map = np.zeros((size_p, size_p), dtype=float)

        def _randomize_floor_plan():
            min_room_size_p = int(10 * mower_radius * pixels_per_meter)
            max_room_size_p = int(32 * mower_radius * pixels_per_meter)
            min_wall_thickness_p = 2
            max_wall_thickness_p = int(2 * mower_radius * pixels_per_meter)
            min_gap = int(4 * mower_radius * pixels_per_meter)
            max_gap = int(8 * mower_radius * pixels_per_meter)
            if size_p > 2 * min_room_size_p:
                room_size_p = random.randint(min_room_size_p, max_room_size_p)
                num_walls = max(1, int(size_p / room_size_p) - 1)
                room_size_p = int(size_p / (num_walls + 1))
                wall_thickness_p = random.randint(min_wall_thickness_p, max_wall_thickness_p)
                vertical_stop = random.uniform(0, 1) < 0.5
                for n in range(num_walls):
                    i1 = room_size_p * (n + 1) - wall_thickness_p // 2
                    i2 = room_size_p * (n + 1) + wall_thickness_p
                    if random.uniform(0, 1) < 0.9:
                        known_obstacle_map[i1:i2, :] = 1
                    if random.uniform(0, 1) < 0.9:
                        known_obstacle_map[:, i1:i2] = 1
                    stop_placed = False
                    for m in range(num_walls + 1):
                        g_min = min_gap
                        g_max = min(max_gap, room_size_p - 2 * wall_thickness_p)
                        j_min = room_size_p * m + wall_thickness_p
                        j_max = room_size_p * (m + 1) - wall_thickness_p
                        p_stop = 1 / (num_walls + 1 - m)
                        place_stop = False
                        if not stop_placed and random.uniform(0, 1) < p_stop:
                            place_stop = True
                            stop_placed = True
                        if (not vertical_stop) or (not place_stop):
                            gap = random.randint(g_min, g_max)
                            j1 = random.randint(j_min, j_max - gap)
                            j2 = j1 + gap
                            known_obstacle_map[i1:i2, j1:j2] = 0
                        if vertical_stop or (not place_stop):
                            gap = random.randint(g_min, g_max)
                            j1 = random.randint(j_min, j_max - gap)
                            j2 = j1 + gap
                            known_obstacle_map[j1:j2, i1:i2] = 0

        def _get_local_neighborhood_indices(pos1_m, pos2_m, radius_m, ppm, size_p_):
            i1 = min(pos1_m[0], pos2_m[0]) - radius_m
            i2 = max(pos1_m[0], pos2_m[0]) + radius_m
            j1 = min(pos1_m[1], pos2_m[1]) - radius_m
            j2 = max(pos1_m[1], pos2_m[1]) + radius_m
            i1 = max(0, min(size_p_, int(i1 * ppm - 10)))
            i2 = max(0, min(size_p_, int(i2 * ppm + 10)))
            j1 = max(0, min(size_p_, int(j1 * ppm - 10)))
            j2 = max(0, min(size_p_, int(j2 * ppm + 10)))
            return i1, i2, j1, j2

        def _randomize_circular_obstacles(use_known, use_unknown):
            known_pos   = np.random.uniform(size=(max_known_obstacles, 2))
            unknown_pos = np.random.uniform(size=(max_unknown_obstacles, 2))
            radius = 2 * mower_radius + obstacle_radius
            for n in range(max(max_known_obstacles, max_unknown_obstacles)):
                if use_known and n < max_known_obstacles:
                    pos_m = 2 * radius + known_pos[n] * (size_m - 4 * radius)
                    i1, i2, j1, j2 = _get_local_neighborhood_indices(pos_m, pos_m, radius, pixels_per_meter, size_p)
                    if known_obstacle_map[i1:i2, j1:j2].sum() == 0 and unknown_obstacle_map[i1:i2, j1:j2].sum() == 0:
                        cv2.circle(
                            known_obstacle_map,
                            center=(np.flip(pos_m * pixels_per_meter)).astype(np.int32),
                            radius=int(obstacle_radius * pixels_per_meter),
                            color=1,
                            thickness=cv2.FILLED,
                            lineType=line_type
                        )
                if use_unknown and n < max_unknown_obstacles:
                    pos_m = 2 * radius + unknown_pos[n] * (size_m - 4 * radius)
                    i1, i2, j1, j2 = _get_local_neighborhood_indices(pos_m, pos_m, radius, pixels_per_meter, size_p)
                    if known_obstacle_map[i1:i2, j1:j2].sum() == 0 and unknown_obstacle_map[i1:i2, j1:j2].sum() == 0:
                        cv2.circle(
                            unknown_obstacle_map,
                            center=(np.flip(pos_m * pixels_per_meter)).astype(np.int32),
                            radius=int(obstacle_radius * pixels_per_meter),
                            color=1,
                            thickness=cv2.FILLED,
                            lineType=line_type
                        )

        use_floor_plans   = random.uniform(0, 1) < p_use_floor_plans
        use_known_obs     = (max_known_obstacles   > 0) and (random.uniform(0, 1) < p_use_known_obstacles)
        use_unknown_obs   = (max_unknown_obstacles > 0) and (random.uniform(0, 1) < p_use_unknown_obstacles)
        
        # Track what was used for curriculum learning
        self._random_map_used_floor_plan = use_floor_plans
        self._random_map_used_obstacles = use_known_obs or use_unknown_obs

        if use_floor_plans:
            _randomize_floor_plan()
        if use_known_obs or use_unknown_obs:
            _randomize_circular_obstacles(use_known_obs, use_unknown_obs)

        # Create the final image (white background with black obstacles)
        img = 255 * np.ones_like(known_obstacle_map, dtype=np.uint8)
        img[known_obstacle_map > 0] = 0
        img[unknown_obstacle_map > 0] = 0
        
        # Apply black border by expanding the image (same as in other image loading methods)
        img = self._add_black_border_to_image(img, border_size=5)
        
        # Save directly as environment attribute instead of PNG file
        self.random_map_image = img.copy()

        return known_obstacle_map, unknown_obstacle_map, size_p, size_m


if para.training_library == 'omnisafe':
    #from omnisafe.envs.core import registry
    from omnisafe.envs.core import CMDP, env_register
    import torch
    height_data_path = os.path.join('.', 'misc', 'geo_data','height_data','height_data.npy')
    height_data = np.load(height_data_path)

    @env_register
    class GymEnvOmniSafe(CMDP):
        need_auto_reset_wrapper: bool = False
        need_time_limit_wrapper: bool = False
        """OmniSafe-compatible wrapper for GymnasiumEnv."""
        _support_envs = ["GymEnvOmniSafe-v0"]

        def __init__(self, radiation_grid_visualization = False, env_id="GymEnvOmniSafe-v0", **kwargs):

            # Pull seed from kwargs (if provided) so it doesn't get passed to the parent
            seed = kwargs.pop('seed', None)

            super().__init__(env_id=env_id, **kwargs)
            self._num_envs = kwargs.get('num_envs', 1)

            # Determine if this is an evaluation environment
            is_evaluation = kwargs.get('is_evaluation', False)

            # If seed provided, apply to global RNGs BEFORE creating the base env so
            # initialization (which uses numpy/random) becomes deterministic.
            if seed is not None:
                try:
                    import random as _random
                    _random.seed(seed)
                except Exception:
                    pass
                try:
                    import numpy as _np
                    _np.random.seed(seed)
                except Exception:
                    pass
                try:
                    import torch as _torch
                    _torch.manual_seed(seed)
                    if _torch.cuda.is_available():
                        _torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass

            # Create the base environment
            base_env = GymnasiumEnv(height_data, radiation_grid_visualization, 
                                   is_evaluation=is_evaluation)

            self._env = AutoResetWrapper(base_env)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
        
            #IMPORTANT: Assertion to ensure the wrapped environment uses uint8 observations
            #GymEnvOmniSafe has been optimized to convert observations to uint8 tensors.
            #If the base environment observation dtype changes, the reset() and step() methods
            #need to be updated to handle the new dtype correctly.
            # Check if we're using fused lidar+image observations
            self._use_fused_obs = getattr(para, 'ablation_study_fuse_lidar_sensor_data_and_image_data', False) and not para.ablation_study_use_radiation_instead_of_lidar
            
            # Only check dtype for Box observation spaces, not Dict
            if not self._use_fused_obs and hasattr(self._observation_space, 'dtype'):
                assert self._observation_space.dtype == np.uint8, (
                    "CRITICAL: Base environment observation dtype has changed from uint8! "
                    "GymEnvOmniSafe.reset() and GymEnvOmniSafe.step() methods have been "
                    "optimized for uint8 observations. If you changed this intentionally, "
                    "please update the tensor conversion logic in these methods. "
                    f"Current dtype: {self._observation_space.dtype}"
                )
            
            self.cost = 0.0
            # Track if this is the first step (for observation space validation)
            # If a seed was passed to constructor, apply it to the environment and RNGs
            if seed is not None:
                try:
                    self.set_seed(seed)
                except Exception:
                    # best-effort seeding; don't raise here
                    pass
        @property
        def observation_space(self):
            return self._observation_space

        @property
        def action_space(self):
            return self._action_space

        def reset(self, *, seed=None, options=None, image_path=None):
            # Pass image_path through options to work with AutoResetWrapper
            if image_path is not None:
                if options is None:
                    options = {}
                options['image_path'] = image_path
            obs, info = self._env.reset(seed=seed, options=options)
            self.cost = 0.0
            # Reset first step flag
            
            if self._use_fused_obs:
                # Return Dict observation with both image and lidar
                obs_image = np.asarray(obs["image"]).astype(np.uint8)
                obs_lidar = np.asarray(obs["lidar"]).astype(np.float32)
                
                obs_tensor = {
                    'image': torch.from_numpy(obs_image),
                    'lidar': torch.from_numpy(obs_lidar)
                }
                return obs_tensor, {}
            else:
                # ALWAYS return Box observation format (uint8 image)
                # Lidar is computed internally but never included in observation
                # Ensure observation is uint8 for optimal RL with images
                obs_array = np.asarray(obs["image"]).astype(np.uint8)
                
                # IMPORTANT: Assertion to verify uint8 tensor creation
                assert obs_array.dtype == np.uint8, (
                    "CRITICAL: Failed to convert observation to uint8! GymEnvOmniSafe.reset() "
                    "has been optimized for uint8 observations. If the base environment returns "
                    "different dtypes, this conversion logic needs to be updated. "
                    f"Current dtype: {obs_array.dtype}"
                )
                
                obs_tensor = torch.from_numpy(obs_array)
                
                # Verify tensor dtype
                assert obs_tensor.dtype == torch.uint8, (
                    "CRITICAL: PyTorch tensor dtype mismatch! Expected torch.uint8 but got "
                    f"{obs_tensor.dtype}. This will cause buffer dtype mismatches."
                )
                
                return obs_tensor, {}

        def step(self, action):
            # Convert action to numpy if it's a torch tensor
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            obs, self.reward, terminated, truncated, info = self._env.step(action)
            if para.loop_step_until_radiation_above_threshold:
                self.cost = self._env.unwrapped.cost_sum
            else:
                self.cost = 0.0 if self._env.unwrapped.current_radiation_exposure < para.radiation_value_to_avoid else 1
            
            if self._use_fused_obs:
                # Return Dict observation with both image and lidar
                obs_image = np.asarray(obs["image"]).astype(np.uint8)
                obs_lidar = np.asarray(obs["lidar"]).astype(np.float32)
                
                obs_tensor = {
                    'image': torch.from_numpy(obs_image),
                    'lidar': torch.from_numpy(obs_lidar)
                }
            else:
                # ALWAYS return Box observation format (uint8 image)
                # Lidar is computed internally but never included in observation
                # Ensure observation is uint8 for optimal RL with images
                obs_array = np.asarray(obs["image"]).astype(np.uint8)
                
                # IMPORTANT: Assertion to verify uint8 tensor creation
                assert obs_array.dtype == np.uint8, (
                    "CRITICAL: Failed to convert observation to uint8! GymEnvOmniSafe.step() "
                    "has been optimized for uint8 observations. If the base environment returns "
                    "different dtypes, this conversion logic needs to be updated. "
                    f"Current dtype: {obs_array.dtype}"
                )
                
                obs_tensor = torch.from_numpy(obs_array)
                
                # Verify tensor dtype
                assert obs_tensor.dtype == torch.uint8, (
                    "CRITICAL: PyTorch tensor dtype mismatch! Expected torch.uint8 but got "
                    f"{obs_tensor.dtype}. This will cause buffer dtype mismatches."
                )
            
            self.reward = torch.tensor(self.reward, dtype=torch.float32)
            self.cost = torch.tensor(float(self.cost), dtype=torch.float32)
            
            if para.reset_after_10_steps:
                if self._env.unwrapped.current_episode_step >= 10:
                    self._env.unwrapped.current_episode_step = 0
                    terminated = True
            num_envs = getattr(self, '_num_envs', 1)
            if isinstance(terminated, (list, np.ndarray)):
                terminated = torch.tensor(terminated, dtype=torch.bool)
            else:
                terminated = torch.full((num_envs,), terminated, dtype=torch.bool)
            if isinstance(truncated, (list, np.ndarray)):
                truncated = torch.tensor(truncated, dtype=torch.bool)
            else:
                truncated = torch.full((num_envs,), truncated, dtype=torch.bool)
            return obs_tensor, self.reward, self.cost, terminated, truncated, {'final_observation': obs_tensor}
        def spec_log(self, logger):
            """Log environment-specific metrics to omnisafe logger.
            
            This method is called by omnisafe adapters to retrieve custom metrics
            from the environment. The metrics are logged under the 'Metrics/' prefix.
            """
            # Access the unwrapped environment to get custom metrics
            unwrapped_env = self._env.unwrapped
            
            # Log goal coverage percentage
            if hasattr(unwrapped_env, 'goal_coverage_percentage_currently'):
                logger.store({
                    'Metrics/GoalCoverage': unwrapped_env.goal_coverage_percentage_currently
                })
            
            # Log average step return using preserved episode statistics
            # This handles AutoResetWrapper timing issues where current stats might be 0
            if (hasattr(unwrapped_env, 'last_episode_steps') and 
                hasattr(unwrapped_env, 'last_episode_total_reward') and
                unwrapped_env.last_episode_steps > 0):
                avg_step_ret = unwrapped_env.last_episode_total_reward / unwrapped_env.last_episode_steps
                logger.store({
                    'Metrics/AvgStepRet': avg_step_ret
                })
            
            # Log episode collisions count using preserved statistics
            # Use last episode collision count to avoid AutoResetWrapper timing issues
            if hasattr(unwrapped_env, 'last_episode_collisions'):
                logger.store({
                    'Metrics/EpCollisions': unwrapped_env.last_episode_collisions
                })

        def close(self):
            self._env.close()
        def render(self, mode="human"):
            self._env.render(mode=mode)
        def set_seed(self, seed=None):
            """
            Set seed for environment and common RNGs (python random, numpy, torch).
            Attempts to set the seed on the wrapped/unwrapped base environment as well.
            """
            import random as _random
            _random.seed(seed)
            try:
                import numpy as _np
                _np.random.seed(seed)
            except Exception:
                pass
            try:
                import torch as _torch
                _torch.manual_seed(seed)
                if _torch.cuda.is_available():
                    _torch.cuda.manual_seed_all(seed)
            except Exception:
                pass

            # Try to set seed on the wrapped environment (AutoResetWrapper) unwrapped
            try:
                if hasattr(self._env, 'unwrapped') and hasattr(self._env.unwrapped, 'set_seed'):
                    self._env.unwrapped.set_seed(seed)
                elif hasattr(self._env, 'set_seed'):
                    # some wrappers may forward set_seed
                    self._env.set_seed(seed)
            except Exception:
                # best-effort: do not fail if underlying env does not support set_seed
                pass