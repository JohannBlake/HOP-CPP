#!/usr/bin/env python3
"""
Lidar Rays Visualization Script

This script visualizes the lidar rays on a map after initializing the gymnasium environment.
It takes random steps through the environment and displays the lidar detection rays,
showing how the sensor perceives obstacles in the environment.

The visualization includes:
- The map background
- The helicopter's current position
- Lidar rays color-coded by detection type:
  * Green: Max range (no detection)
  * Red: Known obstacle detected
  * Orange: Unknown obstacle detected
  * Purple: Out of bounds detection
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from get_parameters import para
import class_gymenv

def compute_lidar_pts_flat(gymenv, position, heading_deg):
    """
    Compute lidar point cloud for flat coordinate system.
    This is a public wrapper that replicates the lidar computation logic.
    
    Parameters:
    -----------
    gymenv : GymnasiumEnv
        The gymnasium environment
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
    heading_deg = (heading_deg) % 360  # Normalize heading to [0, 360)
    
    # Convert lidar range from meters to pixels
    lidar_range_pixels = gymenv.lidar_range / gymenv.meters_per_pixel_mower
    samples = int(lidar_range_pixels)  # Number of samples per ray
    
    # Get map dimensions
    map_height, map_width = gymenv.known_obstacle_map.shape
    
    # Pre-compute all ray angles (vectorized)
    angle_offsets = np.linspace(-gymenv.lidar_fov/2, gymenv.lidar_fov/2, num=gymenv.lidar_rays)
    nav_angles_deg = heading_deg + angle_offsets + 180
    math_angles_deg = 90 - nav_angles_deg
    ang_rads = np.radians(math_angles_deg)
    
    # Compute all search vectors at once (shape: [lidar_rays, 2])
    search_vecs = np.stack([np.cos(ang_rads), -np.sin(ang_rads)], axis=1)
    
    # Initialize outputs
    lidar_pts = np.zeros((gymenv.lidar_rays, 2), dtype=np.int32)
    pts_info = np.zeros(gymenv.lidar_rays, dtype=np.int32)
    
    # Process each ray
    for n in range(gymenv.lidar_rays):
        search_vec = search_vecs[n]
        
        # Generate all sample points along this ray at once
        sample_indices = np.arange(1, samples + 1)
        offsets = sample_indices[:, np.newaxis] * search_vec  # shape: [samples, 2]
        positions = position + offsets  # shape: [samples, 2]
        
        # Convert to integer indices
        j_coords = positions[:, 0].astype(np.int32)  # x coordinates
        i_coords = positions[:, 1].astype(np.int32)  # y coordinates
        
        # Check bounds for all samples at once
        valid_mask = (i_coords >= 0) & (i_coords < map_height) & (j_coords >= 0) & (j_coords < map_width)
        
        # Find first out-of-bounds sample
        if not valid_mask.all():
            first_invalid = np.argmax(~valid_mask)
            if first_invalid == 0 or not valid_mask[0]:
                # First sample is already out of bounds
                pts_info[n] = 3
                lidar_pts[n] = [int(position[0] + search_vec[0]), int(position[1] + search_vec[1])]
                continue
            # Trim to valid samples only
            j_coords = j_coords[:first_invalid]
            i_coords = i_coords[:first_invalid]
        
        # Check for known obstacles (vectorized lookup)
        if len(i_coords) > 0:
            known_obstacles = gymenv.known_obstacle_map[i_coords, j_coords]
            known_hit_idx = np.argmax(known_obstacles > 0)
            
            if known_obstacles[known_hit_idx] > 0:
                pts_info[n] = 1
                lidar_pts[n] = [j_coords[known_hit_idx], i_coords[known_hit_idx]]
                continue
            
            # Check for unknown obstacles (vectorized lookup)
            unknown_obstacles = gymenv.unknown_obstacle_map[i_coords, j_coords]
            unknown_hit_idx = np.argmax(unknown_obstacles > 0)
            
            if unknown_obstacles[unknown_hit_idx] > 0:
                pts_info[n] = 2
                lidar_pts[n] = [j_coords[unknown_hit_idx], i_coords[unknown_hit_idx]]
                continue
        
        # If no obstacle detected, store max range point
        if pts_info[n] == 0:
            offset = samples * search_vec
            pos = position + offset
            lidar_pts[n] = [int(pos[0]), int(pos[1])]
    
    return lidar_pts, pts_info


def visualize_lidar_on_map(gymenv, num_steps=1, save_path=None):
    """
    Visualize lidar rays on the map for a given number of steps.
    
    Parameters:
    -----------
    gymenv : GymnasiumEnv
        The initialized gymnasium environment
    num_steps : int
        Number of steps to visualize
    save_path : str, optional
        Path to save the visualization. If None, displays interactively
    """
    # Get the current map image
    img = gymenv.current_image_for_movement_restriction.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Flip the image vertically to match matplotlib's coordinate system
    img_rgb = np.flip(img_rgb, axis=0)
    img_height = img_rgb.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14), dpi=100)
    ax.imshow(img_rgb, origin='lower')
    
    print(f"\n=== Lidar Visualization ===")
    print(f"Map size: {img.shape}")
    print(f"Lidar range: {gymenv.lidar_range}m = {gymenv.lidar_range / gymenv.meters_per_pixel_mower:.1f} pixels")
    print(f"Lidar FOV: {gymenv.lidar_fov}°")
    print(f"Number of rays: {gymenv.lidar_rays}")
    print(f"\nTaking {num_steps} random steps and visualizing lidar...")
    
    # Color map for different detection types
    colors = {
        0: 'lime',      # Max range (no detection)
        1: 'red',       # Known obstacle
        2: 'orange',    # Unknown obstacle
        3: 'purple'     # Out of bounds
    }
    
    labels = {
        0: 'Max range (no detection)',
        1: 'Known obstacle',
        2: 'Unknown obstacle',
        3: 'Out of bounds'
    }
    
    # Track which types we've seen for legend
    seen_types = set()
    
    # Take random steps and visualize lidar
    for step in range(num_steps):
        # Get current position and heading
        position = gymenv.position
        heading_deg = gymenv.current_bearing
        
        # Compute lidar points using our public wrapper function
        lidar_pts, pts_info = compute_lidar_pts_flat(gymenv, position, heading_deg)
        
        # Flip y-coordinates to match the flipped image
        pos_y_flipped = img_height - 1 - position[1]
        
        # Draw each lidar ray
        for i, (pt, info) in enumerate(zip(lidar_pts, pts_info)):
            pt_y_flipped = img_height - 1 - pt[1]
            
            color = colors.get(info, 'gray')
            seen_types.add(info)
            
            # Draw line from position to detected point
            ax.plot([position[0], pt[0]], [pos_y_flipped, pt_y_flipped],
                   color=color, alpha=0.3, linewidth=0.5)
        
        # Draw current position as a circle
        ax.plot(position[0], pos_y_flipped, 'o', color='blue', 
               markersize=8, markeredgecolor='white', markeredgewidth=1)
        
        # Take a random action
        action = np.random.uniform(-1, 1, size=1)
        obs, reward, done, truncated, info = gymenv.step(action)
        
        if done:
            print(f"Episode ended at step {step + 1}")
            break
    
    # Add legend for detection types we've seen
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[t], label=labels[t]) 
                      for t in sorted(seen_types)]
    legend_elements.append(Patch(facecolor='blue', label='Helicopter position'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Set title
    ax.set_title('Lidar Ray Visualization', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nVisualization saved to: {save_path}")
    else:
        print("\nDisplaying visualization...")
        plt.show()
    
    plt.close(fig)


def main():
    print("=== Lidar Ray Visualization Script ===")
    print(f"RL Type: {para.training_library}")
    print(f"Radiation Type: {para.z_rad_type}")
    
    # Load height data
    height_data_path = os.path.join(os.path.dirname(__file__), 'misc', 'geo_data', 'height_data', 'height_data.npy')
    height_data = np.load(height_data_path)
    
    # Initialize environment based on RL type
    if para.training_library == 'sb3':
        print("\nInitializing classic gymnasium environment...")
        gymenv = class_gymenv.GymnasiumEnv(height_data, radiation_grid_visualization=False)
        
        # Test with a specific image
        test_image_path = os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "train_4_1.png")
        print(f"Resetting with image: {test_image_path}")
        gymenv.reset(options={'image_path': test_image_path})
        
    elif para.training_library == 'omnisafe':
        print("\nInitializing constrained RL (OmniSafe) environment...")
        omnisafe_env = class_gymenv.GymEnvOmniSafe(radiation_grid_visualization=False)
        gymenv = omnisafe_env._env
        
        # Test with a specific image
        test_image_path = os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "train_4_1.png")
        print(f"Resetting with image: {test_image_path}")
        omnisafe_env.reset(options={'image_path': test_image_path})
        
    else:
        raise ValueError(f"Unknown RL type: {para.training_library}")
    
    print(f"Environment initialized successfully!")
    
    # Visualize lidar rays
    save_dir = os.path.join("misc", "logs")
    save_path = os.path.join(save_dir, "lidar_rays_visualization.png")
    
    visualize_lidar_on_map(gymenv, num_steps=1, save_path=save_path)
    
    print("\n=== Visualization Complete ===")


if __name__ == "__main__":
    main()
