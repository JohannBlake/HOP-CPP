"""
Visualization script for ICAPS paper
Creates consistent, publication-quality figures with unified color scheme
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
import torch
import webbrowser
import tempfile
from scipy.ndimage import gaussian_filter1d

# ============================================================================
# GLOBAL COLOR SCHEME AND NAMING CONVENTIONS
# ============================================================================
# CSV Name Mapping - Maps CSV names to paper names
CSV_NAME_MAPPING = {
    'State-of-the-art baseline': 'HOP-CPP w/o VF, OPTV, HM',
    'With HAM': 'HOP-CPP w/o VF, OPTV',
    'With HAM, OPTV': 'HOP-CPP w/o VF',
    'With HAM, OPTV, visit frequency': 'HOP-CPP',
    'With HAM, OPTV, footprint with visit frequency': 'HOP-CPP',
    # Legacy / alternative names
    'State-of-the-art baseline (Radiation)': 'HOP-CPP w/o VF, OPTV, HM',
    'With HAM (Radiation)': 'HOP-CPP w/o VF, OPTV',
    'With HAM, OPTV (Radiation)': 'HOP-CPP w/o VF',
    'With HAM, OPTV, visit frequency (Radiation)': 'HOP-CPP',
}

COLOR_SCHEME = {
    'HOP-CPP w/o VF, OPTV, HM': '#d1495b',
    'HOP-CPP w/o VF, OPTV': '#dcac00',
    'HOP-CPP w/o VF': '#008c64',
    'HOP-CPP': '#30638e',
    # Legacy names for backwards compatibility
    'State-of-the-art baseline': '#d1495b',
    'With HAM': '#dcac00',
    'With HAM, OPTV': '#008c64',
    'With HAM, OPTV, visit frequency': '#30638e',
    'With HAM, OPTV, footprint with visit frequency': '#30638e',
}

# Shortened names for labels
LABEL_NAMES = {
    'HOP-CPP w/o VF, OPTV, HM': 'HOP-CPP w/o VF, OPTV, HM',
    'HOP-CPP w/o VF, OPTV': 'HOP-CPP w/o VF, OPTV',
    'HOP-CPP w/o VF': 'HOP-CPP w/o VF',
    'HOP-CPP': 'HOP-CPP',
    # Legacy names
    'State-of-the-art baseline': 'Baseline',
    'With HAM': 'HAM',
    'With HAM, OPTV': 'HAM + OPTV',
    'With HAM, OPTV, footprint with visit frequency': 'HAM + OPTV + Visit Freq.',
    'With HAM, OPTV, visit frequency': 'HAM + OPTV + Visit Freq.',
}

# ============================================================================
# PUBLICATION-QUALITY MATPLOTLIB SETTINGS
# ============================================================================
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        pass  # Use default style

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'lines.linewidth': 2,
    'lines.markersize': 5,
    'grid.alpha': 0.3
})

# ============================================================================
# ENVIRONMENT INITIALIZATION
# ============================================================================
def init_gymenv():
    """Initialize gym environment similar to step_through_gymenv_and_save_data_for_vis.py"""
    print("Initializing gym environment...")
    
    # Get main folder path
    main_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Add to path if needed
    if main_folder not in sys.path:
        sys.path.insert(0, main_folder)
    
    # Import environment class
    from class_gymenv import GymEnvOmniSafe
    
    # Initialize environment
    env = GymEnvOmniSafe(radiation_grid_visualization=False, is_evaluation=True)
    gymenv = env._env.env
    
    # Reset environment to initialize grids
    print("Resetting environment to initialize grids...")
    env.reset()
    
    # Enable radiation grid visualization after initialization
    gymenv.radiation_grid_visualization = True
    
    # Force creation of multi-scale grids for visualization (even if using HAM ablation)
    if not (hasattr(gymenv, 'multi_scale_grids') and gymenv.multi_scale_grids):
        print("Creating multi-scale grids for visualization...")
        try:
            gymenv.create_multi_scale_grids()
            print(f"Multi-scale grids created: {len(gymenv.multi_scale_grids)} grids")
            if hasattr(gymenv, 'multi_scale_grid_sizes_m'):
                print(f"Grid sizes (meters): {[f'{s:.1f}m' for s in gymenv.multi_scale_grid_sizes_m]}")
        except Exception as e:
            print(f"Error creating multi-scale grids: {e}")
    else:
        print(f"Multi-scale grids initialized: {len(gymenv.multi_scale_grids)} grids")
        if hasattr(gymenv, 'multi_scale_grid_sizes_m'):
            print(f"Grid sizes (meters): {[f'{s:.1f}m' for s in gymenv.multi_scale_grid_sizes_m]}")
    
    print("Environment initialized successfully")
    return env, gymenv


# ============================================================================
# OBSERVATION EXPLANATION FIGURE
# ============================================================================
def create_observation_explanation_figure(output_dir):
    """Create a 2x2 matrix figure showing different observation types."""
    print("\n" + "="*80)
    print("Creating Observation Explanation Figure")
    print("="*80)
    
    img_paths = {
        'radiation': r"C:\Users\johan\Pictures\observation_explanation_radiation.png",
        'lidar': r"C:\Users\johan\Pictures\observation_explanation_lidar.png",
        'lidar_map_rays': r"C:\Users\johan\Pictures\observation_explanation_lidar_map_with_rays.png",
        'normal_map': r"C:\Users\johan\Pictures\observation_explanation_normal_map.png"
    }
    
    # Check if images exist
    missing = [f"{name}: {path}" for name, path in img_paths.items() if not os.path.exists(path)]
    if missing:
        print("Warning: Missing images:")
        for m in missing:
            print(f"  - {m}")
        return None
    
    from PIL import Image
    
    # Create figure
    fig = plt.figure(figsize=(14, 14))
    left, right, top, bottom = 0.10, 0.95, 0.96, 0.05
    hspace = 0.02  # Gap between rows (relative to subplot height)
    wspace = 0.02  # Gap between columns (relative to subplot width)
    gs = fig.add_gridspec(2, 2, hspace=hspace, wspace=wspace, 
                          left=left, right=right, top=top, bottom=bottom)
    
    # Calculate column centers for titles
    total_width = right - left
    col_width = (total_width - wspace) / 2
    col1_center = left + col_width / 2
    col2_center = left + col_width + wspace + col_width / 2
    
    # Calculate row centers for titles
    total_height = top - bottom
    row_height = (total_height - hspace) / 2
    row1_center = bottom + row_height * 1.5 + hspace / 2
    row2_center = bottom + row_height / 2
    
    # Add titles
    fig.text(0.07, row1_center, 'Lidar', fontsize=20, fontweight='bold', rotation=90, va='center', ha='center')
    fig.text(0.07, row2_center, 'Ultrasonic', fontsize=20, fontweight='bold', rotation=90, va='center', ha='center')
    fig.text(col1_center, 0.96, 'Hidden environment state', fontsize=20, fontweight='bold', ha='center', va='bottom')
    fig.text(col2_center, 0.96, 'Observation', fontsize=20, fontweight='bold', ha='center', va='bottom')
    
    # Add images
    images = [
        (gs[0, 0], 'lidar_map_rays'),
        (gs[0, 1], 'lidar'),
        (gs[1, 0], 'normal_map'),
        (gs[1, 1], 'radiation')
    ]
    
    for grid_pos, img_key in images:
        ax = fig.add_subplot(grid_pos)
        ax.imshow(Image.open(img_paths[img_key]), aspect='auto')
        ax.axis('off')
    
    # Save figure
    output_path = output_dir / 'observation_types_radiation_lidar_lidar_with_rays_normal_map_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved observation explanation figure: {output_path}")
    return str(output_path)


# ============================================================================
# AGENT BEHAVIOR COMPARISON FIGURE
# ============================================================================
def create_agent_behavior_comparison_figure(output_dir, folders_dict=None):
    """Create a 1x2 figure showing agent fragmentation and getting stuck.
    Uses second image from baseline and second image from HAM+OPTV+visit frequency."""
    print("\n" + "="*80)
    print("Creating Agent Behavior Comparison Figure")
    print("="*80)
    
    import matplotlib.image as mpimg
    
    # If folders_dict is provided, use images from benchmark folders
    if folders_dict:
        baseline_folder = Path(folders_dict['benchmark_results_baseline'])
        ham_optv_vf_folder = Path(folders_dict['benchmark_results_fisheye_optv_visit_frequ'])
        
        if not baseline_folder.exists():
            print(f"Warning: Baseline folder not found: {baseline_folder}")
            return None
        
        if not ham_optv_vf_folder.exists():
            print(f"Warning: HAM+OPTV+VF folder not found: {ham_optv_vf_folder}")
            return None
        
        # Get second image from each folder (index 1)
        baseline_images = sorted(baseline_folder.glob("*.png"))
        ham_images = sorted(ham_optv_vf_folder.glob("*.png"))
        
        if len(baseline_images) < 2:
            print(f"Warning: Not enough images in baseline folder. Found {len(baseline_images)}")
            return None
        
        if len(ham_images) < 2:
            print(f"Warning: Not enough images in HAM folder. Found {len(ham_images)}")
            return None
        
        baseline_img_path = baseline_images[1]  # Second image
        ham_img_path = ham_images[1]    # Second image
        
        print(f"Using baseline image: {baseline_img_path.name}")
        print(f"Using HAM+OPTV+VF image: {ham_img_path.name}")
        
        # Create figure with 1 row, 2 columns, no spacing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Remove spacing between subplots
        plt.subplots_adjust(wspace=0.02, left=0.02, right=0.98, top=0.98, bottom=0.02)
        
        # Display images
        img1 = mpimg.imread(baseline_img_path)
        ax1.imshow(img1)
        ax1.axis('off')
        
        img2 = mpimg.imread(ham_img_path)
        ax2.imshow(img2)
        ax2.axis('off')
    else:
        # Fallback to hardcoded paths if folders_dict not provided
        img_paths = {
            'fragmentation': r"C:\Users\johan\Pictures\original_tv_reward_step_660.png",
            'stuck': r"C:\Users\johan\Pictures\no_information_on_visit_frequency_agent_gets_stuck.png"
        }
        
        # Check if images exist
        missing = [f"{name}: {path}" for name, path in img_paths.items() if not os.path.exists(path)]
        if missing:
            print("Warning: Missing images:")
            for m in missing:
                print(f"  - {m}")
            return None
        
        from PIL import Image
        
        # Read images
        img_fragmentation = Image.open(img_paths['fragmentation'])
        img_stuck = Image.open(img_paths['stuck'])
        
        # Create figure with 1 row, 2 columns, no spacing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Remove spacing between subplots
        plt.subplots_adjust(wspace=0.02, left=0.02, right=0.98, top=0.98, bottom=0.02)
        
        # Display images with aspect='auto' to allow stretching to square
        ax1.imshow(img_fragmentation, aspect='auto')
        ax1.axis('off')
        
        ax2.imshow(img_stuck, aspect='auto')
        ax2.axis('off')
    
    # Save figure
    output_path = output_dir / 'agent_behavior_fragmentation_issue_and_stuck_behavior_without_visit_frequency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close()
    
    print(f"✓ Saved agent behavior comparison figure: {output_path}")
    return str(output_path)


# ============================================================================
# HYPERBOLIC ATTENTION MAP AND MULTI-SCALE GRIDS VISUALIZATION
# ============================================================================
def create_ham_and_multiscale_grids_figure(gymenv, output_dir):
    """
    Create a figure with two diagrams side by side:
    - Left: Multi-scale grids shown as grid lines
    - Right: HAM distorted grid shown as grid lines
    """
    print("\n" + "="*80)
    print("Creating HAM and multi-scale grids figure...")
    print("="*80)
    
    # Create figure with 1 row, 2 columns
    fig, (ax_multiscale, ax_ham) = plt.subplots(1, 2, figsize=(14, 7))
    
    zoom_half = 280  # pixels for zoom level (30% zoom in from 400)
    
    # ============================================================================
    # LEFT DIAGRAM: Multi-scale grids as grid lines
    # ============================================================================
    if hasattr(gymenv, 'multi_scale_grids') and gymenv.multi_scale_grids:
        print("Plotting multi-scale grids as grid lines...")
        grids = gymenv.multi_scale_grids
        grid_sizes_m = gymenv.multi_scale_grid_sizes_m
        
        # Define colors for each scale (more similar colors)
        scale_colors = [COLOR_SCHEME['State-of-the-art baseline'], COLOR_SCHEME['State-of-the-art baseline'], COLOR_SCHEME['State-of-the-art baseline'], COLOR_SCHEME['State-of-the-art baseline']]
        
        # Plot each scale (largest first as background, smallest in foreground)
        for idx in range(len(grids) - 1, -1, -1):
            grid = grids[idx]
            grid_size_m = grid_sizes_m[idx]
            
            # Extract and center points
            grid_points = grid.reshape(-1, 2)
            grid_center_x = np.mean(grid_points[:, 0])
            grid_center_y = np.mean(grid_points[:, 1])
            grid_points_centered = grid_points - np.array([grid_center_x, grid_center_y])
            
            # Reshape to get grid structure (n x n x 2)
            n = int(np.sqrt(len(grid_points_centered)))
            grid_reshaped = grid_points_centered.reshape(n, n, 2)
            
            # Draw horizontal grid lines (no transparency)
            for i in range(0, n, max(1, n // 20)):  # Subsample lines for clarity
                line_x = grid_reshaped[i, :, 0]
                line_y = grid_reshaped[i, :, 1]
                ax_multiscale.plot(line_x, line_y, color=scale_colors[idx], 
                       alpha=1.0, linewidth=0.9,
                       label=f'{grid_size_m:.1f}m scale' if i == 0 else None)
            
            # Draw vertical grid lines (no transparency)
            for j in range(0, n, max(1, n // 20)):  # Subsample lines for clarity
                line_x = grid_reshaped[:, j, 0]
                line_y = grid_reshaped[:, j, 1]
                ax_multiscale.plot(line_x, line_y, color=scale_colors[idx], 
                       alpha=1.0, linewidth=0.9)
        
        print(f"Plotted {len(grids)} multi-scale grids")
    else:
        print("Warning: Multi-scale grids not available")
        ax_multiscale.text(0.5, 0.5, 'Multi-scale grids not available',
                          ha='center', va='center', transform=ax_multiscale.transAxes,
                          fontsize=12)
    
    # Set properties for left diagram
    ax_multiscale.set_aspect('equal')
    ax_multiscale.set_xlim(-zoom_half, zoom_half)
    ax_multiscale.set_ylim(-zoom_half, zoom_half)
    ax_multiscale.set_facecolor('#f8f8f8')
    # Remove axis labels, ticks, and grid
    ax_multiscale.set_xticks([])
    ax_multiscale.set_yticks([])
    ax_multiscale.spines['top'].set_visible(False)
    ax_multiscale.spines['right'].set_visible(False)
    ax_multiscale.spines['bottom'].set_visible(False)
    ax_multiscale.spines['left'].set_visible(False)
    
    # ============================================================================
    # RIGHT DIAGRAM: HAM distorted grid as grid lines
    # ============================================================================
    print("Plotting HAM grid as grid lines...")
    try:
        # Generate HAM grid
        ham_grid = gymenv.generate_flat_distorted_grid()
        ham_points = ham_grid.reshape(-1, 2)
        
        # Center the HAM grid
        ham_center_x = np.mean(ham_points[:, 0])
        ham_center_y = np.mean(ham_points[:, 1])
        ham_points_centered = ham_points - np.array([ham_center_x, ham_center_y])
        
        # Reshape to get grid structure (n x n x 2)
        n = int(np.sqrt(len(ham_points_centered)))
        ham_reshaped = ham_points_centered.reshape(n, n, 2)
        
        ham_color = COLOR_SCHEME['With HAM']
        
        # Draw horizontal grid lines
        for i in range(0, n, max(1, n // 20)):  # Subsample lines for clarity
            line_x = ham_reshaped[i, :, 0]
            line_y = ham_reshaped[i, :, 1]
            ax_ham.plot(line_x, line_y, color=ham_color, 
                   alpha=0.7, linewidth=1.2,
                   label='HAM Grid' if i == 0 else None)
        
        # Draw vertical grid lines
        for j in range(0, n, max(1, n // 20)):  # Subsample lines for clarity
            line_x = ham_reshaped[:, j, 0]
            line_y = ham_reshaped[:, j, 1]
            ax_ham.plot(line_x, line_y, color=ham_color, 
                   alpha=0.7, linewidth=1.2)
        
        print("HAM grid plotted successfully")
        
    except Exception as e:
        print(f"Error generating HAM grid: {e}")
        ax_ham.text(0.5, 0.5, f'Error generating HAM grid:\n{str(e)}',
                       ha='center', va='center', transform=ax_ham.transAxes,
                       fontsize=12)
        import traceback
        traceback.print_exc()
    
    # Set properties for right diagram
    ax_ham.set_aspect('equal')
    ax_ham.set_xlim(-zoom_half, zoom_half)
    ax_ham.set_ylim(-zoom_half, zoom_half)
    ax_ham.set_facecolor('#f8f8f8')
    # Remove axis labels, ticks, and grid
    ax_ham.set_xticks([])
    ax_ham.set_yticks([])
    ax_ham.spines['top'].set_visible(False)
    ax_ham.spines['right'].set_visible(False)
    ax_ham.spines['bottom'].set_visible(False)
    ax_ham.spines['left'].set_visible(False)
   
    # Save figure
    output_path = output_dir / 'observation_space_multi_scale_grid_levels_and_ham_distortion_comparison.png'
    plt.subplots_adjust(wspace=0.02, left=0.02, right=0.98, top=0.98, bottom=0.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# OPTV EXPLANATION FIGURE
# ============================================================================
def create_optv_explanation_figure(output_dir):
    """
    Create OPTV explanation figures - saves each image separately and returns paths
    """
    print("\n" + "="*80)
    print("Creating OPTV explanation figures...")
    print("="*80)
    
    # Paths to the OPTV images
    optv_image_path = r"C:\Users\johan\Pictures\optv_explanation.png"
    example_image_path = r"C:\Users\johan\Pictures\example_where_original_tv_reward_is_changing_original_objective.png"
    original_tv_path = r"C:\Users\johan\Pictures\original_tv_reward_step_660.png"
    optv_tv_path = r"C:\Users\johan\Pictures\objective_preserving_tv_reward_step_660.png"
    
    output_paths = []
    
    # Define legend items with colors and labels
    legend_items_example = [
        ('#FFAEC9', 'Newly measured area'),
        ('#66edf0', 'Already measured area'),
        ('#f8d304', 'Shorter border before adding newly measured area'),
        ('#008c64', 'Longer border after adding newly measured area'),
        ('#000000', 'Obstacle')
    ]

    legend_items_optv = [
        ('#FFAEC9', 'Newly measured area'),
        ('#66edf0', 'Already measured area'),
        ('#C3C3C3', 'Border length of newly measured area'),
        ('#f8d304', 'Intersection of already and newly measured border'),
        ('#008c64', 'Intersection of obstacle border and newly measured area'),
        ('#000000', 'Obstacle')
    ]
    
    # Import for legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Check if both images exist
    if os.path.exists(optv_image_path) and os.path.exists(example_image_path):
        # Create a single figure with two subplots side by side - square format, 1.5x larger
        fig = plt.figure(figsize=(28, 14))
        
        # Use gridspec for consistent spacing like the observation figure
        left, right, top, bottom = 0.02, 0.98, 0.98, 0.02
        wspace = 0.02  # Gap between columns
        gs = fig.add_gridspec(1, 2, wspace=wspace, 
                              left=left, right=right, top=top, bottom=bottom)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Process OPTV explanation image (left)
        img1 = plt.imread(optv_image_path)
        ax1.imshow(img1, aspect='auto')
        ax1.axis('off')
        
        # Create legend for OPTV explanation
        legend_elements_optv = []
        # First two items are areas - use patches
        for i in range(2):
            color, label = legend_items_optv[i]
            legend_elements_optv.append(Patch(facecolor=color, edgecolor='black', label=label))
        
        # Remaining items are borders/lines - use Line2D
        for i in range(2, len(legend_items_optv)):
            color, label = legend_items_optv[i]
            legend_elements_optv.append(Line2D([0], [0], color=color, linewidth=3, label=label))
        
        # Add legend with 3x larger font (27 instead of 9)
        ax1.legend(handles=legend_elements_optv, loc='upper left', 
                 ncol=1, frameon=True, framealpha=0.675, 
                 edgecolor='black', fontsize=27, fancybox=True)
        
        # Process example image (right)
        img2 = plt.imread(example_image_path)
        ax2.imshow(img2, aspect='auto')
        ax2.axis('off')
        
        # Create legend for example
        legend_elements_example = []
        # First two items are areas - use patches
        for i in range(2):
            color, label = legend_items_example[i]
            legend_elements_example.append(Patch(facecolor=color, edgecolor='black', label=label))
        
        # Remaining items are borders/lines - use Line2D (skip last empty item)
        for i in range(2, len(legend_items_example) - 1):
            color, label = legend_items_example[i]
            legend_elements_example.append(Line2D([0], [0], color=color, linewidth=3, label=label))
        
        # Add empty spacer to match OPTV legend size
        legend_elements_example.append(Line2D([0], [0], color='none', linewidth=0, label=''))
        
        # Add legend with 3x larger font (27 instead of 9)
        ax2.legend(handles=legend_elements_example, loc='upper left', 
                 ncol=1, frameon=True, framealpha=0.675, 
                 edgecolor='black', fontsize=27, fancybox=True)
        
        # Save combined figure
        output_path = output_dir / 'optv_explanation_and_example_combined.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
        plt.close()
        
        print(f"Saved: {output_path}")
        output_paths.append(output_path)
    else:
        if not os.path.exists(optv_image_path):
            print(f"Warning: OPTV explanation image not found at {optv_image_path}")
        if not os.path.exists(example_image_path):
            print(f"Warning: Example image not found at {example_image_path}")
    
    # Process side-by-side comparison of original TV vs OPTV
    if os.path.exists(original_tv_path) and os.path.exists(optv_tv_path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Remove spacing between subplots
        plt.subplots_adjust(wspace=0.02, left=0.02, right=0.98, top=0.98, bottom=0.02)
        
        # Original TV image
        img1 = plt.imread(original_tv_path)
        ax1.imshow(img1, aspect='auto')
        ax1.axis('off')
        
        # OPTV image
        img2 = plt.imread(optv_tv_path)
        ax2.imshow(img2, aspect='auto')
        ax2.axis('off')
        
        # Save figure
        output_path = output_dir / 'agent_path_comparison_original_tv_reward_versus_optv_reward_step_660.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
        plt.close()
        
        print(f"Saved: {output_path}")
        output_paths.append(output_path)
    else:
        if not os.path.exists(original_tv_path):
            print(f"Warning: Original TV image not found at {original_tv_path}")
        if not os.path.exists(optv_tv_path):
            print(f"Warning: OPTV TV image not found at {optv_tv_path}")
    
    return output_paths if output_paths else None








# ============================================================================
# BEST MODEL EXTRACTION FUNCTIONS
# ============================================================================
def get_best_models_info(csv_path):
    """
    Extract best model information for the 4 ablation studies.
    
    Returns:
        tuple: (sweep_ids, model_numbers) where each is a list of 4 entries
               - sweep_ids: List of best sweep IDs for each study
               - model_numbers: List of best constrained model numbers (epochs) for each study
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Clean ablation_study_name
    df['ablation_study_name'] = df['ablation_study_name'].str.strip().str.strip('"')
    
    # Calculate episode duration from steps_end * base_timestep (in seconds)
    df['duration_end'] = df['steps_end'] / 10
    
    # Extract epoch number
    df['epoch_num'] = df['epoch'].str.extract(r'epoch-(\d+)').astype(int)
    
    # Filter for Radiation only (exclude Lidar)
    df_radiation = df[~df['ablation_study_name'].str.contains('Lidar', case=False, na=False)].copy()
    df_radiation['ablation_study_name'] = df_radiation['ablation_study_name'].str.replace(r'\s*\(Radiation\)', '', regex=True)
    
    # Explicitly filter for the 4 ablation studies
    studies_to_include = [
        'State-of-the-art baseline',
        'With HAM',
        'With HAM, OPTV',
        'With HAM, OPTV, footprint with visit frequency',
        'With HAM, OPTV, visit frequency'  # Alternative naming
    ]
    df_radiation = df_radiation[df_radiation['ablation_study_name'].isin(studies_to_include)].copy()
    
    # Get unique ablation studies (in sorted order)
    studies = sorted(df_radiation['ablation_study_name'].unique())
    
    sweep_ids = []
    model_numbers = []
    
    # Find best model for each study
    for study in studies:
        study_data = df_radiation[df_radiation['ablation_study_name'] == study].copy()
        
        # Group by epoch_num (across all sweep_ids) and calculate average metrics
        epoch_stats = study_data.groupby('epoch_num').agg({
            'coverage_end': 'mean',
            'duration_end': 'mean',
            'epoch': 'first'
        }).reset_index()
        
        # Find epoch with maximum coverage, then minimum duration as tiebreaker
        epoch_stats_sorted = epoch_stats.sort_values(['coverage_end', 'duration_end'], 
                                                      ascending=[False, True])
        best_epoch_num = epoch_stats_sorted.iloc[0]['epoch_num']
        
        # Now find the specific sweep_id with best performance for this epoch
        epoch_data = study_data[study_data['epoch_num'] == best_epoch_num].copy()
        
        # Group by sweep_id to find which one performs best for this epoch
        sweep_stats = epoch_data.groupby('sweep_id').agg({
            'coverage_end': 'mean',
            'duration_end': 'mean'
        }).reset_index()
        
        # Sort to find best sweep_id (max coverage, then min duration)
        sweep_stats['path_sort_key'] = sweep_stats['duration_end'].fillna(float('inf'))
        sweep_stats_sorted = sweep_stats.sort_values(['coverage_end', 'path_sort_key'], 
                                                      ascending=[False, True])
        best_sweep_id = sweep_stats_sorted.iloc[0]['sweep_id']
        
        sweep_ids.append(best_sweep_id)
        model_numbers.append(int(best_epoch_num))
    
    return sweep_ids, model_numbers


# ============================================================================
# QUANTITATIVE COMPARISON TABLE - BEST MODEL
# ============================================================================
def create_quantitative_comparison_table_best_model(csv_path, output_dir):
    """
    Create quantitative comparison table using the best epoch from each study.
    Best epoch is determined by highest average coverage (averaged over all seeds),
    with shortest average duration as tiebreaker.
    Includes training steps needed to reach baseline performance.
    """
    print("\n" + "="*80)
    print("Creating quantitative comparison table (Best Epoch)...")
    print("="*80)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Clean ablation_study_name
    df['ablation_study_name'] = df['ablation_study_name'].str.strip().str.strip('"')
    
    # Calculate episode duration from steps_end * base_timestep (in seconds)
    df['duration_end'] = df['steps_end'] / 10
    
    # Extract epoch number and steps
    df['epoch_num'] = df['epoch'].str.extract(r'epoch-(\d+)').astype(int)
    df['steps'] = df['epoch_num'] * 256  # in thousands of steps
    
    # Filter for Radiation only (exclude Lidar)
    df_radiation = df[~df['ablation_study_name'].str.contains('Lidar', case=False, na=False)].copy()
    df_radiation['ablation_study_name'] = df_radiation['ablation_study_name'].str.replace(r'\s*\(Radiation\)', '', regex=True)
    
    # Explicitly filter for the 4 ablation studies
    studies_to_include = [
        'State-of-the-art baseline',
        'With HAM',
        'With HAM, OPTV',
        'With HAM, OPTV, footprint with visit frequency',
        'With HAM, OPTV, visit frequency'  # Alternative naming
    ]
    df_radiation = df_radiation[df_radiation['ablation_study_name'].isin(studies_to_include)].copy()
    
    # Get unique ablation studies (in sorted order)
    studies = sorted(df_radiation['ablation_study_name'].unique())
    
    # Find the best model for each study
    best_models = {}
    steps_to_reach_baseline_performance = {}
    
    # First pass: Find best epoch for each study and get baseline performance
    baseline_coverage = None
    for study in studies:
        study_data = df_radiation[df_radiation['ablation_study_name'] == study].copy()
        
        # Group by epoch and calculate average coverage and duration across all maps and seeds
        epoch_stats = study_data.groupby('epoch_num').agg({
            'coverage_end': 'mean',
            'duration_end': 'mean',
            'epoch': 'first'
        }).reset_index()
        
        # Find epoch with maximum coverage, then minimum duration as tiebreaker
        # Sort by coverage (descending) then duration (ascending)
        epoch_stats_sorted = epoch_stats.sort_values(['coverage_end', 'duration_end'], 
                                                      ascending=[False, True])
        best_epoch_idx = epoch_stats_sorted.index[0]
        best_epoch_num = epoch_stats_sorted.iloc[0]['epoch_num']
        best_epoch_name = epoch_stats_sorted.iloc[0]['epoch']
        
        print(f"\n{study}:")
        print(f"  Best epoch: {best_epoch_num} ({best_epoch_name})")
        print(f"  Avg coverage: {epoch_stats_sorted.iloc[0]['coverage_end']*100:.1f}%")
        print(f"  Avg duration: {epoch_stats_sorted.iloc[0]['duration_end']:.0f} sec")
        
        # Get all data for the best epoch
        best_model_data = study_data[study_data['epoch_num'] == best_epoch_num].copy()
        best_models[study] = best_model_data
        
        # Store baseline coverage (from State-of-the-art baseline study)
        if study == 'State-of-the-art baseline':
            baseline_coverage = epoch_stats_sorted.iloc[0]['coverage_end']
            print(f"  *** Baseline coverage target: {baseline_coverage*100:.1f}% ***")
    
    # Second pass: Calculate steps to reach baseline performance for each study
    for study in studies:
        study_data = df_radiation[df_radiation['ablation_study_name'] == study].copy()
        
        # Group by epoch and calculate average coverage across all maps and seeds
        epoch_stats = study_data.groupby('epoch_num').agg({
            'coverage_end': 'mean',
            'epoch': 'first'
        }).reset_index().sort_values('epoch_num')
        
        # Find first epoch where coverage >= baseline_coverage
        epochs_meeting_baseline = epoch_stats[epoch_stats['coverage_end'] >= baseline_coverage]
        
        if len(epochs_meeting_baseline) > 0:
            first_epoch_num = epochs_meeting_baseline.iloc[0]['epoch_num']
            steps_to_reach_baseline_performance[study] = first_epoch_num * 256
            print(f"  Steps to reach baseline performance: {steps_to_reach_baseline_performance[study]}k (epoch {first_epoch_num})")
        else:
            steps_to_reach_baseline_performance[study] = None
            print(f"  Steps to reach baseline performance: Never reached")
    
    # Create results dictionary
    results = []
    
    for study in studies:
        study_data = best_models[study]
        
        print(f"\n{study} (Best Model):")
        print(f"  Total episodes: {len(study_data)}")
        
        # Calculate metrics
        avg_duration_mean = study_data['duration_end'].mean()
        avg_duration_std = study_data['duration_end'].std()
        
        episodes_99 = study_data[study_data['coverage_end'] >= 0.99]
        total_episodes = len(study_data)
        episodes_reaching_99 = len(episodes_99)
        percentage_99 = (episodes_reaching_99 / total_episodes) * 100 if total_episodes > 0 else 0
        
        # Calculate success rate std from binary outcomes
        success_binary = (study_data['coverage_end'] >= 0.99).astype(int)
        success_rate_std = success_binary.std() * 100
        
        coverage_mean = study_data['coverage_end'].mean() * 100
        coverage_std = study_data['coverage_end'].std() * 100
        
        # Calculate total degrees turned if available
        if 'total_degrees_turned' in study_data.columns:
            degrees_turned_mean = study_data['total_degrees_turned'].mean()
            degrees_turned_std = study_data['total_degrees_turned'].std()
        else:
            degrees_turned_mean = None
            degrees_turned_std = None
        
        print(f"  Avg duration: {avg_duration_mean:.0f} ± {avg_duration_std:.0f} sec")
        print(f"  Success rate (99%): {percentage_99:.1f}% ± {success_rate_std:.1f}%")
        print(f"  Final coverage: {coverage_mean:.1f}% ± {coverage_std:.1f}%")
        if degrees_turned_mean is not None:
            print(f"  Total degrees turned: {degrees_turned_mean/1000:.1f}k ± {degrees_turned_std/1000:.1f}k")
        
        results.append({
            'Study': LABEL_NAMES.get(study, study),
            'Avg Duration (mean)': avg_duration_mean,
            'Avg Duration (std)': avg_duration_std,
            'Success Rate 99% (%)': percentage_99,
            'Success Rate 99% (std)': success_rate_std,
            'Final Coverage (mean %)': coverage_mean,
            'Final Coverage (std %)': coverage_std,
            'Total Degrees Turned (mean)': degrees_turned_mean,
            'Total Degrees Turned (std)': degrees_turned_std,
            'Steps to Reach Baseline Performance': steps_to_reach_baseline_performance.get(study, None)
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Create figure with exact dimensions for table
    fig = plt.figure(figsize=(18, 2.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Prepare table data with training steps column
    table_data = []
    headers = ['Model', 'Final Coverage', 'Success Rate\n(99% Coverage)', 'Avg Duration\n(seconds)', 'Total Degrees\nTurned', 'Steps to Reach\nBaseline Perf.']
    
    for _, row in results_df.iterrows():
        steps_val = row['Steps to Reach Baseline Performance']
        if pd.isna(steps_val) or steps_val is None:
            steps_str = 'N/A'
        else:
            steps_str = f"{steps_val / 1000:.3f} Mio"
        
        degrees_val = row['Total Degrees Turned (mean)']
        if pd.isna(degrees_val) or degrees_val is None:
            degrees_str = 'N/A'
        else:
            degrees_str = f"{degrees_val/1000:.1f}k ± {row['Total Degrees Turned (std)']/1000:.1f}k"
            
        table_row = [
            row['Study'],
            f"{row['Final Coverage (mean %)']:.1f}% ± {row['Final Coverage (std %)']:.1f}%",
            f"{row['Success Rate 99% (%)']:.1f}% ± {row['Success Rate 99% (std)']:.1f}%",
            f"{row['Avg Duration (mean)']:.0f} ± {row['Avg Duration (std)']:.0f}",
            degrees_str,
            steps_str
        ]
        table_data.append(table_row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.20, 0.18, 0.16, 0.16, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3.2)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('black')
        cell.set_text_props(weight='bold', color='white')
        cell.set_edgecolor('black')
    
    # Define row colors
    row_colors = [
        COLOR_SCHEME['State-of-the-art baseline'],
        COLOR_SCHEME['With HAM'],
        COLOR_SCHEME['With HAM, OPTV'],
        COLOR_SCHEME['With HAM, OPTV, visit frequency']
    ]
    
    # Style rows
    for i in range(1, len(table_data) + 1):
        row_color = row_colors[i - 1] if (i - 1) < len(row_colors) else 'white'
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor(row_color)
            cell.set_edgecolor('black')
            cell.set_text_props(color='white')
    
    # Save table
    output_path = output_dir / 'quantitative_results_table_baseline_ham_optv_visit_frequency_best_epoch.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none', pad_inches=0)
    plt.close()
    
    print(f"\nSaved: {output_path}")
    
    # Save CSV
    csv_output_path = output_dir / 'quantitative_comparison_table_best_epoch.csv'
    results_df.to_csv(csv_output_path, index=False)
    print(f"Saved CSV: {csv_output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("QUANTITATIVE COMPARISON SUMMARY (Best Epoch Averaged Over All Seeds)")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    return output_path, results_df


# ============================================================================
# QUANTITATIVE COMPARISON BAR CHART - BEST MODEL
# ============================================================================
def create_quantitative_comparison_bar_chart_best_model(csv_path, output_dir):
    """
    Create quantitative comparison bar chart using the best epoch from each study.
    Best epoch is determined by highest average coverage (averaged over all seeds),
    with shortest average duration as tiebreaker.
    Includes training steps needed to reach baseline performance.
    """
    print("\n" + "="*80)
    print("Creating quantitative comparison bar chart (Best Epoch)...")
    print("="*80)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Clean ablation_study_name
    df['ablation_study_name'] = df['ablation_study_name'].str.strip().str.strip('"')
    
    # Calculate episode duration from steps_end * base_timestep (in seconds)
    df['duration_end'] = df['steps_end'] / 10
    
    # Extract epoch number
    df['epoch_num'] = df['epoch'].str.extract(r'epoch-(\d+)').astype(int)
    df['steps'] = df['epoch_num'] * 256  # in thousands of steps
    
    # Filter for Radiation only
    df_radiation = df[~df['ablation_study_name'].str.contains('Lidar', case=False, na=False)].copy()
    df_radiation['ablation_study_name'] = df_radiation['ablation_study_name'].str.replace(r'\s*\(Radiation\)', '', regex=True)
    
    # Explicitly filter for the 4 ablation studies
    studies_to_include = [
        'State-of-the-art baseline',
        'With fisheye',
        'With HAM',
        'With HAM, OPTV',
        'With HAM, OPTV, footprint with visit frequency',
        'With HAM, OPTV, visit frequency'
    ]
    df_radiation = df_radiation[df_radiation['ablation_study_name'].isin(studies_to_include)].copy()
    
    studies = sorted(df_radiation['ablation_study_name'].unique())
    
    # Find best epoch for each study (averaged over all seeds)
    best_epochs = {}
    steps_to_reach_baseline_performance = {}
    
    # First pass: Find best epoch for each study and get baseline performance
    baseline_coverage = None
    for study in studies:
        study_data = df_radiation[df_radiation['ablation_study_name'] == study].copy()
        
        # Group by epoch and calculate average coverage and duration across all seeds and maps
        epoch_stats = study_data.groupby('epoch_num').agg({
            'coverage_end': 'mean',
            'duration_end': 'mean',
            'epoch': 'first'
        }).reset_index()
        
        # Find epoch with maximum coverage, then minimum duration as tiebreaker
        epoch_stats_sorted = epoch_stats.sort_values(['coverage_end', 'duration_end'], 
                                                      ascending=[False, True])
        best_epoch_num = epoch_stats_sorted.iloc[0]['epoch_num']
        best_epochs[study] = best_epoch_num
        
        # Store baseline coverage (from State-of-the-art baseline study)
        if study == 'State-of-the-art baseline':
            baseline_coverage = epoch_stats_sorted.iloc[0]['coverage_end']
    
    # Second pass: Calculate steps to reach baseline performance for each study
    for study in studies:
        study_data = df_radiation[df_radiation['ablation_study_name'] == study].copy()
        
        # Group by epoch and calculate average coverage across all maps and seeds
        epoch_stats = study_data.groupby('epoch_num').agg({
            'coverage_end': 'mean',
            'epoch': 'first'
        }).reset_index().sort_values('epoch_num')
        
        # Find first epoch where coverage >= baseline_coverage
        epochs_meeting_baseline = epoch_stats[epoch_stats['coverage_end'] >= baseline_coverage]
        
        if len(epochs_meeting_baseline) > 0:
            first_epoch_num = epochs_meeting_baseline.iloc[0]['epoch_num']
            steps_to_reach_baseline_performance[study] = first_epoch_num * 256
        else:
            steps_to_reach_baseline_performance[study] = None
    
    # Create results
    results = []
    
    for study in studies:
        best_epoch_num = best_epochs[study]
        # Get data for the best epoch (all seeds and maps)
        study_data = df_radiation[
            (df_radiation['ablation_study_name'] == study) &
            (df_radiation['epoch_num'] == best_epoch_num)
        ].copy()
        
        avg_duration_mean = study_data['duration_end'].mean()
        avg_duration_std = study_data['duration_end'].std()
        
        episodes_99 = study_data[study_data['coverage_end'] >= 0.99]
        percentage_99 = (len(episodes_99) / len(study_data)) * 100 if len(study_data) > 0 else 0
        
        # Calculate success rate std from binary outcomes
        success_binary = (study_data['coverage_end'] >= 0.99).astype(int)
        success_rate_std = success_binary.std() * 100
        
        coverage_mean = study_data['coverage_end'].mean() * 100
        coverage_std = study_data['coverage_end'].std() * 100
        
        # Calculate total degrees turned if available
        if 'total_degrees_turned' in study_data.columns:
            degrees_turned_mean = study_data['total_degrees_turned'].mean()
            degrees_turned_std = study_data['total_degrees_turned'].std()
        else:
            degrees_turned_mean = None
            degrees_turned_std = None
        
        results.append({
            'Study': study,
            'Label': LABEL_NAMES.get(study, study),
            'Avg Duration (mean)': avg_duration_mean,
            'Avg Duration (std)': avg_duration_std,
            'Success Rate (%)': percentage_99,
            'Success Rate (std)': success_rate_std,
            'Coverage (mean)': coverage_mean,
            'Coverage (std)': coverage_std,
            'Total Degrees Turned (mean)': degrees_turned_mean,
            'Total Degrees Turned (std)': degrees_turned_std,
            'Steps to Reach Baseline Performance': steps_to_reach_baseline_performance.get(study, None)
        })
    
    results_df = pd.DataFrame(results)
    
    # Create figure with 1x5 subplots (added total degrees turned plot)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(30, 6))
    
    x_pos = np.arange(len(results_df))
    bar_width = 0.6
    colors = [COLOR_SCHEME.get(study, '#888888') for study in results_df['Study']]
    
    # Plot 1: Final Coverage
    bars1 = ax1.bar(x_pos, results_df['Coverage (mean)'], bar_width,
                     color=colors, alpha=0.8)
    
    ax1.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Final Coverage Achieved', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df['Label'], rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    y_min = (results_df['Coverage (mean)'] - results_df['Coverage (std)']).min()
    y_max = (results_df['Coverage (mean)'] + results_df['Coverage (std)']).max()
    y_padding = (y_max - y_min) * 0.1
    ax1.set_ylim(0, 110)
    
    for bar, mean, std in zip(bars1, results_df['Coverage (mean)'], results_df['Coverage (std)']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f} ± {std:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Success Rate
    bars2 = ax2.bar(x_pos, results_df['Success Rate (%)'], bar_width,
                     color=colors, alpha=0.8)
    
    ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Runs Reaching 99% Coverage', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df['Label'], rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 110)
    
    for bar, mean, std in zip(bars2, results_df['Success Rate (%)'], results_df['Success Rate (std)']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f} ± {std:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Average Duration
    bars3 = ax3.bar(x_pos, results_df['Avg Duration (mean)'], bar_width,
                     color=colors, alpha=0.8)
    
    ax3.set_ylabel('Duration (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Average Duration', fontsize=13, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(results_df['Label'], rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax3.set_axisbelow(True)
    
    y_min = (results_df['Avg Duration (mean)'] - results_df['Avg Duration (std)']).min()
    y_max = (results_df['Avg Duration (mean)'] + results_df['Avg Duration (std)']).max()
    y_padding = (y_max - y_min) * 0.1
    ax3.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
    
    for bar, mean, std in zip(bars3, results_df['Avg Duration (mean)'], results_df['Avg Duration (std)']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.0f} ± {std:.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 4: Total Degrees Turned
    degrees_values = []
    has_degrees_data = False
    for val in results_df['Total Degrees Turned (mean)']:
        if pd.isna(val) or val is None:
            degrees_values.append(0)
        else:
            degrees_values.append(val)
            has_degrees_data = True
    
    if has_degrees_data:
        bars4 = ax4.bar(x_pos, degrees_values, bar_width,
                         color=colors, alpha=0.8)
        
        ax4.set_ylabel('Total Degrees Turned (k)', fontsize=12, fontweight='bold')
        ax4.set_title('Total Degrees Turned', fontsize=13, fontweight='bold', pad=15)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(results_df['Label'], rotation=15, ha='right')
        ax4.grid(True, alpha=0.3, axis='y', linewidth=0.5)
        ax4.set_axisbelow(True)
        
        # Set y-axis limits
        valid_degrees = [v for v in degrees_values if v > 0]
        if valid_degrees:
            y_min = min(valid_degrees)
            y_max = max(valid_degrees)
            y_padding = (y_max - y_min) * 0.1
            ax4.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
        
        for bar, mean_val, std_val in zip(bars4, results_df['Total Degrees Turned (mean)'], results_df['Total Degrees Turned (std)']):
            height = bar.get_height()
            if pd.isna(mean_val) or mean_val is None:
                label_text = 'N/A'
            else:
                label_text = f'{mean_val/1000:.1f}k ± {std_val/1000:.1f}k'
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    label_text,
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    # Plot 5: Training Steps to Reach Baseline Model Performance
    steps_values = []
    for val in results_df['Steps to Reach Baseline Performance']:
        if pd.isna(val) or val is None:
            steps_values.append(0)
        else:
            steps_values.append(val / 1000)  # Convert to millions
    
    bars5 = ax5.bar(x_pos, steps_values, bar_width,
                     color=colors, alpha=0.8)
    
    ax5.set_ylabel('Training Steps', fontsize=12, fontweight='bold')
    ax5.set_title('Steps to Reach Baseline Performance', fontsize=13, fontweight='bold', pad=15)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(results_df['Label'], rotation=15, ha='right')
    ax5.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax5.set_axisbelow(True)
    
    # Set y-axis limits
    max_steps = max([v for v in steps_values if v > 0], default=1)
    ax5.set_ylim(0, max_steps * 1.15)
    
    for bar, value, orig_val in zip(bars5, steps_values, results_df['Steps to Reach Baseline Performance']):
        height = bar.get_height()
        if pd.isna(orig_val) or orig_val is None:
            label_text = 'N/A'
        else:
            label_text = f'{value:.3f} Mio'
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                label_text,
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'success_rate_bar_chart_comparison_all_ablations_best_epoch.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_path}")
    return output_path





# ============================================================================
# ADDITIONAL FIGURES
# ============================================================================
def include_path_and_observations_figure(output_dir):
    """
    Include path and observations figure with three images side by side:
    - Left: Path
    - Middle: Observation with multi-scale maps
    - Right: Observation with HAM
    """
    print("\n" + "="*80)
    print("Including path and observations figure...")
    print("="*80)
    
    # Paths to the three images
    path_img = r"C:\Users\johan\Pictures\path_leading_to_the_observations.png"
    observation_multiscale = r"C:\Users\johan\Pictures\observation_with_multi_scale_maps.png"
    observation_ham = r"C:\Users\johan\Pictures\observation_with_ham.png"
    
    output_paths = []
    
    # Check if all three images exist
    if os.path.exists(path_img) and os.path.exists(observation_multiscale) and os.path.exists(observation_ham):
        # Read all three images
        img1 = plt.imread(path_img)
        img2 = plt.imread(observation_multiscale)
        img3 = plt.imread(observation_ham)
        
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        
        # Calculate aspect ratios
        aspect1 = w1 / h1
        aspect2 = w2 / h2
        aspect3 = w3 / h3
        
        # Create figure with 3 subplots, width ratios based on aspect ratios to maintain equal height
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), 
                                             gridspec_kw={'width_ratios': [aspect1, aspect2, aspect3], 'wspace': 0.02})
        
        # Display images without titles
        ax1.imshow(img1)
        ax1.axis('off')
        
        ax2.imshow(img2)
        ax2.axis('off')
        
        ax3.imshow(img3)
        ax3.axis('off')
        
        # Save figure
        output_path = output_dir / 'agent_path_with_multi_scale_and_ham_observations_comparison.png'
        plt.tight_layout(pad=0.2)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_path}")
        output_paths.append(output_path)
    else:
        if not os.path.exists(path_img):
            print(f"Warning: Path image not found at {path_img}")
        if not os.path.exists(observation_multiscale):
            print(f"Warning: Observation with multi-scale maps image not found at {observation_multiscale}")
        if not os.path.exists(observation_ham):
            print(f"Warning: Observation with HAM image not found at {observation_ham}")
    
    return output_paths if output_paths else None


def include_additional_figures(output_dir):
    """
    Include additional figures with three visit frequency images side by side with equal height
    """
    print("\n" + "="*80)
    print("Including additional figures...")
    print("="*80)
    
    # Paths to the three visit frequency images
    visit_frequency_path1 = r"C:\Users\johan\Pictures\visit frequency path.png"
    visit_frequency_path2 = r"C:\Users\johan\Pictures\visit frequency ohne blue.png"
    visit_frequency_path3 = r"C:\Users\johan\Pictures\visit frequency mit blue.png"
    
    output_paths = []
    
    # Check if all three images exist
    if os.path.exists(visit_frequency_path1) and os.path.exists(visit_frequency_path2) and os.path.exists(visit_frequency_path3):
        # Read all three images
        img1 = plt.imread(visit_frequency_path1)
        img2 = plt.imread(visit_frequency_path2)
        img3 = plt.imread(visit_frequency_path3)
        
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        
        # Calculate aspect ratios
        aspect1 = w1 / h1
        aspect2 = w2 / h2
        aspect3 = w3 / h3
        
        # Create figure with 3 subplots, width ratios based on aspect ratios to maintain equal height
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), 
                                             gridspec_kw={'width_ratios': [aspect1, aspect2, aspect3], 'wspace': 0.02})
        
        # Display images
        ax1.imshow(img1)
        ax1.axis('off')
        
        ax2.imshow(img2)
        ax2.axis('off')
        
        ax3.imshow(img3)
        ax3.axis('off')
        
        # Save figure
        output_path = output_dir / 'agent_stuck_behavior_comparison_with_and_without_visit_frequency_observations.png'
        plt.tight_layout(pad=0.5)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_path}")
        output_paths.append(output_path)
    else:
        if not os.path.exists(visit_frequency_path1):
            print(f"Warning: Visit frequency image 1 not found at {visit_frequency_path1}")
        if not os.path.exists(visit_frequency_path2):
            print(f"Warning: Visit frequency image 2 not found at {visit_frequency_path2}")
        if not os.path.exists(visit_frequency_path3):
            print(f"Warning: Visit frequency image 3 not found at {visit_frequency_path3}")
    
    return output_paths if output_paths else None


# ============================================================================
# ABLATION STUDY DIAGRAMS (from z_create_ablation_diagrams.py)
# ============================================================================
def create_diagrams_on_axes(df_filtered, title_prefix, ax1, ax2, ax3,
                           legend_ax=None, max_steps=None):
    """Create 3 diagrams for the given filtered dataset on provided axes."""
    
    # Print data summary
    print(f"\n{'='*80}")
    print(f"Data Summary for {title_prefix}:")
    print(f"Total rows: {len(df_filtered)}")
    print(f"\nUnique ablation studies:")
    for study in df_filtered['ablation_study_name'].unique():
        print(f"  - {study}")
    print(f"\nEpochs available: {sorted(df_filtered['epoch_num'].unique())}")
    print(f"Steps range: {df_filtered['steps'].min()}k - {df_filtered['steps'].max()}k")
    print(f"\nUnique seeds: {sorted(df_filtered['seed'].unique())}")
    print(f"{'='*80}")
    
    # Define colors for each ablation study
    # Define custom order for studies (instead of alphabetical)
    all_studies = df_filtered['ablation_study_name'].unique()
    study_order = [
        'HOP-CPP w/o VF, OPTV, HM',
        'HOP-CPP w/o VF, OPTV',
        'HOP-CPP w/o VF',
        'HOP-CPP',
    ]
    # Filter to only include studies that exist in the data, preserving the desired order
    studies = [s for s in study_order if s in all_studies]
    
    # Use global color scheme
    color_map = {}
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(studies)))
    for i, study in enumerate(studies):
        color_map[study] = COLOR_SCHEME.get(study, default_colors[i])
    
    print("\n" + "="*80)
    print(f"Creating {title_prefix} ablation study diagrams")
    print("="*80)
    
    # ============================================================================
    # DIAGRAM 1: End coverage percentage vs training progress
    # ============================================================================
    print("\nDiagram 1: End coverage percentage")
    
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        
        # First group by steps and seed to get average across episodes for each seed
        # Then calculate mean and std across seeds
        grouped = study_data.groupby(['steps', 'seed'])['coverage_end'].mean().reset_index()
        stats = grouped.groupby('steps').agg(
            mean=('coverage_end', 'mean'),
            std=('coverage_end', 'std')
        ).reset_index()
        
        # Convert to percentage
        stats['mean_pct'] = stats['mean'] * 100
        stats['std_pct'] = stats['std'] * 100
        
        print(f"{study}: {len(stats)} data points")
        
        label = LABEL_NAMES.get(study, study)
        
        # Apply Gaussian smoothing to all studies
        mean_pct_values = stats['mean_pct'].values
        std_pct_values = stats['std_pct'].values
        if len(mean_pct_values) > 1:
            mean_pct_values = gaussian_filter1d(mean_pct_values, sigma=2.5)
            std_pct_values = gaussian_filter1d(std_pct_values, sigma=2.5)
        
        # Plot mean line
        ax1.plot(stats['steps'], mean_pct_values, 
                 label=label, color=color_map[study],
                 linewidth=2)
        
        # Add shaded region for standard deviation
        ax1.fill_between(stats['steps'], 
                        mean_pct_values - std_pct_values, 
                        mean_pct_values + std_pct_values,
                        color=color_map[study], alpha=0.2)

    ax1.set_xlabel('Train Steps (in Millions)')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title('Final Coverage Achieved')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    if max_steps is not None and max_steps > 0:
        ax1.set_xlim(0, max_steps)
    ax1.set_ylim(-5, 105)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/8000:.1f}' if x > 0 else '0'))
    
    # ============================================================================
    # DIAGRAM 2: Total degrees turned vs training progress
    # ============================================================================
    print("\nDiagram 2: Total degrees turned")
    
    has_degrees_data = 'total_degrees_turned' in df_filtered.columns
    
    if has_degrees_data:
        for study in studies:
            study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
            
            grouped = study_data.groupby(['steps', 'seed'])['total_degrees_turned'].mean().reset_index()
            stats = grouped.groupby('steps').agg(
                mean=('total_degrees_turned', 'mean'),
                std=('total_degrees_turned', 'std')
            ).reset_index()
            
            print(f"{study}: {len(stats)} data points")
            
            label = LABEL_NAMES.get(study, study)
            
            mean_values = stats['mean'].values / 1000  # Convert to thousands
            std_values = stats['std'].values / 1000
            if len(mean_values) > 1:
                mean_values = gaussian_filter1d(mean_values, sigma=2.5)
                nan_mask = np.isnan(std_values)
                std_values = gaussian_filter1d(np.where(nan_mask, 0, std_values), sigma=2.5)
                std_values[nan_mask] = np.nan
            
            ax2.plot(stats['steps'], mean_values,
                     label=label, color=color_map[study], linewidth=2)
            ax2.fill_between(stats['steps'],
                            mean_values - std_values,
                            mean_values + std_values,
                            color=color_map[study], alpha=0.2)
        
        ax2.set_xlabel('Train Steps (in Millions)')
        ax2.set_ylabel('Total Degrees Turned (k)')
        ax2.set_title('Total Degrees Turned')
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        if max_steps is not None and max_steps > 0:
            ax2.set_xlim(0, max_steps)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/8000:.1f}' if x > 0 else '0'))
    else:
        ax2.text(0.5, 0.5, 'Total degrees turned data not available',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    # ============================================================================
    # DIAGRAM 3: Episode Steps (duration_end) vs training progress
    # ============================================================================
    print("\nDiagram 3: Episode Steps")
    
    all_durations = []
    
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        
        grouped = study_data.groupby(['steps', 'seed'])['duration_end'].mean().reset_index()
        stats = grouped.groupby('steps').agg(
            mean=('duration_end', 'mean'),
            std=('duration_end', 'std')
        ).reset_index()
        
        print(f"{study}: {len(stats)} data points")
        
        label = LABEL_NAMES.get(study, study)
        
        mean_values = stats['mean'].values * 10
        std_values = stats['std'].values * 10
        if len(mean_values) > 1:
            mean_values = gaussian_filter1d(mean_values, sigma=2.5)
            nan_mask = np.isnan(std_values)
            std_values = gaussian_filter1d(np.where(nan_mask, 0, std_values), sigma=2.5)
            std_values[nan_mask] = np.nan
        
        all_durations.extend(mean_values)
        all_durations.extend(mean_values + std_values)
        
        ax3.plot(stats['steps'], mean_values,
                 label=label, color=color_map[study], linewidth=2)
        ax3.fill_between(stats['steps'],
                        mean_values - std_values,
                        mean_values + std_values,
                        color=color_map[study], alpha=0.2)
    
    ax3.set_xlabel('Train Steps (in Millions)')
    ax3.set_ylabel('Episode Steps')
    ax3.set_title('Episode Steps')
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    if max_steps is not None and max_steps > 0:
        ax3.set_xlim(0, max_steps)
    
    if len(all_durations) > 0:
        all_durations = np.array(all_durations)
        all_durations = all_durations[~np.isnan(all_durations)]
        if len(all_durations) > 0:
            y_min = np.min(all_durations)
            y_max = np.max(all_durations)
            y_padding = (y_max - y_min) * 0.05
            ax3.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
    
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/8000:.1f}' if x > 0 else '0'))
    
    # ============================================================================
    # Add legend if legend_ax is provided
    # ============================================================================
    if legend_ax is not None:
        handles, labels = ax1.get_legend_handles_labels()
        legend_ax.axis('off')
        legend_ax.legend(handles, labels, loc='center', ncol=len(handles), frameon=True,
                        framealpha=0.9, edgecolor='gray')
    
    return studies, color_map


def create_ablation_study_diagrams(csv_path, output_dir):
    """Create ablation study diagrams from CSV data"""
    print("\n" + "="*80)
    print("Creating ablation study diagrams...")
    print("="*80)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Clean ablation_study_name: remove quotes if present
    df['ablation_study_name'] = df['ablation_study_name'].str.strip().str.strip('"')
    # Remove "(Radiation)" suffix before name mapping
    df['ablation_study_name'] = df['ablation_study_name'].str.replace(r'\s*\(Radiation\)', '', regex=True)
    # Apply name mapping
    df['ablation_study_name'] = df['ablation_study_name'].replace(CSV_NAME_MAPPING)
    
    # Calculate episode duration
    df['duration_end'] = df['steps_end'] / 10
    
    # Extract epoch number and convert to steps
    df['epoch_num'] = df['epoch'].str.extract(r'epoch-(\d+)').astype(int)
    df['steps'] = df['epoch_num'] * 256  # in thousands of steps
    
    # Filter to the 4 incremental ablation studies
    studies_to_include = ['HOP-CPP w/o VF, OPTV, HM', 'HOP-CPP w/o VF, OPTV', 'HOP-CPP w/o VF', 'HOP-CPP']
    df_radiation = df[df['ablation_study_name'].isin(studies_to_include)].copy()
    
    # Find global max steps across all data for consistent x-axis
    global_max_steps = df_radiation['steps'].max()
    if pd.isna(global_max_steps) or global_max_steps <= 0:
        print(f"Warning: Invalid max_steps ({global_max_steps}). Setting to None for auto-scaling.")
        global_max_steps = None
    
    # Create figure with 2 rows: plots on top, legend below
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.15], hspace=0.4, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    legend_ax = fig.add_subplot(gs[1, :])
    
    create_diagrams_on_axes(df_radiation, 'Radiation', ax1, ax2, ax3, legend_ax,
                            max_steps=global_max_steps)
    
    # Save figure
    output_path = output_dir / 'ablation_study_learning_curves_baseline_ham_optv_visit_frequency_all_maps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def create_ablation_study_diagrams_per_map(csv_path, output_dir, map_name):
    """Create ablation study diagrams for a specific map from CSV data"""
    print("\n" + "="*80)
    print(f"Creating ablation study diagrams for {map_name}...")
    print("="*80)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Filter by map
    df = df[df['map'] == map_name].copy()
    
    if len(df) == 0:
        print(f"Warning: No data found for map {map_name}")
        return None
    
    print(f"Found {len(df)} rows for map {map_name}")
    
    # Clean ablation_study_name: remove quotes if present
    df['ablation_study_name'] = df['ablation_study_name'].str.strip().str.strip('"')
    df['ablation_study_name'] = df['ablation_study_name'].str.replace(r'\s*\(Radiation\)', '', regex=True)
    df['ablation_study_name'] = df['ablation_study_name'].replace(CSV_NAME_MAPPING)
    
    # Calculate episode duration
    df['duration_end'] = df['steps_end'] / 10
    
    # Extract epoch number and convert to steps
    df['epoch_num'] = df['epoch'].str.extract(r'epoch-(\d+)').astype(int)
    df['steps'] = df['epoch_num'] * 256  # in thousands of steps
    
    # Find global max steps
    global_max_steps = df['steps'].max()
    if pd.isna(global_max_steps) or global_max_steps <= 0:
        print(f"Warning: Invalid max_steps ({global_max_steps}). Setting to None for auto-scaling.")
        global_max_steps = None
    
    # Create figure with 2 rows: plots on top, legend below
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.15], hspace=0.4, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    legend_ax = fig.add_subplot(gs[1, :])
    
    create_diagrams_on_axes(df, f'Radiation - {map_name}', ax1, ax2, ax3, legend_ax,
                            max_steps=global_max_steps)
    
    # Save figure
    output_path = output_dir / f'ablation_study_learning_curves_baseline_ham_optv_visit_frequency_{map_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def create_comparison_diagram_transformer_impact(csv_path, output_dir):
    """Create comparison diagram for transformer impact: 
    SAC Baseline + Transformer vs SAC Baseline + Transformer + HAM + OPTV + Visit Frequency
    Light red vs purple color scheme."""
    print("\n" + "="*80)
    print("Creating transformer impact comparison diagram...")
    print("="*80)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Clean ablation_study_name: remove quotes if present
    df['ablation_study_name'] = df['ablation_study_name'].str.strip().str.strip('"')
    
    # Calculate episode duration from steps_end * base_timestep (in seconds)
    df['duration_end'] = df['steps_end'] / 10
    
    # Extract epoch number and convert to steps
    df['epoch_num'] = df['epoch'].str.extract(r'epoch-(\d+)').astype(int)
    df['steps'] = df['epoch_num'] * 256  # in thousands of steps
    
    # Filter for the two studies we want
    studies_to_include = [
        'SOTA + TRANSFORMER',
        'SAC + OUR METHOD + TRANSFORMER'
    ]
    df_filtered = df[df['ablation_study_name'].isin(studies_to_include)].copy()
    
    # Find global max steps
    global_max_steps = df_filtered['steps'].max()
    if pd.isna(global_max_steps) or global_max_steps <= 0:
        global_max_steps = None
    
    # Create figure
    fig = plt.figure(figsize=(20, 4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.2], hspace=0.3, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    legend_ax = fig.add_subplot(gs[0, 4])
    
    # Custom color map: light red and purple
    color_map = {
        'SOTA + TRANSFORMER': '#ff8a8a',  # light red
        'SAC + OUR METHOD + TRANSFORMER': '#9370db'  # purple
    }
    
    # Custom label names
    label_names = {
        'SOTA + TRANSFORMER': 'SOTA + Transformer',
        'SAC + OUR METHOD + TRANSFORMER': 'SAC + Our Method + Transformer'
    }
    
    # Create diagrams (reusing logic from create_diagrams_on_axes)
    studies = df_filtered['ablation_study_name'].unique()
    
    # Diagram 1: End coverage percentage
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        grouped = study_data.groupby(['steps', 'seed'])['coverage_end'].mean().reset_index()
        stats = grouped.groupby('steps').agg(
            mean=('coverage_end', 'mean'),
            std=('coverage_end', 'std')
        ).reset_index()
        
        stats['mean_pct'] = stats['mean'] * 100
        stats['std_pct'] = stats['std'] * 100
        
        label = label_names.get(study, study)
        mean_pct_values = gaussian_filter1d(stats['mean_pct'].values, sigma=2.5) if len(stats) > 1 else stats['mean_pct'].values
        std_pct_values = gaussian_filter1d(stats['std_pct'].values, sigma=2.5) if len(stats) > 1 else stats['std_pct'].values
        
        ax1.plot(stats['steps'], mean_pct_values, label=label, color=color_map[study], linewidth=2)
        ax1.fill_between(stats['steps'], mean_pct_values - std_pct_values, 
                        mean_pct_values + std_pct_values, color=color_map[study], alpha=0.2)
    
    ax1.set_xlabel('Training Steps (in Millions)')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title('Final Coverage Achieved')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    if global_max_steps: ax1.set_xlim(0, global_max_steps)
    ax1.set_ylim(-5, 105)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Diagram 2: Percentage reaching 99% coverage
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        grouped = study_data.groupby(['steps', 'seed']).agg(
            total_runs=('coverage_end', 'count'),
            runs_above_99=('coverage_end', lambda x: (x >= 0.99).sum())
        ).reset_index()
        grouped['percentage_99'] = (grouped['runs_above_99'] / grouped['total_runs']) * 100
        stats = grouped.groupby('steps').agg(mean=('percentage_99', 'mean'), std=('percentage_99', 'std')).reset_index()
        
        label = label_names.get(study, study)
        mean_values = gaussian_filter1d(stats['mean'].values, sigma=2.5) if len(stats) > 1 else stats['mean'].values
        std_values = gaussian_filter1d(stats['std'].values, sigma=2.5) if len(stats) > 1 else stats['std'].values
        
        ax2.plot(stats['steps'], mean_values, label=label, color=color_map[study], linewidth=2)
        ax2.fill_between(stats['steps'], mean_values - std_values, mean_values + std_values, 
                        color=color_map[study], alpha=0.2)
    
    ax2.set_xlabel('Training Steps (in Millions)')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Runs Reaching 99% Coverage')
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    if global_max_steps: ax2.set_xlim(0, global_max_steps)
    ax2.set_ylim(-5, 105)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Diagram 3: Total degrees turned
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        grouped = study_data.groupby(['steps', 'seed'])['total_degrees_turned'].mean().reset_index()
        stats = grouped.groupby('steps').agg(mean=('total_degrees_turned', 'mean'),
                                            std=('total_degrees_turned', 'std')).reset_index()

        label = label_names.get(study, study)
        mean_values = gaussian_filter1d(stats['mean'].values / 1000, sigma=2.5) if len(stats) > 1 else stats['mean'].values / 1000
        std_values = gaussian_filter1d(stats['std'].values / 1000, sigma=2.5) if len(stats) > 1 else stats['std'].values / 1000

        ax3.plot(stats['steps'], mean_values, label=label, color=color_map[study], linewidth=2)
        ax3.fill_between(stats['steps'], mean_values - std_values, mean_values + std_values,
                       color=color_map[study], alpha=0.2)

    ax3.set_xlabel('Training Steps (in Millions)')
    ax3.set_ylabel('Total Degrees Turned (k)')
    ax3.set_title('Total Degrees Turned')
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    if global_max_steps: ax3.set_xlim(0, global_max_steps)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Diagram 4: Duration
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        grouped = study_data.groupby(['steps', 'seed'])['duration_end'].mean().reset_index()
        stats = grouped.groupby('steps').agg(mean=('duration_end', 'mean'), 
                                            std=('duration_end', 'std')).reset_index()
        
        label = label_names.get(study, study)
        mean_values = gaussian_filter1d(stats['mean'].values, sigma=2.5) if len(stats) > 1 else stats['mean'].values
        std_values = gaussian_filter1d(stats['std'].values, sigma=2.5) if len(stats) > 1 else stats['std'].values
        
        ax4.plot(stats['steps'], mean_values, label=label, color=color_map[study], linewidth=2)
        ax4.fill_between(stats['steps'], mean_values - std_values, mean_values + std_values,
                        color=color_map[study], alpha=0.2)
    
    ax4.set_xlabel('Training Steps (in Millions)')
    ax4.set_ylabel('Duration (seconds)')
    ax4.set_title('Duration')
    ax4.grid(True, alpha=0.3, linewidth=0.5)
    if global_max_steps: ax4.set_xlim(0, global_max_steps)
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='center left', frameon=True, framealpha=0.9, edgecolor='gray')
    
    # Save figure
    output_path = output_dir / 'transformer_impact_even_with_transformer_our_tricks_have_significant_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def create_comparison_diagram_ham_impact(csv_path, output_dir):
    """Create comparison diagram for HAM impact with jonnarth's parameters:
    SAC Baseline vs SAC Baseline + HAM
    Light red vs purple color scheme."""
    print("\n" + "="*80)
    print("Creating HAM impact comparison diagram...")
    print("="*80)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Clean ablation_study_name
    df['ablation_study_name'] = df['ablation_study_name'].str.strip().str.strip('"')
    
    # Calculate episode duration from steps_end * base_timestep (in seconds)
    df['duration_end'] = df['steps_end'] / 10
    
    # Extract epoch number and convert to steps
    df['epoch_num'] = df['epoch'].str.extract(r'epoch-(\d+)').astype(int)
    df['steps'] = df['epoch_num'] * 256
    
    # Filter for the two studies we want
    studies_to_include = [
        'SAC Baseline (Radiation)',
        'SAC Baseline + HAM (Radiation)'
    ]
    df_filtered = df[df['ablation_study_name'].isin(studies_to_include)].copy()
    
    # Remove "(Radiation)" from names
    df_filtered['ablation_study_name'] = df_filtered['ablation_study_name'].str.replace(r'\s*\(Radiation\)', '', regex=True)
    
    global_max_steps = df_filtered['steps'].max()
    if pd.isna(global_max_steps) or global_max_steps <= 0:
        global_max_steps = None
    
    # Create figure
    fig = plt.figure(figsize=(20, 4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.2], hspace=0.3, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    legend_ax = fig.add_subplot(gs[0, 4])
    
    # Custom color map: light red and purple
    color_map = {
        'SAC Baseline': '#ff8a8a',  # light red
        'SAC Baseline + HAM': '#9370db'  # purple
    }
    
    label_names = {
        'SAC Baseline': 'SAC Baseline',
        'SAC Baseline + HAM': 'SAC Baseline + HAM'
    }
    
    studies = df_filtered['ablation_study_name'].unique()
    
    # Diagram 1: End coverage percentage
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        grouped = study_data.groupby(['steps', 'seed'])['coverage_end'].mean().reset_index()
        stats = grouped.groupby('steps').agg(mean=('coverage_end', 'mean'), std=('coverage_end', 'std')).reset_index()
        
        stats['mean_pct'] = stats['mean'] * 100
        stats['std_pct'] = stats['std'] * 100
        
        label = label_names.get(study, study)
        mean_pct_values = gaussian_filter1d(stats['mean_pct'].values, sigma=2.5) if len(stats) > 1 else stats['mean_pct'].values
        std_pct_values = gaussian_filter1d(stats['std_pct'].values, sigma=2.5) if len(stats) > 1 else stats['std_pct'].values
        
        ax1.plot(stats['steps'], mean_pct_values, label=label, color=color_map[study], linewidth=2)
        ax1.fill_between(stats['steps'], mean_pct_values - std_pct_values, 
                        mean_pct_values + std_pct_values, color=color_map[study], alpha=0.2)
    
    ax1.set_xlabel('Training Steps (in Millions)')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title('Final Coverage Achieved')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    if global_max_steps: ax1.set_xlim(0, global_max_steps)
    ax1.set_ylim(-5, 105)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Diagram 2: Percentage reaching 99% coverage
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        grouped = study_data.groupby(['steps', 'seed']).agg(
            total_runs=('coverage_end', 'count'),
            runs_above_99=('coverage_end', lambda x: (x >= 0.99).sum())
        ).reset_index()
        grouped['percentage_99'] = (grouped['runs_above_99'] / grouped['total_runs']) * 100
        stats = grouped.groupby('steps').agg(mean=('percentage_99', 'mean'), std=('percentage_99', 'std')).reset_index()
        
        label = label_names.get(study, study)
        mean_values = gaussian_filter1d(stats['mean'].values, sigma=2.5) if len(stats) > 1 else stats['mean'].values
        std_values = gaussian_filter1d(stats['std'].values, sigma=2.5) if len(stats) > 1 else stats['std'].values
        
        ax2.plot(stats['steps'], mean_values, label=label, color=color_map[study], linewidth=2)
        ax2.fill_between(stats['steps'], mean_values - std_values, mean_values + std_values, 
                        color=color_map[study], alpha=0.2)
    
    ax2.set_xlabel('Training Steps (in Millions)')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Runs Reaching 99% Coverage')
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    if global_max_steps: ax2.set_xlim(0, global_max_steps)
    ax2.set_ylim(-5, 105)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Diagram 3: Total degrees turned
    has_degrees_data = 'total_degrees_turned' in df_filtered.columns
    if has_degrees_data:
        for study in studies:
            study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
            grouped = study_data.groupby(['steps', 'seed'])['total_degrees_turned'].mean().reset_index()
            stats = grouped.groupby('steps').agg(mean=('total_degrees_turned', 'mean'), 
                                                std=('total_degrees_turned', 'std')).reset_index()
            
            label = label_names.get(study, study)
            mean_values = gaussian_filter1d(stats['mean'].values / 1000, sigma=2.5) if len(stats) > 1 else stats['mean'].values / 1000
            std_values = gaussian_filter1d(stats['std'].values / 1000, sigma=2.5) if len(stats) > 1 else stats['std'].values / 1000
            
            ax3.plot(stats['steps'], mean_values, label=label, color=color_map[study], linewidth=2)
            ax3.fill_between(stats['steps'], mean_values - std_values, mean_values + std_values,
                           color=color_map[study], alpha=0.2)
        
        ax3.set_xlabel('Training Steps (in Millions)')
        ax3.set_ylabel('Total Degrees Turned (k)')
        ax3.set_title('Total Degrees Turned')
        ax3.grid(True, alpha=0.3, linewidth=0.5)
        if global_max_steps: ax3.set_xlim(0, global_max_steps)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Diagram 4: Duration
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        grouped = study_data.groupby(['steps', 'seed'])['duration_end'].mean().reset_index()
        stats = grouped.groupby('steps').agg(mean=('duration_end', 'mean'), 
                                            std=('duration_end', 'std')).reset_index()
        
        label = label_names.get(study, study)
        mean_values = gaussian_filter1d(stats['mean'].values, sigma=2.5) if len(stats) > 1 else stats['mean'].values
        std_values = gaussian_filter1d(stats['std'].values, sigma=2.5) if len(stats) > 1 else stats['std'].values
        
        ax4.plot(stats['steps'], mean_values, label=label, color=color_map[study], linewidth=2)
        ax4.fill_between(stats['steps'], mean_values - std_values, mean_values + std_values,
                        color=color_map[study], alpha=0.2)
    
    ax4.set_xlabel('Training Steps (in Millions)')
    ax4.set_ylabel('Duration (seconds)')
    ax4.set_title('Duration')
    ax4.grid(True, alpha=0.3, linewidth=0.5)
    if global_max_steps: ax4.set_xlim(0, global_max_steps)
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='center left', frameon=True, framealpha=0.9, edgecolor='gray')
    
    # Save figure
    output_path = output_dir / 'ham_impact_even_with_jonnarths_parameters_our_ham_yields_better_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def create_comparison_diagram_directional_vs_nondirectional(csv_path, output_dir):
    """Create comparison diagram for directional vs non-directional:
    TD3 Baseline (Lidar) vs State-of-the-art baseline (Radiation)
    Light red vs purple color scheme."""
    print("\n" + "="*80)
    print("Creating directional vs non-directional comparison diagram...")
    print("="*80)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Clean ablation_study_name
    df['ablation_study_name'] = df['ablation_study_name'].str.strip().str.strip('"')
    
    # Calculate episode duration from steps_end * base_timestep (in seconds)
    df['duration_end'] = df['steps_end'] / 10
    
    # Extract epoch number and convert to steps
    df['epoch_num'] = df['epoch'].str.extract(r'epoch-(\d+)').astype(int)
    df['steps'] = df['epoch_num'] * 256
    
    # Filter for the two studies we want
    studies_to_include = [
        'TD3 Baseline (Lidar)',
        'State-of-the-art baseline (Radiation)'
    ]
    df_filtered = df[df['ablation_study_name'].isin(studies_to_include)].copy()
    
    # Remove "(Lidar)" and "(Radiation)" from names
    df_filtered['ablation_study_name'] = df_filtered['ablation_study_name'].str.replace(r'\s*\((Lidar|Radiation)\)', '', regex=True)
    
    global_max_steps = df_filtered['steps'].max()
    if pd.isna(global_max_steps) or global_max_steps <= 0:
        global_max_steps = None
    
    # Create figure
    fig = plt.figure(figsize=(20, 4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.2], hspace=0.3, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    legend_ax = fig.add_subplot(gs[0, 4])
    
    # Custom color map: light red and purple
    color_map = {
        'TD3 Baseline': '#ff8a8a',  # light red
        'State-of-the-art baseline': '#9370db'  # purple
    }
    
    label_names = {
        'TD3 Baseline': 'TD3 Baseline (Lidar)',
        'State-of-the-art baseline': 'State-of-the-art baseline (Radiation)'
    }
    
    studies = df_filtered['ablation_study_name'].unique()
    
    # Diagram 1: End coverage percentage
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        grouped = study_data.groupby(['steps', 'seed'])['coverage_end'].mean().reset_index()
        stats = grouped.groupby('steps').agg(mean=('coverage_end', 'mean'), std=('coverage_end', 'std')).reset_index()
        
        stats['mean_pct'] = stats['mean'] * 100
        stats['std_pct'] = stats['std'] * 100
        
        label = label_names.get(study, study)
        mean_pct_values = gaussian_filter1d(stats['mean_pct'].values, sigma=2.5) if len(stats) > 1 else stats['mean_pct'].values
        std_pct_values = gaussian_filter1d(stats['std_pct'].values, sigma=2.5) if len(stats) > 1 else stats['std_pct'].values
        
        ax1.plot(stats['steps'], mean_pct_values, label=label, color=color_map[study], linewidth=2)
        ax1.fill_between(stats['steps'], mean_pct_values - std_pct_values, 
                        mean_pct_values + std_pct_values, color=color_map[study], alpha=0.2)
    
    ax1.set_xlabel('Training Steps (in Millions)')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title('Final Coverage Achieved')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    if global_max_steps: ax1.set_xlim(0, global_max_steps)
    ax1.set_ylim(-5, 105)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Diagram 2: Percentage reaching 99% coverage
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        grouped = study_data.groupby(['steps', 'seed']).agg(
            total_runs=('coverage_end', 'count'),
            runs_above_99=('coverage_end', lambda x: (x >= 0.99).sum())
        ).reset_index()
        grouped['percentage_99'] = (grouped['runs_above_99'] / grouped['total_runs']) * 100
        stats = grouped.groupby('steps').agg(mean=('percentage_99', 'mean'), std=('percentage_99', 'std')).reset_index()
        
        label = label_names.get(study, study)
        mean_values = gaussian_filter1d(stats['mean'].values, sigma=2.5) if len(stats) > 1 else stats['mean'].values
        std_values = gaussian_filter1d(stats['std'].values, sigma=2.5) if len(stats) > 1 else stats['std'].values
        
        ax2.plot(stats['steps'], mean_values, label=label, color=color_map[study], linewidth=2)
        ax2.fill_between(stats['steps'], mean_values - std_values, mean_values + std_values, 
                        color=color_map[study], alpha=0.2)
    
    ax2.set_xlabel('Training Steps (in Millions)')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Runs Reaching 99% Coverage')
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    if global_max_steps: ax2.set_xlim(0, global_max_steps)
    ax2.set_ylim(-5, 105)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Diagram 3: Total degrees turned
    has_degrees_data = 'total_degrees_turned' in df_filtered.columns
    if has_degrees_data:
        for study in studies:
            study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
            grouped = study_data.groupby(['steps', 'seed'])['total_degrees_turned'].mean().reset_index()
            stats = grouped.groupby('steps').agg(mean=('total_degrees_turned', 'mean'), 
                                                std=('total_degrees_turned', 'std')).reset_index()
            
            label = label_names.get(study, study)
            mean_values = gaussian_filter1d(stats['mean'].values / 1000, sigma=2.5) if len(stats) > 1 else stats['mean'].values / 1000
            std_values = gaussian_filter1d(stats['std'].values / 1000, sigma=2.5) if len(stats) > 1 else stats['std'].values / 1000
            
            ax3.plot(stats['steps'], mean_values, label=label, color=color_map[study], linewidth=2)
            ax3.fill_between(stats['steps'], mean_values - std_values, mean_values + std_values,
                           color=color_map[study], alpha=0.2)
        
        ax3.set_xlabel('Training Steps (in Millions)')
        ax3.set_ylabel('Total Degrees Turned (k)')
        ax3.set_title('Total Degrees Turned')
        ax3.grid(True, alpha=0.3, linewidth=0.5)
        if global_max_steps: ax3.set_xlim(0, global_max_steps)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Diagram 4: Duration
    for study in studies:
        study_data = df_filtered[df_filtered['ablation_study_name'] == study].copy()
        grouped = study_data.groupby(['steps', 'seed'])['duration_end'].mean().reset_index()
        stats = grouped.groupby('steps').agg(mean=('duration_end', 'mean'), 
                                            std=('duration_end', 'std')).reset_index()
        
        label = label_names.get(study, study)
        mean_values = gaussian_filter1d(stats['mean'].values, sigma=2.5) if len(stats) > 1 else stats['mean'].values
        std_values = gaussian_filter1d(stats['std'].values, sigma=2.5) if len(stats) > 1 else stats['std'].values
        
        ax4.plot(stats['steps'], mean_values, label=label, color=color_map[study], linewidth=2)
        ax4.fill_between(stats['steps'], mean_values - std_values, mean_values + std_values,
                        color=color_map[study], alpha=0.2)
    
    ax4.set_xlabel('Training Steps (in Millions)')
    ax4.set_ylabel('Duration (seconds)')
    ax4.set_title('Duration')
    ax4.grid(True, alpha=0.3, linewidth=0.5)
    if global_max_steps: ax4.set_xlim(0, global_max_steps)
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}' if x > 0 else '0'))
    
    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='center left', frameon=True, framealpha=0.9, edgecolor='gray')
    
    # Save figure
    output_path = output_dir / 'directional_vs_nondirectional_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


# ============================================================================
# BENCHMARK PATH IMAGES COMBINED FIGURE
# ============================================================================
def create_benchmark_path_images_combined(folders_dict, output_dir):
    """
    Create a single figure with all ablation study paths arranged in a 6x4 grid.
    Methods are shown on top with color labels, maps in rows.
    
    Args:
        folders_dict: Dictionary mapping folder names to folder paths
        output_dir: Directory to save output figures
        
    Returns:
        Path to generated figure (single image)
    """
    import matplotlib.image as mpimg
    
    # Map folder names to display names
    study_names = {
        'benchmark_results_baseline': 'Baseline',
        'benchmark_results_fisheye': 'HAM',
        'benchmark_results_fisheye_optv': 'HAM + OPTV',
        'benchmark_results_fisheye_optv_visit_frequ': 'HAM + OPTV + Visit Frequency'
    }
    
    # Map display names to full names for color scheme lookup
    name_to_full = {
        'Baseline': 'State-of-the-art baseline',
        'HAM': 'With HAM',
        'HAM + OPTV': 'With HAM, OPTV',
        'HAM + OPTV + Visit Frequency': 'With HAM, OPTV, visit frequency'
    }
    
    # Collect all image data organized by study
    all_data = []
    
    for folder_name, folder_path in folders_dict.items():
        print(f"\nProcessing folder: {folder_name}")
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder_path}")
            continue
        
        # Get all PNG images in the folder, sorted
        image_files = sorted(folder_path.glob("*.png"))
        
        if len(image_files) != 6:
            print(f"Warning: Expected 6 images in {folder_name}, found {len(image_files)}")
        
        if len(image_files) == 0:
            print(f"No images found in {folder_path}")
            continue
        
        # Take first 6 images
        image_files = image_files[:6]
        
        display_name = study_names.get(folder_name, folder_name)
        full_name = name_to_full.get(display_name, display_name)
        
        all_data.append({
            'study_name': display_name,
            'images': image_files,
            'color': COLOR_SCHEME.get(full_name, '#888888')
        })
    
    if not all_data:
        print("No images found to combine")
        return None
    
    # Create figure with 6 rows x 4 columns + 1 row for labels on top
    # Grid layout: 7 rows (1 label row + 6 map rows) x 4 columns (methods)
    num_methods = len(all_data)
    num_maps = 6
    fig = plt.figure(figsize=(20, 30))
    
    # Create grid: 7 rows (1 for labels + 6 for maps) x 4 columns (methods)
    gs = fig.add_gridspec(num_maps + 1, num_methods, height_ratios=[0.08] + [1] * num_maps,
                          hspace=0.02, wspace=0.02)
    
    # Add method labels on top with colored background
    from matplotlib.patches import Rectangle
    for col_idx, data in enumerate(all_data):
        ax_label = fig.add_subplot(gs[0, col_idx])
        ax_label.set_facecolor(data['color'])
        ax_label.text(0.5, 0.5, data['study_name'], 
                     rotation=0, va='center', ha='center',
                     fontsize=12, fontweight='bold',
                     color='white')
        ax_label.axis('off')
        # Add a colored patch that fills the entire axes
        rect = Rectangle((0, 0), 1, 1, transform=ax_label.transAxes, 
                        facecolor=data['color'], edgecolor='none', zorder=-1)
        ax_label.add_patch(rect)
    
    # Add images in grid: rows = maps (6), columns = methods (4)
    for col_idx, data in enumerate(all_data):
        for row_idx, img_file in enumerate(data['images']):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])
            img = mpimg.imread(img_file)
            ax.imshow(img)
            ax.axis('off')
    
    # Save figure
    output_filename = "Paths for all eval maps and ablation study combinations.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()
    
    print(f"\nSaved combined figure: {output_path}")
    return output_path


# ============================================================================
# FIRST FIGURE: SECOND IMAGE FROM BASELINE AND HAM+OPTV+VISIT FREQ
# ============================================================================
def create_first_figure_with_two_images(folders_dict, output_dir):
    """
    Create first figure with second image from Baseline and second image from 
    HAM + OPTV + Visit Frequency side by side.
    
    Args:
        folders_dict: Dictionary mapping folder names to folder paths
        output_dir: Directory to save output figures
        
    Returns:
        Path to generated figure
    """
    import matplotlib.image as mpimg
    
    print("\n" + "="*80)
    print("Creating first figure with two comparison images...")
    print("="*80)
    
    # Get the specific folders we need
    baseline_folder = Path(folders_dict['benchmark_results_baseline'])
    ham_optv_vf_folder = Path(folders_dict['benchmark_results_fisheye_optv_visit_frequ'])
    
    if not baseline_folder.exists():
        print(f"Warning: Baseline folder not found: {baseline_folder}")
        return None
    
    if not ham_optv_vf_folder.exists():
        print(f"Warning: HAM+OPTV+VF folder not found: {ham_optv_vf_folder}")
        return None
    
    # Get second image from each folder (index 1)
    baseline_images = sorted(baseline_folder.glob("*.png"))
    ham_images = sorted(ham_optv_vf_folder.glob("*.png"))
    
    if len(baseline_images) < 2:
        print(f"Warning: Not enough images in baseline folder. Found {len(baseline_images)}")
        return None
    
    if len(ham_images) < 2:
        print(f"Warning: Not enough images in HAM folder. Found {len(ham_images)}")
        return None
    
    baseline_img_path = baseline_images[1]  # Second image
    ham_img_path = ham_images[1]    # Second image
    
    print(f"Using baseline image: {baseline_img_path.name}")
    print(f"Using HAM+OPTV+VF image: {ham_img_path.name}")
    
    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0.02, left=0.02, right=0.98, top=0.98, bottom=0.02)
    
    # Display images
    img1 = mpimg.imread(baseline_img_path)
    ax1.imshow(img1)
    ax1.axis('off')
    
    img2 = mpimg.imread(ham_img_path)
    ax2.imshow(img2)
    ax2.axis('off')
    
    # Save figure
    output_filename = "Comparison_Baseline_vs_HAM_OPTV_Visit_Frequency_map_2.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()
    
    print(f"\nSaved first comparison figure: {output_path}")
    return output_path


# ============================================================================
# HTML VIEWER FOR ALL GENERATED FIGURES
# ============================================================================
def create_html_viewer(image_paths, output_dir):
    """Create HTML file to display all generated figures"""
    print("\n" + "="*80)
    print("Creating HTML viewer...")
    print("="*80)
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ICAPS Paper Visualizations</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #000000;
            color: #ffffff;
        }
        h1 {
            color: #ffffff;
            text-align: center;
            border-bottom: 3px solid #30638e;
            padding-bottom: 10px;
        }
        .figure-container {
            background-color: #1a1a1a;
            margin: 30px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(255,255,255,0.1);
            border: 1px solid #333;
        }
        .figure-title {
            color: #30638e;
            font-size: 1.3em;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .figure-image {
            width: 100%;
            height: auto;
            border: 1px solid #444;
            border-radius: 4px;
        }
        .color-legend {
            margin: 20px 0;
            padding: 15px;
            background-color: #1a1a1a;
            border-left: 4px solid #30638e;
            border: 1px solid #333;
        }
        .color-item {
            display: inline-block;
            margin: 5px 15px 5px 0;
        }
        .color-box {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
            border: 1px solid #666;
        }
    </style>
</head>
<body>
    <h1>ICAPS Paper Visualizations</h1>
    
    <div class="color-legend">
        <strong>Color Scheme:</strong><br>
        <div class="color-item">
            <span class="color-box" style="background-color: #d1495b;"></span>
            Baseline
        </div>
        <div class="color-item">
            <span class="color-box" style="background-color: #edae49;"></span>
            HAM
        </div>
        <div class="color-item">
            <span class="color-box" style="background-color: #00798c;"></span>
            OPTV
        </div>
        <div class="color-item">
            <span class="color-box" style="background-color: #30638e;"></span>
            Visit Frequency
        </div>
    </div>
"""
    
    # Add each figure
    for i, img_path in enumerate(image_paths, 1):
        img_name = Path(img_path).name
        
        # For agent behavior comparison, display without text
        if 'agent_behavior_comparison' in img_name:
            html_content += f"""
    <div class="figure-container">
        <img src="{img_path}" alt="Agent Behavior Comparison" class="figure-image">
    </div>
"""
        else:
            title = img_name.replace('_', ' ').replace('.png', '').title()
            html_content += f"""
    <div class="figure-container">
        <div class="figure-title">Figure {i}: {title}</div>
        <img src="{img_path}" alt="{title}" class="figure-image">
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Save HTML file
    html_path = output_dir / 'icaps_visualizations.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Saved HTML viewer: {html_path}")
    return html_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*80)
    print("ICAPS PAPER VISUALIZATION SCRIPT")
    print("="*80)
    
    # Setup paths
    main_folder = os.path.dirname(os.path.abspath(__file__))
    output_dir = Path(r'C:\Users\johan\programming\Simulation\ICAPS_Paper\AuthorKit26-1\AuthorKit26\AnonymousSubmission\LaTeX\figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Check if multi-scale and HAM PNG already exists
    multi_scale_ham_png = output_dir / 'observation_space_multi_scale_grid_levels_and_ham_distortion_comparison.png'
    skip_gymenv_init = multi_scale_ham_png.exists()
    
    if skip_gymenv_init:
        print("\nMulti-scale and HAM PNG already exists. Skipping gymenv initialization.")
        env, gymenv = None, None
    else:
        # Initialize gym environment
        env, gymenv = init_gymenv()
    
    # Store paths of all generated images
    generated_images = []
    
    # Define benchmark folders for reuse
    benchmark_folders = {
        'benchmark_results_baseline': Path(main_folder) / 'misc' / 'logs' / 'benchmark_results' / 'images_of_paths' / 'Archive' / 'benchmark_results_baseline',
        'benchmark_results_fisheye': Path(main_folder) / 'misc' / 'logs' / 'benchmark_results' / 'images_of_paths' / 'Archive' / 'benchmark_results_fisheye',
        'benchmark_results_fisheye_optv': Path(main_folder) / 'misc' / 'logs' / 'benchmark_results' / 'images_of_paths' / 'Archive' / 'benchmark_results_fisheye_optv',
        'benchmark_results_fisheye_optv_visit_frequ': Path(main_folder) / 'misc' / 'logs' / 'benchmark_results' / 'images_of_paths' / 'Archive' / 'benchmark_results_fisheye_optv_visit_frequ'
    }
    
    # 0. Create agent behavior comparison figure (FIRST FIGURE)
    try:
        img_path = create_agent_behavior_comparison_figure(output_dir, benchmark_folders)
        if img_path:
            generated_images.append(img_path)
    except Exception as e:
        print(f"Error creating agent behavior comparison figure: {e}")
        import traceback
        traceback.print_exc()
    
    # 1. Create observation explanation figure (SECOND FIGURE)
    try:
        img_path = create_observation_explanation_figure(output_dir)
        if img_path:
            generated_images.append(img_path)
    except Exception as e:
        print(f"Error creating observation explanation figure: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Create HAM and multi-scale grids figure
    if not skip_gymenv_init:
        try:
            img_path = create_ham_and_multiscale_grids_figure(gymenv, output_dir)
            generated_images.append(img_path)
        except Exception as e:
            print(f"Error creating HAM and multi-scale grids figure: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Add existing PNG to generated images list
        if multi_scale_ham_png.exists():
            generated_images.append(str(multi_scale_ham_png))
    
    # 3. Include path and observations figure (SECOND FIGURE)
    try:
        path_obs_img_paths = include_path_and_observations_figure(output_dir)
        if path_obs_img_paths:
            generated_images.extend(path_obs_img_paths)
    except Exception as e:
        print(f"Error including path and observations figure: {e}")
        import traceback
        traceback.print_exc()

    # 4. Create OPTV explanation figures
    try:
        img_paths = create_optv_explanation_figure(output_dir)
        if img_paths:
            generated_images.extend(img_paths)
    except Exception as e:
        print(f"Error creating OPTV explanation figures: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Include additional figures (visit frequency and final solution)
    try:
        additional_img_paths = include_additional_figures(output_dir)
        if additional_img_paths:
            generated_images.extend(additional_img_paths)
    except Exception as e:
        print(f"Error including additional figures: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Create ablation study diagrams
    csv_path = Path(r'C:\Users\johan\programming\HOP-CPP (State Of Sweep 59etj5gd)\Simulation\ablation_studies_evaluation_dataset copy.csv')
    if csv_path.exists():
        try:
            img_path = create_ablation_study_diagrams(csv_path, output_dir)
            generated_images.append(img_path)
        except Exception as e:
            print(f"Error creating ablation diagrams: {e}")
    else:
        print(f"Warning: CSV file not found: {csv_path}")
    
    # 7. Create quantitative comparison table (best epoch)
    if csv_path.exists():
        try:
            table_path, results_df = create_quantitative_comparison_table_best_model(csv_path, output_dir)
            generated_images.append(table_path)
        except Exception as e:
            print(f"Error creating quantitative comparison table (best epoch): {e}")
            import traceback
            traceback.print_exc()
    
    # 8. Create quantitative comparison bar chart (best epoch)
    if csv_path.exists():
        try:
            bar_chart_path = create_quantitative_comparison_bar_chart_best_model(csv_path, output_dir)
            generated_images.append(bar_chart_path)
        except Exception as e:
            print(f"Error creating quantitative comparison bar chart (best epoch): {e}")
            import traceback
            traceback.print_exc()
    
    # 9. Create ablation study diagrams for each map
    # Maps 9, 10-14 removed as requested
    if csv_path.exists():
        maps = []  # All maps removed
        for map_name in maps:
            try:
                img_path = create_ablation_study_diagrams_per_map(csv_path, output_dir, map_name)
                if img_path:
                    generated_images.append(img_path)
            except Exception as e:
                print(f"Error creating ablation diagrams for {map_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # 9a. Create comparison diagram: Transformer impact
    if csv_path.exists():
        try:
            img_path = create_comparison_diagram_transformer_impact(csv_path, output_dir)
            generated_images.append(img_path)
        except Exception as e:
            print(f"Error creating transformer impact comparison diagram: {e}")
            import traceback
            traceback.print_exc()
    
    # 9b. Create comparison diagram: HAM impact
    if csv_path.exists():
        try:
            img_path = create_comparison_diagram_ham_impact(csv_path, output_dir)
            generated_images.append(img_path)
        except Exception as e:
            print(f"Error creating HAM impact comparison diagram: {e}")
            import traceback
            traceback.print_exc()
    
    # 9c. Create comparison diagram: Directional vs non-directional
    if csv_path.exists():
        try:
            img_path = create_comparison_diagram_directional_vs_nondirectional(csv_path, output_dir)
            generated_images.append(img_path)
        except Exception as e:
            print(f"Error creating directional vs non-directional comparison diagram: {e}")
            import traceback
            traceback.print_exc()
    
    # 10. Create first figure with two comparison images
    try:
        first_fig_path = create_first_figure_with_two_images(benchmark_folders, output_dir)
        if first_fig_path:
            generated_images.append(first_fig_path)
    except Exception as e:
        print(f"Error creating first comparison figure: {e}")
        import traceback
        traceback.print_exc()
    
    # 11. Create benchmark path images combined figures (6x4 grid)
    try:
        combined_img_path = create_benchmark_path_images_combined(benchmark_folders, output_dir)
        if combined_img_path:
            generated_images.append(combined_img_path)
    except Exception as e:
        print(f"Error creating benchmark path images combined: {e}")
        import traceback
        traceback.print_exc()
    
    # 12. Create HTML viewer and open it
    if generated_images:
        html_path = create_html_viewer(generated_images, output_dir)
        
        # Open in browser
        print(f"\nOpening visualizations in browser...")
        webbrowser.open(f'file://{html_path}')
    else:
        print("\nNo images were generated.")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
    
    # Example usage of get_best_models_info function
    # Uncomment to test:
    # csv_path = Path(__file__).parent / 'misc' / 'logs' / 'benchmark_results' / 'ablation_studies_evaluation_files.csv'
    # if csv_path.exists():
    #     sweep_ids, model_numbers = get_best_models_info(csv_path)
    #     print("\n" + "="*80)
    #     print("BEST MODEL INFORMATION")
    #     print("="*80)
    #     print(f"Sweep IDs: {sweep_ids}")
    #     print(f"Model Numbers (Epochs): {model_numbers}")
    #     print("="*80)



