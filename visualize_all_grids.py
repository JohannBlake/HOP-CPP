import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from matplotlib.backend_bases import MouseButton

# Configuration
OBSERVED_IMAGE_SIZE = 32
METERS_PER_PIXEL = 0.0375

def generate_flat_distorted_grid(center_x, center_y, n=OBSERVED_IMAGE_SIZE):
    """
    Generate a fisheye distorted grid in flat coordinate space.
    Recreated from class_gymenv.py without dependencies.
    """
    range_pixels = 40  # Range for the grid in pixels
    
    # === TUNABLE FISHEYE PARAMETERS ===
    threshold_radius = 0.426692    # Radius where scaling starts (0.0 to 1.0)
    linear_scale_factor = 52.174442    # Multiplier for scaling intensity
    distortion_exponent = 3.072314    # Exponent for smooth scaling (1.0 = linear, >1.0 = super-linear)
    
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


def generate_flat_grid(center_x, center_y, grid_size_in_pixels, n=OBSERVED_IMAGE_SIZE):
    """
    Generate a regular flat grid in image pixel coordinates.
    Recreated from class_gymenv.py without dependencies.
    """
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


def create_multi_scale_grids(center_x, center_y):
    """
    Create 4 grids with equal pixel spacing at different scales.
    Recreated from class_gymenv.py without dependencies.
    """
    input_size = OBSERVED_IMAGE_SIZE
    scale_factor = 4
    meters_per_pixel = METERS_PER_PIXEL
    
    # Define 4 different grid sizes in pixels
    multi_scale_grid_sizes_px = [input_size * (scale_factor ** n) for n in range(4)]
    multi_scale_grid_sizes_m = [s * meters_per_pixel for s in multi_scale_grid_sizes_px]
    
    # Create grids at each scale with resolution matching input_size
    n = input_size
    multi_scale_grids = []
    
    for grid_size_px in multi_scale_grid_sizes_px:
        grid = generate_flat_grid(center_x, center_y, grid_size_px, n)
        multi_scale_grids.append(grid)
    
    return multi_scale_grids, multi_scale_grid_sizes_px, multi_scale_grid_sizes_m


def visualize_all_grids():
    """
    Visualize fisheye and all 4 multi-scale grids in a single image.
    """
    # Set center position (arbitrary for visualization)
    center_x, center_y = 200, 200
    
    # Generate fisheye grid with 64 points
    fisheye_grid = generate_flat_distorted_grid(center_x, center_y, n=64)
    
    # Generate multi-scale grids with 32 points
    multi_scale_grids, grid_sizes_px, grid_sizes_m = create_multi_scale_grids(center_x, center_y)
    
    # Create figure with 5 subplots (1 fisheye + 4 multi-scale)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Colors for different grids
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # Plot fisheye grid
    ax = axes[0]
    x_coords = fisheye_grid[:, :, 0]
    y_coords = fisheye_grid[:, :, 1]
    
    # Plot grid points
    ax.scatter(x_coords.flatten(), y_coords.flatten(), c=colors[0], s=1, alpha=0.6)
    
    # Plot grid lines
    for i in range(fisheye_grid.shape[0]):
        ax.plot(x_coords[i, :], y_coords[i, :], color=colors[0], alpha=0.3, linewidth=0.5)
    for j in range(fisheye_grid.shape[1]):
        ax.plot(x_coords[:, j], y_coords[:, j], color=colors[0], alpha=0.3, linewidth=0.5)
    
    # Mark center
    ax.plot(center_x, center_y, 'k*', markersize=15, label='Center')
    
    ax.set_title(f'Fisheye Grid\n(distorted, {OBSERVED_IMAGE_SIZE}x{OBSERVED_IMAGE_SIZE} points)', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.axis('equal')
    
    # Plot each multi-scale grid
    for idx, (grid, size_px, size_m) in enumerate(zip(multi_scale_grids, grid_sizes_px, grid_sizes_m)):
        ax = axes[idx + 1]
        x_coords = grid[:, :, 0]
        y_coords = grid[:, :, 1]
        
        # Plot grid points
        ax.scatter(x_coords.flatten(), y_coords.flatten(), c=colors[idx + 1], s=1, alpha=0.6)
        
        # Plot grid lines
        for i in range(grid.shape[0]):
            ax.plot(x_coords[i, :], y_coords[i, :], color=colors[idx + 1], alpha=0.3, linewidth=0.5)
        for j in range(grid.shape[1]):
            ax.plot(x_coords[:, j], y_coords[:, j], color=colors[idx + 1], alpha=0.3, linewidth=0.5)
        
        # Mark center
        ax.plot(center_x, center_y, 'k*', markersize=15, label='Center')
        
        ax.set_title(f'Multi-Scale Grid {idx + 1}\n({size_px}x{size_px} px, {size_m:.2f}m, {OBSERVED_IMAGE_SIZE}x{OBSERVED_IMAGE_SIZE} points)', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.axis('equal')
    
    # Hide the last empty subplot
    axes[5].axis('off')
    
    # Create overlay plot showing all grids together
    fig2, ax_overlay = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot fisheye
    x_coords = fisheye_grid[:, :, 0]
    y_coords = fisheye_grid[:, :, 1]
    ax_overlay.scatter(x_coords.flatten(), y_coords.flatten(), c=colors[0], s=0.4, alpha=0.4, label='Fisheye')
    
    # Plot multi-scale grids
    for idx, (grid, size_px) in enumerate(zip(multi_scale_grids, grid_sizes_px)):
        x_coords = grid[:, :, 0]
        y_coords = grid[:, :, 1]
        ax_overlay.scatter(x_coords.flatten(), y_coords.flatten(), 
                          c=colors[idx + 1], s=0.4, alpha=0.4, 
                          label=f'Multi-Scale {idx + 1} ({size_px}px)')
    
    ax_overlay.set_title('All Grids Overlaid', fontsize=14, fontweight='bold')
    ax_overlay.set_xlabel('X (pixels)')
    ax_overlay.set_ylabel('Y (pixels)')
    ax_overlay.legend(fontsize=10)
    ax_overlay.grid(True, alpha=0.2)
    ax_overlay.axis('equal')
    
    # Enable interactive zoom/pan
    plt.tight_layout()
    
    # Add mouse wheel zoom functionality
    def on_scroll(event):
        if event.inaxes != ax_overlay:
            return
        
        # Get current axis limits
        xlim = ax_overlay.get_xlim()
        ylim = ax_overlay.get_ylim()
        
        # Calculate zoom factor
        zoom_factor = 1.2 if event.button == 'down' else 1/1.2
        
        # Get mouse position in data coordinates
        xdata, ydata = event.xdata, event.ydata
        
        # Calculate new limits centered on mouse position
        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor
        
        new_xlim = [xdata - (xdata - xlim[0]) * zoom_factor,
                    xdata + (xlim[1] - xdata) * zoom_factor]
        new_ylim = [ydata - (ydata - ylim[0]) * zoom_factor,
                    ydata + (ylim[1] - ydata) * zoom_factor]
        
        ax_overlay.set_xlim(new_xlim)
        ax_overlay.set_ylim(new_ylim)
        fig2.canvas.draw_idle()
    
    # Connect scroll event
    fig2.canvas.mpl_connect('scroll_event', on_scroll)
    
    plt.tight_layout()
    
    # Save figure (only overlay)
    fig2.savefig('all_grids_overlay.png', dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close the separate grids figure without showing
    
    print("\n" + "="*80)
    print("GRID VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nFisheye Grid:")
    print(f"  - Resolution: 64x64 points")
    print(f"  - Distortion: threshold_radius=0.5, linear_scale_factor=18.0")
    print(f"\nMulti-Scale Grids:")
    for idx, (size_px, size_m) in enumerate(zip(grid_sizes_px, grid_sizes_m)):
        print(f"  Grid {idx + 1}: {size_px}x{size_px} pixels ({size_m:.2f}m) with {OBSERVED_IMAGE_SIZE}x{OBSERVED_IMAGE_SIZE} points")
    print(f"\nSaved:")
    print(f"  - all_grids_overlay.png (all grids overlaid)")
    print(f"\nInteractive controls:")
    print(f"  - Mouse wheel: Zoom in/out")
    print(f"  - Click and drag: Pan")
    print(f"  - Home button (toolbar): Reset view")
    print("="*80 + "\n")
    
    plt.show()


if __name__ == "__main__":
    visualize_all_grids()
