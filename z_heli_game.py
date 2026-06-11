
import os
import warnings

# Control variables for warnings and prompts
suppress_warnings = False  # Set to True to suppress all warnings
hide_pygame_prompt = True  # Set to False to show pygame support prompt

if suppress_warnings:
    warnings.filterwarnings("ignore")
if hide_pygame_prompt:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import numpy as np
import pygame
from misc.aid.visualization import append_output_gymenv_values
from get_parameters import para
import class_gymenv
from class_gymenv import *
import misc.aid.create_height_map
import matplotlib
matplotlib.use('Agg')
import cv2
from misc.aid.helpful_geo_functions import compute_position_from_geo_coordinate

display_radiation_grid = True
show_odl_map = True  # Toggle between ODL map and target area visualization
last_image_path = None  # Track when the image changes to update ODL visualization
height_data_path = os.path.join(os.path.dirname(__file__), 'misc', 'geo_data','height_data', 'height_data.npy')
height_data = np.load(height_data_path) 

# Control variables
control_interval_ms = 50   # Reduced interval for more responsive controls
action_step_size    = 0.3  # Step size for actions
max_action_speed    = 1.0  # Maximum action value
acceleration_rate   = 0.15 # How quickly actions build up (increased for more responsiveness)
deceleration_rate   = 0.12 # How quickly actions decay when not pressed
max_action_speed    = 1.0 # Maximum action value
acceleration_rate   = 0.1 # How quickly actions build up
deceleration_rate   = 0.2 # How quickly actions decay when not pressed

def generate_paper_visualizations(env, obs, step_number, output_dir="paper_vis"):
    """Generate 3 paper-style visualization images at every 10th timestep.
    
    Image 1 (State): Environment map + flown path + HM grid overlay
    Image 2 (Internal State): Covered area + past positions + radiation measurements + HM grid (yellow)
    Image 3 (Observed Image): The 3-channel observation as the network sees it
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current HM grid in world coordinates (already rotated for current bearing)
    hm_grid = env.list_of_observation_grids[0]  # shape (64, 64, 2) - world pixel coords
    # Apply offset so grid is centered on heli position
    grid_shape = hm_grid.shape[:2]
    gc_row = (grid_shape[0] - 1) / 2.0
    gc_col = (grid_shape[1] - 1) / 2.0
    r_f, r_c = int(np.floor(gc_row)), int(np.ceil(gc_row))
    c_f, c_c = int(np.floor(gc_col)), int(np.ceil(gc_col))
    if r_f == r_c and c_f == c_c:
        grid_center = hm_grid[r_f, c_f]
    else:
        rw = gc_row - r_f
        cw = gc_col - c_f
        grid_center = ((1-rw)*(1-cw)*hm_grid[r_f, c_f] + (1-rw)*cw*hm_grid[r_f, c_c] +
                        rw*(1-cw)*hm_grid[r_c, c_f] + rw*cw*hm_grid[r_c, c_c])
    offset = env.position - grid_center
    hm_world = hm_grid + offset  # (64,64,2) in world pixel coords
    hm_points = hm_world.reshape(-1, 2)  # (4096, 2)
    
    # =====================================================================
    # IMAGE 1: STATE - environment map + covered area
    # =====================================================================
    S = 4  # upscale factor for high-res output
    base_img = env.current_image_for_movement_restriction
    if base_img is not None:
        img1 = cv2.cvtColor(base_img.copy(), cv2.COLOR_GRAY2RGB)
        img1 = cv2.resize(img1, (img1.shape[1] * S, img1.shape[0] * S), interpolation=cv2.INTER_NEAREST)
    else:
        h, w = 200 * S, 200 * S
        img1 = np.ones((h, w, 3), dtype=np.uint8) * 200
    
    # Draw covered area
    if hasattr(env, 'target_area') and env.target_area is not None and not env.target_area.is_empty:
        polys = [env.target_area] if not hasattr(env.target_area, 'geoms') else list(env.target_area.geoms)
        img1_orig = img1.copy()
        for poly in polys:
            exterior = (np.array(poly.exterior.coords) * S).astype(np.int32)
            cv2.fillPoly(img1, [exterior], (0, 180, 0))
            for interior in poly.interiors:
                hole = (np.array(interior.coords) * S).astype(np.int32)
                hole_mask = np.zeros(img1.shape[:2], dtype=np.uint8)
                cv2.fillPoly(hole_mask, [hole], 255)
                img1[hole_mask == 255] = img1_orig[hole_mask == 255]
    
    # Draw 360-degree lidar rays (every 10 deg) stopping at obstacles
    px, py = int(round(env.position[0] * S)), int(round(env.position[1] * S))
    obstacle_img = base_img if base_img is not None else None
    if obstacle_img is not None:
        oh, ow = obstacle_img.shape[:2]
        max_ray_len = max(oh, ow) * S
        for angle_deg in range(0, 360, 10):
            angle_rad = np.radians(angle_deg)
            dx = np.cos(angle_rad)
            dy = -np.sin(angle_rad)  # image y is flipped
            # Step along ray in original (unscaled) pixel coords
            ex, ey = env.position[0], env.position[1]
            hit = False
            for t in range(1, max(oh, ow)):
                rx = ex + dx * t
                ry = ey - dy * t  # undo flip for image coords
                ix, iy = int(round(rx)), int(round(ry))
                if ix < 0 or ix >= ow or iy < 0 or iy >= oh:
                    rx, ry = ex + dx * (t - 1), ey - dy * (t - 1)
                    hit = True
                    break
                if obstacle_img[iy, ix] == 0:
                    hit = True
                    break
            if not hit:
                rx, ry = ex + dx * max(oh, ow), ey - dy * max(oh, ow)
            end_x, end_y = int(round(rx * S)), int(round(ry * S))
            cv2.line(img1, (px, py), (end_x, end_y), (0, 0, 255), max(1, S // 3), cv2.LINE_AA)
    
    # Draw agent position + heading arrow
    draw_arrow(img1, px, py, -env.current_bearing + 180, arrow_length=15 * S, color=(255, 0, 0), thickness=max(1, 2 * S))
    
    cv2.imwrite(os.path.join(output_dir, f"step_{step_number:05d}_1_state.png"), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    
    # =====================================================================
    # IMAGE 2: INTERNAL STATE - covered area + past positions + radiation + HM overlay
    # =====================================================================
    # Determine canvas size from the base image (at high-res scale S)
    if base_img is not None:
        canvas_h, canvas_w = base_img.shape[0] * S, base_img.shape[1] * S
    else:
        canvas_h, canvas_w = 200 * S, 200 * S
    img2 = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    
    # Draw covered area polygon (solid green, same as state image)
    if hasattr(env, 'target_area') and env.target_area is not None and not env.target_area.is_empty:
        polys = [env.target_area] if not hasattr(env.target_area, 'geoms') else list(env.target_area.geoms)
        for poly in polys:
            exterior = (np.array(poly.exterior.coords) * S).astype(np.int32)
            cv2.fillPoly(img2, [exterior], (0, 180, 0))
            for interior in poly.interiors:
                hole = (np.array(interior.coords) * S).astype(np.int32)
                cv2.fillPoly(img2, [hole], (255, 255, 255))
    
    # Draw past positions with radiation measurements (black dots, 2x size)
    if hasattr(env, 'observed_radiation_points') and env.observed_radiation_points is not None and len(env.observed_radiation_points) > 0:
        rad_pts = env.observed_radiation_points  # (N, 3) - x, y, radiation_value
        for i in range(len(rad_pts)):
            rx, ry = int(round(rad_pts[i, 0] * S)), int(round(rad_pts[i, 1] * S))
            if 0 <= rx < canvas_w and 0 <= ry < canvas_h:
                cv2.circle(img2, (rx, ry), max(2, S), (0, 0, 0), -1)
    
    # Draw visited positions as individual points
    if hasattr(env, 'position_history') and len(env.position_history) > 0:
        for pos in env.position_history:
            ppx, ppy = int(round(pos[0] * S)), int(round(pos[1] * S))
            if 0 <= ppx < canvas_w and 0 <= ppy < canvas_h:
                cv2.circle(img2, (ppx, ppy), max(2, int(S * 1.65)), (30, 100, 255), -1)
    
    
    # Draw HM as dots (every 2nd row/col for 1/4 density)
    for row in range(0, hm_world.shape[0], 2):
        for col in range(0, hm_world.shape[1], 2):
            px_hm = int(round(hm_world[row, col, 0] * S))
            py_hm = int(round(hm_world[row, col, 1] * S))
            if 0 <= px_hm < canvas_w and 0 <= py_hm < canvas_h:
                cv2.circle(img2, (px_hm, py_hm), max(1, int(S * 0.845)), (160, 125, 0), -1)
    
    # Draw agent (same red arrow as state image)
    px, py = int(round(env.position[0] * S)), int(round(env.position[1] * S))
    draw_arrow(img2, px, py, -env.current_bearing + 180, arrow_length=15 * S, color=(255, 0, 0), thickness=max(1, 2 * S))
    
    cv2.imwrite(os.path.join(output_dir, f"step_{step_number:05d}_2_internal_state.png"), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    
    # =====================================================================
    # IMAGE 3: OBSERVED IMAGE - the 3-channel observation the network sees
    # =====================================================================
    img3 = None
    if obs is not None:
        obs_np = obs
        import torch as _torch
        if isinstance(obs_np, _torch.Tensor):
            obs_np = obs_np.cpu().numpy()
        if isinstance(obs_np, dict):
            obs_np = obs_np.get('image', obs_np)
        
        # obs_np is (64, 64, C) with C=2 or C=3, values in uint8
        obs_display = ensure_three_channels(obs_np)
        # Upscale for visibility (64 -> 1024)
        scale = 16
        img3 = cv2.resize(obs_display, (obs_display.shape[1] * scale, obs_display.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, f"step_{step_number:05d}_3_observed.png"), cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
    
    # =====================================================================
    # COMBINED IMAGE: all 3 side-by-side with titles
    # =====================================================================
    titles = ["Environment state", "Internal state of agent + HM", "Observation for policy"]
    panels = [img1, img2]
    if img3 is not None:
        panels.append(img3)
    else:
        titles = titles[:2]
    
    # Resize all panels to same height
    target_h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != target_h:
            scale_f = target_h / p.shape[0]
            new_w = int(p.shape[1] * scale_f)
            p = cv2.resize(p, (new_w, target_h), interpolation=cv2.INTER_NEAREST)
        resized.append(p)
    
    # Title bar height (2x bigger)
    title_h = max(80, target_h // 5)
    gap = 4  # gap between panels
    total_w = sum(p.shape[1] for p in resized) + gap * (len(resized) - 1)
    combined = np.full((target_h + title_h, total_w, 3), 255, dtype=np.uint8)
    
    # Use Computer Modern Roman font (same as ECML/LNCS paper) via PIL
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    cm_font_path = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', 'cmr10.ttf')
    
    x_offset = 0
    for i, (panel, title) in enumerate(zip(resized, titles)):
        pw = panel.shape[1]
        title_lines = title.split('\n')
        longest_title = max(title_lines, key=len)
        # Fit font size to panel width
        font_size = 60
        while font_size > 8:
            pil_font = ImageFont.truetype(cm_font_path, font_size)
            bbox = pil_font.getbbox(longest_title)
            tw = bbox[2] - bbox[0]
            if tw <= int(pw * 0.95):
                break
            font_size -= 1
        # Render title lines onto combined image via PIL
        pil_img = Image.fromarray(combined)
        draw = ImageDraw.Draw(pil_img)
        single_h = pil_font.getbbox('Tg')[3] - pil_font.getbbox('Tg')[1]
        line_spacing = int(single_h * 0.3)
        n_lines = len(title_lines)
        total_text_h = single_h * n_lines + line_spacing * (n_lines - 1)
        y_start = (title_h - total_text_h) // 2
        for li, tline in enumerate(title_lines):
            bbox_line = pil_font.getbbox(tline)
            tw2 = bbox_line[2] - bbox_line[0]
            tx = x_offset + (pw - tw2) // 2
            ty = y_start + li * (single_h + line_spacing)
            draw.text((tx, ty), tline, fill=(0, 0, 0), font=pil_font)
        combined = np.array(pil_img)
        # Place panel
        combined[title_h:title_h + target_h, x_offset:x_offset + pw] = panel
        x_offset += pw + gap
    
    cv2.imwrite(os.path.join(output_dir, f"step_{step_number:05d}_combined.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

metric_data = {}
if para.training_library == 'sb3':
    gymenv = class_gymenv.GymnasiumEnv(height_data, radiation_grid_visualization = display_radiation_grid)
    # Test the image_path parameter with a specific training image
    test_image_path = os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "train_4_1.png")
    gymenv.reset()
elif para.training_library == 'omnisafe':
    omnisafe_env = GymEnvOmniSafe(radiation_grid_visualization= display_radiation_grid,is_evaluation = False) # can add is_evaluation = True
    gymenv = omnisafe_env._env
    # Test the image_path parameter with a specific training image
    test_image_path = os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_14.png")
    omnisafe_env.reset(options={'image_path': test_image_path})
    print("max_episode_steps:", omnisafe_env._env.max_flight_time/para.base_timestep)
metric_data = {}
# print gymenv observation space infos

#gymenv.set_map_tier(map_tier)

# Helper function to get the actual environment (handles both wrapped and unwrapped)
def get_env():
    """Get the actual environment, handling both wrapped and direct cases"""
    if para.training_library == 'sb3':
        return gymenv if not hasattr(gymenv, 'unwrapped') else gymenv.unwrapped
    elif para.training_library == 'omnisafe':
        env = omnisafe_env._env
        # Unwrap through AutoResetWrapper etc. to reach GymnasiumEnv
        while hasattr(env, 'env'):
            env = env.env
        return env

terminated = False
action = 0.0  # Single scalar action for turning angle
smooth_action = 0.0  # Smooth action value that builds up gradually
ball_x = 0.5  # Ball position (0.0 to 1.0), controlled with arrow keys
ball_y = 0.5
ball_step_size = 0.05  # How much ball moves per key press
ball_heaviness = getattr(para, 'ball_heaviness', 10.0)  # From parameters_default.yaml
ball_tension = getattr(para, 'ball_tension', 2.0)        # From parameters_default.yaml
input_mode = None      # None, 'heaviness', or 'tension' — for number input
input_buffer = ''      # Accumulates typed digits/dots
def ensure_three_channels(image):
    # Ensure we're working with a numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    if image.shape[-1] == 2:  # If the image has 2 channels
        return np.stack((image[:, :, 0], image[:, :, 1], np.zeros_like(image[:, :, 0])), axis=-1)
    return image  # If already 3 channels, return as is

def draw_arrow(img, center_x, center_y, bearing_deg, arrow_length=15, color=(255, 0, 0), thickness=2):
    """
    Draw an arrow pointing in the direction of the bearing.

    Parameters:
    - img: image array to draw on
    - center_x, center_y: center position of the arrow
    - bearing_deg: bearing in degrees (0 = North, 90 = East, etc.)
    - arrow_length: length of the arrow
    - color: RGB color tuple
    - thickness: line thickness
    """
    # Convert bearing to math angle (0 = East, 90 = North, etc.)
    # Bearing: 0=North, 90=East, 180=South, 270=West
    # Math angle: 0=East, 90=North, 180=West, 270=South
    math_angle = (90 - bearing_deg) % 360
    angle_rad = np.radians(math_angle)
    
    # Calculate arrow end point
    end_x = int(center_x + arrow_length * np.cos(angle_rad))
    end_y = int(center_y - arrow_length * np.sin(angle_rad))  # minus because image y increases downward
    
    # Draw main arrow line
    cv2.line(img, (center_x, center_y), (end_x, end_y), color, thickness)
    
    # Draw arrowhead
    # Calculate arrowhead points
    arrowhead_length = arrow_length * 0.3
    arrowhead_angle = 25  # degrees
    
    # Left arrowhead line
    left_angle = math_angle + 180 - arrowhead_angle
    left_angle_rad = np.radians(left_angle)
    left_x = int(end_x + arrowhead_length * np.cos(left_angle_rad))
    left_y = int(end_y - arrowhead_length * np.sin(left_angle_rad))
    cv2.line(img, (end_x, end_y), (left_x, left_y), color, thickness)
    
    # Right arrowhead line
    right_angle = math_angle + 180 + arrowhead_angle
    right_angle_rad = np.radians(right_angle)
    right_x = int(end_x + arrowhead_length * np.cos(right_angle_rad))
    right_y = int(end_y - arrowhead_length * np.sin(right_angle_rad))
    cv2.line(img, (end_x, end_y), (right_x, right_y), color, thickness)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1000, 600), pygame.RESIZABLE)
pygame.display.set_caption("Game")
window_width, window_height = screen.get_size()

# Function to render metrics as text on the Pygame window
def render_metrics(screen, metric_data, image_width, window_height):
    font_size = max(12, 18)  # Adjust font size based on window height
    font = pygame.font.SysFont(None, font_size)
    y_offset = 20
    for key, value in metric_data.items():
        text = f'{key}: {value[-1] if isinstance(value, list) else value}'
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(text_surface, (image_width + 640, y_offset))  # Display text to the right of the image
        y_offset += (font_size + 5)

print("Starting game.")
running = True
paused = False
last_control_time = pygame.time.get_ticks()
keys_held = {'a': 0, 'd': 0, 'w': 0, 's': 0, 'o': 0, 'shift_a': 0, 'shift_d': 0}
was_using_shift_keys = [False]  # Track if we were using shift keys (now single value for single action)
sampling_grid_points = None  # Sampling grid for drop_one_ball_experiment overlay

while running:
    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                # Check if shift is held for direct action
                if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]:
                    keys_held['shift_a'] = current_time
                else:
                    keys_held['a'] = current_time
            elif event.key == pygame.K_d:
                # Check if shift is held for direct action
                if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]:
                    keys_held['shift_d'] = current_time
                else:
                    keys_held['d'] = current_time
            elif event.key == pygame.K_w:
                keys_held['w'] = current_time
            elif event.key == pygame.K_s:
                keys_held['s'] = current_time
            elif event.key == pygame.K_o:
                keys_held['o'] = current_time
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:  # Press '+' to increase control interval
                control_interval_ms -= 20
            elif event.key == pygame.K_MINUS:  # Press '-' to decrease control interval
                control_interval_ms += 20
            elif event.key == pygame.K_q:  # Press 'q' to quit the game loop
                running = False
            elif event.key == pygame.K_SPACE:  # Press 'space' to pause/resume
                paused = not paused
            elif event.key == pygame.K_p:  # Press 'p' to toggle radiation grid display
                display_radiation_grid = not display_radiation_grid
            elif event.key == pygame.K_LEFT:
                ball_x = max(0.0, ball_x - ball_step_size)
            elif event.key == pygame.K_RIGHT:
                ball_x = min(1.0, ball_x + ball_step_size)
            elif event.key == pygame.K_UP:
                ball_y = max(0.0, ball_y - ball_step_size)
            elif event.key == pygame.K_DOWN:
                ball_y = min(1.0, ball_y + ball_step_size)
            elif event.key == pygame.K_h and input_mode is None:
                input_mode = 'heaviness'
                input_buffer = ''
            elif event.key == pygame.K_t and input_mode is None:
                input_mode = 'tension'
                input_buffer = ''
            elif input_mode is not None:
                if event.key == pygame.K_RETURN:
                    try:
                        val = float(input_buffer)
                        if input_mode == 'heaviness':
                            ball_heaviness = val
                        elif input_mode == 'tension':
                            ball_tension = val
                        # Update env for current session only (no file write-back)
                        _env = get_env()
                        _env.ball_heaviness = ball_heaviness
                        _env.ball_tension = ball_tension
                        print(f'Updated: heaviness={ball_heaviness}, tension={ball_tension}')
                    except ValueError:
                        print(f'Invalid number: {input_buffer}')
                    input_mode = None
                    input_buffer = ''
                elif event.key == pygame.K_ESCAPE:
                    input_mode = None
                    input_buffer = ''
                elif event.key == pygame.K_BACKSPACE:
                    input_buffer = input_buffer[:-1]
                else:
                    ch = event.unicode
                    if ch in '0123456789.':
                        input_buffer += ch
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_a:
                keys_held['a'] = 0
                keys_held['shift_a'] = 0
            elif event.key == pygame.K_d:
                keys_held['d'] = 0
                keys_held['shift_d'] = 0
            elif event.key == pygame.K_w:
                keys_held['w'] = 0
            elif event.key == pygame.K_s:
                keys_held['s'] = 0
            elif event.key == pygame.K_o:
                keys_held['o'] = 0
            elif event.key == pygame.K_q:  # Press 'q' to quit the game loop
                running = False
            elif event.key == pygame.K_r:  # Press 'r' to reset the environment
                # Render the environment state before resetting
                if para.training_library == 'sb3':
                    gymenv.render()
                elif para.training_library == 'omnisafe':
                    omnisafe_env._env.render()
                
                # Reset the environment
                if para.training_library == 'omnisafe':
                    omnisafe_env.reset()
                else:
                    gymenv.reset()
        elif event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            window_width, window_height = event.w, event.h

    # --- Draw target area polygons every frame (independent of radiation grid) ---
    target_area_img = None
    
    # Handle 'o' key for fast ODL map toggling
    if keys_held['o'] and current_time - keys_held['o'] > 100:  # Toggle every 100ms while held
        show_odl_map = not show_odl_map
        keys_held['o'] = current_time  # Reset timer for next toggle
    
    if para.z_rad_type == 'jon':
        try:
            # Get the actual environment
            env = get_env()
            
            # Reuse the cached grayscale image the environment already uses for ODL generation
            img_gray = env.current_image_for_movement_restriction
            if img_gray is None:
                try:
                    img_gray = gymenv._get_cached_image(env.IMAGE_PATH)
                except TypeError:
                    img_gray = None
            if img_gray is not None:
                # Work on a copy to keep the cached image untouched and convert to RGB for drawing
                img_rgb = cv2.cvtColor(img_gray.copy(), cv2.COLOR_GRAY2RGB)
                
                # Draw target area polygons on the image
                img_height, img_width = img_rgb.shape[:2]
                
                # Handle both Polygon and MultiPolygon cases
                if hasattr(env.target_area, 'geoms'):
                    # MultiPolygon case
                    polygons = env.target_area.geoms
                else:
                    # Single Polygon case
                    polygons = [env.target_area]
                
                for polygon in polygons:
                    # Get exterior coordinates of the polygon
                    coords = list(polygon.exterior.coords)
                    # For flat coordinate system, coordinates are already in image pixel space
                    all_image_coords = []
                    for x, y in coords:
                        # Coordinates are already in flat image space, just convert to integers
                        pixel_x = round(x)
                        pixel_y = round(y)
                        all_image_coords.append((pixel_x, pixel_y))
                    
                    # Draw polygon exterior outline in consistent funky color
                    pts = np.array(all_image_coords, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(img_rgb, [pts], True, (122, 122, 122), 1)
                    
                    # Draw interior holes (if any) in red
                    for interior in polygon.interiors:
                        hole_coords = list(interior.coords)
                        hole_image_coords = []
                        for x, y in hole_coords:
                            # Coordinates are already in flat image space, just convert to integers
                            pixel_x = round(x)
                            pixel_y = round(y)
                            hole_image_coords.append((pixel_x, pixel_y))
                        
                        # Draw hole outline in same funky color to maintain consistency
                        if hole_image_coords:
                            hole_pts = np.array(hole_image_coords, np.int32)
                            hole_pts = hole_pts.reshape((-1, 1, 2))
                            cv2.polylines(img_rgb, [hole_pts], True, (122, 122, 122), 1)

                
                # Get helicopter position in image coordinates
                heli_pos = env.position
                if heli_pos is not None:
                    try:
                        # For flat coordinate system, position is already in image pixel space
                        pixel_x = round(heli_pos[0])
                        pixel_y = round(heli_pos[1])
                        
                        # Mark helicopter position with an arrow pointing in current direction
                        if 0 <= pixel_x < img_width and 0 <= pixel_y < img_height:
                            # Draw an arrow at helicopter position pointing in current bearing direction
                            draw_arrow(img_rgb, pixel_x, pixel_y, - env.current_bearing + 180, 
                                     arrow_length=15, color=(255, 0, 0), thickness=2)
                    except:
                        # If helicopter position marking fails, continue without it
                        pass
                
                # Convert image directly to pygame surface (much faster than matplotlib)
                # Transpose from (height, width, channels) to (width, height, channels) for pygame
                img_transposed = np.transpose(img_rgb, (1, 0, 2))
                target_area_img = pygame.surfarray.make_surface(img_transposed)
        except Exception as e:
            print(f"Error drawing target area: {e}")
            target_area_img = None

    # --- Radiation grid visualization (outside control interval for continuous updates) ---
    radiation_img = None
    if display_radiation_grid:
        if para.z_rad_type == 'jon':
            # Get the actual environment
            env = get_env()
            
            # Check if the environment has been reset (image path changed)
            current_image_path = env.IMAGE_PATH
            
            # Show the precalculated ODL map or target area image based on toggle
            if show_odl_map:
                try:
                    odl_map = env.precalculated_odl_map
 
                    # Normalize ODL map for visualization using radiation_value_to_avoid as max
                    norm_odl = odl_map / para.radiation_value_to_avoid
                    
                    # Apply colormap to the ODL map
                    import matplotlib.cm as cm
                    colored_odl = cm.plasma(norm_odl)[:, :, :3]  # Use plasma colormap, remove alpha
                    colored_odl = (colored_odl * 255).astype(np.uint8)
                    
                    # Mark helicopter position if available
                    heli_pos = env.position
                    if heli_pos is not None:
                        try:
                            # For flat coordinate system, position is already in image pixel space
                            pixel_x = round(heli_pos[0])
                            pixel_y = round(heli_pos[1])
                            
                            # Draw helicopter position with arrow
                            if 0 <= pixel_x < colored_odl.shape[1] and 0 <= pixel_y < colored_odl.shape[0]:
                                draw_arrow(colored_odl, pixel_x, pixel_y, -env.current_bearing + 180,
                                            arrow_length=15, color=(255, 255, 255), thickness=2)
                        except:
                            pass
                    
                    # Convert to pygame surface
                    odl_transposed = np.transpose(colored_odl, (1, 0, 2))
                    radiation_img = pygame.surfarray.make_surface(odl_transposed)
                    
                    # Display ODL map statistics when image changes
                except Exception as e:
                    print(f"Error visualizing ODL map: {e}")
                    radiation_img = target_area_img
            else:
                # Show the original target area image
                radiation_img = target_area_img
        else:
            # Original radiation grid visualization for other types
            env = get_env()
            try:
                radiation_grid = env.radiation_grid
                if radiation_grid is not None:
                    # Normalize and convert to RGB image
                    norm_grid = (radiation_grid) / (para.radiation_value_to_avoid)

                    # --- Mark heli position on the grid ---
                    # Try to get the current heli position in coordinates
                    heli_pos = env.position

                    if heli_pos is not None:
                        try:
                            x, y = env.position_as_xy
                            i, j = round(y), round(x)
                            # Overlay a white pixel on the normalized grid
                            norm_grid = np.array(norm_grid)
                            # Overwrite with the inverse value (1 - value) for the pixel and cross
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    ii, jj = i+di, j+dj
                                    if 0 <= ii < norm_grid.shape[0] and 0 <= jj < norm_grid.shape[1]:
                                        norm_grid[ii, jj] = 1.0 - norm_grid[ii, jj]
                        except:
                            # If position marking fails, continue without it
                            pass

                    # Convert normalized grid to RGB image for pygame
                    # Apply viridis colormap manually (or use any other approach)
                    import matplotlib.cm as cm
                    colored_grid = cm.viridis(norm_grid)[:, :, :3]  # Remove alpha channel
                    colored_grid = (colored_grid * 255).astype(np.uint8)
                    
                    # Draw arrow on the colored grid to show helicopter direction (for non-jon types)
                    if heli_pos is not None and para.z_rad_type != 'jon':
                        try:
                            x, y = env.position_as_xy
                            pixel_x, pixel_y = round(x), round(y)
                            if 0 <= pixel_x < colored_grid.shape[1] and 0 <= pixel_y < colored_grid.shape[0]:
                                # Draw arrow pointing in current bearing direction
                                draw_arrow(colored_grid, pixel_x, pixel_y, env.current_bearing,
                                         arrow_length=8, color=(255, 255, 255), thickness=2)
                        except:
                            pass
                    
                    # Convert to pygame surface directly
                    colored_grid_transposed = np.transpose(colored_grid, (1, 0, 2))
                    radiation_img = pygame.surfarray.make_surface(colored_grid_transposed)
            except AttributeError:
                # radiation_grid doesn't exist for this radiation type
                pass

    if not paused and current_time - last_control_time >= control_interval_ms:
        # Smooth control system - gradually build up and decay actions
        # Action is now a single scalar for turning angle only
        target_action = 0.0
        direct_action = None  # For shift keys that bypass smoothing
        
        # Determine target actions based on currently held keys
        # Check shift keys first for direct action (bypasses smoothing)
        if keys_held['shift_a']:
            direct_action = -1.0
        elif keys_held['shift_d']:
            direct_action = 1.0
        elif keys_held['a']:
            target_action = -max_action_speed
        elif keys_held['d']:
            target_action = max_action_speed
        
        # Note: W and S keys are no longer used since we removed height change
        
        # Smoothly interpolate towards target action
        if direct_action is not None:
            # Direct action bypasses smoothing
            smooth_action = direct_action
            was_using_shift_keys[0] = True
        elif target_action != 0:
            # Accelerate towards target
            if smooth_action < target_action:
                smooth_action = min(target_action, smooth_action + acceleration_rate)
            elif smooth_action > target_action:
                smooth_action = max(target_action, smooth_action - acceleration_rate)
            was_using_shift_keys[0] = False
        else:
            # Check if we were using shift keys in the previous frame
            # If so, immediately stop (no gradual deceleration)
            if was_using_shift_keys[0]:
                smooth_action = 0.0
                was_using_shift_keys[0] = False
            else:
                # Normal deceleration towards zero when no input
                if smooth_action > 0:
                    smooth_action = max(0, smooth_action - deceleration_rate)
                elif smooth_action < 0:
                    smooth_action = min(0, smooth_action + deceleration_rate)
        
        # Apply the smooth action (single scalar)
        action = smooth_action
        
        # Sync heaviness/tension to env before step
        if getattr(para, 'drop_one_ball_experiment', False):
            _env = get_env()
            _env.ball_heaviness = ball_heaviness
            _env.ball_tension = ball_tension

        if para.training_library == 'sb3':
            if getattr(para, 'drop_one_ball_experiment', False):
                output = gymenv.step(np.array([action, ball_x, ball_y]))
            else:
                output = gymenv.step(action)
        elif para.training_library == 'omnisafe':
            if getattr(para, 'drop_one_ball_experiment', False):
                output = omnisafe_env.step([action, ball_x, ball_y])
            else:
                output = omnisafe_env.step([action])  # for testing, use random action
            
            #print(omnisafe_env._env.geo_coordinates_of_radiation_grid.shape[:2])
        
        # Handle observation display - ALWAYS Box format with image
        obs = output[0]

        # Observation is ALWAYS Box format (no lidar dict)
        # Handle dict with 'image' key for SB3 compatibility
        if isinstance(obs, dict):
            obs = obs['image']
        
        # Convert torch tensor to numpy if needed
        import torch
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        
        if para.ablation_study_fisheye_instead_of_multi_scale_maps:
            # Single fisheye observation - convert to 3 channels if needed
            image = ensure_three_channels(obs)
        else:
            # Multi-scale observations - split into 4 separate images and display side by side
            num_channels_per_map = gymenv.image_channels
            
            # Split the observation into 4 maps
            map1 = obs[:, :, 0:num_channels_per_map]
            map2 = obs[:, :, num_channels_per_map:num_channels_per_map*2]
            map3 = obs[:, :, num_channels_per_map*2:num_channels_per_map*3]
            map4 = obs[:, :, num_channels_per_map*3:num_channels_per_map*4]
            
            # Ensure each map has 3 channels for display
            map1 = ensure_three_channels(map1)
            map2 = ensure_three_channels(map2)
            map3 = ensure_three_channels(map3)
            map4 = ensure_three_channels(map4)
            
            # Create grey separator line (1 pixel wide)
            grey_line = np.ones((map1.shape[0], 1, 3), dtype=np.uint8) * 128
            
            # Concatenate all maps horizontally with grey separators
            image = np.concatenate([map1, grey_line, map2, grey_line, map3, grey_line, map4], axis=1)
        
        #image = append_reward_bar_to_image(image=image, reward=output[1])

        # Generate paper visualization images every 10th step
        env_for_vis = get_env()
        if hasattr(env_for_vis, 'current_episode_step') and env_for_vis.current_episode_step % 10 == 0:
            try:
                generate_paper_visualizations(env_for_vis, obs, env_for_vis.current_episode_step)
            except Exception as e:
                print(f"Vis error: {e}")

        white_line = np.ones((1, 1, 3), dtype=np.uint8) * 255
        terminated = output[3]
        if terminated:
            # Render the environment state before resetting
            if para.training_library == 'sb3':
                gymenv.render()
            elif para.training_library == 'omnisafe':
                omnisafe_env._env.render()
            
            # Reset the environment
            if para.training_library == 'omnisafe':
                omnisafe_env.reset()
            else:
                gymenv.reset()
        
        # Create white_line based on the actual observation structure
        # Handle both dict and image observations
        if isinstance(output[0], dict):
            if 'coverage' in output[0]:
                # Dict observation with coverage
                white_line = np.ones((output[0]['coverage'].shape[2], 1, 3), dtype=np.uint8) * 255
            else:
                # Dict observation with image
                white_line = np.ones((output[0]['image'].shape[0], 1, 3), dtype=np.uint8) * 255
        else:
            # Image observation
            white_line = np.ones((output[0].shape[0], 1, 3), dtype=np.uint8) * 255

        # Append reward bar on the relevant image
        #image = append_reward_bar_to_image(image=image, reward=output[1])
        
        if para.training_library == 'sb3':
            metric_data = append_output_gymenv_values(metric_data, gymenv)
        elif para.training_library == 'omnisafe':
            metric_data = append_output_gymenv_values(metric_data, gymenv, omnisafe_env)
        last_elements_of_metric_data = {key: value[-1] for key, value in metric_data.items()}

        # Don't reset action to [0, 0] anymore since we're using smooth_action
        last_control_time = current_time

        # Capture sampling grid for overlay
        if getattr(para, 'drop_one_ball_experiment', False):
            _env = get_env()
            if hasattr(_env, 'list_of_observation_grids') and _env.list_of_observation_grids:
                sampling_grid_points = _env.list_of_observation_grids[0].reshape(-1, 2)

    # While paused, recompute sampling grid preview when ball/params change
    if paused and getattr(para, 'drop_one_ball_experiment', False):
        _env = get_env()
        _env.current_ball_x = ball_x
        _env.current_ball_y = ball_y
        _env.ball_heaviness = ball_heaviness
        _env.ball_tension = ball_tension
        try:
            preview_grid = _env.generate_ball_deformed_grid()
            # Rotate around heli position (center pixel is guaranteed to be there)
            cx, cy = float(_env.position[0]), float(_env.position[1])
            flat = preview_grid.reshape(-1, 2)
            rotated = _env.rotate_coords_flat(flat, cx, cy, _env._current_bearing_key if hasattr(_env, '_current_bearing_key') else 0)
            sampling_grid_points = rotated
        except Exception:
            pass

    # Screen rendering (outside control interval for continuous updates)
    screen.fill((0, 0, 0))

    # Get the latest metric data if available
    if 'last_elements_of_metric_data' in locals():
        current_metrics = last_elements_of_metric_data
    else:
        current_metrics = {}

    if 'image' in locals():
        transposed_image = np.transpose(image, (1, 0, 2))
        #transposed_image = np.flip(transposed_image, axis=1)
        
        # Calculate proper scaling to maintain square pixels (1:1 aspect ratio for each pixel)
        # Get actual dimensions from the image
        img_height, img_width = image.shape[0], image.shape[1]
        
        margin = 10
        available_height = window_height - margin
        half_height = available_height // 2
        
        # Calculate integer scale factor to maintain exact square pixels
        # Find the largest integer scale that fits in the available space
        scale_by_height = half_height // img_height
        scale_by_width = window_width // img_width
        scale_factor = max(1, min(scale_by_height, scale_by_width))
        
        # Calculate exact dimensions with square pixels
        new_width = img_width * scale_factor
        new_height = img_height * scale_factor

        # Use nearest-neighbor scaling to preserve sharp pixels
        scaled_image = pygame.transform.scale(
            pygame.surfarray.make_surface(transposed_image), (new_width, new_height)
        )
        screen.blit(scaled_image, (0, 0))
        # Scale and blit radiation grid below
        if radiation_img is not None and display_radiation_grid:
            # Get original radiation image dimensions
            rad_img_width, rad_img_height = radiation_img.get_size()
            
            # Calculate scale factor to fit in available space while maintaining square pixels
            rad_scale_by_height = half_height // rad_img_height
            rad_scale_by_width = window_width // rad_img_width
            rad_scale_factor = max(1, min(rad_scale_by_height, rad_scale_by_width))
            
            # Calculate exact dimensions with square pixels for radiation grid
            rad_width = rad_img_width * rad_scale_factor
            rad_height = rad_img_height * rad_scale_factor
            
            rad_img_scaled = pygame.transform.scale(radiation_img, (rad_width, rad_height))
            rad_img_scaled = pygame.transform.flip(rad_img_scaled, False, True)  # Flip on y-axis
            screen.blit(rad_img_scaled, (0, new_height + margin))

            # Draw sampling grid overlay for drop_one_ball_experiment
            if getattr(para, 'drop_one_ball_experiment', False) and sampling_grid_points is not None:
                rad_y_offset = new_height + margin
                for gx, gy in sampling_grid_points:
                    sx = int(gx * rad_scale_factor)
                    sy = int(rad_height - gy * rad_scale_factor)
                    if 0 <= sx < rad_width and 0 <= sy < rad_height:
                        pygame.draw.circle(screen, (0, 255, 0), (sx, rad_y_offset + sy), 1)
        elif display_radiation_grid:
            font = pygame.font.SysFont(None, 24)
            text_surface = font.render('No Radiation Grid', True, (255, 0, 0))
            screen.blit(text_surface, (0, new_height + margin))
        
        render_metrics(screen, current_metrics, 0, window_height)

    # Display ball params and input mode
    if getattr(para, 'drop_one_ball_experiment', False):
        info_font = pygame.font.SysFont(None, 22)
        info_y = 5
        for label, val in [('ball_x', f'{ball_x:.2f}'), ('ball_y', f'{ball_y:.2f}'),
                           ('heaviness (h)', f'{ball_heaviness}'), ('tension (t)', f'{ball_tension}')]:
            surf = info_font.render(f'{label}: {val}', True, (200, 200, 200))
            screen.blit(surf, (window_width - surf.get_width() - 10, info_y))
            info_y += 20
        if input_mode is not None:
            prompt = f'Enter {input_mode}: {input_buffer}_'
            prompt_surf = info_font.render(prompt, True, (255, 255, 0))
            screen.blit(prompt_surf, (window_width - prompt_surf.get_width() - 10, info_y))

    pygame.display.flip()

pygame.quit()