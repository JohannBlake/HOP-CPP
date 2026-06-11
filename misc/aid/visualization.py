import numpy as np
import pandas as pd
from geopy.distance import geodesic
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
from plotly import graph_objects as go
import plotly.io as pio
import os
import numpy as np
import plotly.graph_objects as go
import numpy as np
import math

def append_to_metric_data(metric_data, key, value, precision=3):
    if key[:20] not in metric_data.keys():
        metric_data[key[:20]] = []
    if isinstance(value, (int, float, np.float32, np.float64)):
        metric_data[key[:20]].append(round(value, precision))
    elif isinstance(value, bool) or value == None:
        metric_data[key[:20]].append(value)
    else:
        metric_data[key[:20]].append(value)

def append_output_gymenv_values(metric_data, gymenv, omnisafe_env=None):
    append_to_metric_data(metric_data, 'Curr rad exp', gymenv.current_radiation_exposure)
    append_to_metric_data(metric_data, 'Reward', gymenv.reward)
    append_to_metric_data(metric_data, 'rew_meas', gymenv.rew_measurement_via_percentage_of_new_measured_points_scaled_by_height)
    append_to_metric_data(metric_data, 'rew_tv', gymenv.rew_incremental_tv)
    append_to_metric_data(metric_data, "% episode finished", gymenv.percentage_of_episode_finished_by_time_passed)
    append_to_metric_data(metric_data, 'Coverage', str(format(math.floor(100 * (1 - gymenv.percentage_of_target_area_left) * 100) / 100, '.2f')) + '%')
    append_to_metric_data(metric_data, 'Length', str(round(gymenv.length_of_path_in_meters, 10)) + 'm')
    append_to_metric_data(metric_data, 'Collisions', gymenv.num_episode_collisions)
    append_to_metric_data(metric_data, 'meas_area', gymenv.new_measured_area_size)
    append_to_metric_data(metric_data, 'd_len', gymenv.length_new_measured_minus_length_old_measured)
    append_to_metric_data(metric_data, 'l_meas', getattr(gymenv, 'length_measured_area_union_black_polygons', 0))
    append_to_metric_data(metric_data, 'l_old', getattr(gymenv, 'length_old_measured', 0))
    append_to_metric_data(metric_data, 'Step', gymenv.current_episode_step)

    return metric_data
def visualize_heightmap(gridsize, heightmap):
    x = np.arange(0, gridsize[0])
    y = np.arange(0, gridsize[1])
    x, y = np.meshgrid(x, y)
    z = heightmap

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='3D Heightmap', autosize=True,
                        scene=dict(zaxis=dict(range=[0, 150]),
                                    aspectratio=dict(x=1, y=1, z=0.5)))
    fig.show()

def append_reward_bar_to_image(image, reward):
    # Ensure the image has 3 channels
    if image.shape[-1] == 2:  # If the image has 2 channels
        image = np.stack((image[:, :, 0], image[:, :, 1], np.zeros_like(image[:, :, 0])), axis=-1)
    elif len(image.shape) == 2:  # If the image is grayscale (2D)
        image = np.stack((image, image, image), axis=-1)

    image_height, image_width = image.shape[:2]

    # Create a black bar of 3 pixels height and the same width as the image
    reward_bar = np.zeros((3, image_width, 3), dtype=np.uint8)

    # Find the midpoint of the image width
    midpoint = image_width // 2

    # Calculate the length of the reward bar
    reward_length = int((image_width * abs(reward)) / 2)
    reward_length = min(reward_length, midpoint)  # Cap the reward length to the midpoint

    # Draw the reward bar
    if reward > 0:
        # Green bar for positive reward
        reward_bar[:, midpoint:midpoint + reward_length] = [0, 255, 0]  # Green color
    else:
        # Red bar for negative reward
        reward_bar[:, midpoint - reward_length:midpoint] = [255, 0, 0]  # Red color

    # Create a white line of 1 pixel height and the same width as the image
    white_line_horizontal = np.ones((1, image_width, 3), dtype=np.uint8) * 255

    # Use np.vstack to add the white line and the black bar with the reward visualization below the image
    image = np.vstack((image, white_line_horizontal, reward_bar))
    return image