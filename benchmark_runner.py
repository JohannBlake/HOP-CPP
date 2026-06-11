"""
Benchmark runner module for running ML model benchmarks.
"""
import os
import sys
import numpy as np
import time
import torch
import json
import wandb
import importlib.util
import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any


def get_model_path_via_scp(sweep_id: str, run_name: str, cluster: str, constrained_model_number: str, base_folder: str) -> str:
    """Download model from cluster using SCP - same logic as visualization phase."""
    # Import parameters here to avoid circular imports
    from get_parameters import para
    
    # SCP command setup - identical to visualization phase
    if cluster == 'lmu':
        remote_user = 'blake@madeira.dbs.ifi.lmu.de'
        if para.training_library == 'sb3':
            remote_base_path = f"/home/stud/blake/git_clones/Simulation_{sweep_id}/logs/{run_name}"
            remote_model_zip_path = f"{remote_base_path}/best_model.zip"
        elif para.training_library == 'omnisafe':
            if constrained_model_number == "":
                list_cmd = (
                    f'ssh blake@madeira.dbs.ifi.lmu.de "ls /home/stud/blake/git_clones/Simulation_{sweep_id}/misc/logs/runs/*/*/torch_save/epoch-*.pt"'
                )
            else:
                list_cmd = (
                    f'ssh blake@madeira.dbs.ifi.lmu.de "ls /home/stud/blake/git_clones/Simulation_{sweep_id}/misc/logs/runs/*/*/torch_save/epoch-{constrained_model_number}.pt"'
                )
            result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True, encoding='utf-8', check=True)
            files = [f for f in result.stdout.strip().split('\n') if f]
            
            if files:
                # Select file with highest epoch number
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
        scp_command = f'scp {remote_user}:"{remote_model_zip_path}" "{destination_model_zip_path}"'
        
    elif cluster == 'lrz':
        remote_user = 'di97sog@login.ai.lrz.de'
        ssh_key = "C:\\Users\\johan\\.ssh\\id_rsa_lrz"
        if para.training_library == 'sb3':
            remote_base_path = f"/dss/dsshome1/0C/di97sog/git_clones/Simulation_{sweep_id}/misc/logs/runs/{run_name}"
            remote_model_zip_path = f"{remote_base_path}/best_model.zip"
        elif para.training_library == 'omnisafe':
            if constrained_model_number == "":
                list_cmd = (
                    f'ssh -i "{ssh_key}" {remote_user} "ls /dss/dsshome1/0C/di97sog/git_clones/Simulation_{sweep_id}/misc/logs/runs/*/*/torch_save/epoch-*.pt"'
                )
            else:
                list_cmd = (
                    f'ssh -i "{ssh_key}" {remote_user} "ls /dss/dsshome1/0C/di97sog/git_clones/Simulation_{sweep_id}/misc/logs/runs/*/*/torch_save/epoch-{constrained_model_number}.pt"'
                )
            result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True, encoding='utf-8', check=True)
            files = [f for f in result.stdout.strip().split('\n') if f]
            
            if files:
                # Select file with highest epoch number
                def extract_epoch_num(fname):
                    match = re.search(r'epoch-(\d+)\.pt$', fname)
                    return int(match.group(1)) if match else -1
                files_with_epoch = [(f, extract_epoch_num(f)) for f in files if extract_epoch_num(f) != -1]
                latest = max(files_with_epoch, key=lambda x: x[1])[0]
            else:
                latest = None
            remote_model_zip_path = latest

        destination_model_zip_path = os.path.join(base_folder, f"{run_name}-best_model.zip")
        scp_command = f'scp -i "{ssh_key}" {remote_user}:"{remote_model_zip_path}" "{destination_model_zip_path}"'
    
    subprocess.run(scp_command, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
    
    return destination_model_zip_path


def setup_omnisafe_import(main_folder: str) -> None:
    """Set up omnisafe import from main repository."""
    OMNISAFE_DIR = os.path.join(main_folder, "omnisafe")
    spec = importlib.util.spec_from_file_location("omnisafe", os.path.join(OMNISAFE_DIR, "__init__.py"))
    omnisafe = importlib.util.module_from_spec(spec)
    sys.modules["omnisafe"] = omnisafe
    spec.loader.exec_module(omnisafe)


def create_path_length_histograms(path_lengths_per_image: Dict[str, List[float]], run_id: str, base_folder: str) -> None:
    """
    Create histograms showing distribution of path lengths for each reset image.
    NOW INCLUDES: Path length at 90% coverage (if available) and coverage at end of episode.
    
    Args:
        path_lengths_per_image: Dictionary mapping reset image paths to lists of path lengths
        run_id: The WandB run ID for labeling
        base_folder: Base folder to save histograms
    """
    # Create output directory for histograms
    histogram_dir = os.path.join(base_folder, 'path_length_histograms')
    os.makedirs(histogram_dir, exist_ok=True)
    
    for reset_image_path, path_lengths in path_lengths_per_image.items():
        if not path_lengths:
            continue
            
        # Extract just the filename for the title
        image_name = os.path.basename(reset_image_path).replace('.png', '')
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(path_lengths, bins=20, alpha=0.7, edgecolor='black')
        plt.title(f'Path Length Distribution - {image_name}\nRun: {run_id}')
        plt.xlabel('Path Length (meters)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics as text
        if path_lengths:
            mean_length = np.mean(path_lengths)
            std_length = np.std(path_lengths)
            min_length = np.min(path_lengths)
            max_length = np.max(path_lengths)
            
            stats_text = f'Mean: {mean_length:.1f}m\nStd: {std_length:.1f}m\nMin: {min_length:.1f}m\nMax: {max_length:.1f}m\nN: {len(path_lengths)}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save histogram
        safe_filename = f'path_lengths_{image_name}_{run_id}.png'
        histogram_path = os.path.join(histogram_dir, safe_filename)
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[HISTOGRAM] Histogram saved: {histogram_path}")
        
        # Also print summary statistics to console
        if path_lengths:
            print(f"[STATS] {image_name} path lengths - Mean: {mean_length:.1f}m, Std: {std_length:.1f}m, N: {len(path_lengths)}")
    
    print(f"[COMPLETE] All histograms saved to: {histogram_dir}")


def create_combined_histogram(all_path_lengths: List[float], base_folder: str, sweep_id: str) -> None:
    """
    Create a histogram showing path lengths from ALL runs and ALL episodes combined.
    
    Args:
        all_path_lengths: List of all path lengths from all runs and episodes
        base_folder: Base folder to save histogram
        sweep_id: Sweep ID for labeling
    """
    # Create output directory for histograms
    histogram_dir = os.path.join(base_folder, 'path_length_histograms')
    os.makedirs(histogram_dir, exist_ok=True)
    
    if not all_path_lengths:
        print("⚠️ No path lengths available for combined histogram")
        return
    
    # Create combined histogram
    plt.figure(figsize=(12, 7))
    plt.hist(all_path_lengths, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    plt.title(f'Combined Path Length Distribution - All Runs & Episodes\nSweep: {sweep_id}')
    plt.xlabel('Path Length (meters)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    mean_length = np.mean(all_path_lengths)
    std_length = np.std(all_path_lengths)
    min_length = np.min(all_path_lengths)
    max_length = np.max(all_path_lengths)
    median_length = np.median(all_path_lengths)
    
    stats_text = (f'Mean: {mean_length:.1f}m\nMedian: {median_length:.1f}m\n'
                 f'Std: {std_length:.1f}m\nMin: {min_length:.1f}m\n'
                 f'Max: {max_length:.1f}m\nTotal Episodes: {len(all_path_lengths)}')
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            fontsize=10, fontweight='bold')
    
    # Save histogram
    histogram_path = os.path.join(histogram_dir, f'path_lengths_COMBINED_ALL_RUNS_{sweep_id}.png')
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[HISTOGRAM] COMBINED histogram saved: {histogram_path}")
    print(f"[STATS] Combined statistics - Mean: {mean_length:.1f}m, Median: {median_length:.1f}m, Std: {std_length:.1f}m, N: {len(all_path_lengths)}")



def run_benchmark(sweep_id: str, 
                 num_episodes: int, 
                 num_runs_displayed: int, 
                 run_ids: List[str], 
                 run_names: List[str],  # NEW: Pass run names from benchmark.py
                 reset_images: List[str], 
                 cluster: str, 
                 constrained_model_number: str,
                 base_folder: str,
                 main_folder: str) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Run benchmark episodes for multiple runs without visualization processing.
    
    Args:
        sweep_id: WandB sweep ID
        num_episodes: Number of episodes to run per model
        num_runs_displayed: Maximum number of runs to process
        run_ids: List of WandB run IDs to benchmark
        run_names: List of WandB run names corresponding to run_ids
        reset_images: List of reset image paths to alternate between
        cluster: Cluster name ('lmu' or 'lrz')
        constrained_model_number: Model epoch number for constrained RL
        base_folder: Base folder path for the cloned repository
        main_folder: Main folder path for the original repository
        
    Returns:
        Tuple of (List of benchmark results with performance metrics, DataFrame with detailed metrics)
    """
    # Add base folder to path
    if base_folder not in sys.path:
        sys.path.insert(0, base_folder)
    
    # Set up device configuration (same as visualization phase)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # GPU setup for benchmark phase
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set up omnisafe import
    setup_omnisafe_import(main_folder)
    
    # Import required modules
    from omnisafe.models.actor.actor_builder import ActorBuilder
    from get_parameters import para
    from class_gymenv import GymnasiumEnv, GymEnvOmniSafe
    
    # Load height data
    height_data_path = os.path.join(os.path.dirname(__file__), 'misc', 'geo_data', 'height_data', 'height_data.npy')
    height_data = np.load(height_data_path)
    
    benchmark_results = []
    all_path_lengths_combined = []  # NEW: Collect all path lengths across all runs
    
    # NEW: DataFrame to store all metrics
    df_data = []
    
    for run_idx, (run_id, run_name) in enumerate(zip(run_ids[:num_runs_displayed], run_names[:num_runs_displayed])):
        # NEW: No wandb API calls - use passed run names
        try:            
            # Download model using SCP - same as visualization phase
            model_local_path = get_model_path_via_scp(sweep_id, run_name, cluster, constrained_model_number, base_folder)
                
        except Exception as e:
            print(f"Error downloading model for run {run_id}: {e}")
            continue
            
        # Load model - use same logic as visualization phase
        try:
            # Use ActorBuilder from omnisafe like in visualization phase
            if para.training_library == 'sb3':
                gymenv = GymnasiumEnv(height_data, radiation_grid_visualization=False)
                env = gymenv
            elif para.training_library == 'omnisafe':
                gymenv = GymEnvOmniSafe(radiation_grid_visualization=False)
                env = gymenv
            else:
                continue
            
            # Build policy using ActorBuilder from visualization phase
            actor_builder = ActorBuilder(env.observation_space, env.action_space, hidden_sizes=[256, 256])
            policy = actor_builder.build_actor(actor_type="cnn")
            
            # Load model weights - same as visualization phase
            state = torch.load(model_local_path, map_location="cpu")
            policy.load_state_dict(state.get("pi", state))
            policy.eval()
            
            # Move policy to GPU if available (same as visualization phase)
            if torch.cuda.is_available():
                policy = policy.to(DEVICE)
            
        except Exception as e:
            continue
        
        # Run benchmark episodes
        run_start_time = time.time()
        total_steps = 0
        episodes_completed = 0
        
        # Dictionary to track performance per reset image
        reset_image_performance = {}
        path_lengths_per_image = {}  # NEW: Track path lengths for histograms
        path_lengths_at_90_per_image = {}  # NEW: Track path lengths at 90% coverage
        coverage_at_end_per_image = {}  # NEW: Track coverage at end of episode
        
        for reset_img in reset_images:
            reset_image_performance[reset_img] = {
                'episode_steps': [],
                'episode_times': [],
                'episode_rewards': [],
                'episode_costs': [] if para.training_library == 'omnisafe' else None
            }
            path_lengths_per_image[reset_img] = []  # NEW: Initialize path length tracking
            path_lengths_at_90_per_image[reset_img] = []  # NEW: Initialize 90% coverage path length tracking
            coverage_at_end_per_image[reset_img] = []  # NEW: Initialize coverage tracking
        
        print(f"Starting {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Reset with alternating images
            reset_image = reset_images[episode % len(reset_images)]
            full_reset_path = os.path.join(main_folder, reset_image)
            
            try:
                obs, info = env.reset(options={'image_path': full_reset_path})
            except Exception as e:
                obs, info = env.reset()
            
            episode_start_time = time.time()
            episode_steps = 0
            episode_reward = 0
            episode_cost = 0
            terminated = False
            truncated = False
            path_length_at_90 = None  # NEW: Track path length when 90% coverage reached
            reached_90_coverage = False  # NEW: Flag to track if 90% coverage was reached
            
            # Step through episode (NO VISUALIZATION PROCESSING)
            while not terminated and not truncated and episode_steps < 10000:
                try:
                    with torch.no_grad():
                        action = policy.predict(obs, deterministic=True)
                    
                    # NEW: Check coverage before step to see if we're crossing 90% threshold
                    if para.training_library == 'sb3':
                        current_coverage = (1 - env.unwrapped.percentage_of_target_area_left) * 100
                        current_path_length = env.unwrapped.length_of_path_in_meters
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                    else:  # constrained
                        current_coverage = (1 - env._env.unwrapped.percentage_of_target_area_left) * 100
                        current_path_length = env._env.unwrapped.length_of_path_in_meters
                        obs, reward, cost, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        episode_cost += cost
                    
                    # NEW: Check if we just crossed 90% coverage threshold
                    if not reached_90_coverage and current_coverage >= 90.0:
                        path_length_at_90 = current_path_length
                        reached_90_coverage = True
                        print(f"Episode {episode + 1} reached 90% coverage at path length: {path_length_at_90:.2f}m")
                    
                    episode_steps += 1
                    total_steps += 1
                    
                    # Log step number and episode progress every 200th step
                    if episode_steps % 200 == 0:
                        print(f"Episode {episode + 1}/{num_episodes}, Step {episode_steps}")
                    
                    # NEW: If episode is ending, collect the path length and coverage BEFORE they get reset
                    if terminated or truncated:
                        # Store the CURRENT values as final values (before environment resets)
                        final_coverage = current_coverage
                        final_path_length = current_path_length
                        
                        path_lengths_per_image[reset_image].append(final_path_length)
                        coverage_at_end_per_image[reset_image].append(final_coverage)
                        
                        # Store path length at 90% (or None if never reached)
                        path_lengths_at_90_per_image[reset_image].append(path_length_at_90)
                        
                        print(f"Episode {episode + 1} ended. Path length: {final_path_length:.2f}m, Coverage: {final_coverage:.2f}%")
                    
                except Exception as e:
                    print(f"ERROR during episode step: {e}")
                    break
            
            episode_time = time.time() - episode_start_time
            episodes_completed += 1
            
            # Store performance metrics per reset image
            reset_image_performance[reset_image]['episode_steps'].append(episode_steps)
            reset_image_performance[reset_image]['episode_times'].append(episode_time)
            reset_image_performance[reset_image]['episode_rewards'].append(float(episode_reward))
            if para.training_library == 'omnisafe':
                reset_image_performance[reset_image]['episode_costs'].append(float(episode_cost))
            
            print(f"Episode {episode + 1}/{num_episodes} completed with {episode_steps} steps, reward: {float(episode_reward):.2f}")
        
        run_time = time.time() - run_start_time
        
        # NEW: Collect all path lengths from this run into the combined list
        for reset_img_path, lengths in path_lengths_per_image.items():
            all_path_lengths_combined.extend(lengths)
        
        # NEW: Create DataFrame entries for this run
        map_run_counters = {}  # Track run number per map
        for reset_img_path in reset_images:
            map_name = os.path.basename(reset_img_path).replace('.png', '')
            
            # Get all metrics for this map
            path_lengths_end = path_lengths_per_image[reset_img_path]
            path_lengths_90 = path_lengths_at_90_per_image[reset_img_path]
            coverages_end = coverage_at_end_per_image[reset_img_path]
            
            # Create DataFrame row for each episode
            for idx, (pl_end, pl_90, cov_end) in enumerate(zip(path_lengths_end, path_lengths_90, coverages_end)):
                if map_name not in map_run_counters:
                    map_run_counters[map_name] = 1
                else:
                    map_run_counters[map_name] += 1
                
                df_data.append({
                    'model': "Ours",
                    'map': map_name,
                    'map_run_id': map_run_counters[map_name],
                    'path_length_90_pct': pl_90,  # Can be None if never reached 90%
                    'path_length_end': pl_end,
                    'coverage_end': cov_end
                })
        
        # NEW: Generate histograms for path lengths per reset image
        create_path_length_histograms(path_lengths_per_image, run_id, base_folder)
        
        result = {
            'run_id': run_id,
            'episodes_completed': episodes_completed,
            'total_steps': total_steps,
            'total_time': run_time,
            'avg_steps_per_second': total_steps / run_time if run_time > 0 else 0,
            'avg_episode_time': run_time / episodes_completed if episodes_completed > 0 else 0,
            'reset_image_performance': reset_image_performance,
            'path_lengths_per_image': path_lengths_per_image,  # Path length at end of episode
            'path_lengths_at_90_per_image': path_lengths_at_90_per_image,  # NEW: Path length at 90% coverage
            'coverage_at_end_per_image': coverage_at_end_per_image  # NEW: Coverage at end of episode
        }
        
        benchmark_results.append(result)
    
    # NEW: Create combined histogram showing ALL path lengths from ALL runs
    create_combined_histogram(all_path_lengths_combined, base_folder, sweep_id)
    
    # NEW: Create DataFrame from collected data
    df = pd.DataFrame(df_data)
    
    # Save DataFrame to CSV
    csv_path = os.path.join(base_folder, f'benchmark_results_{sweep_id}.csv')
    df.to_csv(csv_path, index=False)
    print(f"[DATAFRAME] Benchmark DataFrame saved to: {csv_path}")
    print(f"[DATAFRAME] DataFrame shape: {df.shape}")
    print(f"[DATAFRAME] DataFrame preview:\n{df.head(10)}")
    
    return benchmark_results, df


def main():
    """Command line interface for the benchmark runner."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-id', required=True)
    parser.add_argument('--num-episodes', type=int, default=2)
    parser.add_argument('--num-runs-displayed', type=int, default=2)
    parser.add_argument('--run-ids', required=True)
    parser.add_argument('--run-names', required=True)  # NEW: Add run names argument
    parser.add_argument('--reset-images', required=True)
    parser.add_argument('--cluster', default='lmu')
    parser.add_argument('--constrained-model-number', default='')
    parser.add_argument('--base-folder', required=True)
    parser.add_argument('--main-folder', required=True)
    args = parser.parse_args()
    
    run_ids = json.loads(args.run_ids)
    run_names = json.loads(args.run_names)  # NEW: Parse run names
    reset_images = json.loads(args.reset_images)
    
    # Run benchmark
    results, df = run_benchmark(
        sweep_id=args.sweep_id,
        num_episodes=args.num_episodes,
        num_runs_displayed=args.num_runs_displayed,
        run_ids=run_ids,
        run_names=run_names,  # NEW: Pass run names
        reset_images=reset_images,
        cluster=args.cluster,
        constrained_model_number=args.constrained_model_number,
        base_folder=args.base_folder,
        main_folder=args.main_folder
    )
    
    return results, df


if __name__ == "__main__":
    main()