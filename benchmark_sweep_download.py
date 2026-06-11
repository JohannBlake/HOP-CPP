import wandb
import os
import subprocess
import shutil
import re
import yaml
import torch
import sys
import glob
from pathlib import Path
from types import SimpleNamespace
import time

# Add current directory to path so we can import local modules
sys.path.append(os.getcwd())

# Import necessary modules from benchmark_local
# We import these to reuse the logic
try:
    from benchmark_local import (
        run_single_benchmark, 
        initialize_global_state, 
        setup_omnisafe_import,
        init_logging,
        load_model
    )
    import get_parameters
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Constants
SWEEP_ID = "59etj5gd"
CLUSTER_ALIAS = "c" # As per user instruction "ssh c"
YAML_PATH = "parameters_default.yaml"
LOCAL_MODEL_PATH = "downloaded_model.pt"
WANDB_ENTITY = "johanndavidblake-ludwig-maximilianuniversity-of-munich"
WANDB_PROJECT = "Heli-Logs"

class SimpleConfig:
    def __init__(self):
        self.reset_image_paths = []
        self.model_base_path = ""

def get_remote_base_path(sweep_id, algo, run_name):
    """Constructs the remote base path based on algorithm and run details."""
    # Logic adapted from step_through_gymenv_and_save_data_for_vis.py
    # Note: Assuming user is 'blake' based on step_through script, but using 'c' alias
    
    # We need to handle the wildcards on the remote side.
    # The step_through script uses ls to find files.
    
    if algo.startswith("SAC"):
        # Pattern: /home/stud/blake/git_clones/Simulation_{sweep_id}/.experiments/*/*/torch_save
        return f"/home/stud/blake/git_clones/Simulation_{sweep_id}/.experiments"
    else:
        # Pattern: /home/stud/blake/git_clones/Simulation_{sweep_id}/misc/logs/runs/*/*/torch_save
        # Wait, step_through uses: .../misc/logs/runs/*/*/torch_save
        # But also .../misc/logs/runs/{run_name} for SB3
        pass
    
    # For OmniSafe (which seems to be the default based on imports)
    return f"/home/stud/blake/git_clones/Simulation_{sweep_id}/misc/logs/runs"

def download_last_model(sweep_id):
    print(f"Connecting to WandB to find run for sweep {sweep_id}...")
    api = wandb.Api()
    sweep_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}"
    sweep = api.sweep(sweep_path)
    runs = sorted(sweep.runs, key=lambda run: run.summary.get("_step", 0), reverse=True)
    
    if not runs:
        print("No runs found for sweep.")
        sys.exit(1)
        
    last_run = runs[0]
    print(f"Found last run: {last_run.id} ({last_run.name})")
    
    config = last_run.config
    omnisafe_alg = config.get("omnisafe_alg", "TD3Lag") # Default to TD3Lag if not found
    print(f"Algorithm: {omnisafe_alg}")

    # Determine remote path pattern
    # The step_through script uses 'ssh ls' to find the path.
    # We will do the same.
    search_path_base = get_remote_base_path(sweep_id, omnisafe_alg, last_run.name)
    
    # We need to find the specific folder. 
    # Usually: .../runs/<algo>-<env>-<tags>/seed-000-...
    # The run name in wandb often matches part of the folder or is used in it.
    # Let's search for the folder containing the run files.
    
    # We can try to list files in the base path and find a folder that looks correct.
    # Or strict path traversal.
    
    print("Searching for model on cluster...")
    
    # First, list the directories in the base path to find the correct experiment folder
    # Note: The path might vary.
    
    # Strategy: Find valid torch_save folders deep in the structure
    # Finding model 99 specially
    find_cmd = f'ssh {CLUSTER_ALIAS} "find {search_path_base} -name epoch-99.pt -type f | head -n 1"'
    print(f"Executing: {find_cmd}")
    
    try:
        # Added encoding and errors handling
        result = subprocess.run(find_cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        remote_file_path = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error finding model: {e}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)
        
    if not remote_file_path:
        print("\nCould not find epoch-99.pt file in the expected path.")
        
        # Fallback for SAC location if first attempt failed
        if "SAC" not in search_path_base and "SAC" in omnisafe_alg:
             # Try SAC path if we didn't use it
             pass 
        sys.exit(1)
        
    print(f"Found latest model: {remote_file_path}")
    
    # Download using ssh cat to avoid scp path issues
    local_abs_path = os.path.abspath(LOCAL_MODEL_PATH)
    print(f"Downloading with ssh cat to {local_abs_path}...")
    
    ssh_cmd = f'ssh {CLUSTER_ALIAS} "cat {remote_file_path}"'
    print(f"Executing: {ssh_cmd}")
    
    try:
        with open(local_abs_path, 'wb') as f:
            # Execute ssh cat and pipe stdout directly to the file
            # This handles large files efficiently without loading into memory
            process = subprocess.Popen(ssh_cmd, shell=True, stdout=f, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"SSH Cat failed with code {process.returncode}")
                # print(f"Stderr: {stderr.decode('utf-8', errors='replace')}")
                # Try reading stderr from the pipe if communicated
                if stderr:
                     print(f"Stderr: {stderr.decode('utf-8', errors='replace')}")
                sys.exit(1)
                
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)
        
    return local_abs_path

def update_yaml_action_noise(std_dev):
    print(f"Updating {YAML_PATH} with action_noise_std = {std_dev}")
    with open(YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)
    
    data['action_noise_std'] = std_dev
    
    with open(YAML_PATH, 'w') as f:
        yaml.safe_dump(data, f)
        
def main():
    # 0. Setup
    model_path = download_last_model(SWEEP_ID)
    init_logging()
    
    # 1. Initialize global state ONCE with the downloaded model
    print("Initializing global state with downloaded model...")
    setup_omnisafe_import()
    initialize_global_state(model_path)
    
    noises = [0.1, 0.05, 0.01]
    maps = [f"misc/radiation_data/bw_jon_images_from_paper/eval_mowing_{i}.png" for i in range(9, 15)]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dummy args for run_single_benchmark
    # We need to simulate the args object
    Args = type('Args', (), {})
    args = Args()
    args.num_episodes = 3 # 3 times per map? "each 3 times" - per map?
    # User said: "benchmarks it on all 6 maps with current gymenv eaach 3 times" -> 3 episodes per map
    # run_single_benchmark runs 'num_episodes' total. 
    # If we pass all maps to config, need num_episodes = 3 * 6 = 18?
    # benchmark_local logic:
    # "When num_episodes equals number of maps, each map gets exactly 1 episode"
    # "If num_episodes > len(maps), it distributes them."
    # So if we want 3 per map, we need 18 episodes.
    args.num_episodes = 3 * len(maps) 
    args.processes = 1 # Sequential or parallel? user didn't specify. Safe default 1 or 2.
    args.max_steps_per_episode = 3000 # Default/Safe
    args.constrained_model_number = None
    args.save_csv = "benchmark_sweep_results.csv"
    args.wandb_logging = False
    args.sweep_id = None
    args.wandb_project = None
    args.continuous_loop = False
    args.plot = False
    
    # We need to reload policy between noise changes?
    # run_single_benchmark reloads only if reload_model=True via find_latest.
    # Since we have a static file, we can pass it (checkpoint_path).
    
    # We need to manually load the policy/env initially or let run_single_benchmark handle it?
    # run_single_benchmark takes policy, env.
    
    loaded_policy, loaded_env = load_model(device)
    
    for noise in noises:
        print(f"\n{'='*20} Benchmarking with Noise {noise} {'='*20}\n")
        update_yaml_action_noise(noise)
        
        # Reloading parameters often requires reloading the environment wrapper
        # The environment reads parameters on __init__. 
        # run_single_benchmark receives 'env'. If we pass the existing 'env', it checks if it needs reset?
        # run_single_benchmark -> run_episodes_sequential
        # The WORKERS create new envs via initialize_global_state -> GymEnvOmniSafe()
        # So the global state must be re-initialized?
        # Actually initialize_global_state stores the env config, not the env instance itself (except purely for arch detection).
        # The workers inside run_single_benchmark call run_episodes_sequential, which calls run_episode_subprocess?
        # Prepare config
        config = SimpleConfig()
        config.reset_image_paths = maps
        config.model_base_path = os.path.abspath(os.path.dirname(model_path)) # Just directory
        
        # Override args for this run
        args.save_csv = f"benchmark_results_noise_{noise}.csv"
        
        # Run
        # We pass loaded_policy, loaded_env - but workers will recreate them?
        # Workers create their own policy/env. The passed policy/env in run_single_benchmark is mostly for the main process loop?
        # Actually run_single_benchmark runs episodes in PARALLEL using workers.
        # The passed policy/env are returned at end.
        
        run_single_benchmark(
            args=args,
            config=config,
            device=device,
            policy=loaded_policy,
            env=loaded_env,
            reload_model=False,
            run_number=1,
            checkpoint_path=model_path,
            epoch_number=99 # Dummy
        )

if __name__ == "__main__":
    main()
