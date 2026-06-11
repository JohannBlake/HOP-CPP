"""
Local benchmark script - loads model from local directory and runs test episodes.
No directory changes, no cloning, no SCP downloads.
"""
import os
import sys
import time
import glob
import re
import csv
import argparse
import statistics
import shutil
import subprocess
from pathlib import Path
import numpy as np
import torch
from typing import List, Optional, Dict
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial
import copy
import traceback
import importlib.util
from get_parameters import para
# Global log file handle
_LOG_FILE = None
_JOB_ID = None  # Store job ID for SLURM status checking
_TRAINING_JOB_ID = None  # Store training job ID for monitoring training status

def log(msg: str) -> None:
    """Simple logging to both console and file."""
    print(msg)
    if _LOG_FILE:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()

def is_slurm_job_active(job_id: str) -> bool:
    """
    Check if a SLURM job is still active by querying squeue.
    
    Args:
        job_id: SLURM job ID to check
        
    Returns:
        True if job is still in the queue (running or pending), False otherwise
    """
    try:
        # Run squeue to check if job is still active
        # Use encoding='utf-8' with errors='ignore' to handle special characters on Windows
        result = subprocess.run(
            f'ssh blake@madeira.dbs.ifi.lmu.de "squeue -h -j {job_id}"',
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',  # Ignore decoding errors from SSH output
            timeout=30,
            shell=True
        )
        
        # Check if we got output - the presence of output means job is active
        # We ignore returncode because SSH connection issues can cause non-zero returns
        # even when the job is actually active
        has_output = result.stdout.strip() != ''
        
        if has_output:
            log(f"   Job {job_id} is active (found in squeue)")
            return True
        
        # SSH connection failure (returncode 255) should be treated as "assume active"
        # to avoid false negatives due to temporary connection issues
        if result.returncode == 255:
            log(f"   Warning: SSH connection failed (returncode=255), assuming job {job_id} is still active")
            return True
        
        # No output and successful SSH connection means job is not in queue anymore
        log(f"   Job {job_id} is not active (no output from squeue)")
        return False
        
    except subprocess.TimeoutExpired:
        log("   Warning: Timeout while checking SLURM job status, assuming job is still active")
        return True
    except Exception as e:
        log(f"   Warning: Failed to check SLURM job status: {e}, assuming job is still active")
        return True

def init_logging(job_id: Optional[str] = None, training_job_id: Optional[str] = None) -> None:
    """Initialize logging to job_id.txt file."""
    global _LOG_FILE, _JOB_ID, _TRAINING_JOB_ID
    _JOB_ID = job_id  # Store job ID globally for status checking
    _TRAINING_JOB_ID = training_job_id  # Store training job ID for monitoring
    if job_id:
        log_filename = f"{job_id}.txt"
    else:
        log_filename = f"benchmark_{int(time.time())}.txt"
    _LOG_FILE = open(log_filename, "w", encoding="utf-8")
    log(f"Logging to: {log_filename}")
    if job_id:
        log(f"SLURM Job ID: {job_id}")
    if training_job_id:
        log(f"Training Job ID: {training_job_id}")

# wait
#hours = 8 * 60 * 60
#log(f"Waiting for {hours/3600} hours before starting benchmark...")
#time.sleep(hours)


def clear_png_folder(constrained_model_number: Optional[int]) -> None:
    """Clear the PNG folder for the specified model number before starting benchmark."""
    if constrained_model_number is not None and isinstance(constrained_model_number, int):
        images_dir = f"images_of_paths/benchmark_results_{constrained_model_number}"
        
        if os.path.exists(images_dir):
            try:
                shutil.rmtree(images_dir)
                log(f"Cleared existing PNG folder: {images_dir}")
            except Exception as e:
                log(f"Warning: Failed to clear PNG folder {images_dir}: {e}")
        
        # Recreate the directory
        try:
            os.makedirs(images_dir, exist_ok=True)
            log(f"Created fresh PNG folder: {images_dir}")
        except Exception as e:
            log(f"Warning: Failed to create PNG folder {images_dir}: {e}")


# Global variables to store loaded state and environment configuration
_GLOBAL_MODEL_STATE = None
_GLOBAL_ENV_CONFIG = None
_GLOBAL_POLICY_ARCHITECTURE = None  # Store policy architecture info
_LAST_LOADED_EPOCH = -1  # Track the last loaded epoch for finding next checkpoint
_EPISODE_SEED_COUNTER = 0  # Track episode seeds, increments before each reset


def init_worker_global_state(model_state, env_config, policy_architecture):
    """Initialize global state in each worker process."""
    global _GLOBAL_MODEL_STATE, _GLOBAL_ENV_CONFIG, _GLOBAL_POLICY_ARCHITECTURE
    _GLOBAL_MODEL_STATE = model_state
    _GLOBAL_ENV_CONFIG = env_config
    _GLOBAL_POLICY_ARCHITECTURE = policy_architecture

@dataclass
class Config:
    """Configuration for local benchmark."""
    # Model path
    if para.omnisafe_alg == "TD3Lag":
        model_base_path: str = r"misc\logs\runs\TD3Lag-{GymEnvOmniSafe-v0}"
    elif para.omnisafe_alg[:3] == "SAC":
        model_base_path: str = r".experiments\SAC-{GymEnvOmniSafe-v0}"
    # Episode settings
    max_steps_per_episode: int = 18000
    
    # Reset images
    reset_image_paths: List[str] = None
    
    # Map filter (optional)
    map_filter: List[int] = None
    
    def __post_init__(self):
        if self.reset_image_paths is None:
            self.reset_image_paths = [
                os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_4.png"),
                os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_5.png"),
                os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_6.png"),
                os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_7.png"),
                os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_8.png"),
                os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_15.png")
            ]

        def _normalize_path(path_str: str) -> str:
            cleaned = path_str.replace("\\", "/")
            if os.name == 'nt':
                return str(Path(cleaned))
            return Path(cleaned).as_posix()

        normalized_model_path = Path(_normalize_path(self.model_base_path))

        if normalized_model_path.name.startswith("seed-"):
            seed_parent = normalized_model_path.parent
            seed_dir_candidate = normalized_model_path
        else:
            seed_parent = normalized_model_path
            seed_dir_candidate = None

        def _pick_seed_directory(parent: Path) -> Path:
            if not parent.exists():
                raise FileNotFoundError(f"Model run root not found: {parent}")
            seed_dirs = [d for d in parent.iterdir() if d.is_dir() and d.name.startswith("seed-")]
            if not seed_dirs:
                raise FileNotFoundError(f"No seed-* directory found under {parent}")
            seed_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            return seed_dirs[0]

        if seed_dir_candidate is None or not seed_dir_candidate.exists():
            seed_dir_candidate = _pick_seed_directory(seed_parent)

        self.model_base_path = _normalize_path(str(seed_dir_candidate))
        self.reset_image_paths = [_normalize_path(p) for p in self.reset_image_paths]
        
        # Apply map filter if provided
        if self.map_filter:
            log(f"[Filter] Applying map filter: {self.map_filter}")
            filtered_paths = []
            for map_id in self.map_filter:
                # Find the path that matches this map ID
                map_filename = f"eval_mowing_{map_id}.png"
                matching_path = None
                for path in self.reset_image_paths:
                    if map_filename in path:
                        matching_path = path
                        break
                if matching_path:
                    filtered_paths.append(matching_path)
                else:
                    log(f"[Filter] Warning: Map ID {map_id} not found in reset_image_paths")
            
            if filtered_paths:
                self.reset_image_paths = filtered_paths
                log(f"[Filter] Filtered to {len(self.reset_image_paths)} maps: {[os.path.basename(p) for p in self.reset_image_paths]}")
            else:
                log(f"[Filter] Warning: No maps matched filter, keeping all maps")
        
        # Verify all reset image paths exist
        missing_images = []
        for img_path in self.reset_image_paths:
            full_path = img_path if os.path.isabs(img_path) else os.path.join(os.getcwd(), img_path)
            if not os.path.exists(full_path):
                missing_images.append(full_path)
        
        if missing_images:
            log(" Warning: Missing evaluation images:")
            for missing in missing_images:
                log(f"   {missing}")
        else:
            log(f"[OK] Verified {len(self.reset_image_paths)} evaluation images exist")


def find_next_model_checkpoint(model_base_path: str, max_wait_hours: float = 2.0) -> Optional[tuple[str, int]]:
    """
    Find the next model checkpoint (epoch n+1 where n is the currently loaded epoch).
    Starts by waiting for epoch-0 if no model has been loaded yet.
    After that, only loads epoch-{n+1} when it becomes available.
    Waits 1 minute and retries for up to max_wait_hours if the next checkpoint doesn't exist yet.
    
    If training_job_id is provided, will check if training job is still active before waiting.
    Exits when checkpoint is found OR when training job has stopped OR when max wait time exceeded.
    
    Special handling for epoch-0: If epoch-0 doesn't exist yet, keeps waiting indefinitely (ignores timeout)
    until it appears or training job stops. This ensures the benchmark can start even if training hasn't
    created the first checkpoint yet.
    
    Args:
        model_base_path: Base path to the model directory (e.g., misc/logs/runs/TD3Lag-.../seed-000-...)
        max_wait_hours: Maximum time to wait for next checkpoint (default: 2 hours)
        
    Returns:
        Tuple of (checkpoint_path, epoch_number) or None if training stopped/timeout without finding next checkpoint
    """
    global _LAST_LOADED_EPOCH, _TRAINING_JOB_ID
    
    torch_save_dir = os.path.join(model_base_path, "torch_save")
    
    # Wait for torch_save directory to be created if it doesn't exist yet
    if not os.path.exists(torch_save_dir):
        log(f" Torch save directory not found yet: {torch_save_dir}")
        log(f"   Waiting for training to create directory...")
        # Don't return None immediately - wait for directory to be created
        start_wait_time = time.time()
        while not os.path.exists(torch_save_dir):
            # Check if training job is still active
            job_id_to_check = _TRAINING_JOB_ID if _TRAINING_JOB_ID else _JOB_ID
            if job_id_to_check:
                if not is_slurm_job_active(job_id_to_check):
                    log(f" Training job {job_id_to_check} stopped before creating torch_save directory")
                    return None
            
            elapsed_minutes = (time.time() - start_wait_time) / 60
            if elapsed_minutes % 5 < 0.017:  # Log every ~5 minutes
                log(f"   Still waiting for torch_save directory (elapsed: {elapsed_minutes:.1f} minutes)...")
            
            time.sleep(60)  # Check every minute
            
        log(f" Torch save directory created: {torch_save_dir}")
    
    # Determine the next target epoch to load
    target_epoch = _LAST_LOADED_EPOCH + 1
    log(f" Looking for next checkpoint: epoch-{target_epoch}.pt (current loaded epoch: {_LAST_LOADED_EPOCH})")
    
    # Special handling for epoch-0: no timeout, wait indefinitely until it appears
    if target_epoch == 0:
        log(f"   Waiting for first checkpoint (epoch-0) - no timeout")
        log(f"   Will keep checking until training creates epoch-0 or training job stops")
    else:
        log(f"   Maximum wait time: {max_wait_hours:.1f} hours")
    
    # Track start time for timeout
    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600
    
    while True:
        # Check if we've exceeded the maximum wait time (skip for epoch-0)
        elapsed_time = time.time() - start_time
        if target_epoch != 0 and elapsed_time > max_wait_seconds:
            elapsed_hours = elapsed_time / 3600
            log(f" Maximum wait time of {max_wait_hours:.1f} hours exceeded (waited {elapsed_hours:.2f} hours)")
            log(f"   Checkpoint epoch-{target_epoch}.pt did not appear within timeout period")
            return None
        
        # Find all epoch-*.pt files
        pattern = os.path.join(torch_save_dir, "epoch-*.pt")
        checkpoint_files = glob.glob(pattern)
        
        # Extract epoch numbers
        def extract_epoch_num(filepath):
            match = re.search(r'epoch-(\d+)\.pt$', filepath)
            return int(match.group(1)) if match else -1
        
        # Look for the target epoch
        target_checkpoint = None
        for checkpoint_file in checkpoint_files:
            if extract_epoch_num(checkpoint_file) == target_epoch:
                target_checkpoint = checkpoint_file
                break
        
        if target_checkpoint is not None:
            # Found the target epoch
            log(f"[OK] Found next checkpoint: {os.path.basename(target_checkpoint)} (epoch {target_epoch})")
            _LAST_LOADED_EPOCH = target_epoch
            return target_checkpoint, target_epoch
        
        # Checkpoint not found, check if training is still running before waiting
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        log(f" Checkpoint epoch-{target_epoch}.pt not found yet ({current_time})")
        if checkpoint_files:
            available_epochs = sorted([extract_epoch_num(f) for f in checkpoint_files])
            log(f"   Available checkpoints: {available_epochs}")
        
        # Show different message for epoch-0 (no timeout) vs other epochs (with timeout)
        if target_epoch == 0:
            log(f"   Waiting for epoch-0 to be created by training (no timeout)...")
        else:
            remaining_hours = (max_wait_seconds - elapsed_time) / 3600
            log(f"   Waiting for epoch-{target_epoch} to become available (timeout in {remaining_hours:.1f} hours)...")

        # Check if training job is still active (prioritize training job ID over benchmark job ID)
        job_id_to_check = _TRAINING_JOB_ID if _TRAINING_JOB_ID else _JOB_ID
        
        if job_id_to_check:
            log(f"   Checking if training job {job_id_to_check} is still active...")
            if not is_slurm_job_active(job_id_to_check):
                log(f" Training job {job_id_to_check} is no longer active - stopping benchmark")
                log(f"   Training appears to have stopped. No new checkpoints expected.")
                return None
            else:
                log(f"   Training job {job_id_to_check} is still active, continuing to wait for checkpoint")

        # Wait 60 seconds before retrying (1 minute as requested)
        log(f"   Waiting 60 seconds before checking again...")
        time.sleep(60)


def find_latest_model_checkpoint(model_base_path: str, constrained_model_number: Optional[int] = None, max_wait_hours: float = 2.0) -> Optional[tuple[str, int]]:
    """
    Find the latest epoch checkpoint in the torch_save directory, or a specific checkpoint if constrained_model_number is provided.
    
    If constrained_model_number is None, find the next model starting from epoch-0 (waits up to max_wait_hours for new models).
    
    Args:
        model_base_path: Base path to the model directory (e.g., misc/logs/runs/TD3Lag-.../seed-000-...)
        constrained_model_number: If provided, use this specific checkpoint number instead of latest/next
        max_wait_hours: Maximum time to wait for next checkpoint when constrained_model_number is None (default: 2 hours)
        
    Returns:
        Tuple of (checkpoint_path, epoch_number) or None if not found
    """
    torch_save_dir = os.path.join(model_base_path, "torch_save")
    
    if not os.path.exists(torch_save_dir):
        log(f" Torch save directory not found: {torch_save_dir}")
        return None
    
    # Extract epoch numbers
    def extract_epoch_num(filepath):
        match = re.search(r'epoch-(\d+)\.pt$', filepath)
        return int(match.group(1)) if match else -1
    
    if constrained_model_number is not None:
        # Look for specific checkpoint number
        pattern = os.path.join(torch_save_dir, "epoch-*.pt")
        checkpoint_files = glob.glob(pattern)
        
        if not checkpoint_files:
            log(f" No checkpoint files found in: {torch_save_dir}")
            return None
        
        target_epoch = constrained_model_number
        target_checkpoint = None
        
        for checkpoint_file in checkpoint_files:
            if extract_epoch_num(checkpoint_file) == target_epoch:
                target_checkpoint = checkpoint_file
                break
        
        if target_checkpoint is None:
            log(f" Constrained checkpoint epoch-{target_epoch}.pt not found in: {torch_save_dir}")
            log(f"Available checkpoints: {[os.path.basename(f) for f in checkpoint_files]}")
            return None
        
        log(f"[OK] Found constrained checkpoint: {os.path.basename(target_checkpoint)} (epoch {target_epoch})")
        return target_checkpoint, target_epoch
    else:
        # Use find_next_model_checkpoint to get the next model (waits up to max_wait_hours)
        return find_next_model_checkpoint(model_base_path, max_wait_hours)


def setup_omnisafe_import() -> None:
    """Set up omnisafe import from local repository."""
    main_folder = os.getcwd()
    OMNISAFE_DIR = os.path.join(main_folder, "omnisafe")
    
    if not os.path.exists(OMNISAFE_DIR):
        raise FileNotFoundError(f"Omnisafe directory not found: {OMNISAFE_DIR}")
    
    spec = importlib.util.spec_from_file_location("omnisafe", os.path.join(OMNISAFE_DIR, "__init__.py"))
    omnisafe = importlib.util.module_from_spec(spec)
    sys.modules["omnisafe"] = omnisafe
    spec.loader.exec_module(omnisafe)
    log("[OK] Omnisafe imported successfully")


def initialize_global_state(checkpoint_path: str) -> None:
    """
    Initialize global state by loading model checkpoint and building policy architecture once.
    This function should be called once at the start of the script.
    """
    global _GLOBAL_MODEL_STATE, _GLOBAL_ENV_CONFIG, _GLOBAL_POLICY_ARCHITECTURE
    
    log(f"   Loading model state globally from checkpoint...")
    
    # Load model state only once
    _GLOBAL_MODEL_STATE = torch.load(checkpoint_path, map_location="cpu")
    log(f" Loaded model state globally from: {os.path.basename(checkpoint_path)}")
    
    # Load height data (needed for environment initialization)
    height_data_path = os.path.join(os.path.dirname(__file__), 'misc', 'geo_data', 'height_data', 'height_data.npy')
    if not os.path.exists(height_data_path):
        raise FileNotFoundError(f"Height data not found: {height_data_path}")
    
    height_data = np.load(height_data_path)
    log(f" Loaded height data globally from: {height_data_path}")
    
    # Build policy architecture once by creating a temporary environment
    log(f"   Building policy architecture once...")
    
    # Import modules after omnisafe is set up
    from omnisafe.models.actor.actor_builder import ActorBuilder
    from get_parameters import para
    from class_gymenv import GymEnvOmniSafe
    
    # Create temporary environment to get observation/action spaces
    temp_env = GymEnvOmniSafe(radiation_grid_visualization=False, is_evaluation=True)
    log(f" Created temporary environment for architecture detection")
    obs_space = temp_env.observation_space
    _obs_space_desc = str(obs_space) if not hasattr(obs_space, 'shape') else str(obs_space.shape)
    log(f"   Environment observation space: {_obs_space_desc}")
    
    # Determine correct actor type based on algorithm and observation space type
    # SAC uses stochastic policy; TD3Lag uses deterministic policy.
    # When fused lidar+image observations are used (Dict obs space), use fused_sac.
    from gymnasium import spaces as _spaces
    _use_fused_obs = isinstance(temp_env.observation_space, _spaces.Dict)
    if para.omnisafe_alg == "SAC":
        actor_type = "fused_sac" if _use_fused_obs else "cnn_sac"
    else:
        actor_type = "cnn"
    log(f"   Using actor type: {actor_type} for algorithm: {para.omnisafe_alg} (fused_obs={_use_fused_obs})")
    
    # Build policy architecture and store its configuration
    actor_builder = ActorBuilder(
        temp_env.observation_space,
        temp_env.action_space,
        hidden_sizes=[256, 256]
    )
    policy = actor_builder.build_actor(actor_type=actor_type)
    log(f" Built policy architecture")
    
    # Store policy architecture configuration for subprocess reconstruction
    _GLOBAL_POLICY_ARCHITECTURE = {
        'observation_space': temp_env.observation_space,
        'action_space': temp_env.action_space,
        'hidden_sizes': [256, 256],
        'actor_type': actor_type
    }
    
    # Clean up temporary environment
    del temp_env
    log(f" Cleaned up temporary environment")
    
    # Store environment configuration
    _GLOBAL_ENV_CONFIG = {
        'height_data_path': height_data_path,
        'height_data': height_data
    }
    log(" Global state initialized successfully")


def load_model(device: torch.device):
    """
    Load the trained model from global state using pre-built architecture.
    
    Args:
        device: Torch device to load model on
        
    Returns:
        Loaded policy model in evaluation mode and environment
    """
    global _GLOBAL_MODEL_STATE, _GLOBAL_ENV_CONFIG, _GLOBAL_POLICY_ARCHITECTURE
    
    if _GLOBAL_MODEL_STATE is None or _GLOBAL_ENV_CONFIG is None or _GLOBAL_POLICY_ARCHITECTURE is None:
        raise RuntimeError("Global state not initialized. Call initialize_global_state() first.")
    
    # Import modules after omnisafe is set up
    from omnisafe.models.actor.actor_builder import ActorBuilder
    from get_parameters import para
    from class_gymenv import GymEnvOmniSafe
    
    log(f"   Using global model state and pre-built architecture...")
    
    # Create environment (quick operation since height data is cached in global state)
    gymenv = GymEnvOmniSafe(radiation_grid_visualization=False, is_evaluation=True)
    log(f" Created environment: GymEnvOmniSafe (evaluation mode)")
    
    # Rebuild policy from stored architecture configuration (no I/O, just object construction)
    actor_builder = ActorBuilder(
        _GLOBAL_POLICY_ARCHITECTURE['observation_space'],
        _GLOBAL_POLICY_ARCHITECTURE['action_space'],
        hidden_sizes=_GLOBAL_POLICY_ARCHITECTURE['hidden_sizes']
    )
    policy = actor_builder.build_actor(actor_type=_GLOBAL_POLICY_ARCHITECTURE['actor_type'])
    log(f" Rebuilt policy from global architecture")
    
    # Load checkpoint weights from global state (no file I/O)
    log(f"   Loading weights from global state...")
    
    # Try to load with strict=False to see mismatches
    try:
        policy.load_state_dict(_GLOBAL_MODEL_STATE.get("pi", _GLOBAL_MODEL_STATE), strict=True)
    except RuntimeError as e:
        log(f" Strict loading failed: {e}")
        log(f" Attempting to load with strict=False...")
        
        # Load with strict=False to allow partial loading
        missing_keys, unexpected_keys = policy.load_state_dict(_GLOBAL_MODEL_STATE.get("pi", _GLOBAL_MODEL_STATE), strict=False)
        
        if missing_keys:
            log(f" Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            log(f" Unexpected keys in current model: {unexpected_keys}")
        
        # Check if critical keys are missing
        critical_missing = [k for k in missing_keys if 'encoder' in k or 'net.0' in k]
        if critical_missing:
            log(f" Critical encoder/network keys are missing - model architecture mismatch!")
            raise RuntimeError("Critical model architecture mismatch - cannot proceed")
        
        log(f" Loaded checkpoint with warnings (non-strict mode)")
    
    policy.eval()
    log(f" Loaded checkpoint weights from global state")
    
    # Move to device
    policy = policy.to(device)
    log(f" Moved policy to device: {device}")
    
    return policy, gymenv


def run_episode_subprocess(reset_image_path: str, episode_num: int, max_steps: int, config_dict: dict, constrained_model_number: Optional[int] = None, csv_base_path: Optional[str] = None, run_number: int = 1, epoch_number: Optional[int] = None) -> dict:
    """
    Run a single episode in a subprocess - creates model and env from global state to avoid pickling issues.
    
    Args:
        reset_image_path: Path to reset image
        episode_num: Episode number for logging
        max_steps: Maximum steps per episode
        config_dict: Configuration dictionary with model paths etc.
        constrained_model_number: Model number for PNG rendering
        csv_base_path: Base path for subprocess CSV files (if provided, writes CSV immediately)
        run_number: Current run number for multi-run scenarios
        epoch_number: Model epoch number for CSV output
        
    Returns:
        Dictionary with episode metrics
    """
    try:
        # Set matplotlib backend to non-interactive for parallel processing
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Initialize random seed for this process to ensure randomization
        import random
        import numpy as np
        process_seed = episode_num  # Use episode number for unique seeds
        random.seed(process_seed)
        np.random.seed(process_seed)
        
        # Set threading limits for this process - 4 cores per process
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # Force CPU-only processing to avoid GPU conflicts
        device = torch.device('cpu')
        
        # Setup omnisafe import
        setup_omnisafe_import()
        
        # Load model and environment from global state (no file I/O)
        policy, env = load_model(device)
        
        # Run the episode
        metrics = run_episode(env, policy, reset_image_path, episode_num, max_steps, device, constrained_model_number)
        
        # If CSV base path provided, write subprocess CSV immediately
        if csv_base_path:
            write_subprocess_csv(metrics, csv_base_path, episode_num, run_number, epoch_number)
        
        return metrics
        
    except Exception as e:
        log(f" Error in subprocess episode {episode_num}: {e}")
        import traceback
        traceback.print_exc()
        # Return minimal error metrics
        error_metrics = {
            'episode_num': episode_num,
            'steps_end': 0,
            'steps_at_90': None,
            'steps_at_99': None,
            'time': 0,
            'reward': 0,
            'cost': 0,
            'final_coverage': 0,
            'final_path_length': 0,
            'path_length_at_90': None,
            'reached_90_coverage': False,
            'terminated': False,
            'truncated': False,
            'map_name': Path(reset_image_path).stem if reset_image_path else "error",
            'map_run_id': "error",
            'collision_count': None,
            'wall_gliding_distance': None,
            'sum_optv': None,
            'sum_optv_no_black': None,
            'sum_reward_if_optv_no_black': None,
            'effective_border_length': None,
        }
        
        # Still write subprocess CSV even for errors
        if csv_base_path:
            write_subprocess_csv(error_metrics, csv_base_path, episode_num, run_number, epoch_number)
        
        return error_metrics


def run_episodes_sequential(episode_args_list):
    """
    Run multiple episodes sequentially in a single process.
    This function must be at module level to be picklable by multiprocessing.
    
    Args:
        episode_args_list: List of tuples containing arguments for each episode
        
    Returns:
        List of episode metrics dictionaries
    """
    results = []
    for args in episode_args_list:
        result = run_episode_subprocess(*args)
        results.append(result)
    return results


def run_episode(env, policy, reset_image_path: str, episode_num: int, max_steps: int, device: torch.device, constrained_model_number: Optional[int] = None) -> dict:
    """
    Run a single episode and collect metrics.
    
    Args:
        env: Gymnasium environment
        policy: Trained policy model
        reset_image_path: Path to reset image
        episode_num: Episode number for logging
        max_steps: Maximum steps per episode
        device: Torch device
        
    Returns:
        Dictionary with episode metrics
    """
    from get_parameters import para
    
    # Reset environment with randomization
    full_reset_path = reset_image_path if os.path.isabs(reset_image_path) else os.path.join(os.getcwd(), reset_image_path)
    log(f" Resetting with image: {full_reset_path}")
    
    # Generate a random seed for this episode to ensure different starting conditions
    # Increment counter before each reset, starting from 0
    global _EPISODE_SEED_COUNTER
    episode_seed = _EPISODE_SEED_COUNTER
    _EPISODE_SEED_COUNTER += 1
    
    obs, info = env.reset(options={'image_path': full_reset_path}, seed=episode_seed)
    env._env.unwrapped.goal_coverage_percentage_currently = 0.990  # Reset coverage
    env._env.unwrapped.early_termination_episode_step_limit_currently = 35000  # Reset step limit
    log(f"[OK] Episode {episode_num}: Reset with image: {os.path.basename(reset_image_path)} (seed: {episode_seed})")
    
    # Verify the image path was set correctly
    actual_image_path = getattr(env._env.unwrapped, 'IMAGE_PATH', 'UNKNOWN')
    if actual_image_path != full_reset_path:
        log(f" Warning: Expected image {full_reset_path}, but environment has {actual_image_path}")
    else:
        log(f" Image path verified: {os.path.basename(actual_image_path)}")

    # Extract metadata about the active map from the underlying environment
    base_env = None
    if hasattr(env, "_env"):
        base_env = getattr(env._env, "env", None)
        if base_env is None:
            base_env = getattr(env._env, "unwrapped", None)
    map_name = Path(reset_image_path).stem if reset_image_path else ""
    map_run_identifier = ""
    if base_env is not None:
        image_path_attr = getattr(base_env, "IMAGE_PATH", None)
        if image_path_attr:
            try:
                map_name = Path(image_path_attr).stem
            except Exception:
                pass
        map_run_identifier = getattr(base_env, "gymenv_id", "") or ""
    
    # Episode tracking
    episode_start_time = time.time()
    episode_steps = 0
    episode_reward = 0
    episode_cost = 0
    terminated = False
    truncated = False
    path_length_at_90 = None
    steps_at_90 = None
    reached_90_coverage = False
    steps_at_99 = None
    reached_99_coverage = False
    total_action_sum = 0.0  # For calculating total degrees turned
    sum_optv_accumulator = 0.0      # Accumulated OPTV (normalizer * delta_border_length) per step
    sum_optv_no_black_accumulator = 0.0  # Accumulated OPTV using main_length - 2*target_length (no black penalty)
    wall_gliding_accumulator = 0.0  # Accumulated lost distance per collision step: sum(expected_dist - actual_dist)
    wall_gliding_step_count = 0     # Number of steps where wall gliding occurred (each = 1 collision stop)
    total_distance_accumulator = 0.0  # Sum of actual distance travelled per step (metres)
    
    # Run episode
    while not terminated and not truncated and episode_steps < max_steps:
        # Get action from policy
        with torch.no_grad():
            action = policy.predict(obs, deterministic=True)
        
        # Accumulate absolute action for degrees turned calculation
        # Ensure action is scalar by extracting from array/tensor if needed
        action_scalar = action[0] if hasattr(action, '__getitem__') else action
        if isinstance(action_scalar, torch.Tensor):
            action_scalar = action_scalar.item()
        elif hasattr(action_scalar, '__getitem__'):  # Still an array/list
            action_scalar = float(action_scalar[0])
        total_action_sum += abs(float(action_scalar))
        
        # Check current coverage before step (keep as decimal between 0 and 1)
        current_coverage = 1 - env._env.unwrapped.percentage_of_target_area_left
        current_path_length = env._env.unwrapped.length_of_path_in_meters

        # Capture pre-step state for per-step metric accumulation
        _pre_path_length = float(getattr(env._env.unwrapped, 'length_of_path_in_meters', 0.0))
        _pre_collisions = getattr(env._env.unwrapped, 'num_episode_collisions', 0)

        # Take step
        obs, reward, cost, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_cost += cost
        episode_steps += 1

        # Accumulate sum_optv, wall_gliding_distance, and total_distance from env state after each step
        try:
            _b = env._env.unwrapped
            _normalizer = float(getattr(_b, 'incremental_tv_reward_normalizer', 0.0))
            _delta_len = float(getattr(_b, 'length_new_measured_minus_length_old_measured', 0.0))
            _delta_len_no_black = float(getattr(_b, 'length_new_measured_minus_length_old_measured_no_black', 0.0))
            if getattr(para, 'use_tv_reward_decay', False):
                _decay = max(0.0, 1.0 - float(getattr(_b, 'current_overall_step', 0)) /
                             max(1.0, float(getattr(para, 'weight_incremental_tv_reward_zero_after_steps', 1))))
                sum_optv_accumulator += _decay * _normalizer * _delta_len
                sum_optv_no_black_accumulator += _decay * _normalizer * _delta_len_no_black
            else:
                sum_optv_accumulator += _normalizer * _delta_len
                sum_optv_no_black_accumulator += _normalizer * _delta_len_no_black
            _post_path_length = float(getattr(_b, 'length_of_path_in_meters', 0.0))
            _actual_dist = max(0.0, _post_path_length - _pre_path_length)  # metres
            total_distance_accumulator += _actual_dist
            _mpp = float(getattr(_b, 'meters_per_pixel_mower', 1.0))
            _expected_dist = float(getattr(_b, 'distance_to_move_per_step', 0.0)) * _mpp  # px -> metres
            wall_gliding_accumulator += max(0.0, _expected_dist - _actual_dist)  # loss this step
            _post_collisions = getattr(_b, 'num_episode_collisions', 0)
            if _post_collisions > _pre_collisions:
                wall_gliding_step_count += 1
        except Exception:
            pass

    # Check if we crossed 0.90 coverage fraction
        if not reached_90_coverage and current_coverage >= 0.90:
            path_length_at_90 = current_path_length
            steps_at_90 = episode_steps
            reached_90_coverage = True
            log(
                f"   Coverage reached 0.90 at path length: {path_length_at_90:.2f}m (step {episode_steps})"
            )
        if not reached_99_coverage and current_coverage >= 0.99:
            steps_at_99 = episode_steps
            reached_99_coverage = True
            log(f"   Coverage reached 0.99 at step {episode_steps}")
        
        # Log progress
        if episode_steps % 500 == 0:
            log(f"  Step {episode_steps}: Coverage {current_coverage:.3f}, Path {current_path_length:.1f}m")
    
    # Collect final metrics
    episode_time = time.time() - episode_start_time
    final_coverage = current_coverage
    final_path_length = current_path_length
    
    # Calculate total degrees turned
    total_degrees_turned = para.largest_horizontal_turning_angle_deg * total_action_sum
    
    # Convert tensors to scalars
    if isinstance(episode_reward, torch.Tensor):
        episode_reward = episode_reward.item()
    if isinstance(episode_cost, torch.Tensor):
        episode_cost = episode_cost.item()
    if isinstance(final_coverage, torch.Tensor):
        final_coverage = final_coverage.item()
    if isinstance(final_path_length, torch.Tensor):
        final_path_length = final_path_length.item()
    if isinstance(path_length_at_90, torch.Tensor):
        path_length_at_90 = path_length_at_90.item()
    
    # -------------------------------------------------------
    # Extract additional metrics: collision frequency,
    # wall-gliding distance, sum(OPTV), effective border length
    # -------------------------------------------------------
    try:
        _base = env._env.unwrapped
        collision_count = wall_gliding_step_count  # Each step with wall gliding = 1 collision stop
        wall_gliding_distance = wall_gliding_accumulator
        sum_optv = sum_optv_accumulator
        sum_optv_no_black = sum_optv_no_black_accumulator
        total_distance_travelled = total_distance_accumulator
        # Border lengths and effective border length
        try:
            _prev = getattr(_base, 'previous_episode_data', {})
            _ta = _prev.get('target_area', None)  # covered region saved before auto-reset
            _obs = getattr(_base, 'black_polygons_dilated', None)  # static per map, survives reset
            _mpp = float(getattr(_base, 'meters_per_pixel_mower', 1.0))
            if _ta is not None and not _ta.is_empty:
                # Compute target area boundary length component-by-component (mirrors obstacle intersection approach)
                try:
                    _ta_boundary = _base.safe_boundary(_ta)
                    _ta_len_px = 0.0
                    _ta_lines = list(_ta_boundary.geoms) if hasattr(_ta_boundary, 'geoms') else [_ta_boundary]
                    for _ta_line in _ta_lines:
                        if _ta_line is None or _ta_line.is_empty:
                            continue
                        _ta_len_px += _ta_line.length
                    border_length_target_area = _ta_len_px * _mpp
                except Exception as _e_ta:
                    log(f"   [BORDER] target area boundary calc failed ({type(_e_ta).__name__}: {_e_ta})")
                    border_length_target_area = None
                if _obs is not None and not _obs.is_empty:
                    try:
                        # Obstacle boundary clipped to target area = obstacle perimeter inside the field.
                        # Iterate per-component to avoid MultiLineString.intersection(MultiPolygon)
                        # topology failures on older GEOS versions.
                        _obs_boundary = _base.safe_boundary(_obs)
                        _overlap_len_px = 0.0
                        _lines = list(_obs_boundary.geoms) if hasattr(_obs_boundary, 'geoms') else [_obs_boundary]
                        for _line in _lines:
                            if _line is None or _line.is_empty:
                                continue
                            try:
                                _result = _line.intersection(_ta)
                                _overlap_len_px += _result.length
                            except Exception:
                                try:
                                    from shapely.validation import make_valid
                                    _result = make_valid(_line).intersection(make_valid(_ta))
                                    _overlap_len_px += _result.length
                                except Exception:
                                    pass
                    except Exception as _e3:
                        log(f"   [BORDER] obs boundary calc failed ({type(_e3).__name__}: {_e3})")
                        _overlap_len_px = 0.0
                else:
                    _overlap_len_px = 0.0
                border_length_obstacle_intersection = _overlap_len_px * _mpp
                # Effective border = perimeter of free area (target area minus obstacles)
                try:
                    _free_area = _base.safe_geometry_difference(_ta, _obs)
                    _free_boundary = _base.safe_boundary(_free_area)
                    effective_border_length = _free_boundary.length * _mpp if (_free_boundary is not None and not _free_boundary.is_empty) else border_length_target_area
                except Exception:
                    effective_border_length = border_length_target_area
            else:
                border_length_target_area = None
                border_length_obstacle_intersection = None
                effective_border_length = None
        except Exception:
            border_length_target_area = None
            border_length_obstacle_intersection = None
            effective_border_length = None
    except Exception:
        collision_count = None
        wall_gliding_distance = None
        sum_optv = None
        sum_optv_no_black = None
        total_distance_travelled = None
        border_length_target_area = None
        border_length_obstacle_intersection = None
        effective_border_length = None

    metrics = {
        'episode_num': episode_num,
        'steps_end': episode_steps,
        'steps_at_90': steps_at_90,
        'steps_at_99': steps_at_99,
        'time': episode_time,
        'reward': episode_reward,
        'cost': episode_cost,
        'final_coverage': final_coverage,
        'final_path_length': final_path_length,
        'path_length_at_90': path_length_at_90,
        'reached_90_coverage': reached_90_coverage,
        'terminated': terminated,
        'truncated': truncated,
        'map_name': map_name,
        'map_run_id': map_run_identifier,
        'total_degrees_turned': total_degrees_turned,
        'collision_count': collision_count,
        'wall_gliding_distance': wall_gliding_distance,
        'sum_optv': sum_optv,
        'sum_optv_no_black': sum_optv_no_black,
        'sum_reward_if_optv_no_black': (episode_reward - sum_optv + sum_optv_no_black) if (sum_optv is not None and sum_optv_no_black is not None) else None,
        'effective_border_length': effective_border_length,
        'total_distance_travelled': total_distance_travelled,
        'border_length_target_area': border_length_target_area,
        'border_length_obstacle_intersection': border_length_obstacle_intersection,
        'sum_reward': episode_reward,
    }
    
    log(f" Episode {episode_num} complete: {episode_steps} steps, "
        f"{final_coverage:.3f} coverage, {final_path_length:.1f}m path, "
          f"{episode_time:.1f}s")
    
    # Save PNG render if constrained_model_number is an integer (not None or list)
    if constrained_model_number is not None and isinstance(constrained_model_number, int):
        try:
            # Create images directory
            images_dir = f"images_of_paths/benchmark_results_{constrained_model_number}"
            os.makedirs(images_dir, exist_ok=True)
            
            # Generate filename based on episode and map
            map_name = Path(reset_image_path).stem if reset_image_path else "unknown_map"
            png_filename = f"episode_{episode_num:03d}_{map_name}_model_{constrained_model_number}.png"
            png_path = os.path.join(images_dir, png_filename)
            
            # Render the environment and save as PNG
            try:
                # Access the underlying GymnasiumEnv for rendering with save_path
                # The env is GymEnvOmniSafe -> AutoResetWrapper -> GymnasiumEnv
                base_env = None
                if hasattr(env, '_env') and hasattr(env._env, 'env'):
                    # GymEnvOmniSafe._env is AutoResetWrapper, AutoResetWrapper.env is GymnasiumEnv
                    base_env = env._env.env
                elif hasattr(env, '_env'):
                    # Try direct access if structure is different
                    base_env = env._env
                elif hasattr(env, 'unwrapped'):
                    # Try using unwrapped attribute
                    base_env = env.unwrapped
                
                if base_env and hasattr(base_env, 'render'):
                    # Try calling render with save_path on the base environment
                    # Use use_previous_episode=True to get the completed episode data
                    base_env.render(save_path=png_path)
                    
                    # Check if file was created successfully
                    if os.path.exists(png_path):
                        log(f"   Saved render: {png_path}")
                    else:
                        log(f"   Warning: render() did not create file at {png_path}")
                    
                    # Critical: Clean up matplotlib resources to prevent memory leaks in parallel processes
                    try:
                        import matplotlib.pyplot as plt
                        plt.close('all')  # Close all matplotlib figures
                        import gc
                        gc.collect()  # Force garbage collection to free memory
                    except Exception as cleanup_error:
                        log(f"   Note: Cleanup warning: {cleanup_error}")
                else:
                    log(f"   Warning: Could not access base environment for rendering")
                    
            except Exception as e:
                # Handle any other rendering errors
                log(f"   Warning: Failed to render episode {episode_num}: {e}")
                log(f"   Skipping PNG save for episode {episode_num}")
                # Still try to clean up matplotlib
                try:
                    import matplotlib.pyplot as plt
                    plt.close('all')
                except:
                    pass
                
        except Exception as e:
            log(f"   Warning: Failed to save render for episode {episode_num}: {e}")
    
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the local benchmark and optionally store metrics to CSV or log to wandb.")
    parser.add_argument("--save-csv", type=str, default=None,
                        help="Optional path to save per-episode metrics as CSV. Relative paths resolve against the working directory.")
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Optional sweep identifier to embed into the CSV output or use for wandb logging.")
    parser.add_argument("--job-id", type=str, default=None,
                        help="Optional job ID for logging filename (creates {job_id}.txt).")
    parser.add_argument("--training-job-id", type=str, default=None,
                        help="Optional training job ID to monitor. When provided, benchmark will wait for new models while training job is still running.")
    parser.add_argument("--constrained-model-number", type=int, default=None,
                        help="Optional specific model checkpoint number to use instead of latest. "
                             "When specified, continuous-loop is disabled and CSV filename includes model number.")
    parser.add_argument("--constrained-model-numbers", type=str, default=None,
                        help="Optional comma-separated list of model checkpoint numbers to evaluate sequentially. "
                             "Example: '315,320,325'. When specified, runs through each model in order.")
    parser.add_argument("--wandb-logging", action="store_true",
                        help="Log metrics to wandb instead of saving to CSV. Requires --sweep-id.")
    parser.add_argument("--wandb-project", type=str, default="Heli-Logs",
                        help="Wandb project name for logging (default: Heli-Logs).")
    parser.add_argument("--continuous-loop", action="store_true",
                        help="Run benchmark in an infinite loop with 3-hour intervals between runs.")
    parser.add_argument("--loop-interval", type=int, default=10800,
                        help="Interval between benchmark runs in continuous loop mode (default: 10800 seconds = 3 hours).")
    parser.add_argument("--num-episodes", type=int, default=600,
                        help="Total number of episodes to run (default: 600).")
    parser.add_argument("--processes", type=int, default=100,
                        help="Number of parallel processes (default: 100).")
    parser.add_argument("--max-steps-per-episode", type=int, default=50000,
                        help="Maximum steps per episode (default: 50000).")
    parser.add_argument("--map-filter", type=str, default=None,
                        help="Optional comma-separated list of map IDs to include (e.g., '9,10,11,12,13'). "
                             "Map IDs correspond to eval_mowing_X.png files. When specified, only these maps are used.")
    parser.add_argument("--noise-type", type=str, default="none",
                        choices=["none", "action", "observation", "pose", "pose_heading", "pose_position", "pose_combined"],
                        help="Noise study type: 'none' (default), 'action', 'observation', 'pose', "
                             "'pose_heading' (absolute std in degrees), 'pose_position' (absolute std in meters).")
    parser.add_argument("--noise-intensity", type=float, default=0.0,
                        help="Noise intensity as a fraction (e.g., 0.05 = 5%%, 0.30 = 30%%). Default: 0.0.")
    return parser.parse_args(argv)


def resolve_csv_path(csv_path: str) -> Path:
    path = Path(csv_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_subprocess_csv(metrics: dict, csv_base_path: str, episode_num: int, run_number: int, epoch_number: Optional[int]) -> None:
    """
    Write a single episode's metrics to a subprocess-specific CSV file.
    Each subprocess creates its own CSV to avoid contention.
    
    Args:
        metrics: Episode metrics dictionary
        csv_base_path: Base CSV path (e.g., "benchmark_results_sweep123.csv")
        episode_num: Episode number for unique filename
        run_number: Run number for multi-run scenarios
        epoch_number: Model epoch number
    """
    try:
        # Load omnisafe config to get steps_per_epoch and vector_env_nums
        from get_parameters import para
        import yaml
        
        omnisafe_alg = para.omnisafe_alg
        config_path = os.path.join(os.getcwd(), 'omnisafe', 'configs', 'off-policy', f'{omnisafe_alg}.yaml')
        
        steps_per_epoch = None
        vector_env_nums = None
        env_steps_per_epoch = None
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if 'defaults' in config_data:
                    if 'algo_cfgs' in config_data['defaults']:
                        steps_per_epoch = config_data['defaults']['algo_cfgs'].get('steps_per_epoch')
                    if 'train_cfgs' in config_data['defaults']:
                        vector_env_nums = config_data['defaults']['train_cfgs'].get('vector_env_nums')
            
            if steps_per_epoch is not None and vector_env_nums is not None and vector_env_nums > 0:
                env_steps_per_epoch = steps_per_epoch / vector_env_nums
        
        base_timestep = para.base_timestep
        from get_parameters import para
        
        # Create subprocess CSV filename by inserting episode number before extension
        base_path = Path(csv_base_path)
        subprocess_csv = base_path.parent / f"{base_path.stem}_ep{episode_num:04d}{base_path.suffix}"
        
        # Ensure parent directory exists
        subprocess_csv.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            "run_number",
            "timestamp",
            "model",
            "epoch",
            "map",
            "map_run_id",
            "ablation_study_name",
            "seed",
            "noise_type",
            "noise_intensity",
            "steps_90_pct",
            "steps_99_pct",
            "steps_end",
            "base_timestep",
            "env_steps_per_epoch",
            "training_steps_per_epoch",
            "path_length_90_pct",
            "path_length_end",
            "coverage_end",
            "total_degrees_turned",
            "collision_count",
            "wall_gliding_distance",
            "sum_optv",
            "sum_optv_no_black",
            "effective_border_length",
            "total_distance_travelled",
            "border_length_target_area",
            "border_length_obstacle_intersection",
            "sum_reward",
            "sum_reward_if_optv_no_black",
        ]
        
        # Prepare row data
        current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Access parameters safely with fallback
        try:
            ablation_name = para.ablation_study_name
        except AttributeError:
            ablation_name = ""
        
        try:
            seed_value = para.seed
        except AttributeError:
            seed_value = ""
        
        row = {
            "run_number": run_number,
            "timestamp": current_timestamp,
            "model": "Ours",
            "epoch": f"epoch-{epoch_number}" if epoch_number is not None else "",
            "map": metrics.get("map_name", ""),
            "map_run_id": metrics.get("map_run_id", ""),
            "ablation_study_name": ablation_name,
            "seed": seed_value,
            "noise_type": os.environ.get('BENCHMARK_NOISE_TYPE', 'none'),
            "noise_intensity": os.environ.get('BENCHMARK_NOISE_INTENSITY', '0.0'),
            "steps_90_pct": metrics.get("steps_at_90"),
            "steps_99_pct": metrics.get("steps_at_99"),
            "steps_end": metrics.get("steps_end"),
            "base_timestep": base_timestep,
            "env_steps_per_epoch": env_steps_per_epoch,
            "training_steps_per_epoch": steps_per_epoch,
            "path_length_90_pct": metrics.get("path_length_at_90"),
            "path_length_end": metrics.get("final_path_length"),
            "coverage_end": metrics.get("final_coverage"),
            "total_degrees_turned": metrics.get("total_degrees_turned"),
            "collision_count": metrics.get("collision_count"),
            "wall_gliding_distance": metrics.get("wall_gliding_distance"),
            "sum_optv": metrics.get("sum_optv"),
            "sum_optv_no_black": metrics.get("sum_optv_no_black"),
            "effective_border_length": metrics.get("effective_border_length"),
            "total_distance_travelled": metrics.get("total_distance_travelled"),
            "border_length_target_area": metrics.get("border_length_target_area"),
            "border_length_obstacle_intersection": metrics.get("border_length_obstacle_intersection"),
            "sum_reward": metrics.get("sum_reward"),
            "sum_reward_if_optv_no_black": metrics.get("sum_reward_if_optv_no_black"),
        }
        
        # Write CSV file
        with subprocess_csv.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Normalize values
            normalized_row = {}
            for key in fieldnames:
                value = row.get(key, "")
                if key in ("sum_optv", "sum_optv_no_black", "sum_reward_if_optv_no_black") and isinstance(value, float):
                    normalized_row[key] = f"{value:.10f}"
                elif isinstance(value, float):
                    normalized_row[key] = f"{value:.3f}"
                elif value is None:
                    normalized_row[key] = ""
                else:
                    normalized_row[key] = str(value)
            writer.writerow(normalized_row)
        
    except Exception as e:
        # Don't fail the episode if CSV writing fails
        log(f"   Warning: Failed to write subprocess CSV for episode {episode_num}: {e}")





def combine_subprocess_csvs(csv_base_path: str, num_episodes: int, append_mode: bool = False) -> Path:
    """
    Combine individual subprocess CSV files into a single master CSV.
    
    Args:
        csv_base_path: Base CSV path (e.g., "benchmark_results_sweep123.csv")
        num_episodes: Total number of episodes (number of subprocess CSV files)
        append_mode: If True, append to existing master CSV; if False, create new
        
    Returns:
        Path to the combined CSV file
    """
    base_path = Path(csv_base_path)
    output_path = resolve_csv_path(csv_base_path)
    
    log(f"   Combining {num_episodes} subprocess CSV files into: {output_path.name}")
    
    fieldnames = [
        "run_number",
        "timestamp",
        "model",
        "epoch",
        "map",
        "map_run_id",
        "ablation_study_name",
        "seed",
        "noise_type",
        "noise_intensity",
        "steps_90_pct",
        "steps_99_pct",
        "steps_end",
        "base_timestep",
        "env_steps_per_epoch",
        "training_steps_per_epoch",
        "path_length_90_pct",
        "path_length_end",
        "coverage_end",
        "total_degrees_turned",
        "collision_count",
        "wall_gliding_distance",
        "sum_optv",
            "sum_optv_no_black",
            "effective_border_length",
            "total_distance_travelled",
            "border_length_target_area",
            "border_length_obstacle_intersection",
            "sum_reward",
            "sum_reward_if_optv_no_black",
        ]
    
    # Check if master file exists and has data for append mode
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    mode = "a" if append_mode and file_exists else "w"
    
    rows_written = 0
    with output_path.open(mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if creating new file or file is empty
        if mode == "w" or not file_exists:
            writer.writeheader()
        
        # Read and combine all subprocess CSV files
        for episode_num in range(1, num_episodes + 1):
            subprocess_csv = base_path.parent / f"{base_path.stem}_ep{episode_num:04d}{base_path.suffix}"
            
            if not subprocess_csv.exists():
                log(f"     Warning: Subprocess CSV not found: {subprocess_csv.name}")
                continue
            
            try:
                with subprocess_csv.open("r", newline="") as subprocess_file:
                    reader = csv.DictReader(subprocess_file)
                    for row in reader:
                        writer.writerow(row)
                        rows_written += 1
                
                # Delete subprocess CSV after successful read
                subprocess_csv.unlink()
                
            except Exception as e:
                log(f"     Warning: Failed to read subprocess CSV {subprocess_csv.name}: {e}")
    
    action = "Appended" if append_mode and file_exists else "Wrote"
    log(f"   {action} {rows_written} rows to master CSV: {output_path}")
    return output_path


def write_results_csv(rows: List[dict], csv_path: str, append_mode: bool = False) -> Path:
    """Legacy function - kept for backward compatibility or direct row writing."""
    output_path = resolve_csv_path(csv_path)
    fieldnames = [
        "run_number",
        "timestamp", 
        "model",
        "epoch",
        "map",
        "map_run_id",
        "ablation_study_name",
        "seed",
        "noise_type",
        "noise_intensity",
        "collision_count",
        "wall_gliding_distance",
        "sum_optv",
        "sum_optv_no_black",
        "effective_border_length",
        "total_distance_travelled",
        "border_length_target_area",
        "border_length_obstacle_intersection",
        "sum_reward",
        "sum_reward_if_optv_no_black",
    ]

    # Check if file exists and has data for append mode
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    mode = "a" if append_mode and file_exists else "w"
    
    with output_path.open(mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        
        # Write header only if creating new file or file is empty
        if mode == "w" or not file_exists:
            writer.writeheader()
            
        for row in rows:
            normalized_row = {}
            for key in fieldnames:
                value = row.get(key, "")
                if key in ("sum_optv", "sum_optv_no_black", "sum_reward_if_optv_no_black") and isinstance(value, float):
                    normalized_row[key] = f"{value:.10f}"
                elif isinstance(value, float):
                    normalized_row[key] = f"{value:.3f}"
                elif value is None:
                    normalized_row[key] = ""
                else:
                    normalized_row[key] = str(value)
            writer.writerow(normalized_row)

    action = "Appended" if append_mode and file_exists else "Wrote"
    log(f" {action} benchmark metrics to CSV: {output_path}")
    return output_path


def find_wandb_run_by_sweep_id(sweep_id: str) -> Optional[str]:
    """Find the correct wandb run ID for the given sweep ID."""
    try:
        import wandb
        
        # Try to use wandb API to find the specific sweep and its runs
        try:
            log(f" Connecting to wandb API to find sweep '{sweep_id}'...")
            
            api = wandb.Api(timeout=30)
            project = "Heli-Logs"
            entity = "johanndavidblake-ludwig-maximilianuniversity-of-munich"
            
            log(f" Searching for sweep '{sweep_id}' in {entity}/{project}")
            
            # First, try to get the sweep directly
            try:
                sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
                log(f" Found sweep: {sweep.name} ({sweep.id})")
                
                # Get runs from the sweep
                runs = list(sweep.runs)
                log(f"   Sweep has {len(runs)} runs")
                
                if runs:
                    # Get the first (or most recent) run from the sweep
                    target_run = runs[0]
                    log(f" Found run ID: {target_run.id} from sweep {sweep_id}")
                    log(f"   Run name: {target_run.name}")
                    log(f"   Run state: {target_run.state}")
                    return target_run.id
                else:
                    log(" Sweep found but has no runs")
                    
            except wandb.errors.CommError as sweep_error:
                log(f"   Could not access sweep directly: {sweep_error}")
                log("   Trying alternative method...")
                
                # Alternative: search through all runs for ones with the sweep ID
                log(f" Searching all runs in {entity}/{project} for sweep {sweep_id}")
                
                runs = api.runs(f"{entity}/{project}")
                found_runs = []
                
                for run in runs:
                    if hasattr(run, 'sweep') and run.sweep:
                        if run.sweep.id == sweep_id:
                            found_runs.append(run)
                            log(f"   Found run {run.id} ({run.name}) in sweep {sweep_id}")
                
                if found_runs:
                    # Use the first found run
                    target_run = found_runs[0]
                    log(f" Selected run ID: {target_run.id} from sweep {sweep_id}")
                    return target_run.id
                else:
                    log(f" No runs found for sweep {sweep_id}")
            
        except Exception as api_error:
            log(f" Wandb API search failed: {type(api_error).__name__}: {api_error}")
            
        # Fallback to local wandb directory search
        log("   Falling back to local wandb directory search")
        
        # Try to find runs in wandb directory by sweep ID pattern
        wandb_dir = Path("wandb")
        if not wandb_dir.exists():
            log(" No wandb directory found")
            return None
            
        # Look for runs that contain the sweep_id
        run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
        
        if not run_dirs:
            log(" No wandb run directories found")
            return None
        
        log(f" Searching {len(run_dirs)} local run directories for sweep {sweep_id}")
        
        # Search through run directories for sweep ID
        for run_dir in sorted(run_dirs, reverse=True):  # Most recent first
            try:
                # Extract run ID from directory name
                run_id = run_dir.name.split("-")[-1]
                
                # Check if any files in the run directory contain the sweep_id
                files_dir = run_dir / "files"
                if files_dir.exists():
                    # Check wandb-summary.json and config.yaml specifically
                    for filename in ["wandb-summary.json", "config.yaml", "wandb-metadata.json"]:
                        file_path = files_dir / filename
                        if file_path.exists():
                            try:
                                content = file_path.read_text(encoding="utf-8", errors="ignore")
                                if sweep_id in content:
                                    log(f" Found sweep {sweep_id} in {file_path}")
                                    log(f" Using run ID: {run_id}")
                                    return run_id
                            except Exception:
                                continue
                    
                    # Also check other files as fallback
                    for file_path in files_dir.rglob("*"):
                        if file_path.is_file() and file_path.suffix in ['.yaml', '.json', '.txt', '.log']:
                            try:
                                content = file_path.read_text(encoding="utf-8", errors="ignore")
                                if sweep_id in content:
                                    log(f" Found sweep {sweep_id} in {file_path}")
                                    log(f" Using run ID: {run_id}")
                                    return run_id
                            except Exception:
                                continue
            except Exception as dir_error:
                log(f"   Error checking {run_dir}: {dir_error}")
                continue
        
        log(f" Could not find any run associated with sweep {sweep_id}")
        return None
        
    except Exception as e:
        log(f" Error finding wandb run: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_metrics_from_episodes(all_metrics: List[dict]) -> Dict[str, List[Dict[str, any]]]:
    """Parse metrics from episode list and group by map_name (PNG filename without extension)."""
    metrics_by_map = {}
    
    for episode_metric in all_metrics:
        map_name = episode_metric.get('map_name', 'unknown')
        map_run_id = episode_metric.get('map_run_id', 'unknown')
        
        # Convert metric format to match CSV structure
        metric = {
            'model': 'Ours',
            'map': map_name,
            'map_run_id': map_run_id,
            'path_length_90_pct': episode_metric.get('path_length_at_90'),
            'path_length_end': episode_metric.get('final_path_length'),
            'coverage_end': episode_metric.get('final_coverage'),
            'reward': episode_metric.get('reward'),
            'steps_end': episode_metric.get('steps_end'),
            'total_degrees_turned': episode_metric.get('total_degrees_turned'),
        }
        
        # Group by map_name instead of map_run_id for wandb logging
        if map_name not in metrics_by_map:
            metrics_by_map[map_name] = []
        metrics_by_map[map_name].append(metric)
    
    return metrics_by_map


def calculate_averages(metrics_list: List[Dict[str, any]]) -> Dict[str, float]:
    """Calculate averages for metrics, handling None values."""
    def safe_avg(values):
        valid_values = [v for v in values if v is not None]
        return statistics.mean(valid_values) if valid_values else None
    
    path_90_values = [m['path_length_90_pct'] for m in metrics_list]
    path_end_values = [m['path_length_end'] for m in metrics_list]
    coverage_values = [m['coverage_end'] for m in metrics_list]
    degrees_turned_values = [m['total_degrees_turned'] for m in metrics_list if m.get('total_degrees_turned') is not None]
    
    # Calculate average reward per step
    reward_per_step_values = []
    for m in metrics_list:
        if m.get('reward') is not None and m.get('steps_end') is not None and m['steps_end'] > 0:
            reward_per_step_values.append(m['reward'] / m['steps_end'])
    
    return {
        'avg_path_length_90_pct': safe_avg(path_90_values),
        'avg_path_length_end': safe_avg(path_end_values),
        'avg_coverage_end': safe_avg(coverage_values),
        'avg_reward_per_step': safe_avg(reward_per_step_values),
        'avg_total_degrees_turned': safe_avg(degrees_turned_values),
        'count': len(metrics_list)
    }


def log_metrics_to_wandb(all_metrics: List[dict], sweep_id: str, project_name: str = "Heli-Logs") -> None:
    """Log benchmark metrics directly to wandb run without CSV intermediary."""
    try:
        import wandb
    except ImportError:
        log(" Wandb not available. Please install wandb: pip install wandb")
        return
    
    if not sweep_id:
        log(" No sweep ID provided for wandb logging")
        return
    
    log(f" Logging metrics to wandb for sweep: {sweep_id}")
    
    # Find corresponding wandb run
    run_id = find_wandb_run_by_sweep_id(sweep_id)
    if run_id is None:
        log(" Could not find corresponding wandb run. Metrics not logged.")
        return
    
    # Parse metrics from episodes (similar to CSV parsing)
    metrics_by_map = parse_metrics_from_episodes(all_metrics)
    
    if not metrics_by_map:
        log(" No metrics found to log")
        return
    
    run = None
    try:
        log(f"   Connecting to wandb project: {project_name}")
        log(f"   Attempting to connect to run with ID: {run_id}")
        
        # Try multiple approaches to connect to the run
        connection_methods = [
            # Method 1: Try to resume existing run
            {"id": run_id, "resume": "allow", "project": project_name, "reinit": True},
            # Method 2: Try to resume with must resume
            {"id": run_id, "resume": "must", "project": project_name, "reinit": True},
            # Method 3: Create new run if others fail
            {"project": project_name, "reinit": True, "name": f"benchmark_logging_{run_id}"}
        ]
        
        for i, method in enumerate(connection_methods):
            try:
                log(f"     Trying connection method {i+1}...")
                run = wandb.init(**method)
                
                if run is not None:
                    log(f" Connected to wandb run: {run.name} ({run.id}) using method {i+1}")
                    break
                    
            except Exception as method_error:
                log(f"     Method {i+1} failed: {type(method_error).__name__}")
                if run:
                    try:
                        wandb.finish()
                    except:
                        pass
                    run = None
                continue
        
        if run is None:
            log(" Failed to connect to any wandb run")
            return
        
        # Use current timestamp for step value
        current_time = time.time()
        step_value = int(current_time)
        
        # Log metrics for each map (grouped by map name)
        for map_name, metrics_list in metrics_by_map.items():
            if not metrics_list:
                continue
                
            averages = calculate_averages(metrics_list)
            
            log(f"   Map {map_name}: {averages['count']} episodes")
            
            # Log individual map metrics under Test Runs subsection using PNG filename
            log_data = {}
            
            if averages['avg_path_length_90_pct'] is not None:
                log_data[f"Test_Runs/path_length_90_pct_{map_name}"] = averages['avg_path_length_90_pct']
                
            if averages['avg_path_length_end'] is not None:
                log_data[f"Test_Runs/path_length_end_{map_name}"] = averages['avg_path_length_end']
                
            if averages['avg_coverage_end'] is not None:
                log_data[f"Test_Runs/coverage_end_{map_name}"] = averages['avg_coverage_end']
            
            if averages['avg_reward_per_step'] is not None:
                log_data[f"Test_Runs/avg_reward_per_step_{map_name}"] = averages['avg_reward_per_step']
            
            if averages['avg_total_degrees_turned'] is not None:
                log_data[f"Test_Runs/total_degrees_turned_{map_name}"] = averages['avg_total_degrees_turned']
            
            if log_data:
                log(f"     Logging: {log_data}")
                try:
                    wandb.log(log_data, step=step_value)
                    log(f"      Successfully logged data for {map_name} at step {step_value}")
                except Exception as log_error:
                    log(f"      Failed to log data for {map_name}: {log_error}")
        
        # Calculate and log combined averages over all maps
        all_combined_metrics = []
        for metrics_list in metrics_by_map.values():
            all_combined_metrics.extend(metrics_list)
        
        if all_combined_metrics:
            combined_averages = calculate_averages(all_combined_metrics)
            
            log(f"   Combined average over all maps: {combined_averages['count']} total episodes")
            
            combined_log_data = {}
            
            if combined_averages['avg_path_length_90_pct'] is not None:
                combined_log_data["Test_Runs/path_length_90_pct_combined"] = combined_averages['avg_path_length_90_pct']
                
            if combined_averages['avg_path_length_end'] is not None:
                combined_log_data["Test_Runs/path_length_end_combined"] = combined_averages['avg_path_length_end']
                
            if combined_averages['avg_coverage_end'] is not None:
                combined_log_data["Test_Runs/coverage_end_combined"] = combined_averages['avg_coverage_end']
            
            if combined_averages['avg_reward_per_step'] is not None:
                combined_log_data["Test_Runs/avg_reward_per_step_combined"] = combined_averages['avg_reward_per_step']
            
            if combined_averages['avg_total_degrees_turned'] is not None:
                combined_log_data["Test_Runs/total_degrees_turned_combined"] = combined_averages['avg_total_degrees_turned']
            
            if combined_log_data:
                log(f"     Logging combined: {combined_log_data}")
                try:
                    wandb.log(combined_log_data, step=step_value)
                    log(f"      Successfully logged combined data at step {step_value}")
                except Exception as log_error:
                    log(f"      Failed to log combined data: {log_error}")
        
        log(" Successfully logged metrics to wandb")
        
    except Exception as e:
        log(f" Error logging to wandb: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Don't finish the run if it was already active - let the original training continue
        try:
            if run and run.id != run_id:
                # Only finish if we created a new run, not if we resumed an existing one
                wandb.finish()
            else:
                log("   Keeping wandb run active (not finishing)")
        except Exception as finish_error:
            log(f"   Note: {finish_error}")


def run_single_benchmark(args, config, device, policy=None, env=None, reload_model=False, run_number: int = 1, checkpoint_path: str = None, epoch_number: int = None):
    """Run a single benchmark session with all episodes."""
    
    # If reload_model is True, find and load the latest checkpoint
    if reload_model:
        log(f"\n[R] Reloading latest model checkpoint...")
        
        # Find the latest checkpoint (ignore constrained_model_number in continuous mode)
        latest_checkpoint_result = find_latest_model_checkpoint(config.model_base_path, None)
        if latest_checkpoint_result is None:
            log(" Failed to find latest model checkpoint.")
            log("   This typically means the training job has stopped and no new checkpoints are being generated.")
            # Return None for epoch_number to signal that training has stopped
            return policy, env, None
        else:
            try:
                # Update global state with new checkpoint
                latest_checkpoint, latest_epoch = latest_checkpoint_result
                initialize_global_state(latest_checkpoint)
                
                # Load the new model from updated global state
                new_policy, new_env = load_model(device)
                policy = new_policy
                env = new_env
                checkpoint_path = latest_checkpoint  # Update checkpoint path for logging
                epoch_number = latest_epoch  # Update epoch number
                log(f" Reloaded model from: {os.path.basename(latest_checkpoint)}")
            except Exception as e:
                log(f" Failed to reload model: {e}")
                log("   Using previous model...")
    
    # Ensure we have a checkpoint path for parallel execution
    if checkpoint_path is None:
        log(" No checkpoint path available for parallel execution. This should not happen.")
        return policy, env, None
    
    # Run benchmark episodes in parallel
    num_episodes = args.num_episodes
    processes = args.processes
    log(f"\n[5/5] Running {num_episodes} benchmark episodes in parallel ({processes} processes)...")
    
    # Show episode distribution across images
    log(f"   Using {len(config.reset_image_paths)} evaluation images:")
    # When num_episodes equals number of maps, each map gets exactly 1 episode
    if num_episodes == len(config.reset_image_paths):
        for img_path in config.reset_image_paths:
            log(f"    {os.path.basename(img_path)}: 1 episode")
    else:
        episodes_per_image = num_episodes // len(config.reset_image_paths)
        remaining_episodes = num_episodes % len(config.reset_image_paths)
        for i, img_path in enumerate(config.reset_image_paths):
            extra = 1 if i < remaining_episodes else 0
            count = episodes_per_image + extra
            log(f"    {os.path.basename(img_path)}: {count} episodes")
    
    log("-" * 80)
    
    # Prepare arguments for parallel execution
    episode_args = []
    
    # Determine CSV base path for subprocess CSV files
    csv_base_path = args.save_csv if args.save_csv else None
    
    # FIXED: Split episodes into two groups for controlled process assignment
    # Group 1: Maps 9-12 (4 maps) - will run on Process 1
    # Group 2: Maps 13-14 (2 maps) - will run on Process 2
    log(f"   Using controlled process assignment:")
    log(f"   - Process 1: Maps 9-12 ({min(4, len(config.reset_image_paths)-2)} episodes)")
    log(f"   - Process 2: Maps 13-14 (2 episodes)")
    
    # Create episode args for all maps
    for episode in range(num_episodes):
        # Each episode gets its own map (no repetition)
        map_index = episode % len(config.reset_image_paths)
        reset_image = config.reset_image_paths[map_index]
        
        episode_args.append((
            reset_image,
            episode + 1,
            args.max_steps_per_episode,
            {},  # config_dict - not currently used but kept for compatibility
            args.constrained_model_number,  # Pass constrained_model_number for PNG rendering
            csv_base_path,  # Base path for subprocess CSV files
            run_number,  # Run number for multi-run scenarios
            epoch_number  # Epoch number for CSV output
        ))
    
    # Split episodes into two groups based on map
    # Group 1: First 4 episodes (maps 9-12)
    # Group 2: Last 2 episodes (maps 13-14)
    episode_args_group1 = episode_args[:4] if len(episode_args) >= 4 else episode_args
    episode_args_group2 = episode_args[4:] if len(episode_args) > 4 else []
    
    # Run episodes in parallel using multiprocessing with controlled assignment
    log(f"   Starting 2 parallel processes with split workload...")
    
    # Use spawn method to avoid issues with CUDA/PyTorch
    mp_context = mp.get_context('spawn')
    
    # Get current global state to pass to workers
    global _GLOBAL_MODEL_STATE, _GLOBAL_ENV_CONFIG, _GLOBAL_POLICY_ARCHITECTURE
    
    all_metrics = []
    
    # Run both groups in parallel - each group runs sequentially within its process
    # This enforces the 4+2 split: Process 1 does maps 9-12, Process 2 does maps 13-14
    with mp_context.Pool(
        processes=2,  # Always use 2 processes for the split workload
        initializer=init_worker_global_state, 
        initargs=(_GLOBAL_MODEL_STATE, _GLOBAL_ENV_CONFIG, _GLOBAL_POLICY_ARCHITECTURE)
    ) as pool:
        try:
            # Submit both batches asynchronously - each batch runs on a dedicated process
            if episode_args_group1:
                log(f"   Submitting {len(episode_args_group1)} episodes to Process 1 (maps 9-12) - will run sequentially...")
                result1 = pool.apply_async(run_episodes_sequential, (episode_args_group1,))
            
            if episode_args_group2:
                log(f"   Submitting {len(episode_args_group2)} episodes to Process 2 (maps 13-14)...")
                result2 = pool.apply_async(run_episodes_sequential, (episode_args_group2,))
            
            # Wait for both to complete
            if episode_args_group1:
                metrics1 = result1.get()
                all_metrics.extend(metrics1)
            
            if episode_args_group2:
                metrics2 = result2.get()
                all_metrics.extend(metrics2)
            
            log(f"   Both processes completed successfully")
        except Exception as e:
            log(f" Error in parallel execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Explicitly close and join the pool to ensure clean shutdown
            log(f"   Closing pool and waiting for workers to finish...")
            pool.close()
            pool.join()
            log(f"   Pool closed and joined successfully")
    
    log(f" Completed {len(all_metrics)} episodes in parallel")
    log("-" * 80)
    
    headers = [
        "model",
        "epoch",
        "map",
        "map_run_id",
        "path_length_90_pct",
        "path_length_end",
        "coverage_end",
    ]

    def _format_value(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    model_identifier = Path(config.model_base_path).name
    current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    table_rows: List[dict] = []
    for metrics in all_metrics:
        row = {
            "run_number": run_number,
            "timestamp": current_timestamp,
            "model": "Ours",
            "epoch": f"epoch-{epoch_number}" if epoch_number is not None else "",
            "map": metrics.get("map_name", ""),
            "map_run_id": metrics.get("map_run_id", ""),
            "path_length_90_pct": metrics.get("path_length_at_90"),
            "path_length_end": metrics.get("final_path_length"),
            "coverage_end": metrics.get("final_coverage"),
        }
        if args.sweep_id:
            suffix = str(args.sweep_id)
            base_id = row["map_run_id"]
            row["map_run_id"] = f"{base_id}-{suffix}" if base_id else suffix
        table_rows.append(row)

    # Display headers (excluding run_number and timestamp for console output)
    display_headers = [h for h in headers if h not in ["run_number", "timestamp"]]
    formatted_rows: List[dict] = [{h: _format_value(row[h]) for h in display_headers} for row in table_rows]
    column_widths = {h: len(h) for h in display_headers}
    for row in formatted_rows:
        for header in display_headers:
            column_widths[header] = max(column_widths[header], len(row[header]))

    log("")
    log(f"Run #{run_number} at {current_timestamp}")
    header_line = "  ".join(header.ljust(column_widths[header]) for header in display_headers)
    log(header_line)
    for row in formatted_rows:
        log("  ".join(row[header].ljust(column_widths[header]) for header in display_headers))

    # Add summary of image usage
    image_usage = {}
    for metrics in all_metrics:
        img_name = metrics.get("map_name", "unknown")
        image_usage[img_name] = image_usage.get(img_name, 0) + 1
    
    log(f"\n Image Usage Summary:")
    log("-" * 40)
    for img_name in sorted(image_usage.keys()):
        count = image_usage[img_name]
        log(f"  {img_name}: {count} episodes")
    log(f"  Total: {sum(image_usage.values())} episodes")
    
    expected_images = {f"eval_mowing_{i}" for i in range(9, 15)}
    actual_images = set(image_usage.keys())
    if actual_images == expected_images:
        log(" Using correct evaluation images (eval_mowing_9-14)")
    else:
        log(" WARNING: Not using expected evaluation images!")
        log(f"  Expected: {sorted(expected_images)}")
        log(f"  Actual: {sorted(actual_images)}")

    # Handle output - either wandb logging or CSV saving
    if args.wandb_logging:
        if all_metrics:
            if not args.sweep_id:
                log(" --wandb-logging requires --sweep-id to be specified")
                return
            log(f"\n Logging to wandb...")
            log_metrics_to_wandb(all_metrics, args.sweep_id, args.wandb_project)
        else:
            log(" No metrics available to log to wandb.")
    elif args.save_csv:
        # Combine subprocess CSV files into master CSV
        log(f"\n Combining subprocess CSV files...")
        append_mode = run_number > 1
        combine_subprocess_csvs(args.save_csv, num_episodes, append_mode=append_mode)
        log(" CSV combination complete")
    else:
        log(" No output specified (use --save-csv or --wandb-logging)")
    
    # Return the (potentially updated) policy, env, and epoch_number for reuse in continuous mode
    return policy, env, epoch_number


def main(argv=None):
    """Main function to run local benchmark."""
    args = parse_args(argv)
    
    # Initialize logging
    init_logging(args.job_id, args.training_job_id)
    
    # Parse constrained model numbers list if provided
    constrained_model_numbers_list = None
    if args.constrained_model_numbers is not None:
        try:
            constrained_model_numbers_list = [int(x.strip()) for x in args.constrained_model_numbers.split(',')]
            log(f" Parsed constrained model numbers: {constrained_model_numbers_list}")
        except ValueError as e:
            log(f" Error parsing constrained model numbers '{args.constrained_model_numbers}': {e}")
            return
        
        # Override single constrained_model_number if list is provided
        if args.constrained_model_number is not None:
            log(f" Both --constrained-model-number ({args.constrained_model_number}) and --constrained-model-numbers ({args.constrained_model_numbers}) specified. Using the list.")
        args.constrained_model_number = None  # Clear single value to avoid conflicts
    
    # Generate appropriate CSV filename for constrained models
    if constrained_model_numbers_list is not None and args.save_csv and args.sweep_id:
        # Check if the CSV path looks like a default pattern (contains sweep_id but not model number)
        if f"benchmark_results_{args.sweep_id}.csv" in args.save_csv:
            # Replace with model range version for multiple models
            model_range = f"{min(constrained_model_numbers_list)}-{max(constrained_model_numbers_list)}"
            args.save_csv = args.save_csv.replace(
                f"benchmark_results_{args.sweep_id}.csv", 
                f"benchmark_results_{args.sweep_id}_models_{model_range}.csv"
            )
            log(f" Using multi-model CSV filename: {args.save_csv}")
    elif args.constrained_model_number is not None and args.save_csv and args.sweep_id:
        # Check if the CSV path looks like a default pattern (contains sweep_id but not model number)
        if f"benchmark_results_{args.sweep_id}.csv" in args.save_csv:
            # Replace with constrained model number version
            args.save_csv = args.save_csv.replace(
                f"benchmark_results_{args.sweep_id}.csv", 
                f"benchmark_results_{args.sweep_id}_{args.constrained_model_number}.csv"
            )
            log(f" Using constrained model CSV filename: {args.save_csv}")
    
    # Set noise study environment variables for class_gymenv.py to read
    os.environ['BENCHMARK_NOISE_TYPE'] = args.noise_type
    os.environ['BENCHMARK_NOISE_INTENSITY'] = str(args.noise_intensity)
    if args.noise_type != 'none':
        log(f"[NOISE STUDY] type={args.noise_type}, intensity={args.noise_intensity} ({args.noise_intensity*100:.0f}%)")

    log("=" * 80)
    log("LOCAL BENCHMARK - Testing Model from Local Directory")
    log("=" * 80)
    
    # Initialize random seed for reproducible yet varied runs
    import random
    import numpy as np
    import time
    run_seed = 0
    random.seed(run_seed)
    np.random.seed(run_seed)
    log(f"[OK] Initialized random seed: {run_seed}")
    
    # Parse map filter if provided
    map_filter = None
    if args.map_filter:
        try:
            map_filter = [int(x.strip()) for x in args.map_filter.split(',')]
            log(f"[OK] Parsed map filter: {map_filter}")
        except ValueError as e:
            log(f"[ERROR] Invalid map filter format: {args.map_filter}. Expected comma-separated integers.")
            return
    
    # Initialize configuration
    config = Config(max_steps_per_episode=args.max_steps_per_episode, map_filter=map_filter)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"\n[OK] Using device: {device}")
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # Handle multiple model numbers case
    if constrained_model_numbers_list is not None:
        log(f"\n Multi-model evaluation mode enabled for models: {constrained_model_numbers_list}")
        
        # Set up omnisafe once
        log(f"\n[1/2] Setting up OmniSafe...")
        try:
            setup_omnisafe_import()
        except Exception as e:
            log(f" Failed to set up OmniSafe: {e}")
            return
        
        log(f"\n[2/2] Running evaluation for each model...")
        
        for i, model_number in enumerate(constrained_model_numbers_list):
            log(f"\n{'=' * 80}")
            log(f"EVALUATING MODEL {i+1}/{len(constrained_model_numbers_list)}: {model_number}")
            log(f"{'=' * 80}")
            
            # Find checkpoint for this specific model
            log(f"\n[1/4] Finding model checkpoint {model_number}...")
            checkpoint_result = find_latest_model_checkpoint(config.model_base_path, model_number)
            if checkpoint_result is None:
                log(f" Failed to find model checkpoint {model_number}. Skipping.")
                continue
            
            checkpoint_path, epoch_number = checkpoint_result
            
            # Initialize global state with this checkpoint
            log(f"\n[2/4] Initializing global state for model {model_number}...")
            try:
                initialize_global_state(checkpoint_path)
            except Exception as e:
                log(f" Failed to initialize global state for model {model_number}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Load model from global state
            log(f"\n[3/4] Loading model {model_number} from global state...")
            try:
                policy, env = load_model(device)
            except Exception as e:
                log(f" Failed to load model {model_number}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Run benchmark for this model
            log(f"\n[4/4] Running benchmark for model {model_number}...")
            try:
                # Calculate run number (models are run sequentially, not in loop mode)
                run_number = i + 1
                
                # Clear PNG folder for this model before starting
                clear_png_folder(model_number)
                
                # Temporarily set constrained_model_number for rendering
                original_constrained_model_number = args.constrained_model_number
                args.constrained_model_number = model_number
                
                policy, env, updated_epoch = run_single_benchmark(args, config, device, policy, env, 
                                                                 reload_model=False, run_number=run_number, 
                                                                 checkpoint_path=checkpoint_path, epoch_number=epoch_number)
                
                # Restore original value
                args.constrained_model_number = original_constrained_model_number
                
                log(f" Completed benchmark for model {model_number}")
            except Exception as e:
                log(f" Error in benchmark for model {model_number}: {e}")
                import traceback
                traceback.print_exc()
        
        log(f"\n Completed evaluation of all {len(constrained_model_numbers_list)} models")
        return
    
    # Original single model logic
    # Find checkpoint (latest or constrained)
    log(f"\n[1/4] Finding model checkpoint...")
    if args.constrained_model_number is not None:
        log(f"   Using constrained model number: {args.constrained_model_number}")
    checkpoint_result = find_latest_model_checkpoint(config.model_base_path, args.constrained_model_number)
    if checkpoint_result is None:
        log(" Failed to find model checkpoint. Exiting.")
        return
    
    checkpoint_path, epoch_number = checkpoint_result
    
    # Set up omnisafe
    log(f"\n[2/4] Setting up OmniSafe...")
    try:
        setup_omnisafe_import()
    except Exception as e:
        log(f" Failed to set up OmniSafe: {e}")
        return
    
    # Initialize global state with checkpoint
    log(f"\n[3/4] Initializing global state...")
    try:
        initialize_global_state(checkpoint_path)
    except Exception as e:
        log(f" Failed to initialize global state: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load model from global state
    log(f"\n[4/4] Loading model from global state...")
    try:
        policy, env = load_model(device)
    except Exception as e:
        log(f" Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check if continuous loop mode is enabled (but disable if constrained model number is specified)
    if args.continuous_loop and (args.constrained_model_number is not None or constrained_model_numbers_list is not None):
        if args.constrained_model_number is not None:
            log(f"\n Continuous loop mode disabled because constrained model number ({args.constrained_model_number}) was specified")
        if constrained_model_numbers_list is not None:
            log(f"\n Continuous loop mode disabled because constrained model numbers ({constrained_model_numbers_list}) were specified")
        args.continuous_loop = False
    
    if args.continuous_loop:
        loop_hours = args.loop_interval / 3600
        log(f"\n Continuous loop mode enabled - will run every {loop_hours:.1f} hours until job time limit")
        log("   Press Ctrl+C to stop manually if needed")
        
        if args.save_csv:
            log(f"   Results will be appended to CSV file: {args.save_csv}")
        if args.wandb_logging:
            log(f"   Results will be logged to wandb sweep: {args.sweep_id}")
        
        run_count = 0
        
        # Force wandb logging in continuous mode if sweep_id provided
        if not args.wandb_logging and args.sweep_id and not args.save_csv:
            args.wandb_logging = True
            log("    Auto-enabled wandb logging for continuous mode")
        
        # Clear PNG folder once at the start of continuous mode
        clear_png_folder(args.constrained_model_number)
        
        while True:
            run_count += 1
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            log(f"\n{'=' * 80}")
            log(f"BENCHMARK RUN #{run_count} - {current_time}")
            log(f"{'=' * 80}")
            
            try:
                # For first run, use loaded policy/env; for subsequent runs, reload
                reload_model = (run_count > 1)
                policy, env, updated_epoch = run_single_benchmark(args, config, device, policy, env, reload_model=reload_model, run_number=run_count, checkpoint_path=checkpoint_path, epoch_number=epoch_number)
                
                # Check if training has stopped (updated_epoch is None)
                if updated_epoch is None and reload_model:
                    log(f"\n Training job has stopped (no new checkpoints available)")
                    log(f"   Completed {run_count} benchmark runs")
                    log(f"   Exiting continuous loop - benchmark complete")
                    break
                
                # Update epoch_number for subsequent runs in case model was reloaded
                if updated_epoch is not None:
                    epoch_number = updated_epoch
                log(f" Completed benchmark run #{run_count}")
            except Exception as e:
                log(f" Error in benchmark run #{run_count}: {e}")
                import traceback
                traceback.print_exc()
            
            # Wait for specified interval before next run
            wait_hours = args.loop_interval / 3600
            log(f"\n Waiting {wait_hours:.1f} hours until next run...")
            log(f"   Next run will start at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + args.loop_interval))}")
            
            try:
                time.sleep(args.loop_interval)
            except KeyboardInterrupt:
                log(f"\n Received interrupt signal - stopping continuous loop")
                log(f"   Completed {run_count} benchmark runs")
                break
    else:
        # Single run mode (original behavior)
        # Clear PNG folder before starting single run
        clear_png_folder(args.constrained_model_number)
        
        policy, env, updated_epoch = run_single_benchmark(args, config, device, policy, env, checkpoint_path=checkpoint_path, epoch_number=epoch_number)
        
if __name__ == "__main__":
    main()