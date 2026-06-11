import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import wandb
import git
import subprocess
import sys
import json
import datetime
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class Config:
    """Configuration for vis and benchmark script."""
    # Core settings
    num_runs_displayed: int = 10000
    constrained_model_number: Optional[int] = None

    # Episode settings
    num_episode_repeats_for_benchmark: int = 2  # 
    
    # Environment settings
    debug_mode: bool = False
    cluster: str = 'lmu'
    
    # Paths    
    def __post_init__(self):
        self.sweep_ids = ["n8xr0dci"]
        
        self.reset_image_paths = [
            os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_1.png"),
            os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "eval_mowing_2.png")
        ]

        # Handle constrained model number conversion
        if self.constrained_model_number is not None:
            self.constrained_model_number = str(self.constrained_model_number + 1)
        else:
            self.constrained_model_number = ""


# Initialize configuration
config = Config()

# Define main folders
main_folder = os.getcwd()
git_clones_folder = os.path.join(main_folder, 'git_clones')
html_data_folder = os.path.join(main_folder, 'html_data')
run_ids_to_be_considered = []

def get_timestamp() -> str:
    """Get current timestamp formatted as string."""
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def get_commit_depth(repo_url: str, commit_id: str) -> int:
    """Get the depth needed to clone a repository up to a specific commit."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        subprocess.run(
            ["git", "clone", "--bare", "--filter=blob:none", "--no-checkout", repo_url, tmpdirname],
            check=True, capture_output=True
        )
        repo = git.Repo(tmpdirname)
        try:
            default_branch_ref = repo.git.symbolic_ref("HEAD")
            default_branch = default_branch_ref.split("/")[-1]
        except Exception:
            default_branch = 'main' if 'refs/heads/main' in repo.refs else 'master'
        count = repo.git.rev_list("--count", f"{default_branch}", f"^{commit_id}")
        return int(count) + 1

def clone_repo(repo_url: str, folder_name: str, commit_id: str) -> None:
    """Clone a repository and checkout to a specific commit."""
    os.system('git config --global http.postBuffer 157286400')
    if os.path.exists(folder_name):
        return
    depth = get_commit_depth(repo_url, commit_id)
    repo = git.Repo.clone_from(repo_url, folder_name, depth=depth, branch ='main')
    repo.git.checkout(commit_id)


def run_benchmark_directly(base_folder: str, main_folder: str, cluster: str, constrained_model_number: str,
                           sweep_id: str, run_ids: List[str], run_names: List[str], num_episodes: int, 
                           num_runs_displayed: int, reset_images: List[str]) -> tuple:
    """
    Run benchmark using the benchmark_runner module directly.
    
    Args:
        base_folder: Path to the cloned repository base folder
        main_folder: Path to the original repository folder
        cluster: Cluster name ('lmu' or 'lrz')
        constrained_model_number: Model epoch number for constrained RL
        sweep_id: WandB sweep ID
        run_ids: List of run IDs to benchmark
        run_names: List of run names corresponding to run_ids
        num_episodes: Number of episodes per run
        num_runs_displayed: Maximum number of runs to process
        reset_images: List of reset image paths
        
    Returns:
        Tuple of (results, dataframe)
    """
    # Import benchmark runner
    from benchmark_runner import run_benchmark
    
    # Run benchmark directly
    results, df = run_benchmark(
        sweep_id=sweep_id,
        num_episodes=num_episodes,
        num_runs_displayed=num_runs_displayed,
        run_ids=run_ids,
        run_names=run_names,
        reset_images=reset_images,
        cluster=cluster,
        constrained_model_number=constrained_model_number,
        base_folder=base_folder,
        main_folder=main_folder
    )
    
    return results, df


def execute_subprocess_with_error_handling(command: List[str], phase_name: str) -> None:
    """Execute a subprocess command with proper error handling."""
    result = subprocess.run(command, check=True, capture_output=False, text=True)
    
    #print(f"{phase_name} output:")
    #print(result.stdout)
    #if result.stderr:
    #    print(f"{phase_name} warnings/errors:")
    #    print(result.stderr)
        


def setup_sweep_environment(sweep_id: str, api) -> Tuple[List[str], List[str], str, str]:
    """
    Set up the sweep environment and return necessary paths and info.
    
    Args:
        sweep_id: The wandb sweep ID
        api: The wandb API instance
        
    Returns:
        Tuple of (run_ids, run_names, commit_id_fitting_to_model, base_folder)
    """
    sweep_path = f"johanndavidblake-ludwig-maximilianuniversity-of-munich/Heli-Logs/{sweep_id}"
    sweep = api.sweep(sweep_path)
    runs = sweep.runs
    run_ids = [run.id for run in runs]
    run_names = [run.name for run in runs]  # NEW: Collect run names
    
    if run_ids_to_be_considered:
        # Filter both run_ids and run_names consistently
        filtered_data = [(run_id, run_name) for run_id, run_name in zip(run_ids, run_names) 
                        if run_id in run_ids_to_be_considered]
        run_ids, run_names = zip(*filtered_data) if filtered_data else ([], [])
        run_ids, run_names = list(run_ids), list(run_names)

    first_run_id = run_ids[0]
    run = api.run(f"{sweep_path}/{first_run_id}")
    commit_id_fitting_to_model = run.config['commit_id']
    
    sweep_base_folder = os.path.join(git_clones_folder, sweep_id)
    sweep_base_folder_test_file = os.path.join(sweep_base_folder, 'Simulation', 'parameters_default.yaml')

    if os.path.exists(sweep_base_folder) and not os.path.exists(sweep_base_folder_test_file):
        shutil.rmtree(sweep_base_folder)

    base_folder = os.path.join(sweep_base_folder, 'Simulation')
    
    return run_ids, run_names, commit_id_fitting_to_model, base_folder


def run_benchmark_phase(sweep_id: str, api, timestamp: str, repo_url: str) -> tuple:
    """Run benchmark phase - only stepping through episodes without visualization processing
    
    Returns:
        Tuple of (results, dataframe)
    """
    
    run_ids, run_names, commit_id_fitting_to_model, base_folder = setup_sweep_environment(sweep_id, api)
    clone_repo(repo_url, base_folder, commit_id_fitting_to_model)

    # Copy the benchmark runner to the cloned repository
    import shutil
    source_benchmark_path = os.path.join(main_folder, 'benchmark_runner.py')
    target_benchmark_path = os.path.join(base_folder, 'benchmark_runner.py')
    shutil.copy2(source_benchmark_path, target_benchmark_path)

    # Run benchmark directly using the benchmark_runner module
    try:
        # Change to base folder directory for proper imports
        original_cwd = os.getcwd()
        os.chdir(base_folder)
        
        # Add base folder to path for imports
        if base_folder not in sys.path:
            sys.path.insert(0, base_folder)
        
        # Import and run benchmark
        from benchmark_runner import run_benchmark
        
        # Calculate total episodes: each reset image gets num_episode_repeats_for_benchmark episodes
        total_episodes = len(config.reset_image_paths) * config.num_episode_repeats_for_benchmark
        
        results, df = run_benchmark(
            sweep_id=sweep_id,
            num_episodes=total_episodes,
            num_runs_displayed=config.num_runs_displayed,
            run_ids=run_ids,
            run_names=run_names,  # NEW: Pass run names
            reset_images=config.reset_image_paths,
            cluster=config.cluster,
            constrained_model_number=config.constrained_model_number,
            base_folder=base_folder,
            main_folder=main_folder
        )
        
        return results, df
        
    except Exception as e:
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)



def process_sweep(sweep_id: str, api, timestamp: str, repo_url: str) -> tuple:
    """
    Process a sweep with benchmark phase only.
    
    Args:
        sweep_id: The wandb sweep ID to process
        api: The wandb API instance
        timestamp: Current timestamp string
        repo_url: Git repository URL for cloning
        
    Returns:
        Tuple of (results, dataframe)
    """
    # Benchmark phase
    benchmark_start_time = time.time()
    results, df = run_benchmark_phase(sweep_id, api, timestamp, repo_url)
    benchmark_time = time.time() - benchmark_start_time
    
    print(f"[COMPLETE] Benchmark phase completed in {benchmark_time:.2f} seconds")
    
    return results, df

def main() -> None:
    """
    Main function to run benchmarking for configured sweeps.
    
    This processes each sweep ID in the configuration, running the benchmark
    phase for performance analysis.
    """
    timestamp = get_timestamp()
    api = wandb.Api()
    repo_url = "https://github.com/JohannBlake/Simulation.git"
    
    all_dataframes = []
    
    for sweep_id in config.sweep_ids:
        results, df = process_sweep(sweep_id, api, timestamp, repo_url)
        all_dataframes.append(df)
    
    # Combine all DataFrames if multiple sweeps were processed
    if len(all_dataframes) > 1:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_csv_path = os.path.join(main_folder, f'benchmark_results_combined_{timestamp}.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"[DATAFRAME] Combined benchmark results saved to: {combined_csv_path}")
    
    print(f"[SUCCESS] All sweeps processed successfully!")

if __name__ == "__main__":
    main()