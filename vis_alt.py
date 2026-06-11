###############################################################################
###############################################################################
num_runs_displayed = 1
sweep_ids = ["u1vlrqrh"] # y4amqq8z
step_number_from_wandb = None
num_episodes_to_skip = 8
visualization_step_interval = 10  # Save visualization data every Nth step (1 = every step)
# Path to saved actions from previous run
saved_actions_path = r"misc\3d_vis_archive\offline\vis_2025-11-10_13-03-24\html_data\u1vlrqrh\blhhecp7\actions.npy"
# Seed for environment initialization (for reproducibility)
env_seed = 0
###############################################################################
###############################################################################

if step_number_from_wandb is not None:
    constrained_model_number = str(step_number_from_wandb + 1)
else:
    constrained_model_number = ""
debug_mode = False
cluster = 'lmu'

import wandb
import os
import git
import subprocess
import sys
import json
import datetime
import shutil
import tempfile

# Define main folders
main_folder = os.getcwd()
git_clones_folder = os.path.join(main_folder, 'git_clones')
html_data_folder = os.path.join(main_folder, 'html_data')
run_ids_to_be_considered = []

def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_commit_depth(repo_url, commit_id):
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

def clone_repo(repo_url, folder_name, commit_id):
    os.system('git config --global http.postBuffer 157286400')
    if os.path.exists(folder_name):
        print(f"Skip clone (Already exists).")
        return
    print(f"Finding depth for commit {commit_id}...")
    depth = get_commit_depth(repo_url, commit_id)
    print(f"Cloning with depth={depth} to include commit {commit_id}")
    repo = git.Repo.clone_from(repo_url, folder_name, depth=depth, branch ='main')
    print(f"Repository cloned to {folder_name}")
    repo.git.checkout(commit_id)
    print(f"Checked out to commit {commit_id}")

def process_sweep(sweep_id, api, timestamp, repo_url):
    sweep_path = f"johanndavidblake-ludwig-maximilianuniversity-of-munich/Heli-Logs/{sweep_id}"
    sweep = api.sweep(sweep_path)
    runs = sweep.runs
    run_ids = [run.id for run in runs]
    if run_ids_to_be_considered:
        run_ids = [run_id for run_id in run_ids if run_id in run_ids_to_be_considered]

    first_run_id = run_ids[0]
    run = api.run(f"{sweep_path}/{first_run_id}")
    commit_id_fitting_to_model = run.config['commit_id']
    sweep_base_folder = os.path.join(git_clones_folder, sweep_id)
    sweep_base_folder_test_file = os.path.join(sweep_base_folder, 'Simulation', 'parameters_default.yaml')

    if os.path.exists(sweep_base_folder) and not os.path.exists(sweep_base_folder_test_file):
        shutil.rmtree(sweep_base_folder)

    base_folder = os.path.join(sweep_base_folder, 'Simulation')
    clone_repo(repo_url, base_folder, commit_id_fitting_to_model)

    target_file_path = os.path.join(base_folder, 'step_through_gymenv_and_save_data_for_vis.py')
    source_file_path = os.path.join(main_folder, "misc", "aid", 'step_through_gymenv_and_save_data_for_vis_with_saved_actions.py')
    shutil.copy2(source_file_path, target_file_path)

    # Convert relative path to absolute path
    abs_saved_actions_path = os.path.abspath(os.path.join(main_folder, saved_actions_path))
    
    try:
        result = subprocess.run([
            sys.executable, target_file_path,
            '--sweep-id', sweep_id,
            '--num-runs', str(num_runs_displayed),
            '--cluster', cluster,
            '--main-folder', main_folder,
            '--git-clones-folder', git_clones_folder,
            '--html-data-folder', html_data_folder,
            '--timestamp', timestamp,
            '--base-folder', base_folder,
            '--commit-id', commit_id_fitting_to_model,
            '--run-ids', json.dumps(run_ids),
            '--run-ids-to-be-considered', json.dumps(run_ids_to_be_considered),
            '--constrained-model-number', constrained_model_number,
            '--debug-mode', str(debug_mode),
            '--num-episodes-to-skip', str(num_episodes_to_skip),
            '--visualization-step-interval', str(visualization_step_interval),
            '--saved-actions-path', abs_saved_actions_path,
            '--env-seed', str(env_seed),
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running subprocess:")
        print(f"Return code: {e.returncode}")
        print(f"Command: {' '.join(e.cmd)}")
        raise

def main():
    timestamp = get_timestamp()
    api = wandb.Api()
    repo_url = "https://github.com/JohannBlake/Simulation.git"
    
    # Verify saved actions file exists
    abs_saved_actions_path = os.path.abspath(os.path.join(main_folder, saved_actions_path))
    if not os.path.exists(abs_saved_actions_path):
        print(f"ERROR: Saved actions file not found at: {abs_saved_actions_path}")
        print("Please run vis.py first to generate and save actions.")
        return
    
    print(f"Using saved actions from: {abs_saved_actions_path}")
    
    for sweep_id in sweep_ids:
        process_sweep(sweep_id, api, timestamp, repo_url)

if __name__ == "__main__":
    main()
