import subprocess
import re
import os
from get_parameters import para
import yaml
# get current directory. if it starts with /dss we are in lrz, if we are in /home/stud we are in lmu assign varibale cluster
# Get the current directory
current_dir = os.getcwd()

# Determine the cluster based on the directory path
if current_dir.startswith('/dss'):
    cluster = 'lrz'
elif current_dir.startswith('/home/stud'):
    cluster = 'lmu'
else:
    cluster = 'unknown'
if cluster == 'unknown':
    print(f"Cluster unknown")

if os.name == 'nt':  # Windows
    file_path_sweep_txt = 'D:\DownloadsSAVE\Johann\Programming_Space\Simulation\misc\sweep_id.txt'
    file_path_sweep_yaml = 'D:\DownloadsSAVE\Johann\Programming_Space\Simulation\parameters_sweep.yaml'
    file_path_default_yaml = 'D:\DownloadsSAVE\Johann\Programming_Space\Simulation\parameters_default.yaml'
else:
    if cluster == 'lmu':
        file_path_sweep_txt = '/home/stud/blake/git_clones/Simulation/misc/sweep_id.txt'
        file_path_sweep_yaml = '/home/stud/blake/git_clones/Simulation/parameters_sweep.yaml'
        file_path_default_yaml = '/home/stud/blake/git_clones/Simulation/parameters_default.yaml'
    elif cluster == 'lrz':
        file_path_sweep_txt = '/dss/dsshome1/0C/di97sog/git_clones/Simulation/misc/sweep_id.txt'
        file_path_sweep_yaml = '/dss/dsshome1/0C/di97sog/git_clones/Simulation/parameters_sweep.yaml'
        file_path_default_yaml = '/dss/dsshome1/0C/di97sog/git_clones/Simulation/parameters_default.yaml'

# Retrieve the latest commit ID
def get_latest_commit_id():
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
        return commit_id
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while getting commit ID: {e}")
        return None


# Update the commit_id in the YAML file
def update_commit_id_in_yaml(file_path, commit_id):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Ensure 'parameters' and 'commit_id' exist in the YAML structure
        if 'parameters' not in data:
            data['parameters'] = {}
        if 'commit_id' not in data['parameters']:
            data['parameters']['commit_id'] = {}

        # Update the commit_id value
        data['parameters']['commit_id']['values'] = [commit_id]

        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file)

    except Exception as e:
        print(f"Error occurred while updating YAML file: {e}")

# Retrieve the latest commit ID
commit_id = get_latest_commit_id()
if commit_id:
    yaml_file_path = 'parameters_sweep.yaml'
    update_commit_id_in_yaml(yaml_file_path, commit_id)
else:
    print("Failed to get commit ID.")
    
# Function to run wandb sweep and capture the output
def run_wandb_sweep_and_capture_id():
    # Execute the wandb sweep command and capture the output
    result = subprocess.run(['wandb', 'sweep', file_path_sweep_yaml], capture_output=True, text=True)
    # Check if the command was executed successfully
    if result.returncode == 0:
        # Use regular expression to find the sweep ID in the output
        match = re.search(r'sweep with ID: (\w+)', result.stderr)
        if match:
            # Extract the sweep ID
            sweep_id = match.group(1)
            # Save the sweep ID to a file
            with open(file_path_sweep_txt, 'w') as file:
                file.write(sweep_id)
            return sweep_id
        else:
            print("Sweep ID not found in the output.")
    else:
        print("Failed to execute wandb sweep command.")
        print("Error:", result.stderr)

# Run the function
sweep_id = run_wandb_sweep_and_capture_id()
print(f"Sweep ID: {sweep_id}")
sweep_id_with_instance_project = f"johanndavidblake-ludwig-maximilianuniversity-of-munich/Heli-Logs/{sweep_id}"

# Command to run in each terminal
command = f'wandb agent {sweep_id_with_instance_project}'

# Check the operating system
if os.name == 'nt':  # Windows
    for _ in range(min(22,1)):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        subprocess.Popen(['cmd.exe', '/c', command], startupinfo=startupinfo)
else:  # Unix-like (Linux, macOS)
    if cluster == 'lmu':
        old_dir_path = os.path.join("/home/stud/blake/git_clones", "Simulation")
        new_dir_path = os.path.join("/home/stud/blake/git_clones", f"Simulation_{sweep_id}")
    elif cluster == 'lrz':
        old_dir_path = os.path.join("/dss/dsshome1/0C/di97sog/git_clones", "Simulation")
        new_dir_path = os.path.join("/dss/dsshome1/0C/di97sog/git_clones", f"Simulation_{sweep_id}")
    # Rename the directory
    os.rename(old_dir_path, new_dir_path)
    max_number_agents = 48
    run_agent_n_times = ""
    for _ in range(para.agents_per_gpu):
        run_agent_n_times += f"{{\n  {command}\n}} &\n\n"
    if cluster == 'lmu':
        path_to_simulation = "/home/stud/blake/"
        partition_statement = "#SBATCH --partition=major"
        qos_statement = ""
        mem_statement = "" #SBATCH --mem=120G
        gpu_statement = "#SBATCH --gres=gpu:1" # 
        out_statement = "#SBATCH --output=/home/stud/blake/%j.txt"
        which_node = f"{para.worker_to_use_with_slurm_command}" #SBATCH --nodelist=worker-8,worker-1,worker-2,worker-3,worker-4,worker-5,worker-6,worker-7,Quality with GPU: w8 (first run), w4,w3 "#SBATCH --exclude=worker-1,worker-2,worker-5,worker-6,worker-9" ##SBATCH --exclude=worker-1,worker-2,worker-5, worker-6,worker-9"
        hint_statement = ""
    elif cluster == 'lrz':
        path_to_simulation = "/dss/dsshome1/0C/di97sog/"
        partition_statement = "#SBATCH --partition=mcml-hgx-h100-94x4,mcml-dgx-a100-40x8,mcml-hgx-a100-80x4" #mcml-dgx-a100-40x8,mcml-hgx-a100-80x4,mcml-hgx-a100-80x4-mig,mcml-hgx-h100-94x4
        qos_statement = "#SBATCH --qos=mcml"
        mem_statement = "#SBATCH --mem=30G"
        gpu_statement = "#SBATCH --gres=gpu:1" # No GPU for CPU partition
        out_statement = "#SBATCH --output=/dss/dsshome1/0C/di97sog/%j.txt"
        which_node = ""
        hint_statement = "" # "#SBATCH --hint=nomultithread"
    slurm_command = f"""#!/usr/bin/env bash
#
#SBATCH --job-name=agents_parallel
{partition_statement}
#SBATCH --cpus-per-task={para.cpus_per_task}
{gpu_statement}
{out_statement}
#SBATCH --ntasks=1
#SBATCH --time='{para.time_for_each_agent}'
{mem_statement}
{qos_statement}
{which_node}
{hint_statement}

# Activate conda environment
source {path_to_simulation}anaconda3/etc/profile.d/conda.sh
conda activate e

export PYTHONUNBUFFERED=1
exec 2>&1
{run_agent_n_times}
wait
"""
    process = subprocess.run(['sbatch'], input=slurm_command, text=True)