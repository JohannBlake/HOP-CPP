####################################################################################################
time_for_each_agent = '9-23:59:00'
agents_per_gpu = 1
cpus_per_task = 16
cluster = 'lmu'  # 'lmu' or 'lrz'
####################################################################################################

if cluster == 'lrz':
    time_for_each_agent = '1-23:59:00'
    print("Time for each agent set to 1 day and 23 hours for LRZ cluster.")
from get_parameters import para
import subprocess
import yaml
if cluster == 'lmu':
    ssh_command = 'ssh c "rm -rf /home/stud/blake/git_clones/Simulation"'
elif cluster == 'lrz':
    ssh_command = 'ssh h "rm -rf /dss/dsshome1/0C/di97sog/git_clones/Simulation"'
process = subprocess.Popen(
    ssh_command,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,  # Ensures the output is returned as a string
    errors='replace'
)

# Wait for the process to complete
stdout, stderr = process.communicate()

if process.returncode == 0:
    pass
else:
    print(f"Error occurred during deletion. Return code: {process.returncode}")
    print(f"Error details: {stderr.strip()}")

# Path to the YAML file
yaml_file_path = r'.\parameters_default.yaml'
# Load the existing YAML file
with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file)

# Update the values
data['time_for_each_agent'] = time_for_each_agent
data['agents_per_gpu'] = agents_per_gpu
data['cpus_per_task'] = cpus_per_task

# Save the updated YAML file
with open(yaml_file_path, 'w') as file:
    yaml.safe_dump(data, file)

# Define the commit message
commit_message = "commit to run on cluster"
# Add the changes to the staging area
subprocess.run(["git", "add", "."], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Check if there are any changes to commit
result = subprocess.run(["git", "diff", "--cached", "--exit-code"], capture_output=True)
if result.returncode != 0:
    # Commit the changes with the specified comment
    subprocess.run(["git", "commit", "-m", commit_message], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Push the changes to the remote repository
    subprocess.run(["git", "push"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if cluster == 'lmu':
    ssh_command = (
        'ssh c '
        '"source /home/stud/blake/anaconda3/etc/profile.d/conda.sh && '
        'conda activate e && '
        '/home/stud/blake/update_git_and_run_locally_created_bash_script_run_file_on_cluster.sh"'
    )
elif cluster == 'lrz':
    ssh_command = (
        'ssh h '
        '"source /dss/dsshome1/0C/di97sog/anaconda3/etc/profile.d/conda.sh && '
        'conda activate e && '
        '/dss/dsshome1/0C/di97sog/update_git_and_run_locally_created_bash_script_run_file_on_cluster.sh"'
    )


# Execute the SSH command
process = subprocess.Popen(
    ssh_command,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,  # Ensures the output is returned as a string
    encoding='utf-8',
    errors='replace'
)

#Read the output in real-times. I dont know why exactly but apparently the following lines need to run in order for the model to run.
for line in process.stderr: 
    print(line, end='')
for line in process.stdout:
    print(line, end='')