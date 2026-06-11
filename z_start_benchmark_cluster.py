#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

# Can be a single sweep_id string or space-separated sweep_ids
sweep_id = "vlxj01cf"  # Single: "gkasuo3t" or Multiple: "sweep1 sweep2 sweep3"
# Can be None, single int, or space-separated ints. If sweep_id has multiple values, this should have same count
constrained_model_number = None  # None, 220, or "120 150 200" for multiple models
cores_per_process = 16                # CPU cores per process
processes = 2                        # Number of parallel processes (Process 1: maps 9-13, Process 2: map 14)
num_episodes = 6                    # Total number of episodes to run (maps 9, 10, 11, 12, 13, 14)
max_steps_per_episode = 18000        # Maximum steps per episode
total_cores = processes * cores_per_process # Total CPU cores to allocate for the job

import argparse
import os
import subprocess
import sys
import time
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
import shlex
import tempfile

# Set UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from get_parameters import para
REMOTE_HOST = "c"  # "c" for LMU cluster, "h" for LRZ cluster

# Cluster-specific configurations
CLUSTER_CONFIGS = {
    "c": {  # LMU cluster
        "simulation_root": "/home/stud/blake/git_clones",
        "conda_source": "source /home/stud/blake/anaconda3/etc/profile.d/conda.sh",
        "conda_env": "e",
        "partition": "major",
        "qos": "",
        "exclude_nodes": "worker-1,worker-2,worker-3,worker-4,worker-5,worker-6,worker-7,worker-9,worker-10",
        "stdout_path": "/home/stud/blake/%j.txt"
    },
    "h": {  # LRZ cluster
        "simulation_root": "/dss/dsshome1/0C/di97sog/git_clones",
        "conda_source": "source /dss/dsshome1/0C/di97sog/miniconda3/etc/profile.d/conda.sh",
        "conda_env": "e",
        "partition": "lrz-cpu",
        "qos": "cpu",
        "exclude_nodes": "",
        "stdout_path": "/dss/dsshome1/0C/di97sog/%j.txt"
    }
}

# Get cluster config based on REMOTE_HOST
if REMOTE_HOST not in CLUSTER_CONFIGS:
    raise ValueError(f"Unknown REMOTE_HOST '{REMOTE_HOST}'. Must be 'c' (LMU) or 'h' (LRZ)")

CLUSTER_CONFIG = CLUSTER_CONFIGS[REMOTE_HOST]
REMOTE_SIMULATION_ROOT = CLUSTER_CONFIG["simulation_root"]

LOCAL_BENCHMARK_FILENAME = "benchmark_local.py"
LOCAL_GYMENV_FILENAME = "class_gymenv.py"
SLURM_POLL_INTERVAL_SECONDS = 10

@dataclass
class ClusterPaths:
    sweep_id: str

    @property
    def remote_simulation_dir(self) -> str:
        return f"{REMOTE_SIMULATION_ROOT}/Simulation_{self.sweep_id}"

    @property
    def remote_benchmark_path(self) -> str:
        return f"{self.remote_simulation_dir}/{LOCAL_BENCHMARK_FILENAME}"

    @property
    def remote_gymenv_path(self) -> str:
        return f"{self.remote_simulation_dir}/{LOCAL_GYMENV_FILENAME}"

    @property
    def remote_csv_path(self) -> str:
        if constrained_model_number is not None:
            if isinstance(constrained_model_number, list):
                # For multiple models, use a generic filename that covers all
                model_range = f"{min(constrained_model_number)}-{max(constrained_model_number)}"
                return f"{self.remote_simulation_dir}/benchmark_results_{self.sweep_id}_models_{model_range}.csv"
            else:
                return f"{self.remote_simulation_dir}/benchmark_results_{self.sweep_id}_{constrained_model_number}.csv"
        return f"{self.remote_simulation_dir}/benchmark_results_{self.sweep_id}.csv"

    @property
    def remote_slurm_stdout(self) -> str:
        return CLUSTER_CONFIG["stdout_path"]

    @property
    def remote_slurm_script_path(self) -> str:
        return f"{self.remote_simulation_dir}/_benchmark_job_{self.sweep_id}.sh"
    
    def get_remote_slurm_script_path(self, suffix: str = "") -> str:
        """Get remote SLURM script path with optional suffix."""
        return f"{self.remote_simulation_dir}/_benchmark_job_{self.sweep_id}{suffix}.sh"

    def local_csv_path(self, destination_dir: Path) -> Path:
        if constrained_model_number is not None:
            if isinstance(constrained_model_number, list):
                # For multiple models, use a generic filename that covers all
                model_range = f"{min(constrained_model_number)}-{max(constrained_model_number)}"
                return destination_dir / f"benchmark_results_{self.sweep_id}_models_{model_range}.csv"
            else:
                return destination_dir / f"benchmark_results_{self.sweep_id}_{constrained_model_number}.csv"
        return destination_dir / f"benchmark_results_{self.sweep_id}.csv"


class CommandError(RuntimeError):
    pass


def run_command(
    cmd: list[str],
    *,
    input_text: Optional[str] = None,
    check: bool = True,
    text: bool = True,
    capture_output: bool = True,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> subprocess.CompletedProcess:
    run_kwargs: dict[str, object] = {
        "capture_output": capture_output,
    }

    if input_text is not None:
        run_kwargs["input"] = input_text
        run_kwargs["text"] = True
        run_kwargs["encoding"] = encoding
        run_kwargs["errors"] = errors
    else:
        run_kwargs["text"] = text
        if text:
            run_kwargs["encoding"] = encoding
            run_kwargs["errors"] = errors

    result = subprocess.run(cmd, **run_kwargs)
    if check and result.returncode != 0:
        stdout = result.stdout
        stderr = result.stderr
        if not text:
            stdout = stdout.decode("utf-8", "replace") if isinstance(stdout, (bytes, bytearray)) else stdout
            stderr = stderr.decode("utf-8", "replace") if isinstance(stderr, (bytes, bytearray)) else stderr
        raise CommandError(
            f"Command failed: {' '.join(cmd)}\n"
            f"Exit code: {result.returncode}\n"
            f"stdout:\n{stdout or ''}\n"
            f"stderr:\n{stderr or ''}"
        )
    return result


def run_ssh(command: str) -> subprocess.CompletedProcess:
    ssh_command = ["ssh", REMOTE_HOST, command]
    return run_command(ssh_command)


def ensure_remote_directory(paths: ClusterPaths) -> None:
    quoted_dir = shlex.quote(paths.remote_simulation_dir)
    run_ssh(f"bash -lc 'mkdir -p {quoted_dir}'")
    print(f"✓ Ensured remote directory exists: {paths.remote_simulation_dir}")

def copy_benchmark_to_remote(paths: ClusterPaths, local_benchmark_path: Path) -> None:
    ensure_remote_directory(paths)
    remote_target = f"[{REMOTE_HOST}]:{paths.remote_benchmark_path}"
    local_posix_path = local_benchmark_path.resolve().as_posix()
    run_command(["scp", local_posix_path, remote_target], text=False)
    print(f"✓ Copied {LOCAL_BENCHMARK_FILENAME} to {paths.remote_benchmark_path}")


def copy_gymenv_to_remote(paths: ClusterPaths, local_gymenv_path: Path) -> None:
    """Copy class_gymenv.py to remote cluster."""
    ensure_remote_directory(paths)
    remote_target = f"[{REMOTE_HOST}]:{paths.remote_gymenv_path}"
    local_posix_path = local_gymenv_path.resolve().as_posix()
    run_command(["scp", local_posix_path, remote_target], text=False)
    print(f"✓ Copied {LOCAL_GYMENV_FILENAME} to {paths.remote_gymenv_path}")


def build_slurm_script(paths: ClusterPaths, continuous_loop: bool = False, loop_interval: int = 60, training_job_id: Optional[str] = None, training_job_file: Optional[str] = None, wait_for_sweep_id: bool = False) -> str:
    """
    Build a SLURM script for benchmark execution.
    
    Args:
        paths: Cluster paths configuration
        continuous_loop: Whether to run in continuous loop mode
        loop_interval: Interval between loops in seconds
        training_job_id: Optional training job ID to monitor
        training_job_file: Optional path to training job output file for sweep_id monitoring
        wait_for_sweep_id: Whether to wait for sweep_id in training job output before starting
    """
    remote_dir_quoted = shlex.quote(paths.remote_simulation_dir)
    benchmark_filename_quoted = shlex.quote(LOCAL_BENCHMARK_FILENAME)
    if constrained_model_number is not None:
        if isinstance(constrained_model_number, list):
            # For multiple models, use a generic filename that covers all
            model_range = f"{min(constrained_model_number)}-{max(constrained_model_number)}"
            csv_filename = f"benchmark_results_{paths.sweep_id}_models_{model_range}.csv"
        else:
            csv_filename = f"benchmark_results_{paths.sweep_id}_{constrained_model_number}.csv"
    else:
        csv_filename = f"benchmark_results_{paths.sweep_id}.csv"
    csv_filename_quoted = shlex.quote(csv_filename)
    sweep_id_quoted = shlex.quote(paths.sweep_id)

    # Build sweep_id waiting logic if needed
    sweep_id_wait_section = ""
    if wait_for_sweep_id and training_job_file:
        training_job_file_quoted = shlex.quote(training_job_file)
        sweep_id_wait_section = f"""
# Wait for sweep_id to appear in training job output
echo "Waiting for sweep_id to appear in {training_job_file}..."
SWEEP_ID_PATTERN='maximilianuniversity-of-munich/Heli-Logs/sweeps/([a-z0-9]{{8}})'
MAX_WAIT_SECONDS=6000000
POLL_INTERVAL=10
START_TIME=$(date +%s)
SWEEP_ID_FOUND=""

while [ -z "$SWEEP_ID_FOUND" ]; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    if [ $ELAPSED -ge $MAX_WAIT_SECONDS ]; then
        echo "ERROR: Sweep ID not found after $MAX_WAIT_SECONDS seconds"
        exit 1
    fi
    
    # Check if file exists and search for sweep_id
    if [ -f {training_job_file_quoted} ]; then
        SWEEP_ID_FOUND=$(grep -oP "$SWEEP_ID_PATTERN" {training_job_file_quoted} | head -1 | grep -oP '[a-z0-9]{{8}}$' || true)
        
        if [ -n "$SWEEP_ID_FOUND" ]; then
            echo "Found sweep_id: $SWEEP_ID_FOUND"
            export SWEEP_ID="$SWEEP_ID_FOUND"
            
            # Update working directory to use discovered sweep_id
            cd "/home/stud/blake/git_clones/Simulation_$SWEEP_ID_FOUND" || {{
                echo "ERROR: Failed to change to directory for sweep_id $SWEEP_ID_FOUND"
                exit 1
            }}
            echo "Changed to directory: $(pwd)"
            break
        fi
    fi
    
    echo "Attempt at ${{ELAPSED}}s - sweep_id not found yet, waiting..."
    sleep $POLL_INTERVAL
done
"""

    # Build the python command
    # If waiting for sweep_id, use the variable; otherwise use the provided sweep_id
    if wait_for_sweep_id and training_job_file:
        # Use bash variable for sweep_id discovered at runtime
        python_cmd = f"python {benchmark_filename_quoted} --sweep-id \"$SWEEP_ID_FOUND\" --job-id $SLURM_JOB_ID"
        
        # CSV filename will also use the discovered sweep_id
        if constrained_model_number is not None:
            if isinstance(constrained_model_number, list):
                model_range = f"{min(constrained_model_number)}-{max(constrained_model_number)}"
                python_cmd += f" --save-csv \"benchmark_results_${{SWEEP_ID_FOUND}}_models_{model_range}.csv\""
            else:
                python_cmd += f" --save-csv \"benchmark_results_${{SWEEP_ID_FOUND}}_{constrained_model_number}.csv\""
        else:
            python_cmd += f" --save-csv \"benchmark_results_${{SWEEP_ID_FOUND}}.csv\""
    else:
        # Normal mode: use provided sweep_id
        python_cmd = f"python {benchmark_filename_quoted} --save-csv {csv_filename_quoted} --sweep-id {sweep_id_quoted} --job-id $SLURM_JOB_ID"

    # Add training job ID if provided
    if training_job_id:
        training_job_id_quoted = shlex.quote(training_job_id)
        python_cmd += f" --training-job-id {training_job_id_quoted}"

    # Add benchmark configuration parameters
    python_cmd += f" --num-episodes {num_episodes}"
    python_cmd += f" --processes {processes}"
    python_cmd += f" --max-steps-per-episode {max_steps_per_episode}"

    if constrained_model_number is not None:
        if isinstance(constrained_model_number, list):
            # Pass the list as comma-separated values
            model_list_str = ",".join(map(str, constrained_model_number))
            constrained_model_number_quoted = shlex.quote(model_list_str)
            python_cmd += f" --constrained-model-numbers {constrained_model_number_quoted}"
        else:
            constrained_model_number_quoted = shlex.quote(str(constrained_model_number))
            python_cmd += f" --constrained-model-number {constrained_model_number_quoted}"
    if continuous_loop:
        python_cmd += f" --continuous-loop --loop-interval {loop_interval}"
    
    # Adjust time limit based on continuous loop mode and cluster
    if REMOTE_HOST == "h":  # LRZ cluster - shorter time limits
        time_limit = "1-23:59:00" if continuous_loop else "23:59:00"  # ~2 days for continuous, ~1 day for single run
    else:  # LMU cluster - longer time limits
        time_limit = "7-00:00:00" if continuous_loop else "1-00:59:00"  # 7 days for continuous, 1 day for single run
    
    # Build exclude nodes directive if needed
    exclude_directive = f"#SBATCH --exclude={CLUSTER_CONFIG['exclude_nodes']}" if CLUSTER_CONFIG['exclude_nodes'] else ""
    
    # Build QOS directive if needed
    qos_directive = f"#SBATCH --qos={CLUSTER_CONFIG['qos']}" if CLUSTER_CONFIG['qos'] else ""
    
    script = textwrap.dedent(
        f"""#!/usr/bin/env bash
#SBATCH --job-name=benchmark_{paths.sweep_id}
#SBATCH --partition={CLUSTER_CONFIG['partition']}
#SBATCH --cpus-per-task={total_cores}
#SBATCH --output={paths.remote_slurm_stdout}
#SBATCH --error={paths.remote_slurm_stdout}
#SBATCH --ntasks=1
#SBATCH --time={time_limit}
#SBATCH --mem=6G
{qos_directive}
{exclude_directive}

{CLUSTER_CONFIG['conda_source']}
conda activate {CLUSTER_CONFIG['conda_env']}
set -euo pipefail
export PYTHONUNBUFFERED=1

# Install missing dependencies if not already installed (OmniSafe dependencies)
python -c "import moviepy" 2>/dev/null || pip install --quiet moviepy
python -c "import tensorboard" 2>/dev/null || pip install --quiet tensorboard

export SWEEP_ID={sweep_id_quoted}
export OMP_NUM_THREADS={cores_per_process}
export OPENBLAS_NUM_THREADS={cores_per_process}
export MKL_NUM_THREADS={cores_per_process}
export NUMEXPR_NUM_THREADS={cores_per_process}

cd {remote_dir_quoted}

echo "Job ID: $SLURM_JOB_ID"
echo "Sweep: {paths.sweep_id}"
echo "Model: {constrained_model_number if constrained_model_number else 'latest'}"
echo "Episodes: {num_episodes}"
echo "Processes: {processes}"
echo "Started: $(date)"
echo "Logging to: $SLURM_JOB_ID.txt"
echo ""
{sweep_id_wait_section}
# Run benchmark - all output goes to job_id.txt
{python_cmd}

echo ""
echo "Finished: $(date)"
"""
    ).strip()
    return script + "\n"


def stage_slurm_script(paths: ClusterPaths, slurm_script: str, remote_script_path: str = None) -> str:
    """
    Stage a SLURM script to the remote cluster.
    
    Args:
        paths: Cluster paths configuration
        slurm_script: The script content to upload
        remote_script_path: Optional custom remote path. If None, uses paths.remote_slurm_script_path
        
    Returns:
        The remote path where the script was uploaded
    """
    if remote_script_path is None:
        remote_script_path = paths.remote_slurm_script_path
    
    remote_target = f"[{REMOTE_HOST}]:{remote_script_path}"
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="\n") as tmp_file:
        tmp_file.write(slurm_script)
        tmp_path = Path(tmp_file.name)
    try:
        run_command(["scp", tmp_path.as_posix(), remote_target], text=False)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"✓ Uploaded Slurm script to {remote_script_path}")
    return remote_script_path


def submit_slurm_job(remote_script_path: str) -> int:
    result = run_command(["ssh", REMOTE_HOST, "sbatch", remote_script_path])
    stdout = result.stdout.strip()
    print(stdout)
    if not stdout.startswith("Submitted batch job"):
        raise CommandError(f"Unexpected sbatch response: {stdout}\n{result.stderr}")
    job_id = int(stdout.split()[3])
    print(f"✓ Submitted job {job_id}")
    return job_id


def cleanup_remote_file(remote_path: str) -> None:
    quoted_path = shlex.quote(remote_path)
    run_ssh(f"bash -lc 'rm -f {quoted_path}'")


def wait_for_job_completion(job_id: int) -> None:
    print(f"⏳ Waiting for job {job_id} to finish...")
    while True:
        result = run_command(["ssh", REMOTE_HOST, f"squeue -h -j {job_id}"], check=False)
        if result.returncode != 0:
            raise CommandError(f"Failed to poll job state for {job_id}: {result.stderr}")
        if not result.stdout.strip():
            break
        time.sleep(SLURM_POLL_INTERVAL_SECONDS)
    sacct_result = run_command([
        "ssh",
        REMOTE_HOST,
        f"sacct -j {job_id} --format=JobID,State --noheader --parsable2"
    ])
    state_lines = [line.strip() for line in sacct_result.stdout.splitlines() if line.strip()]
    if not state_lines:
        raise CommandError(f"Unable to determine final state for job {job_id}.")
    primary_state = state_lines[0].split("|")[-1]
    print(f"✓ Job {job_id} finished with state: {primary_state}")
    if primary_state not in {"COMPLETED"}:
        raise CommandError(f"Job {job_id} did not complete successfully (state={primary_state}).")


def fetch_csv(paths: ClusterPaths, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    local_csv = paths.local_csv_path(destination_dir).resolve()
    remote_source = f"[{REMOTE_HOST}]:{paths.remote_csv_path}"
    run_command(["scp", remote_source, local_csv.as_posix()], text=False)
    if not local_csv.exists():
        raise CommandError(f"CSV download failed; file missing at {local_csv}")
    print(f"✓ Downloaded CSV to {local_csv}")
    return local_csv


def normalize_config() -> tuple[list[str], list[Optional[int]]]:
    """Normalize sweep_id and constrained_model_number to lists and validate."""
    # Convert sweep_id to list
    if isinstance(sweep_id, str):
        # Split by whitespace if multiple IDs are provided
        sweep_ids = sweep_id.split()
    elif isinstance(sweep_id, list):
        sweep_ids = sweep_id
    else:
        raise ValueError("sweep_id must be a string or list of strings")
    
    # Convert constrained_model_number to list
    if constrained_model_number is None:
        model_numbers = [None] * len(sweep_ids)
    elif isinstance(constrained_model_number, int):
        model_numbers = [constrained_model_number] * len(sweep_ids)
    elif isinstance(constrained_model_number, str):
        # Split by whitespace and convert to ints
        model_numbers = [int(x) for x in constrained_model_number.split()]
    elif isinstance(constrained_model_number, list):
        model_numbers = constrained_model_number
    else:
        raise ValueError("constrained_model_number must be None, int, string, or list of ints")
    
    # Validate lengths match
    if len(sweep_ids) != len(model_numbers):
        raise ValueError(f"Length mismatch: {len(sweep_ids)} sweep_ids but {len(model_numbers)} model_numbers. Must be equal.")
    
    return sweep_ids, model_numbers


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Launch benchmark_local.py on remote cluster (currently: REMOTE_HOST='{REMOTE_HOST}').")
    parser.add_argument("--sweep-id", default=None,
                        help=f"Sweep identifier; overrides sweep_id if provided (default: {sweep_id}).")
    parser.add_argument("--training-job-id", type=str, default=None,
                        help="Training job ID to monitor. Benchmark will wait for new models while this job is running.")
    parser.add_argument("--training-job-file", type=str, default=None,
                        help="Path to training job output file to monitor for sweep_id.")
    parser.add_argument("--wait-for-sweep-id", action="store_true", default=False,
                        help="Wait for sweep_id to appear in training job output before starting benchmark.")
    parser.add_argument("--local-dir", type=Path, default=Path.cwd(),
                        help="Directory containing benchmark_local.py (default: current working directory).")
    parser.add_argument("--wait", action="store_true", default=False,
                        help="Wait for job completion and download CSV (default: False).")
    parser.add_argument("--continuous-loop", action="store_true", default=True,
                        help="Run benchmark in continuous loop mode on the cluster (default: True).")
    parser.add_argument("--loop-interval", type=int, default=60,
                        help="Interval between benchmark runs in continuous loop mode (default: 60 seconds = 1 minute).")
    return parser.parse_args(argv)


def run_single_benchmark(current_sweep_id: str, current_model_number: Optional[int], args: argparse.Namespace, local_benchmark_path: Path, local_gymenv_path: Path) -> int:
    """Run a single benchmark job for one sweep_id and model_number combination."""
    # Temporarily override the global constrained_model_number for this run
    global constrained_model_number
    original_model_number = constrained_model_number
    constrained_model_number = current_model_number
    
    try:
        paths = ClusterPaths(sweep_id=current_sweep_id)
        copy_benchmark_to_remote(paths, local_benchmark_path)
        copy_gymenv_to_remote(paths, local_gymenv_path)

        slurm_script = build_slurm_script(
            paths, 
            args.continuous_loop, 
            args.loop_interval, 
            args.training_job_id,
            args.training_job_file,
            args.wait_for_sweep_id
        )
        remote_script_path = stage_slurm_script(paths, slurm_script)
        try:
            job_id = submit_slurm_job(remote_script_path)
        finally:
            cleanup_remote_file(remote_script_path)
        
        return job_id
    finally:
        constrained_model_number = original_model_number


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    
    # Display cluster configuration
    cluster_name = "LMU" if REMOTE_HOST == "c" else "LRZ" if REMOTE_HOST == "h" else "Unknown"
    print(f"="*60)
    print(f"Cluster Configuration: {cluster_name} (REMOTE_HOST={REMOTE_HOST})")
    print(f"Simulation Root: {REMOTE_SIMULATION_ROOT}")
    print(f"Partition: {CLUSTER_CONFIG['partition']}")
    print(f"Conda Env: {CLUSTER_CONFIG['conda_env']}")
    if CLUSTER_CONFIG['exclude_nodes']:
        print(f"Exclude Nodes: {CLUSTER_CONFIG['exclude_nodes']}")
    print(f"="*60)
    print()
    
    # Handle wait_for_sweep_id mode (single run only)
    if args.wait_for_sweep_id:
        if not args.training_job_file:
            raise SystemExit("--training-job-file is required when --wait-for-sweep-id is used")
        current_sweep_id = args.sweep_id if args.sweep_id else "pending"
        
        local_dir = args.local_dir.resolve()
        local_benchmark_path = (local_dir / LOCAL_BENCHMARK_FILENAME)
        if not local_benchmark_path.exists():
            raise SystemExit(f"Local benchmark script not found at {local_benchmark_path}")
        local_gymenv_path = (local_dir / LOCAL_GYMENV_FILENAME)
        if not local_gymenv_path.exists():
            raise SystemExit(f"Local gymenv file not found at {local_gymenv_path}")
        
        job_id = run_single_benchmark(current_sweep_id, constrained_model_number, args, local_benchmark_path, local_gymenv_path)
        
        if args.wait:
            print("Waiting for job completion as requested...")
            wait_for_job_completion(job_id)
            print(f"Job {job_id} completed. Use the post-processing script to download results and run comparison.")
        else:
            print(f"Job {job_id} submitted successfully.")
        return
    
    # Normal mode: process sweep_id and model_number combinations
    # Override with command line argument if provided
    if args.sweep_id:
        current_sweep_ids = [args.sweep_id]
        current_model_numbers = [constrained_model_number]
    else:
        # Use configuration from top of file
        current_sweep_ids, current_model_numbers = normalize_config()
    
    # Validate sweep_ids
    for sid in current_sweep_ids:
        if not sid or sid == "REPLACE_WITH_SWEEP_ID":
            raise SystemExit("Please set valid sweep_id values before running.")
    
    local_dir = args.local_dir.resolve()
    local_benchmark_path = (local_dir / LOCAL_BENCHMARK_FILENAME)
    if not local_benchmark_path.exists():
        raise SystemExit(f"Local benchmark script not found at {local_benchmark_path}")
    local_gymenv_path = (local_dir / LOCAL_GYMENV_FILENAME)
    if not local_gymenv_path.exists():
        raise SystemExit(f"Local gymenv file not found at {local_gymenv_path}")
    
    # Run benchmarks for all combinations
    job_ids = []
    print(f"\n🚀 Starting {len(current_sweep_ids)} benchmark job(s)...\n")
    
    for i, (sid, model_num) in enumerate(zip(current_sweep_ids, current_model_numbers), 1):
        print(f"--- Job {i}/{len(current_sweep_ids)} ---")
        print(f"Sweep ID: {sid}")
        print(f"Model: {model_num if model_num else 'latest'}")
        
        job_id = run_single_benchmark(sid, model_num, args, local_benchmark_path, local_gymenv_path)
        job_ids.append((sid, model_num, job_id))
        print()
    
    # Summary
    print(f"\n✅ All {len(job_ids)} job(s) submitted successfully:")
    for sid, model_num, jid in job_ids:
        model_str = str(model_num) if model_num else "latest"
        print(f"  - Job {jid}: sweep={sid}, model={model_str}")
    
    if args.wait:
        print(f"\n⏳ Waiting for all {len(job_ids)} job(s) to complete...\n")
        for sid, model_num, jid in job_ids:
            model_str = str(model_num) if model_num else "latest"
            print(f"Waiting for job {jid} (sweep={sid}, model={model_str})...")
            wait_for_job_completion(jid)
        print(f"\n✅ All jobs completed. Use the post-processing script to download results.")
    else:
        print(f"\n💡 Use post-processing script to check status and download results later.")

if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except CommandError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)