#!/usr/bin/env python3
"""
Script that:
1. Runs z_start_sweep_cluster.py to start a sweep
2. Retrieves the sweep_id from the cluster job output
3. Runs z_start_benchmark_cluster.py with the retrieved sweep_id
"""

import subprocess
import sys
import time
import re
import os
from pathlib import Path

REMOTE_HOST = "c"
SWEEP_ID_PATTERN = r'maximilianuniversity-of-munich/Heli-Logs/sweeps/([a-z0-9]{8})'
JOB_ID_PATTERN = r'Submitted batch job (\d+)'

def run_sweep_script():
    """Run the sweep cluster script and extract job ID."""
    print("=" * 80)
    print("STEP 1: Starting sweep on cluster...")
    print("=" * 80)
    
    sweep_script = Path(__file__).parent / "z_start_sweep_cluster.py"
    if not sweep_script.exists():
        raise FileNotFoundError(f"Sweep script not found: {sweep_script}")
    
    # Run the sweep script with proper environment settings
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    result = subprocess.run(
        [sys.executable, str(sweep_script)],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        raise RuntimeError(f"Sweep script failed with return code {result.returncode}")
    
    # Extract job ID from the output
    combined_output = result.stdout + result.stderr
    match = re.search(JOB_ID_PATTERN, combined_output)
    if not match:
        raise RuntimeError("Could not find job ID in sweep script output")
    
    job_id = match.group(1)
    
    # Set job file path based on cluster
    if REMOTE_HOST == "c":
        job_file = f"/home/stud/blake/{job_id}.txt"
    elif REMOTE_HOST == "h":
        job_file = f"/dss/dsshome1/0C/di97sog/{job_id}.txt"
    else:
        raise ValueError(f"Unknown REMOTE_HOST: {REMOTE_HOST}")
    
    print(f"\n✓ Sweep script completed successfully")
    print(f"Job ID: {job_id}")
    print(f"Job file: {job_file}")
    
    return job_id, job_file


def wait_for_sweep_id(job_file, max_wait_seconds=300, poll_interval=10):
    """
    Monitor the job output file for the sweep_id.
    
    Args:
        job_file: Path to the job output file on the remote server
        max_wait_seconds: Maximum time to wait for sweep_id
        poll_intepoll_intervalrval: Time between checks in seconds
    
    Returns:
        The sweep_id (8 character string)
    """
    print(f"\nMonitoring {job_file} for sweep_id...")
    print(f"Will check every {poll_interval} seconds for up to {max_wait_seconds} seconds")
    
    start_time = time.time()
    attempts = 0
    
    while time.time() - start_time < max_wait_seconds:
        attempts += 1
        
        # Read the job output file
        cmd = ["ssh", REMOTE_HOST, f"cat {job_file} 2>/dev/null"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if result.returncode == 0 and result.stdout:
            # Search for sweep_id pattern
            match = re.search(SWEEP_ID_PATTERN, result.stdout)
            if match:
                sweep_id = match.group(1)
                print(f"\n✓ Found sweep_id: {sweep_id}")
                return sweep_id
        
        elapsed = int(time.time() - start_time)
        print(f"  Attempt {attempts} (elapsed: {elapsed}s) - sweep_id not found yet, waiting...", end='\r')
        time.sleep(poll_interval)
    
    raise TimeoutError(f"Sweep_id not found after {max_wait_seconds} seconds")


def run_benchmark_script(sweep_id, training_job_id):
    """Run the benchmark cluster script with the given sweep_id and training_job_id."""
    print("\n" + "=" * 80)
    print(f"STEP 2: Starting benchmark for sweep_id: {sweep_id}")
    print(f"Training job ID: {training_job_id}")
    print("=" * 80)
    
    benchmark_script = Path(__file__).parent / "z_start_benchmark_cluster.py"
    if not benchmark_script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_script}")
    
    # Run the benchmark script with the sweep_id, training_job_id and proper environment settings
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    result = subprocess.run(
        [sys.executable, str(benchmark_script), "--sweep-id", sweep_id, "--training-job-id", training_job_id],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark script failed with return code {result.returncode}")
    
    print(f"\n✓ Benchmark script completed successfully for sweep_id: {sweep_id}")


def run_benchmark_script_without_sweep_id(training_job_id, job_file):
    """
    Run the benchmark cluster script without sweep_id.
    The benchmark job will wait internally for sweep_id to appear in the training job output.
    
    Args:
        training_job_id: The SLURM job ID of the training job
        job_file: Path to the training job output file where sweep_id will appear
    """
    print("\n" + "=" * 80)
    print(f"STEP 2: Starting benchmark job (will wait for sweep_id internally)")
    print(f"Training job ID: {training_job_id}")
    print(f"Will monitor: {job_file}")
    print("=" * 80)
    
    benchmark_script = Path(__file__).parent / "z_start_benchmark_cluster.py"
    if not benchmark_script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_script}")
    
    # Run the benchmark script with training_job_id and job_file for sweep_id monitoring
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    result = subprocess.run(
        [sys.executable, str(benchmark_script), 
         "--training-job-id", training_job_id,
         "--training-job-file", job_file,
         "--wait-for-sweep-id"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark script failed with return code {result.returncode}")
    
    print(f"\n✓ Benchmark job submitted successfully (monitoring training job {training_job_id})")

def main():
    """Main execution flow."""
    try:
        # Step 1: Run sweep script and get job ID
        job_id, job_file = run_sweep_script()
        
        # Step 2: Run benchmark script immediately with training job_id
        # The benchmark job will wait internally for sweep_id to appear
        run_benchmark_script_without_sweep_id(job_id, job_file)
        
        print("\n" + "=" * 80)
        print("SUCCESS! Both sweep and benchmark jobs have been started.")
        print(f"Training Job ID: {job_id}")
        print(f"Training Job File: {job_file}")
        print("The benchmark job will wait for the sweep_id to appear in the training job output.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()