"""Post-process benchmark cluster jobs: check status, download CSV, and run comparison.

This script handles the completion workflow for jobs submitted by z_start_benchmark_cluster.py:
1. Check job status and wait for completion if needed
2. Download the benchmark CSV results
3. Run the comparison script and open the HTML results

Usage:
    python z_benchmark_postprocess.py --job-id 12345 --sweep-id m41x6eo4
    python z_benchmark_postprocess.py --sweep-id m41x6eo4 --check-all
    python z_benchmark_postprocess.py --sweep-id afa2itty --csv-filename benchmark_results_afa2itty_42.csv --analyze-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union
import shlex

# Can be a single value, a list of values, or a space-separated string
# If lists, sweep_ids and constrained_model_numbers must have the same length
# Examples: 
#   String: "eahy35d3 yk5c7m57 393ktsaa"
#   List: ["3jnn3wfi", "rtj99q87", "lmfuq8s0"]
#   Single: "lmfuq8s0"
# Can be None, single int, or space-separated ints. If sweep_id has multiple values, this should have same count
# hohbatwc sez8wxj2 86wkmczr vlxj01cf lzn82k54
# "xtp0ynx3 j02glm93 dlueb6wq oudm03a0 dcv9nj05 e1m3ahnc 5ly4rxye l2t32wgv o8lm3uj1 uk3de0ya tufywdie m7q2m5go v9xag6uc fr7cjdg1 zriqlh0r qmksu12f nxopxyiz evti3k6f 5o38em8e lkiz98og" 
sweep_ids: Union[str, List[str]] = "lzn82k54" 
constrained_model_numbers: Union[int, List[int], None] = None # Example: [64, 120, 150] or single value or None
csv_filename = None  # Set to specific filename (e.g., "benchmark_results_afa2itty_42.csv") or None for auto-generation

do_noise_study = False  # Set to True to download all 15 noise-study CSVs per sweep_id

# Noise study configurations: (noise_type, noise_intensity)
# Combined pose noise: intensity = position std (m), heading std = intensity * 4 (deg)
NOISE_STUDY_CONFIGS = [
    ("pose_combined", 0.0),
    ("pose_combined", 0.5),
    ("pose_combined", 1.0),
    ("pose_combined", 1.5),
    ("pose_combined", 2.0),
]

# Legacy single-value variables for backward compatibility (used internally)
sweep_id = None
constrained_model_number = None
create_html_script = False  # Set to True to generate HTML visualization script, False otherwise
REMOTE_HOST = "c"  # "c" for LMU cluster, "h" for LRZ cluster

# Cluster-specific configurations
CLUSTER_CONFIGS = {
    "c": {  # LMU cluster
        "simulation_root": "/home/stud/blake/git_clones",
        "user": "blake"
    },
    "h": {  # LRZ cluster
        "simulation_root": "/dss/dsshome1/0C/di97sog/git_clones",
        "user": "di97sog"
    }
}

# Get cluster config based on REMOTE_HOST
if REMOTE_HOST not in CLUSTER_CONFIGS:
    raise ValueError(f"Unknown REMOTE_HOST '{REMOTE_HOST}'. Must be 'c' (LMU) or 'h' (LRZ)")

CLUSTER_CONFIG = CLUSTER_CONFIGS[REMOTE_HOST]
REMOTE_SIMULATION_ROOT = CLUSTER_CONFIG["simulation_root"]
REMOTE_USER = CLUSTER_CONFIG["user"]

SLURM_POLL_INTERVAL_SECONDS = 1

@dataclass
class ClusterPaths:
    sweep_id: str
    model_number: Optional[int] = None
    custom_csv_filename: Optional[str] = None

    @property
    def remote_simulation_dir(self) -> str:
        return f"{REMOTE_SIMULATION_ROOT}/Simulation_{self.sweep_id}"

    @property
    def remote_csv_path(self) -> str:
        if self.custom_csv_filename:
            filename = self.custom_csv_filename
        elif self.model_number is not None:
            filename = f"benchmark_results_{self.sweep_id}_{self.model_number}.csv"
        else:
            filename = f"benchmark_results_{self.sweep_id}.csv"
        return f"{self.remote_simulation_dir}/{filename}"

    def local_csv_path(self, destination_dir: Path) -> Path:
        if self.custom_csv_filename:
            filename = self.custom_csv_filename
        elif self.model_number is not None:
            filename = f"benchmark_results_{self.sweep_id}_{self.model_number}.csv"
        else:
            filename = f"benchmark_results_{self.sweep_id}.csv"
        return destination_dir / filename


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


def check_job_status(job_id: int) -> tuple[str, bool]:
    """Check job status. Returns (state, is_running)."""
    result = run_command(["ssh", REMOTE_HOST, f"squeue -h -j {job_id}"], check=False)
    if result.returncode != 0:
        # Job not in queue, check sacct for final state
        sacct_result = run_command([
            "ssh",
            REMOTE_HOST,
            f"sacct -j {job_id} --format=JobID,State --noheader --parsable2"
        ], check=False)
        if sacct_result.returncode == 0:
            state_lines = [line.strip() for line in sacct_result.stdout.splitlines() if line.strip()]
            if state_lines:
                primary_state = state_lines[0].split("|")[-1]
                return primary_state, False
        return "UNKNOWN", False
    
    if result.stdout.strip():
        # Job is still in queue
        return "RUNNING", True
    else:
        # Job finished, get final state
        sacct_result = run_command([
            "ssh",
            REMOTE_HOST,
            f"sacct -j {job_id} --format=JobID,State --noheader --parsable2"
        ])
        state_lines = [line.strip() for line in sacct_result.stdout.splitlines() if line.strip()]
        if not state_lines:
            return "UNKNOWN", False
        primary_state = state_lines[0].split("|")[-1]
        return primary_state, False


def wait_for_job_completion(job_id: int) -> str:
    """Wait for job to complete and return final state."""
    print(f"⏳ Waiting for job {job_id} to finish...")
    while True:
        state, is_running = check_job_status(job_id)
        if not is_running:
            print(f"✓ Job {job_id} finished with state: {state}")
            return state
        time.sleep(SLURM_POLL_INTERVAL_SECONDS)


def fetch_csv(paths: ClusterPaths, destination_dir: Path) -> Path:
    """Download CSV results from remote cluster."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    local_csv = paths.local_csv_path(destination_dir).resolve()
    remote_source = f"[{REMOTE_HOST}]:{paths.remote_csv_path}"
    run_command(["scp", remote_source, local_csv.as_posix()], text=False)
    if not local_csv.exists():
        raise CommandError(f"CSV download failed; file missing at {local_csv}")
    print(f"✓ Downloaded CSV to {local_csv}")
    return local_csv


def fetch_and_combine_subprocess_csvs(paths: ClusterPaths, destination_dir: Path, num_episodes: int = 600) -> Path:
    """
    Download subprocess CSV files from cluster and combine them into a master CSV.
    Falls back to fetching the master CSV if subprocess files don't exist.
    
    Args:
        paths: ClusterPaths object with remote paths
        destination_dir: Local destination directory
        num_episodes: Total number of episodes (number of subprocess CSV files)
        
    Returns:
        Path to the combined local CSV file
    """
    import csv
    from pathlib import Path
    
    destination_dir.mkdir(parents=True, exist_ok=True)
    local_csv = paths.local_csv_path(destination_dir).resolve()
    
    # If constrained_model_number is None, skip subprocess CSV logic and fetch master CSV directly
    if constrained_model_number is None:
        print(f"📥 Fetching master CSV directly (model_number is None)...")
        return fetch_csv(paths, destination_dir)
    
    # Check if subprocess CSV files exist on remote - use Unix-style paths
    # Extract just the filename (stem) and directory from the remote path string
    remote_csv_path = paths.remote_csv_path
    base_csv_name = remote_csv_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]  # Get filename without extension
    remote_dir = remote_csv_path.rsplit('/', 1)[0]  # Get directory path
    
    print(f"🔍 Checking for subprocess CSV files on cluster...")
    print(f"  → Looking for pattern: {remote_dir}/{base_csv_name}_ep*.csv")
    check_result = run_command([
        "ssh", REMOTE_HOST,
        f"ls {remote_dir}/{base_csv_name}_ep*.csv 2>/dev/null | wc -l"
    ], check=False)
    
    if check_result.returncode != 0:
        subprocess_count = 0
    else:
        subprocess_count = int(check_result.stdout.strip())
    
    if subprocess_count == 0:
        # No subprocess files, try to fetch master CSV directly
        print(f"  → No subprocess CSV files found, attempting to fetch master CSV...")
        return fetch_csv(paths, destination_dir)
    
    print(f"  → Found {subprocess_count} subprocess CSV files on cluster")
    print(f"📥 Downloading and combining subprocess CSV files...")
    
    # Create temporary directory for subprocess CSVs
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Download all subprocess CSV files in bulk using scp with wildcard
        remote_pattern = f"{remote_dir}/{base_csv_name}_ep*.csv"
        remote_source = f"[{REMOTE_HOST}]:{remote_pattern}"
        
        print(f"  → Downloading from: {remote_pattern}")
        print(f"  → Downloading subprocess CSV files to temporary directory...")
        try:
            run_command(["scp", remote_source, temp_path.as_posix()], text=False, check=False)
        except Exception as e:
            print(f"  ⚠ Warning: Bulk download may have failed: {e}")
            print(f"  → Falling back to master CSV...")
            return fetch_csv(paths, destination_dir)
        
        # Find all downloaded subprocess CSV files
        subprocess_csvs = sorted(temp_path.glob(f"{base_csv_name}_ep*.csv"))
        
        if not subprocess_csvs:
            print(f"  ⚠ No subprocess CSV files downloaded")
            print(f"  → Falling back to master CSV...")
            return fetch_csv(paths, destination_dir)
        
        print(f"  → Downloaded {len(subprocess_csvs)} subprocess CSV files")
        print(f"  → Combining into master CSV: {local_csv.name}")
        
        # Detect fieldnames dynamically from the first subprocess CSV
        fieldnames = None
        for candidate in subprocess_csvs:
            try:
                with candidate.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    if fieldnames:
                        break
            except Exception:
                pass
        if not fieldnames:
            fieldnames = ["run_number", "timestamp", "model", "epoch", "map",
                          "map_run_id", "ablation_study_name", "seed",
                          "path_length_90_pct", "path_length_end", "coverage_end"]
        
        rows_written = 0
        with local_csv.open("w", newline="") as master_csvfile:
            writer = csv.DictWriter(master_csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            
            for subprocess_csv in subprocess_csvs:
                try:
                    with subprocess_csv.open("r", newline="") as subprocess_file:
                        reader = csv.DictReader(subprocess_file)
                        for row in reader:
                            writer.writerow(row)
                            rows_written += 1
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to read {subprocess_csv.name}: {e}")
        
        print(f"✓ Combined {len(subprocess_csvs)} subprocess CSV files into master CSV")
        print(f"  → Total rows: {rows_written}")
        print(f"  → Output: {local_csv}")
    
    return local_csv


def fetch_png_images(paths: ClusterPaths, destination_dir: Path, timestamp: str) -> bool:
    """Download PNG images from remote cluster if they exist."""
    if paths.model_number is None:
        return False
        
    try:
        # Construct remote and local image paths with timestamped subfolder
        remote_images_dir = f"{paths.remote_simulation_dir}/images_of_paths/benchmark_results_{paths.model_number}"
        local_images_dir = destination_dir / "images_of_paths" / f"postprocess_{timestamp}" / f"benchmark_results_{paths.sweep_id}_{paths.model_number}"
        
        # Check if remote directory exists and count files
        check_result = run_command([
            "ssh", REMOTE_HOST, 
            f"test -d {remote_images_dir} && find {remote_images_dir} -name '*.png' -type f | wc -l || echo '0'"
        ], check=False)
        
        if check_result.returncode != 0:
            print(f"  → Error checking remote directory: {remote_images_dir}")
            return False
        
        file_count = int(check_result.stdout.strip())
        
        if file_count == 0:
            print(f"  → No PNG images found on remote (directory {remote_images_dir})")
            return False
        
        print(f"  → Found {file_count} PNG files on remote, downloading in bulk...")
        
        # Create the full timestamped directory structure
        local_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Delete existing contents if directory already had files
        if any(local_images_dir.iterdir()):
            import shutil
            for item in local_images_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            print(f"  → Cleared existing contents for clean download")
        
        # Use scp with -r flag to recursively copy entire directory in one operation
        remote_source = f"[{REMOTE_HOST}]:{remote_images_dir}"
        local_dest = local_images_dir.as_posix()
        
        try:
            run_command(["scp", "-r", remote_source, local_dest], text=False)
        except CommandError as e:
            print(f"  ❌ Failed to download PNG images: {e}")
            return False
        
        # Verify download by counting local files
        if local_images_dir.exists():
            local_png_count = len(list(local_images_dir.glob("*.png")))
            if local_png_count > 0:
                print(f"  ✓ Downloaded {local_png_count} PNG images to {local_images_dir}")
                return True
            else:
                print(f"  ❌ Download completed but no PNG files found locally")
                return False
        else:
            print(f"  ❌ Local directory was not created: {local_images_dir}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error downloading PNG images: {e}")
        return False





def analyze_performance_over_time(csv_file: Path, sweep_id: str) -> Optional[str]:
    """Analyze performance metrics over multiple benchmark runs and create tracking visualization.
    
    Returns HTML string of the performance tracking plot if successful, None otherwise.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio
    from datetime import datetime
    
    if not csv_file or not csv_file.exists():
        print("⚠ No CSV file to analyze")
        return None
    
    print(f"📊 Analyzing performance from: {csv_file.name}")
    
    try:
        # Read CSV data
        df = pd.read_csv(csv_file)
        if df.empty:
            print("❌ CSV file is empty")
            return None
        
        # Check if run_number column exists (new format)
        if 'run_number' not in df.columns:
            print("❌ CSV file doesn't contain run_number column - cannot analyze over time")
            return None
        
        # Group by run_number and calculate averages for each run
        run_data = []
        for run_number in sorted(df['run_number'].unique()):
            run_df = df[df['run_number'] == run_number]
            
            # Calculate average metrics for this run
            avg_path_90 = run_df['path_length_90_pct'].replace('', float('nan')).astype(float).mean()
            avg_path_end = run_df['path_length_end'].replace('', float('nan')).astype(float).mean()
            avg_coverage = run_df['coverage_end'].replace('', float('nan')).astype(float).mean()
            
            # Get timestamp if available
            timestamp = run_df['timestamp'].iloc[0] if 'timestamp' in run_df.columns else 'Unknown'
            
            run_data.append({
                'run_number': run_number,
                'timestamp': timestamp,
                'avg_path_length_90_pct': avg_path_90,
                'avg_path_length_end': avg_path_end,
                'avg_coverage_end': avg_coverage,
                'episode_count': len(run_df)
            })
        
        if not run_data:
            print("❌ No valid run data found")
            return None
        
        # Only create HTML if flag is enabled
        performance_html = None
        if create_html_script:
            # Create interactive performance tracking plot using Plotly
            fig = go.Figure()
            fig.update_layout(
                title=f'Performance Metrics Over Time - Sweep {sweep_id}',
                xaxis_title='Benchmark Run',
                yaxis=dict(
                    title=dict(text='Path Length (m)', font=dict(color='red')),
                    tickfont=dict(color='red'),
                    side='left'
                ),
                yaxis2=dict(
                    title=dict(text='Coverage (%)', font=dict(color='green')),
                    tickfont=dict(color='green'),
                    overlaying='y',
                    side='right',
                    tickformat='.1%'
                ),
                hovermode='x unified'
            )
            
            run_numbers = [d['run_number'] for d in run_data]
            
            # Plot 1: Average final path length (left y-axis)
            path_end_values = [d['avg_path_length_end'] for d in run_data]
            valid_end = [(r, v) for r, v in zip(run_numbers, path_end_values) if not pd.isna(v)]
            if valid_end:
                runs_end, vals_end = zip(*valid_end)
                fig.add_trace(go.Scatter(
                    x=runs_end,
                    y=vals_end,
                    mode='lines+markers',
                    name='Avg Path Length End',
                    line=dict(color='red', width=2),
                    marker=dict(size=8, color='red'),
                    text=[f'{val:.1f}m' for val in vals_end],
                    textposition='top center',
                    hovertemplate='<b>Run %{x}</b><br>Path Length: %{y:.1f}m<extra></extra>'
                ))
            
            # Plot 2: Average coverage (right y-axis)
            coverage_values = [d['avg_coverage_end'] for d in run_data]
            valid_cov = [(r, v) for r, v in zip(run_numbers, coverage_values) if not pd.isna(v)]
            if valid_cov:
                runs_cov, vals_cov = zip(*valid_cov)
                fig.add_trace(go.Scatter(
                    x=runs_cov,
                    y=vals_cov,
                    mode='lines+markers',
                    name='Avg Coverage End',
                    line=dict(color='green', width=2),
                    marker=dict(size=8, color='green'),
                    yaxis='y2',
                    text=[f'{val:.1%}' for val in vals_cov],
                    textposition='bottom center',
                    hovertemplate='<b>Run %{x}</b><br>Coverage: %{y:.1%}<extra></extra>'
                ))
            
            # Convert plot to HTML
            performance_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
            
            print(f"✓ Performance tracking plot generated")
        else:
            print(f"⊘ HTML script creation disabled (create_html_script=False)")
        
        # Print summary statistics - only for the last run
        if run_data:
            last_run = run_data[-1]  # Get the most recent run
            print("\n📈 Performance Summary (Last Run Only):")
            print("-" * 80)
            path_90_str = f"{last_run['avg_path_length_90_pct']:.1f}m" if not pd.isna(last_run['avg_path_length_90_pct']) else "nan"
            print(f"Run {last_run['run_number']:3d} ({last_run['timestamp']}): "
                  f"Path90={path_90_str}, "
                  f"PathEnd={last_run['avg_path_length_end']:.1f}m, "
                  f"Coverage={last_run['avg_coverage_end']:.1%}, "
                  f"Episodes={last_run['episode_count']}")
        
        return performance_html
                  
    except Exception as e:
        print(f"❌ Error analyzing performance: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_comparison_and_open_results(sweep_id: str, specific_csv: Optional[Path] = None, performance_html: Optional[str] = None) -> None:
    """Run benchmark comparison script and open HTML results."""
    import pandas as pd
    import tempfile
    
    print("Running benchmark comparison script...")
    comparison_script_path = r"C:\Users\johan\programming\Simulation\compare_benchmark_plotly.py"
    
    # Determine which CSV file to use
    original_csv_path = None
    if specific_csv and specific_csv.exists():
        original_csv_path = specific_csv
        print(f"  → Using specific CSV: {specific_csv.name}")
    else:
        # Look for the standard CSV file in the benchmark results directory
        standard_csv = Path.cwd() / "misc" / "logs" / "benchmark_results" / f"benchmark_results_{sweep_id}.csv"
        if standard_csv.exists():
            original_csv_path = standard_csv
            print(f"  → Using CSV: {standard_csv.name}")
        else:
            print(f"❌ No CSV file found for comparison: benchmark_results_{sweep_id}.csv")
            return
    
    # Filter CSV to only include the last run_number for benchmark comparison
    try:
        df = pd.read_csv(original_csv_path)
        if 'run_number' in df.columns:
            # Get the maximum run_number
            max_run_number = df['run_number'].max()
            filtered_df = df[df['run_number'] == max_run_number].copy()
            
            # Create a temporary filtered CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as temp_file:
                filtered_df.to_csv(temp_file, index=False)
                filtered_csv_path = temp_file.name
            
            print(f"  → Filtered to run_number {max_run_number} only ({len(filtered_df)} entries)")
            ours_csv_path = filtered_csv_path
        else:
            # No run_number column, use original file
            ours_csv_path = str(original_csv_path)
            filtered_csv_path = None
            print("  → No run_number column found, using all data")
    except Exception as e:
        print(f"Warning: Could not filter CSV data: {e}")
        ours_csv_path = str(original_csv_path)
        filtered_csv_path = None
    
    try:
        comparison_result = run_command([
            "python", comparison_script_path, "--ours", ours_csv_path
        ], check=False)
        
        if comparison_result.returncode != 0:
            print(f"Warning: Comparison script failed with exit code {comparison_result.returncode}")
            print(f"stderr: {comparison_result.stderr}")
        else:
            print("✓ Benchmark comparison completed successfully")
            
            # If we have performance tracking HTML, integrate it into the benchmark comparison
            html_file_path = Path(r"C:\Users\johan\programming\Simulation\benchmark_comparison.html")
            if performance_html and html_file_path.exists():
                try:
                    # Read the existing benchmark comparison HTML
                    existing_html = html_file_path.read_text(encoding="utf-8")
                    
                    # Insert the performance tracking section before the closing </body> tag
                    performance_section = f'''
        <section>
            <h2>Performance Tracking Over Time</h2>
            <div class="description">
                This chart shows how the performance metrics have evolved across multiple benchmark runs for sweep {sweep_id}.
                The red line represents the average path length at episode end, while the green line shows the average coverage achieved.
            </div>
            {performance_html}
        </section>
    </body>'''
                    
                    # Replace the closing </body> tag with our new section
                    modified_html = existing_html.replace('</body>', performance_section)
                    
                    # Write the combined HTML file
                    html_file_path.write_text(modified_html, encoding="utf-8")
                    print("✓ Integrated performance tracking into benchmark comparison")
                    
                except Exception as e:
                    print(f"Warning: Could not integrate performance tracking: {e}")
            
            # Open HTML file only if create_html_script is True
            if create_html_script:
                if html_file_path.exists():
                    print("Opening combined benchmark comparison in default browser...")
                    try:
                        os.startfile(str(html_file_path))
                        print("✓ Opened benchmark_comparison.html in default browser")
                    except Exception as e:
                        print(f"Warning: Could not open HTML file: {e}")
                else:
                    print(f"Warning: HTML file not found at {html_file_path}")
            else:
                print("⊘ Skipping HTML file opening (create_html_script=False)")
    
    finally:
        # Clean up temporary file if it was created
        if filtered_csv_path:
            try:
                os.unlink(filtered_csv_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")


def find_latest_benchmark_job(sweep_id: str) -> Optional[int]:
    """Find the most recent benchmark job for the given sweep_id."""
    job_name_pattern = f"benchmark_{sweep_id}"
    result = run_command([
        "ssh", REMOTE_HOST, 
        f"sacct -u {REMOTE_USER} -S $(date -d '7 days ago' '+%Y-%m-%d') --format=JobID,JobName,State,Submit --noheader --parsable2 | "
        f"grep '{job_name_pattern}' | sort -t'|' -k4 -r | head -1"
    ], check=False)
    
    if result.stdout.strip():
        job_line = result.stdout.strip()
        job_id_str = job_line.split("|")[0]
        try:
            return int(job_id_str)
        except ValueError:
            pass
    return None


def list_recent_jobs() -> None:
    """List recent benchmark jobs for the current user."""
    print("Recent benchmark jobs:")
    result = run_command([
        "ssh", REMOTE_HOST, 
        f"squeue -u {REMOTE_USER} -o '%.10i %.15j %.8T %.10M %.20S' --noheader || true; "
        f"sacct -u {REMOTE_USER} -S $(date -d '1 day ago' '+%Y-%m-%d') --format=JobID,JobName,State,Elapsed,Start --noheader | head -10"
    ], check=False)
    if result.stdout.strip():
        print(result.stdout)
    else:
        print("No recent jobs found.")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process benchmark cluster jobs.")
    parser.add_argument("--job-id", type=int,
                        help="Job ID to check and process (optional - will auto-detect if not provided).")
    parser.add_argument("--sweep-id", default=None,
                        help=f"Sweep identifier for the benchmark run (default: from config).")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "misc" / "logs" / "benchmark_results",
                        help="Local directory to store the fetched CSV (default: misc/logs/benchmark_results).")
    parser.add_argument("--no-wait", action="store_true", default=False,
                        help="Don't wait for job completion, just check status (default: False).")
    parser.add_argument("--skip-comparison", action="store_true", default=False,
                        help="Skip running the comparison script and opening results (default: False).")
    parser.add_argument("--check-all", action="store_true", default=False,
                        help="List recent jobs instead of processing a specific job (default: False).")
    parser.add_argument("--poll-interval", type=int, default=1,
                        help="Polling interval in seconds while waiting for job completion (default: 1).")
    parser.add_argument("--analyze-only", action="store_true", default=False,
                        help="Skip job processing and CSV download, analyze existing local CSV files only (default: False).")
    parser.add_argument("--no-analyze-only", dest="analyze_only", action="store_false",
                        help="Process jobs and download CSV instead of analyzing existing files.")
    parser.add_argument("--csv-filename", type=str, default=None,
                        help="Custom CSV filename to use instead of default benchmark_results_{sweep_id}.csv format.")
    parser.add_argument("--num-episodes", type=int, default=600,
                        help="Total number of episodes that were run (for combining subprocess CSV files, default: 600).")
    return parser.parse_args(argv)


def process_single_sweep(sweep_id_val: str, model_number_val: Optional[int], args: argparse.Namespace, timestamp: str) -> None:
    """Process a single sweep ID and model number combination."""
    global sweep_id, constrained_model_number
    sweep_id = sweep_id_val
    constrained_model_number = model_number_val
    
    print(f"\n{'='*80}")
    print(f"Processing: sweep_id={sweep_id}, model_number={model_number_val}")
    print(f"{'='*80}\n")
    
    job_id = args.job_id
    
    # Auto-detect job ID if not provided (but we won't use it for status checking)
    if not job_id:
        print(f"Finding latest benchmark job for sweep_id: {sweep_id}")
        job_id = find_latest_benchmark_job(sweep_id)
        if job_id:
            print(f"Found job ID: {job_id} (but skipping status check)")
        else:
            print(f"No benchmark jobs found for sweep_id: {sweep_id} (continuing anyway)")
    
    # Skip all job status checking - just try to download CSV immediately
    print("⚡ Skipping job status check - downloading CSV in current state...")
    
    # Handle different processing modes
    if args.analyze_only:
        # Analyze existing local CSV file
        print(f"🔍 Analyzing existing CSV file for sweep {sweep_id}...")
        filename = args.csv_filename or csv_filename
        if filename is None and model_number_val is not None:
            filename = f"benchmark_results_{sweep_id}_{model_number_val}.csv"
        elif filename is None:
            filename = f"benchmark_results_{sweep_id}.csv"
        csv_file = args.output_dir / filename
        
        if csv_file.exists():
            performance_html = analyze_performance_over_time(csv_file, sweep_id)
            if not args.skip_comparison:
                print(f"📊 Running comparison with CSV: {csv_file.name}")
                run_comparison_and_open_results(sweep_id, specific_csv=csv_file, performance_html=performance_html)
        else:
            print(f"❌ No local CSV file found: {csv_file}")
        return
    
    # Download CSV - auto-generate filename if model number is specified
    final_csv_filename = args.csv_filename or csv_filename
    if final_csv_filename is None and model_number_val is not None:
        final_csv_filename = f"benchmark_results_{sweep_id}_{model_number_val}.csv"
    paths = ClusterPaths(sweep_id=sweep_id, model_number=model_number_val, custom_csv_filename=final_csv_filename)
    
    csv_file = None
    performance_html = None
    
    try:
        # Use the new function that handles subprocess CSV combining
        csv_file = fetch_and_combine_subprocess_csvs(paths, args.output_dir.resolve(), num_episodes=args.num_episodes)
        print(f"✓ CSV file ready: {csv_file.name}")
        
        # Create performance tracking analysis
        performance_html = analyze_performance_over_time(csv_file, sweep_id)
    except CommandError as e:
        print(f"Failed to download/combine CSV: {e}")
        print("Continuing to check for PNG images...")
    
    # Download PNG images if model_number is specified (regardless of CSV success)
    if model_number_val is not None:
        print(f"🖼️ Checking for PNG images for model {model_number_val}...")
        png_success = fetch_png_images(paths, args.output_dir.resolve(), timestamp)
        
        # If CSV failed but PNG succeeded, we still have some results
        if csv_file is None and png_success:
            print("✓ PNG images downloaded successfully (even though CSV failed)")
        elif csv_file is None and not png_success:
            print("❌ Both CSV and PNG downloads failed")
            return
    
    # Run comparison and open results (only if we have a CSV file)
    if not args.skip_comparison and csv_file is not None:
        run_comparison_and_open_results(sweep_id, specific_csv=csv_file, performance_html=performance_html)
    elif csv_file is None:
        print("⚠ Skipping comparison due to missing CSV file")
    
    print(f"✓ Post-processing completed for sweep {sweep_id}\n")


def main(argv: Optional[list[str]] = None) -> None:
    from datetime import datetime
    
    args = parse_args(argv)
    
    # Generate a single timestamp for this entire post-processing session
    # This ensures all sweeps' images go into the same timestamped folder
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Display cluster configuration
    cluster_name = "LMU" if REMOTE_HOST == "c" else "LRZ" if REMOTE_HOST == "h" else "Unknown"
    print(f"="*60)
    print(f"Cluster Configuration: {cluster_name} (REMOTE_HOST={REMOTE_HOST})")
    print(f"Simulation Root: {REMOTE_SIMULATION_ROOT}")
    print(f"User: {REMOTE_USER}")
    print(f"Session Timestamp: {session_timestamp}")
    print(f"="*60)
    print()
    
    global SLURM_POLL_INTERVAL_SECONDS
    SLURM_POLL_INTERVAL_SECONDS = max(5, args.poll_interval)
    
    if args.check_all:
        list_recent_jobs()
        return
    
    # Determine sweep IDs and model numbers to process
    current_sweep_ids = args.sweep_id if args.sweep_id else sweep_ids
    current_model_numbers = constrained_model_numbers
    
    # Convert to lists for uniform processing
    # Handle space-separated string format (e.g., "id1 id2 id3")
    if isinstance(current_sweep_ids, str):
        # Check if it's a space-separated string with multiple IDs
        if ' ' in current_sweep_ids:
            sweep_id_list = current_sweep_ids.split()
        else:
            sweep_id_list = [current_sweep_ids]
    else:
        sweep_id_list = current_sweep_ids
    
    if current_model_numbers is None:
        model_number_list = [None] * len(sweep_id_list)
    elif isinstance(current_model_numbers, int):
        model_number_list = [current_model_numbers] * len(sweep_id_list)
    else:
        model_number_list = current_model_numbers
    
    # Validate that lists have the same length
    if len(sweep_id_list) != len(model_number_list):
        print(f"❌ Error: sweep_ids and model_numbers must have the same length")
        print(f"   Got {len(sweep_id_list)} sweep IDs and {len(model_number_list)} model numbers")
        sys.exit(1)
    
    # ----- Noise study mode: expand pairs to cover all 15 noise configs -----
    if do_noise_study:
        args.skip_comparison = True  # no comparison script for noise CSVs
        base_pairs = list(zip(sweep_id_list, model_number_list))
        total = len(base_pairs) * len(NOISE_STUDY_CONFIGS)
        print(f"Noise study mode: downloading {total} CSVs ({len(base_pairs)} sweeps × {len(NOISE_STUDY_CONFIGS)} noise configs)...\n")
        done = 0
        for sweep_id_val, model_number_val in base_pairs:
            for noise_type, noise_intensity in NOISE_STUDY_CONFIGS:
                done += 1
                noise_suffix = f"_noise_{noise_type}_{noise_intensity:.2f}"
                if model_number_val is not None:
                    args.csv_filename = f"benchmark_results_{sweep_id_val}_{model_number_val}{noise_suffix}.csv"
                else:
                    args.csv_filename = f"benchmark_results_{sweep_id_val}{noise_suffix}.csv"
                print(f"\n[{done}/{total}] sweep={sweep_id_val}  {noise_type} {noise_intensity:.0%}")
                try:
                    process_single_sweep(sweep_id_val, model_number_val, args, session_timestamp)
                except Exception as e:
                    print(f"  ⚠ Failed: {e}")
        print(f"\n{'='*80}")
        print(f"✓ Noise study download complete ({total} configs)")
        print(f"{'='*80}")
        return
    # -------------------------------------------------------------------------

    # Process each sweep/model pair
    total_pairs = len(sweep_id_list)
    print(f"Processing {total_pairs} sweep/model pair(s)...\n")
    
    for idx, (sweep_id_val, model_number_val) in enumerate(zip(sweep_id_list, model_number_list), 1):
        print(f"\n{'#'*80}")
        print(f"# Processing pair {idx}/{total_pairs}")
        print(f"{'#'*80}")
        try:
            process_single_sweep(sweep_id_val, model_number_val, args, session_timestamp)
        except Exception as e:
            print(f"❌ Error processing sweep {sweep_id_val} with model {model_number_val}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Continuing with next pair...\n")
    
    print(f"\n{'='*80}")
    print(f"✓ All post-processing completed ({total_pairs} pair(s))")
    print(f"{'='*80}")
    
    # Ask user if they want to combine all tables into one
    if total_pairs > 1:
        print(f"\n📊 You have {total_pairs} separate CSV files.")
        combine_response = input("Do you want to combine them into one table? (y/n): ").strip().lower()
        
        if combine_response == 'y':
            combine_csv_files(sweep_id_list, model_number_list, args.output_dir.resolve())


def combine_csv_files(sweep_id_list: List[str], model_number_list: List[Optional[int]], output_dir: Path) -> Optional[Path]:
    """Combine multiple CSV files into one table."""
    import pandas as pd
    
    print(f"\n{'='*80}")
    print("COMBINING CSV FILES")
    print(f"{'='*80}")
    
    # Collect all CSV files to combine
    csv_files: List[tuple[Path, str]] = []
    for sweep_id_val, model_number_val in zip(sweep_id_list, model_number_list):
        if model_number_val is not None:
            filename = f"benchmark_results_{sweep_id_val}_{model_number_val}.csv"
        else:
            filename = f"benchmark_results_{sweep_id_val}.csv"
        
        csv_path = output_dir / filename
        if csv_path.exists():
            csv_files.append((csv_path, sweep_id_val))
            print(f"  → Found: {filename}")
        else:
            print(f"  ⚠ Missing: {filename}")
    
    if len(csv_files) == 0:
        print("❌ No CSV files found to combine")
        return None
    
    if len(csv_files) == 1:
        print("⚠ Only one CSV file found, nothing to combine")
        return csv_files[0][0]
    
    print(f"\n📥 Combining {len(csv_files)} CSV files...")
    
    try:
        # Read and combine all CSV files
        dfs = []
        for csv_file, sweep_id_val in csv_files:
            df = pd.read_csv(csv_file)
            df.insert(0, 'sweep_id', sweep_id_val)
            dfs.append(df)
            print(f"  → Read {len(df)} rows from {csv_file.name}")
        
        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Generate output filename - use consistent naming with z_ablation_study_postprocess.py
        combined_filename = "ablation_studies_evaluation_files.csv"
        combined_path = output_dir / combined_filename
        
        # Save combined CSV
        combined_df.to_csv(combined_path, index=False)
        print(f"\n✓ Combined CSV saved: {combined_filename}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Total columns: {len(combined_df.columns)}")
        
        return combined_path
        
    except Exception as e:
        print(f"❌ Error combining CSV files: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except CommandError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        sys.exit(1)