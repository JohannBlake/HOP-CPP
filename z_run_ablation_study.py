#!/usr/bin/env python3
"""
Script that runs ablation study by testing different parameter combinations.
For each combination:
1. Updates parameters_default.yaml with the configuration
2. Runs z_run_sweep_and_benchmark.py
3. Only proceeds to next combination if previous run was successful
"""

import subprocess
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime
import winsound
import time

how_many_seeds_for_ablation_study = 2
# first ablation study run was 51 and how_many_seeds_for_ablation_study = 2
# second ablation study run was 53 and how_many_seeds_for_ablation_study = 4
starting_seed = 51 


# 9 combinations for all ablation studies but best run jonnarth.
COMBINATIONS = [
    {
        "name": "SOTA",
        "params": {
            "ablation_study_name": "SOTA",
            "ablation_study_fisheye_instead_of_multi_scale_maps": False,
            "ablation_study_use_visit_frequency": False,
            "ablation_study_use_optv": False,
            "ablation_study_use_radiation_instead_of_lidar": False,
            "omnisafe_alg": "SAC",
            "update_omnisafe_config_world_model_parameter_to": False,
            "ablation_study_fuse_lidar_sensor_data_and_image_data": True,
            "ablation_study_use_frontier_maps": True,
            "worker_to_use_with_slurm_command": "#SBATCH --exclude=worker-1,worker-2,worker-3,worker-4,worker-5,worker-6,worker-7,worker-9,worker-10"
        },
    }
]


def update_yaml_parameters(yaml_file, params):
    """
    Update the parameters in the YAML file.
    
    Args:
        yaml_file: Path to parameters_default.yaml
        params: Dictionary of parameters to update
    """
    print(f"Updating {yaml_file.name}...")
    
    # Read the current YAML file
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Update the parameters
    for key, value in params.items():
        config[key] = value
        print(f"  {key}: {value}")
    
    # Write back to the file
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✓ Parameters updated successfully\n")

def update_sac_world_model_parameter(sac_yaml_file, use_world_model):
    """
    Update the use_world_model parameter in SAC.yaml.
    
    Args:
        sac_yaml_file: Path to SAC.yaml
        use_world_model: Boolean value to set for use_world_model
    """
    print(f"Updating {sac_yaml_file.name}...")
    
    # Read the current YAML file
    with open(sac_yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Update the use_world_model parameter
    if 'defaults' in config and 'model_cfgs' in config['defaults'] and 'world_model' in config['defaults']['model_cfgs']:
        config['defaults']['model_cfgs']['world_model']['use_world_model'] = use_world_model
        print(f"  use_world_model: {use_world_model}")
    else:
        print("  ⚠ Warning: world_model configuration not found in SAC.yaml")
        print(f"  Config structure: {list(config.keys())}")
    
    # Write back to the file
    with open(sac_yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✓ SAC.yaml updated successfully\n")


def run_sweep_and_benchmark(script_path):
    """
    Run the z_run_sweep_and_benchmark.py script.
    
    Args:
        script_path: Path to z_run_sweep_and_benchmark.py
    
    Returns:
        Tuple of (success: bool, sweep_id: str or None)
    """
    print("Running sweep and benchmark...")
    print("-" * 80)
    
    try:
        # Set PYTHONIOENCODING to utf-8 for the subprocess
        env = dict(os.environ)
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            capture_output=True,
            env=env
        )
        
        # Extract sweep ID from output
        sweep_id = None
        for line in result.stdout.splitlines():
            if "Sweep ID:" in line:
                sweep_id = line.split("Sweep ID:")[-1].strip()
                break
        
        print(result.stdout)
        print("-" * 80)
        print("✓ Sweep and benchmark completed successfully\n")
        return True, sweep_id
        
    except subprocess.CalledProcessError as e:
        print(e.stdout if e.stdout else "")
        print(e.stderr if e.stderr else "")
        print("-" * 80)
        print(f"❌ Sweep and benchmark failed with return code {e.returncode}\n")
        return False, None
    except Exception as e:
        print("-" * 80)
        print(f"❌ Unexpected error: {e}\n")
        return False, None


def main():
    """Main execution flow."""
    script_dir = Path(__file__).parent
    yaml_file = script_dir / "parameters_default.yaml"
    sweep_script = script_dir / "z_run_sweep_and_benchmark.py"
    
    # Verify files exist
    if not yaml_file.exists():
        print(f"❌ ERROR: {yaml_file} not found")
        sys.exit(1)
    
    if not sweep_script.exists():
        print(f"❌ ERROR: {sweep_script} not found")
        sys.exit(1)
    
    total_seeds = how_many_seeds_for_ablation_study
    total_combinations = len(COMBINATIONS)
    total_runs = 0
    successful_runs = 0
    sweep_ids = []  # Store all sweep IDs
    
    print("=" * 80)
    print("ABLATION STUDY - Running parameter combinations")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total seeds: {total_seeds}")
    print(f"Total combinations per seed: {total_combinations}")
    print(f"Total runs: {total_seeds * total_combinations}")
    print("=" * 80)
    print()
    
    for i in range(how_many_seeds_for_ablation_study):
        seed = starting_seed + i
        print("\n" + "=" * 80)
        print(f"SEED {i+1}/{total_seeds}: {seed}")
        print("=" * 80)
        
        # overwrite seed in YAML
        print(f"Setting seed to {seed}")
        update_yaml_parameters(yaml_file, {"seed": seed})
        
        for idx, combination in enumerate(COMBINATIONS, 1):
            total_runs += 1
            print("\n" + "-" * 80)
            print(f"RUN {total_runs}/{total_seeds * total_combinations}")
            print(f"Seed {i+1}/{total_seeds}, Combination {idx}/{total_combinations}: {combination['name']}")
            print("-" * 80)
            print()
            
            # Update YAML file with current combination
            update_yaml_parameters(yaml_file, combination['params'])
            
            # Update SAC.yaml if update_omnisafe_config_world_model_parameter_to is specified
            if 'update_omnisafe_config_world_model_parameter_to' in combination['params']:
                sac_yaml_file = script_dir / "omnisafe" / "configs" / "off-policy" / "SAC.yaml"
                if sac_yaml_file.exists():
                    update_sac_world_model_parameter(
                        sac_yaml_file,
                        combination['params']['update_omnisafe_config_world_model_parameter_to']
                    )
                else:
                    print(f"⚠ Warning: {sac_yaml_file} not found")
            
            # Run sweep and benchmark
            success, sweep_id = run_sweep_and_benchmark(sweep_script)
            
            if success:
                successful_runs += 1
                if sweep_id:
                    sweep_ids.append(sweep_id)
                print(f"✓ Run {total_runs}/{total_seeds * total_combinations} completed successfully")
                print()
            else:
                print(f"❌ Run {total_runs}/{total_seeds * total_combinations} FAILED")
                print("Stopping ablation study due to failure.")
                print()
                # Print final summary before exiting
                print("=" * 80)
                print("ABLATION STUDY SUMMARY (INCOMPLETE)")
                print("=" * 80)
                print(f"Total expected runs: {total_seeds * total_combinations}")
                print(f"Completed runs: {total_runs}")
                print(f"Successful runs: {successful_runs}")
                print(f"Failed runs: {total_runs - successful_runs}")
                print(f"Stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)
                sys.exit(1)
    
    # Final summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"Total seeds: {total_seeds}")
    print(f"Total combinations: {total_combinations}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {total_runs - successful_runs}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Print sweep IDs in requested format
    if sweep_ids:
        print("\nSWEEP IDS:")
        print(" ".join(sweep_ids))
        print("=" * 80)
    
    if successful_runs == total_runs:
        print("\n✓ All combinations completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Study incomplete.")
        sys.exit(1)


if __name__ == "__main__":
    main()
    # make 3 ping sounds when done
    for _ in range(3):
        winsound.Beep(1000, 300)  # 1000 Hz frequency, 300 ms duration
        time.sleep(0.2)  # Short pause between beeps