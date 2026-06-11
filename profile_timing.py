#!/usr/bin/env python3
"""
Timing profiler script for the gymnasium environment.
This script initializes omnisafe_env, makes random steps with train_4_1.png,
and profiles timing for update_radiation_exposure_history_target_area and get_observation_as_images methods.

The script runs 1000 warmup steps first, then profiles the next 100 steps to measure
average timing info. Reset detection and logging are included.
"""

import os
import numpy as np
import time
from get_parameters import para
import class_gymenv
from class_gymenv import *

def main():
    print("=== Environment Performance Profiler ===")
    print(f"RL Type: {para.training_library}")
    print(f"Radiation Type: {para.z_rad_type}")
    
    # Load height data
    height_data_path = os.path.join(os.path.dirname(__file__), 'misc', 'geo_data','height_data', 'height_data.npy')    
    height_data = np.load(height_data_path) 
    
    # Initialize environment based on RL type
    if para.training_library == 'sb3':
        print("Initializing classic gymnasium environment...")
        gymenv = class_gymenv.GymnasiumEnv(height_data, radiation_grid_visualization=False)
        
        # Test the image_path parameter with train_4_1.png
        test_image_path = os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "train_4_1.png")
        print(f"Resetting with image: {test_image_path}")
        gymenv.reset(options={'image_path': test_image_path})
        
        env_wrapper = None
        
    elif para.training_library == 'omnisafe':
        print("Initializing constrained RL (OmniSafe) environment...")
        omnisafe_env = GymEnvOmniSafe(radiation_grid_visualization=False)
        gymenv = omnisafe_env._env
        
        # Test the image_path parameter with train_4_1.png
        test_image_path = os.path.join("misc", "radiation_data", "bw_jon_images_from_paper", "train_4_1.png")
        print(f"Resetting with image: {test_image_path}")
        omnisafe_env.reset(options={'image_path': test_image_path})
        
        env_wrapper = omnisafe_env
        
    else:
        raise ValueError(f"Unknown RL type: {para.training_library}")
    
    print(f"Environment initialized successfully!")
    
    # Print base grid distances (raw and normalized)
    if hasattr(gymenv, 'base_grid_distances_raw') and hasattr(gymenv, 'base_grid_distances'):
        print(f"\n=== Base Grid Distances ===")
        print(f"Grid shape: {gymenv.base_grid_distances.shape}")
        
        # Find and print center values
        grid_shape = gymenv.base_grid_distances.shape
        center_row = grid_shape[0] // 2
        center_col = grid_shape[1] // 2
        print(f"Center position: ({center_row}, {center_col})")
        
        print(f"\nRaw distances (not normalized):")
        print(f"  Center value: {gymenv.base_grid_distances_raw[center_row, center_col]:.6f}")
        print(f"  Min: {np.min(gymenv.base_grid_distances_raw):.6f}, Max: {np.max(gymenv.base_grid_distances_raw):.6f}")
        print(f"  First 5x5 corner:")
        for i in range(5):
            row = "    " + " ".join(f"{gymenv.base_grid_distances_raw[i, j]:7.3f}" for j in range(5))
            print(row)
        
        print(f"\nNormalized distances (center = 1.0):")
        print(f"  Center value: {gymenv.base_grid_distances[center_row, center_col]:.6f}")
        print(f"  Min: {np.min(gymenv.base_grid_distances):.6f}, Max: {np.max(gymenv.base_grid_distances):.6f}")
        print(f"  First 5x5 corner:")
        for i in range(5):
            row = "    " + " ".join(f"{gymenv.base_grid_distances[i, j]:7.3f}" for j in range(5))
            print(row)
    else:
        print("⚠️  Base grid distances not found - check if distance calculation was successful")
    
    # Initialize counters and tracking variables
    total_steps = 0
    reset_count = 0
    warmup_steps = 20
    profiling_steps = 1000
    
    print(f"\n=== Starting Warmup Phase ({warmup_steps} steps) ===")
    print("This allows the environmeaussian_blob_applicatint to reach steady state before profiling...")
    
    # Warmup phase - 1000 steps
    start_warmup = time.perf_counter()
    
    for step in range(warmup_steps):
        # Generate random action (single scalar for turning angle)
        action = np.random.uniform(-1, 1, size=1)
        
        # Step the environment
        if para.training_library == 'sb3':
            obs, reward, terminated, truncated, info = gymenv.step(action)
        else:  # constrained
            obs, reward, cost, terminated, truncated, info = env_wrapper.step(action)
        
        total_steps += 1
        
        # Check for reset
        if terminated or truncated:
            reset_count += 1
            print(f"  Reset #{reset_count} occurred at step {total_steps}")
            
            if para.training_library == 'sb3':
                gymenv.reset(options={'image_path': test_image_path})
            else:
                env_wrapper.reset(options={'image_path': test_image_path})
        
        # Progress update every 100 steps
        if (step + 1) % 100 == 0:
            print(f"  Warmup progress: {step + 1}/{warmup_steps} steps completed")
    
    end_warmup = time.perf_counter()
    warmup_duration = end_warmup - start_warmup
    
    print(f"Warmup completed in {warmup_duration:.2f} seconds")
    print(f"Total resets during warmup: {reset_count}")
    print(f"Average steps per second during warmup: {warmup_steps/warmup_duration:.1f}")
    
    # Clear any existing timing data before profiling
    if hasattr(gymenv, 'get_observation_as_images_times'):
        gymenv.get_observation_as_images_times.clear()
    if hasattr(gymenv, 'update_radiation_exposure_history_target_area_times'):
        gymenv.update_radiation_exposure_history_target_area_times.clear()
    if hasattr(gymenv, 'observation_detailed_times'):
        gymenv.observation_detailed_times.clear()
    if hasattr(gymenv, 'update_radiation_detailed_times'):
        gymenv.update_radiation_detailed_times.clear()
    
    print(f"\n=== Starting Profiling Phase ({profiling_steps} steps) ===")
    print("Measuring timing for key methods...")
    
    # Profiling phase - 100 steps with timing
    profiling_reset_count = 0
    start_profiling = time.perf_counter()
    
    for step in range(profiling_steps):
        # Generate random action (single scalar for turning angle)
        action = np.random.uniform(-1, 1, size=1)
        
        # Step the environment
        if para.training_library == 'sb3':
            obs, reward, terminated, truncated, info = gymenv.step(action)
        else:  # constrained
            obs, reward, cost, terminated, truncated, info = env_wrapper.step(action)
        
        total_steps += 1
        
        # Check for reset
        if terminated or truncated:
            profiling_reset_count += 1
            print(f"  Reset occurred during profiling at step {step + 1}")
            
            if para.training_library == 'sb3':
                gymenv.reset(options={'image_path': test_image_path})
            else:
                env_wrapper.reset(options={'image_path': test_image_path})
        
        # Progress update every 25 steps
        if (step + 1) % 25 == 0:
            print(f"  Profiling progress: {step + 1}/{profiling_steps} steps completed")
    
    end_profiling = time.perf_counter()
    profiling_duration = end_profiling - start_profiling
    
    print(f"\n=== Profiling Results ===")
    print(f"Profiling completed in {profiling_duration:.2f} seconds")
    print(f"Resets during profiling: {profiling_reset_count}")
    print(f"Average steps per second during profiling: {profiling_steps/profiling_duration:.1f}")
    
    # Analyze timing data
    if hasattr(gymenv, 'get_observation_as_images_times') and gymenv.get_observation_as_images_times:
        obs_times = gymenv.get_observation_as_images_times
        print(f"\n--- get_observation_as_images Timing ---")
        print(f"Total calls: {len(obs_times)}")
        print(f"Average time: {np.mean(obs_times)*1000:.2f} ms")
        print(f"Median time: {np.median(obs_times)*1000:.2f} ms")
        print(f"Min time: {np.min(obs_times)*1000:.2f} ms")
        print(f"Max time: {np.max(obs_times)*1000:.2f} ms")
        print(f"Standard deviation: {np.std(obs_times)*1000:.2f} ms")
        print(f"95th percentile: {np.percentile(obs_times, 95)*1000:.2f} ms")
        print(f"Total time spent: {np.sum(obs_times)*1000:.2f} ms")
        print(f"Percentage of total step time: {(np.sum(obs_times)/profiling_duration)*100:.1f}%")
        
        # Display granular timing for get_observation_as_images
        if hasattr(gymenv, 'observation_detailed_times') and gymenv.observation_detailed_times:
            print(f"\n  ↳ Detailed Breakdown:")
            total_detailed = 0
            for operation, times in gymenv.observation_detailed_times.items():
                if times:
                    avg_time = np.mean(times) * 1000
                    total_time = np.sum(times) * 1000
                    percentage = (np.sum(times) / np.sum(obs_times)) * 100
                    total_detailed += np.sum(times)
                    print(f"    {operation}: {avg_time:.2f} ms avg, {total_time:.2f} ms total ({percentage:.1f}%)")
            overhead = (np.sum(obs_times) - total_detailed) * 1000
            overhead_pct = (overhead / (np.sum(obs_times) * 1000)) * 100
            print(f"    overhead/other: {overhead:.2f} ms total ({overhead_pct:.1f}%)")
    else:
        print("\n--- get_observation_as_images Timing ---")
        print("No timing data collected (method might not have been called)")
    
    if hasattr(gymenv, 'update_radiation_exposure_history_target_area_times') and gymenv.update_radiation_exposure_history_target_area_times:
        update_times = gymenv.update_radiation_exposure_history_target_area_times
        print(f"\n--- update_radiation_exposure_history_target_area Timing ---")
        print(f"Total calls: {len(update_times)}")
        print(f"Average time: {np.mean(update_times)*1000:.2f} ms")
        print(f"Median time: {np.median(update_times)*1000:.2f} ms")
        print(f"Min time: {np.min(update_times)*1000:.2f} ms")
        print(f"Max time: {np.max(update_times)*1000:.2f} ms")
        print(f"Standard deviation: {np.std(update_times)*1000:.2f} ms")
        print(f"95th percentile: {np.percentile(update_times, 95)*1000:.2f} ms")
        print(f"Total time spent: {np.sum(update_times)*1000:.2f} ms")
        print(f"Percentage of total step time: {(np.sum(update_times)/profiling_duration)*100:.1f}%")
        
        # Display granular timing for update_radiation_exposure_history_target_area
        if hasattr(gymenv, 'update_radiation_detailed_times') and gymenv.update_radiation_detailed_times:
            print(f"\n  ↳ Detailed Breakdown:")
            total_detailed = 0
            for operation, times in gymenv.update_radiation_detailed_times.items():
                if times:
                    avg_time = np.mean(times) * 1000
                    total_time = np.sum(times) * 1000
                    percentage = (np.sum(times) / np.sum(update_times)) * 100
                    total_detailed += np.sum(times)
                    print(f"    {operation}: {avg_time:.2f} ms avg, {total_time:.2f} ms total ({percentage:.1f}%)")
            overhead = (np.sum(update_times) - total_detailed) * 1000
            overhead_pct = (overhead / (np.sum(update_times) * 1000)) * 100
            print(f"    overhead/other: {overhead:.2f} ms total ({overhead_pct:.1f}%)")
    else:
        print("\n--- update_radiation_exposure_history_target_area Timing ---")
        print("No timing data collected (method might not have been called)")
    
    # Overall summary
    total_time = warmup_duration + profiling_duration
    total_measured_steps = warmup_steps + profiling_steps
    total_resets = reset_count + profiling_reset_count
    
    print(f"\n=== Overall Summary ===")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Total steps: {total_measured_steps}")
    print(f"Total resets: {total_resets}")
    print(f"Overall average steps per second: {total_measured_steps/total_time:.1f}")
    
    if hasattr(gymenv, 'get_observation_as_images_times') and gymenv.get_observation_as_images_times:
        obs_total_time = np.sum(gymenv.get_observation_as_images_times)
        print(f"Time spent in get_observation_as_images: {obs_total_time*1000:.2f} ms ({(obs_total_time/total_time)*100:.1f}% of total)")
    
    if hasattr(gymenv, 'update_radiation_exposure_history_target_area_times') and gymenv.update_radiation_exposure_history_target_area_times:
        update_total_time = np.sum(gymenv.update_radiation_exposure_history_target_area_times)
        print(f"Time spent in update_radiation_exposure_history_target_area: {update_total_time*1000:.2f} ms ({(update_total_time/total_time)*100:.1f}% of total)")
    
    # Performance recommendations
    print(f"\n=== Performance Analysis ===")
    
    # Collect all granular timing data for bottleneck analysis
    all_operations = []
    
    if hasattr(gymenv, 'get_observation_as_images_times') and gymenv.get_observation_as_images_times:
        avg_obs_time = np.mean(gymenv.get_observation_as_images_times)
        if avg_obs_time > 0.01:  # > 10ms
            print("⚠️  get_observation_as_images is taking significant time (>10ms avg)")
        elif avg_obs_time > 0.005:  # > 5ms
            print("⚠️  get_observation_as_images is moderately slow (>5ms avg)")
        else:
            print("✅ get_observation_as_images performance looks good (<5ms avg)")
            
        # Add granular operations
        if hasattr(gymenv, 'observation_detailed_times'):
            for op, times in gymenv.observation_detailed_times.items():
                if times:
                    all_operations.append({
                        'method': 'get_observation_as_images',
                        'operation': op,
                        'avg_time_ms': np.mean(times) * 1000,
                        'total_time_ms': np.sum(times) * 1000,
                        'calls': len(times)
                    })
    
    if hasattr(gymenv, 'update_radiation_exposure_history_target_area_times') and gymenv.update_radiation_exposure_history_target_area_times:
        avg_update_time = np.mean(gymenv.update_radiation_exposure_history_target_area_times)
        if avg_update_time > 0.01:  # > 10ms
            print("⚠️  update_radiation_exposure_history_target_area is taking significant time (>10ms avg)")
        elif avg_update_time > 0.005:  # > 5ms
            print("⚠️  update_radiation_exposure_history_target_area is moderately slow (>5ms avg)")  
        else:
            print("✅ update_radiation_exposure_history_target_area performance looks good (<5ms avg)")
            
        # Add granular operations
        if hasattr(gymenv, 'update_radiation_detailed_times'):
            for op, times in gymenv.update_radiation_detailed_times.items():
                if times:
                    all_operations.append({
                        'method': 'update_radiation_exposure_history_target_area',
                        'operation': op,
                        'avg_time_ms': np.mean(times) * 1000,
                        'total_time_ms': np.sum(times) * 1000,
                        'calls': len(times)
                    })
    
    # Display top bottlenecks
    if all_operations:
        print(f"\n=== Top Performance Bottlenecks ===")
        # Sort by total time spent
        all_operations.sort(key=lambda x: x['total_time_ms'], reverse=True)
        
        print("🔥 Operations taking the most total time:")
        for i, op in enumerate(all_operations[:10]):  # Top 10
            print(f"{i+1:2d}. {op['method']}.{op['operation']}: "
                  f"{op['total_time_ms']:.2f} ms total "
                  f"({op['avg_time_ms']:.3f} ms avg × {op['calls']} calls)")
        
        print(f"\n⏱️  Operations with highest average time:")
        # Sort by average time
        sorted_by_avg = sorted(all_operations, key=lambda x: x['avg_time_ms'], reverse=True)
        for i, op in enumerate(sorted_by_avg[:10]):  # Top 10
            print(f"{i+1:2d}. {op['method']}.{op['operation']}: "
                  f"{op['avg_time_ms']:.3f} ms avg "
                  f"({op['total_time_ms']:.2f} ms total)")
    
    print("\n🎯 Profiling complete!")
    print("💡 Focus optimization efforts on operations with highest total time first.")

if __name__ == "__main__":
    main()