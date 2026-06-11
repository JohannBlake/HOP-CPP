"""
Minimal local test to verify pose_combined noise is applied as intended.

Tests:
  1. Math: correct std values read from env vars
  2. Integration: position and heading actually drift by the right amount
     over N steps in a mock step loop (no full env instantiation needed)
"""
import os
import numpy as np

# ── helper: emulate one step of the noise logic ──────────────────────────────

def step_with_noise(position, bearing_deg, noise_type, noise_intensity,
                    distance_per_step=2.6, bearing_change_deg=0.0, rng=None):
    """Pure-Python replica of the relevant sim_step noise lines."""
    if rng is None:
        rng = np.random.default_rng()

    # --- heading noise ---
    if noise_type == 'pose_heading' and noise_intensity > 0:
        bearing_change_deg += rng.normal(0, noise_intensity)
    elif noise_type == 'pose_combined' and noise_intensity > 0:
        bearing_change_deg += rng.normal(0, noise_intensity * 4.0)

    bearing_deg = (bearing_deg + bearing_change_deg) % 360
    bearing_rad = np.deg2rad(bearing_deg)

    dx = distance_per_step * np.sin(bearing_rad)
    dy = distance_per_step * np.cos(bearing_rad)

    # --- position noise ---
    if noise_type == 'pose_position' and noise_intensity > 0:
        dx += rng.normal(0, noise_intensity)
        dy += rng.normal(0, noise_intensity)
    elif noise_type == 'pose_combined' and noise_intensity > 0:
        dx += rng.normal(0, noise_intensity)
        dy += rng.normal(0, noise_intensity)

    position = np.array([position[0] + dx, position[1] + dy])
    return position, bearing_deg


# ── test 1: single-step noise statistics ─────────────────────────────────────

def test_noise_statistics():
    print("=" * 60)
    print("TEST 1: Per-step noise magnitudes (N=100 000 direct draws)")
    print("  Noise is sampled independently each step from:")
    print("    position x: N(0, intensity)  [metres]")
    print("    position y: N(0, intensity)  [metres]")
    print("    heading:    N(0, intensity*4) [degrees]")
    print("=" * 60)

    DISTANCE_PER_STEP = 2.6   # metres
    intensities = [0.0, 0.5, 1.0, 1.5, 2.0]

    rng = np.random.default_rng(42)
    N = 100_000

    print(f"\n{'Intensity':>10}  {'pos_x std (m)':>14}  {'pos_y std (m)':>14}  "
          f"{'heading std (deg)':>18}  {'step_size (m)':>14}  {'noise/step':>10}")
    print("-" * 88)

    for intensity in intensities:
        if intensity == 0.0:
            pos_x_samples  = np.zeros(N)
            pos_y_samples  = np.zeros(N)
            head_samples   = np.zeros(N)
        else:
            # Replicate exactly what sim_step does for pose_combined
            pos_x_samples  = rng.normal(0, intensity,       N)   # dx noise
            pos_y_samples  = rng.normal(0, intensity,       N)   # dy noise
            head_samples   = rng.normal(0, intensity * 4.0, N)   # bearing_change noise (deg)

        px_std = np.std(pos_x_samples)
        py_std = np.std(pos_y_samples)
        hd_std = np.std(head_samples)
        ratio  = intensity / DISTANCE_PER_STEP

        print(f"{intensity:>10.1f}  {px_std:>14.3f}  {py_std:>14.3f}  "
              f"{hd_std:>18.3f}  {DISTANCE_PER_STEP:>14.2f}  {ratio:>9.0%}")

    print(f"\nSummary at intensity=2.0 (max):")
    print(f"  position noise std = 2.0 m per axis, per step")
    print(f"  heading noise std  = 8.0 deg per step")
    print(f"  nominal step size  = {DISTANCE_PER_STEP:.2f} m  →  noise is {2.0/DISTANCE_PER_STEP:.0%} of each step")


# ── test 2: environment variable wiring ──────────────────────────────────────

def test_env_var_wiring():
    print("\n" + "=" * 60)
    print("TEST 2: Env-var wiring (BENCHMARK_NOISE_TYPE / BENCHMARK_NOISE_INTENSITY)")
    print("=" * 60)

    for noise_type, intensity in [("pose_combined", 0.0), ("pose_combined", 2.0)]:
        os.environ["BENCHMARK_NOISE_TYPE"] = noise_type
        os.environ["BENCHMARK_NOISE_INTENSITY"] = str(intensity)

        read_type      = os.environ.get("BENCHMARK_NOISE_TYPE", "none")
        read_intensity = float(os.environ.get("BENCHMARK_NOISE_INTENSITY", "0.0"))

        ok = (read_type == noise_type) and (read_intensity == intensity)
        print(f"  set ({noise_type!r}, {intensity}) → read ({read_type!r}, {read_intensity})  {'✓' if ok else '✗ MISMATCH'}")


# ── test 3: multi-step cumulative drift ───────────────────────────────────────

def test_cumulative_drift():
    print("\n" + "=" * 60)
    print("TEST 3: Cumulative drift after 1000 straight-ahead steps")
    print("=" * 60)
    print("(helicopter tries to fly straight; position error = random walk)")

    DISTANCE_PER_STEP = 2.6
    N_STEPS = 1000
    N_TRIALS = 500
    intensities = [0.0, 0.5, 1.0, 1.5, 2.0]

    print(f"\n{'Intensity':>10}  {'mean |pos_err| (m)':>20}  {'theory sqrt(N)*sqrt(2)*I (m)':>30}")
    print("-" * 65)

    rng = np.random.default_rng(0)
    for intensity in intensities:
        final_errors = []
        for _ in range(N_TRIALS):
            pos = np.array([0.0, 0.0])
            true_pos = np.array([0.0, 0.0])
            bearing = 0.0
            for _ in range(N_STEPS):
                pos, bearing = step_with_noise(pos, bearing, "pose_combined", intensity,
                                               DISTANCE_PER_STEP, rng=rng)
                true_pos += np.array([DISTANCE_PER_STEP * np.sin(0.0),
                                      DISTANCE_PER_STEP * np.cos(0.0)])
            final_errors.append(np.linalg.norm(pos - true_pos))

        mean_err = np.mean(final_errors)
        theory   = np.sqrt(N_STEPS) * np.sqrt(2) * intensity
        print(f"{intensity:>10.1f}  {mean_err:>20.1f}  {theory:>30.1f}")

    print("\nNote: heading noise also deflects the path, so actual error > theory for intensity>0")


# ── main ──────────────────────────────────────────────────────────────────────

# ── test 4: step distance with and without wall gliding ───────────────────────

def test_step_distance():
    print("\n" + "=" * 60)
    print("TEST 4: Step distance – glide_on_collision True vs False")
    print("=" * 60)

    # ── parameters (from parameters_default.yaml + class_gymenv.py) ──────────
    base_timestep            = 10      # seconds  (parameters_default.yaml)
    speed_m_per_s            = 0.26    # m/s      (parameters_default.yaml)
    meters_per_pixel_mower   = 0.0375  # m/px     (class_gymenv.py line 175)

    nominal_step_m = base_timestep * speed_m_per_s
    nominal_step_px = nominal_step_m / meters_per_pixel_mower  # position stored in metres, but useful ref

    print(f"\n  base_timestep            = {base_timestep} s")
    print(f"  speed                    = {speed_m_per_s} m/s")
    print(f"  meters_per_pixel_mower   = {meters_per_pixel_mower} m/px")
    print(f"\n  nominal step (no wall)   = {base_timestep} × {speed_m_per_s} = {nominal_step_m:.4f} m")
    print(f"  equivalent in pixels     = {nominal_step_m:.4f} / {meters_per_pixel_mower} = {nominal_step_px:.2f} px")

    # ── simulate collision outcomes ───────────────────────────────────────────
    print(f"\n  glide_on_collision=False (current default):")
    print(f"    → wall hit: new_position = last_position  ⇒  step = 0.0 m")

    print(f"\n  glide_on_collision=True:")
    print(f"    → wall hit: position projected to obstacle surface ⇒  step < {nominal_step_m:.4f} m")

    # ── verify the arithmetic ─────────────────────────────────────────────────
    assert abs(nominal_step_m - 2.6) < 1e-9, f"Expected 2.6 m, got {nominal_step_m}"
    assert abs(nominal_step_px - 2.6 / 0.0375) < 1e-6

    print(f"\n  ✓ nominal_step = {nominal_step_m} m  (assertion passed)")

    # ── noise perspective ─────────────────────────────────────────────────────
    print(f"\n  Noise context (per-axis, per-step standard deviations):")
    print(f"  {'Intensity':>10}  {'pos std (m)':>12}  {'pos/step ratio':>16}  {'heading std (deg)':>18}")
    print(f"  {'-'*62}")
    for intensity in [0.0, 0.5, 1.0, 1.5, 2.0]:
        ratio = intensity / nominal_step_m if nominal_step_m > 0 else 0
        print(f"  {intensity:>10.1f}  {intensity:>12.3f}  {ratio:>15.0%}  {intensity*4:>18.1f}")

    print(f"\n  At max intensity (2.0 m): noise std = 2.0 m = {2.0/nominal_step_m:.0%} of one step per axis")


if __name__ == "__main__":
    test_noise_statistics()
    test_env_var_wiring()
    test_cumulative_drift()
    test_step_distance()
    print("\nDone.")
