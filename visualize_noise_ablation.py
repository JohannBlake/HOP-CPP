#!/usr/bin/env python3
"""
Noise robustness study visualization.
Loads all  benchmark_results_*_noise_pose_combined_*.csv  files,
groups by noise_intensity, and plots mean +/- std for each metric.

noise_intensity encodes:
  position std  = intensity  (meters)
  heading  std  = intensity * 4  (degrees)
So intensity=0 is the clean no-noise baseline run under identical conditions.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "misc" / "logs" / "benchmark_results"
OUTPUT_PNG  = Path(__file__).parent / "noise_robustness.png"
OUTPUT_HTML = Path(__file__).parent / "noise_robustness.html"

LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]   # noise_intensity values (meters)

STEP_SIZE_M = 2.6          # base_timestep (10 s) * speed (0.26 m/s) – no wall gliding
METERS_PER_PIXEL = 0.0375  # class_gymenv.py line 175

METRICS = [
    # (y-axis label,              csv column,    lower_is_better)
    ("Steps to 90% coverage",   "steps_90_pct", True),
    ("Steps to 99% coverage",   "steps_99_pct", True),
    ("Steps to final coverage", "steps_end",    True),
]

COLOR   = "#9b5de5"
COLOR_0 = "#555555"   # baseline marker
# ---------------------------------------------------------------------------


def load_noise_csvs(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("*_noise_pose_combined_*.csv"))
    if not files:
        raise FileNotFoundError(f"No pose_combined noise CSVs found in {results_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "sweep_id" not in df.columns:
            m = re.match(r"benchmark_results_([a-z0-9]+)_", f.name)
            if m:
                df["sweep_id"] = m.group(1)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    for col in ["noise_intensity", "steps_90_pct", "steps_99_pct", "steps_end"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    # coverage_end: convert 0-1 to 0-100 %
    if "coverage_end" in combined.columns and combined["coverage_end"].max() <= 1.01:
        combined["coverage_end"] = combined["coverage_end"] * 100.0
    return combined


def compute_stats(df: pd.DataFrame, col: str):
    intensities, means, stds, ns = [], [], [], []
    for ni in LEVELS:
        rows = df[np.isclose(df["noise_intensity"], ni, atol=1e-4)][col].dropna()
        intensities.append(ni)
        means.append(rows.mean() if len(rows) else np.nan)
        stds.append(rows.std(ddof=1) if len(rows) > 1 else 0.0)
        ns.append(len(rows))
    return np.array(intensities), np.array(means), np.array(stds), np.array(ns)


def make_figure(df: pd.DataFrame) -> plt.Figure:
    n = len(METRICS)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3.2 * n), squeeze=False)
    fig.suptitle("Pose Noise Robustness\n(pos std = x m,  heading std = 4x deg)",
                 fontsize=13, fontweight="bold", y=1.01)

    x_labels = [f"pos σ={ni/METERS_PER_PIXEL:.1f}px\nhdg σ={ni*4:.0f}°" for ni in LEVELS]

    for ri, (ylabel, col, lower_is_better) in enumerate(METRICS):
        ax = axes[ri][0]
        if col not in df.columns:
            ax.set_visible(False)
            continue

        intensities, means, stds, ns = compute_stats(df, col)
        has = ~np.isnan(means)

        if has.any():
            ax.plot(intensities[has], means[has],
                    color=COLOR, linewidth=2, marker="o", markersize=6,
                    zorder=3, label="mean")
            ax.fill_between(intensities[has],
                            means[has] - stds[has],
                            means[has] + stds[has],
                            color=COLOR, alpha=0.20, zorder=2, label="+/-1 std")

        # Mark baseline distinctly
        if has[0]:
            ax.plot(intensities[0], means[0], marker="D", markersize=9,
                    color=COLOR_0, zorder=4, label=f"no noise (n={ns[0]})")

        # Annotate n per point
        for xi, mi, ni_val in zip(intensities[has], means[has], ns[has]):
            ax.annotate(f"n={ni_val}", xy=(xi, mi),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7, color="#555555")

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(LEVELS)
        ax.set_xticklabels(x_labels, fontsize=8)
        if ri == n - 1:
            ax.set_xlabel("Position noise  (pos σ [px], hdg σ = 4×pos [deg])", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="upper left" if lower_is_better else "lower left")

    fig.tight_layout()
    return fig


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("NOISE ROBUSTNESS RESULTS  (mean +/- std,  n = number of episodes)")
    print("noise_intensity: position std (m),  heading std = intensity x 4 deg")
    print("=" * 78)
    col_w = 22
    header = f"  {'Intensity':<22} | " + " | ".join(
        f"{lbl[:col_w]:<{col_w}}" for lbl, _, _ in METRICS
    )
    print(header)
    print("-" * len(header))
    for ni in LEVELS:
        rows = df[np.isclose(df["noise_intensity"], ni, atol=1e-4)]
        n = len(rows)
        tag = " (baseline)" if ni == 0.0 else ""
        ni_px = ni / METERS_PER_PIXEL
        label = f"{ni_px:.1f}px / hdg {ni*4:.0f}°{tag}"
        parts = []
        for _, col, _ in METRICS:
            if col not in df.columns:
                parts.append(f"{'N/A (col missing)':<{col_w}}")
                continue
            v = rows[col].dropna()
            if v.empty:
                parts.append(f"{'N/A':<{col_w}}")
            elif len(v) > 1:
                parts.append(f"{v.mean():7.1f} +/-{v.std(ddof=1):5.1f} n={n:<4}")
            else:
                parts.append(f"{v.mean():7.1f}           n={n:<4}")
        print(f"  {label:<22} | " + " | ".join(parts))
    print("=" * 78)


def make_plotly_html(df: pd.DataFrame) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("(plotly not available - HTML skipped)")
        return

    n = len(METRICS)
    metric_labels = [lbl for lbl, _, _ in METRICS]
    fig = make_subplots(rows=n, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.06,
                        subplot_titles=metric_labels)

    x_tick_vals = LEVELS
    x_tick_text = [f"pos σ={ni/METERS_PER_PIXEL:.1f}px / hdg σ={ni*4:.0f}°" for ni in LEVELS]

    for ri, (ylabel, col, _) in enumerate(METRICS, start=1):
        if col not in df.columns:
            continue
        intensities, means, stds, ns = compute_stats(df, col)
        has = ~np.isnan(means)
        xp = intensities[has].tolist()
        yp = means[has].tolist()
        ep = stds[has].tolist()

        fig.add_trace(go.Scatter(
            x=xp, y=yp,
            error_y=dict(type="data", array=ep, visible=True),
            mode="lines+markers",
            name=ylabel,
            line=dict(color=COLOR),
            marker=dict(size=8),
            legendgroup=ylabel,
            showlegend=(ri == 1),
        ), row=ri, col=1)

        fig.update_yaxes(title_text=ylabel, row=ri, col=1)

    fig.update_xaxes(
        tickvals=x_tick_vals,
        ticktext=x_tick_text,
        title_text="Position noise  (pos σ [px]  /  hdg σ = 4×pos [deg])",
        row=n, col=1,
    )
    fig.update_layout(
        height=280 * n,
        title_text="Pose Noise Robustness Study",
        template="plotly_white",
    )
    fig.write_html(str(OUTPUT_HTML))
    print(f"Saved interactive HTML -> {OUTPUT_HTML}")


def main() -> None:
    print(f"Loading from: {RESULTS_DIR}")
    df = load_noise_csvs(RESULTS_DIR)

    counts = df.groupby("noise_intensity").size()
    print(f"Loaded {len(df)} rows across {len(counts)} intensity levels:")
    for ni, cnt in counts.items():
        print(f"  {ni:.1f}m / {ni*4:.0f}deg  ->  {cnt} episodes")

    print_summary(df)

    fig = make_figure(df)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure -> {OUTPUT_PNG}")

    make_plotly_html(df)


if __name__ == "__main__":
    main()