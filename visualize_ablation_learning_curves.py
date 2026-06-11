#!/usr/bin/env python3
"""
Learning curve visualization for ablation studies.
Loads ablation_studies_evaluation_files.csv, groups by ablation_study_name
and epoch, averages over maps and seeds, then plots mean +/- std bands.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------------------------------------------------------------------------
CSV_PATH    = Path(__file__).parent / "misc" / "logs" / "benchmark_results" / "ablation_studies_evaluation_files.csv"
OUTPUT_HTML = Path(__file__).parent / "ablation_learning_curves.html"

METRICS = [
    # (y-axis label,                csv column,           lower_is_better, scale_factor)
    ("Coverage at end (%)",         "coverage_end",       False,           100.0),
    ("Sum reward",                  "sum_reward",         False,           1.0),
    ("Steps to 90% coverage",       "steps_90_pct",       True,            1.0),
    ("Steps to final coverage",     "steps_end",          True,            1.0),
    ("Total degrees turned",        "total_degrees_turned", True,          1.0),
    ("Sum OPTV (no black)",         "sum_optv_no_black",  False,           1.0),
]

# Colour palette — one per ablation condition
PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
# ---------------------------------------------------------------------------


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Symmetric edge-padded moving average that preserves array length."""
    if window <= 1 or len(arr) == 0:
        return arr.copy()
    half = window // 2
    padded = np.pad(arr, (half, window - half - 1), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "coverage_end", "sum_reward", "steps_90_pct", "sum_optv",
        "sum_optv_no_black", "run_number", "env_steps_per_epoch",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert coverage_end from 0-1 to 0-100 % if needed
    if "coverage_end" in df.columns and df["coverage_end"].max() <= 1.01:
        df["coverage_end"] = df["coverage_end"] * 100.0

    # Training steps on x-axis
    if "env_steps_per_epoch" in df.columns:
        df["training_steps"] = df["run_number"] * df["env_steps_per_epoch"]
    else:
        df["training_steps"] = df["run_number"]

    return df


def compute_curve(df: pd.DataFrame, ablation: str, col: str):
    """Return (epochs, training_steps, means, stds) averaged over maps and seeds."""
    sub = df[df["ablation_study_name"] == ablation].copy()
    grouped = (
        sub.groupby("run_number")[col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)

    # Retrieve matching training_steps (constant per run_number)
    step_map = (
        sub.groupby("run_number")["training_steps"].first().reset_index()
    )
    grouped = grouped.merge(step_map, on="run_number")
    grouped = grouped.sort_values("run_number")

    epochs   = grouped["run_number"].values
    steps    = grouped["training_steps"].values
    means    = grouped["mean"].values
    stds     = grouped["std"].values
    return epochs, steps, means, stds


def show_interactive(df: pd.DataFrame) -> None:
    """Open an interactive matplotlib window with a moving-average slider."""
    ablation_names = sorted(df["ablation_study_name"].unique())
    n_metrics = len(METRICS)

    # Pre-compute raw (x, means, stds) for every metric × ablation
    raw: dict = {}   # col -> ablation -> (x, m, s)
    for _ylabel, col, _lib, _scale in METRICS:
        if col not in df.columns:
            continue
        raw[col] = {}
        for ablation in ablation_names:
            _epochs, steps, means, stds = compute_curve(df, ablation, col)
            valid = ~np.isnan(means)
            if valid.any():
                raw[col][ablation] = (steps[valid], means[valid], stds[valid])

    fig, axes = plt.subplots(
        n_metrics, 1,
        figsize=(10, 3.5 * n_metrics),
        squeeze=False,
    )
    fig.suptitle("Ablation Study — Learning Curves", fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.10, top=0.94, hspace=0.45)

    # plot_items[(ri, ablation)] = {"line": ..., "fill": ..., "x": ..., "m": ..., "s": ...}
    plot_items: dict = {}

    for ri, (ylabel, col, lower_is_better, _scale) in enumerate(METRICS):
        ax = axes[ri][0]
        if col not in raw or not raw[col]:
            ax.text(0.5, 0.5, f"No data for '{col}'",
                    ha="center", va="center", transform=ax.transAxes, color="grey")
            ax.set_visible(True)
        else:
            for ai, ablation in enumerate(ablation_names):
                if ablation not in raw[col]:
                    continue
                color = PALETTE[ai % len(PALETTE)]
                x, m, s = raw[col][ablation]
                line, = ax.plot(x, m, color=color, linewidth=2,
                                marker="o", markersize=4, label=ablation, zorder=3)
                fill = ax.fill_between(x, m - s, m + s,
                                       color=color, alpha=0.15, zorder=2)
                plot_items[(ri, ablation)] = dict(line=line, fill=fill, x=x, m=m, s=s, ax=ax)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        direction = "↑ higher is better" if not lower_is_better else "↓ lower is better"
        ax.set_title(direction, fontsize=8, loc="right", color="#777777", pad=2)
        if ri == n_metrics - 1:
            ax.set_xlabel("Training steps", fontsize=10)
        ax.legend(fontsize=9,
                  loc="upper left" if lower_is_better else "lower right",
                  framealpha=0.8)

    # ---- Slider ----
    ax_slider = fig.add_axes([0.15, 0.03, 0.70, 0.025])
    slider = Slider(ax_slider, "Moving avg window", 1, 30,
                    valinit=1, valstep=1, color="#4477AA")

    def _update(_val):
        w = int(slider.val)
        for (ri, ablation), item in plot_items.items():
            x, m, s = item["x"], item["m"], item["s"]
            sm = moving_average(m, w)
            ss = moving_average(s, w)
            item["line"].set_ydata(sm)
            item["fill"].remove()
            color = item["line"].get_color()
            item["fill"] = item["ax"].fill_between(
                x, sm - ss, sm + ss, color=color, alpha=0.15, zorder=2)
            item["ax"].relim()
            item["ax"].autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(_update)
    plt.show(block=True)


def save_html(fig: plt.Figure, path: Path) -> None:
    """Save an interactive HTML via mpld3 if available, else a static image embed."""
    try:
        import mpld3
        html = mpld3.fig_to_html(fig)
        path.write_text(html, encoding="utf-8")
        print(f"Saved interactive HTML: {path}")
    except ImportError:
        import base64, io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        b64 = base64.b64encode(buf.getvalue()).decode()
        html = f'<html><body><img src="data:image/png;base64,{b64}"/></body></html>'
        path.write_text(html, encoding="utf-8")
        print(f"Saved static HTML (mpld3 not installed): {path}")


def print_summary(df: pd.DataFrame) -> None:
    ablation_names = sorted(df["ablation_study_name"].unique())
    print("\n" + "=" * 72)
    print("ABLATION STUDY — FINAL EPOCH SUMMARY")
    print("=" * 72)
    last_epoch = df["run_number"].max()
    last = df[df["run_number"] == last_epoch]
    for col_label, col, lower_is_better, _ in METRICS:
        if col not in df.columns:
            continue
        print(f"\n  {col_label}  (epoch={last_epoch})")
        for ablation in ablation_names:
            vals = last[last["ablation_study_name"] == ablation][col].dropna()
            if len(vals):
                print(f"    {ablation:<25s}  mean={vals.mean():.3f}  std={vals.std(ddof=1):.3f}  n={len(vals)}")
    print("=" * 72)


def main():
    print(f"Loading: {CSV_PATH}")
    df = load_data(CSV_PATH)
    print(f"Rows: {len(df)}  |  Epochs: {df['run_number'].min()}–{df['run_number'].max()}"
          f"  |  Ablation conditions: {sorted(df['ablation_study_name'].unique())}")

    print_summary(df)
    show_interactive(df)


if __name__ == "__main__":
    main()
