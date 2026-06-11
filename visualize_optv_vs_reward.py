import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ── publication style (from vis_for_icaps_paper.py) ──────────────────────────
try:
    plt.style.use('seaborn-v0_8-paper')
except Exception:
    try:
        plt.style.use('seaborn-paper')
    except Exception:
        pass

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'lines.linewidth': 2,
    'lines.markersize': 5,
    'grid.alpha': 0.3,
})

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "misc", "logs", "benchmark_results", "ablation_studies_evaluation_files.csv"
)

df = pd.read_csv(CSV_PATH)

# Compute per-map averages (across seeds/sweeps), then mean±std over maps
COLS = ["sum_optv", "sum_reward",
        "sum_optv_no_black", "sum_reward_if_optv_no_black",
        "total_distance_travelled", "wall_gliding_distance",
        "border_length_target_area", "border_length_obstacle_intersection"]

per_map = df.groupby("map")[COLS].mean()

stats = {}
for c in COLS:
    stats[c] = {"mean": per_map[c].mean(), "std": per_map[c].std()}

# wall-glide fraction per map, then mean±std
wg_frac = per_map["wall_gliding_distance"] / per_map["total_distance_travelled"] * 100
wg_frac_mean, wg_frac_std = wg_frac.mean(), wg_frac.std()

# ratio per map, then mean±std
r1 = per_map["sum_optv"] / per_map["sum_reward"] * 100
r2 = per_map["sum_optv_no_black"] / per_map["sum_reward_if_optv_no_black"] * 100
r3 = per_map["border_length_obstacle_intersection"] / per_map["border_length_target_area"] * 100

# ── print ─────────────────────────────────────────────────────────────────────
W = 55
print("=" * W)
print("AVERAGES  (mean ± std across maps, seeds averaged per map)")
print("=" * W)
print(f"  sum_optv                      : {stats['sum_optv']['mean']:9.3f}  ± {stats['sum_optv']['std']:.3f}")
print(f"  sum_reward                    : {stats['sum_reward']['mean']:9.3f}  ± {stats['sum_reward']['std']:.3f}")
print(f"  ratio sum_optv / sum_reward   : {r1.mean():+.2f}%  ± {r1.std():.2f}%")
print()
print(f"  sum_optv_no_black             : {stats['sum_optv_no_black']['mean']:9.3f}  ± {stats['sum_optv_no_black']['std']:.3f}")
print(f"  sum_reward_if_optv_no_black   : {stats['sum_reward_if_optv_no_black']['mean']:9.3f}  ± {stats['sum_reward_if_optv_no_black']['std']:.3f}")
print(f"  ratio optv_nb / reward_nb     : {r2.mean():+.2f}%  ± {r2.std():.2f}%")
print()
print(f"  total_distance_travelled      : {stats['total_distance_travelled']['mean']:9.3f}  ± {stats['total_distance_travelled']['std']:.3f}")
print(f"  wall_gliding_distance         : {stats['wall_gliding_distance']['mean']:9.3f}  ± {stats['wall_gliding_distance']['std']:.3f}")
print(f"  wall-glide % of total dist    : {wg_frac_mean:+.2f}%  ± {wg_frac_std:.2f}%")
print()
print(f"  border_length_target_area     : {stats['border_length_target_area']['mean']:9.3f}  ± {stats['border_length_target_area']['std']:.3f}")
print(f"  border_length_obstacle_inter  : {stats['border_length_obstacle_intersection']['mean']:9.3f}  ± {stats['border_length_obstacle_intersection']['std']:.3f}")
print(f"  obstacle_inter % of target    : {r3.mean():+.2f}%  ± {r3.std():.2f}%")
print("=" * W)

# ── plot ──────────────────────────────────────────────────────────────────────
# Three grouped-bar panels, each with 2 bars (the two compared quantities)
# Error bars = std across maps

COLORS = {
    "sum_optv":                              "#4C72B0",
    "sum_reward":                            "#DD8452",
    "sum_optv_no_black":                     "#55A868",
    "sum_reward_if_optv_no_black":           "#C44E52",
    "total_distance_travelled":              "#8172B2",
    "wall_gliding_distance":                 "#CCB974",
    "border_length_target_area":             "#64B5CD",
    "border_length_obstacle_intersection":   "#E07340",
}

panels = [
    {
        "title": "sum_optv  vs  sum_reward",
        "pairs": [("sum_optv", "sum_optv"), ("sum_reward", "sum_reward")],
        "ylabel": "Value",
        "ratio_label": "optv / reward",
        "ratio_mean": r1.mean(), "ratio_std": r1.std(),
    },
    {
        "title": "sum_optv_no_black  vs  sum_reward_if_optv_no_black",
        "pairs": [("sum_optv_no_black", "sum_optv_no_black"),
                  ("sum_reward_if_optv_no_black", "sum_reward_if_optv_no_black")],
        "ylabel": "Value",
        "ratio_label": "optv_nb / reward_nb",
        "ratio_mean": r2.mean(), "ratio_std": r2.std(),
    },
    {
        "title": "total_distance_travelled  vs  wall_gliding_distance",
        "pairs": [("total_distance_travelled", "total_distance_travelled"),
                  ("wall_gliding_distance", "wall_gliding_distance")],
        "ylabel": "Distance",
        "ratio_label": "wall-glide % of total",
        "ratio_mean": wg_frac_mean, "ratio_std": wg_frac_std,
    },
    {
        "title": "border_length_target_area  vs  border_length_obstacle_intersection",
        "pairs": [("border_length_target_area", "border_length_target_area"),
                  ("border_length_obstacle_intersection", "border_length_obstacle_intersection")],
        "ylabel": "Border Length",
        "ratio_label": "obstacle_inter % of target_area",
        "ratio_mean": r3.mean(), "ratio_std": r3.std(),
    },
]

fig, axes_grid = plt.subplots(2, 2, figsize=(13, 10))
axes = [axes_grid[0, 0], axes_grid[0, 1], axes_grid[1, 0], axes_grid[1, 1]]
fig.suptitle("OPTV / Reward / Wall-glide / Border Length — mean ± std across maps",
             fontsize=12, fontweight="bold")

width = 0.35
x = np.array([0.0])   # single group per panel

for ax, panel in zip(axes, panels):
    for offset, (col, label) in zip([-width/2, width/2], panel["pairs"]):
        m = stats[col]["mean"]
        s = stats[col]["std"]
        ax.bar(x + offset, m, width,
               color=COLORS[col], alpha=0.85, label=label,
               yerr=s, capsize=6, error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="#333333"))

    # annotate ratio
    rm, rs = panel["ratio_mean"], panel["ratio_std"]
    top = max(stats[p[0]]["mean"] for p in panel["pairs"])
    top = max(top, 0)
    ax.text(0, top * 1.06,
            f"{rm:+.1f}% ± {rs:.1f}%",
            ha="center", va="bottom", fontsize=9, color="#333333", fontweight="bold")
    ax.text(0, top * 1.12,
            f"({panel['ratio_label']})",
            ha="center", va="bottom", fontsize=7.5, color="#666666")

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title(panel["title"], pad=6)
    ax.set_ylabel(panel["ylabel"])
    ax.set_xticks([])
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
