"""Interactive comparison of benchmark results using Plotly.

This script pulls data from two CSV files (our results and the
state-of-the-art reference results) and produces interactive scatter plots
with 90% coverage indicators to help compare performance across different maps.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRICS: Dict[str, str] = {
    "path_length_90_pct": "Path Length (90% Coverage)",
    "path_length_end": "Path Length (End)",
    "coverage_end": "Coverage (End)",
}

MODEL_COLORS = {
    "State Of The Art": "#d62728",  # red
    "Ours": "#2ca02c",  # green
}


def convert_map_name(map_name: str) -> str:
    """Convert eval_mowing_X format to Map Y format."""
    if map_name.startswith("eval_mowing_"):
        try:
            map_num = int(map_name.replace("eval_mowing_", ""))
            # Convert eval_mowing_9 -> Map 1, eval_mowing_10 -> Map 2, etc.
            display_num = map_num - 8
            return f"Map {display_num}"
        except ValueError:
            pass
    return map_name


def read_results(path: Path, *, sep: str = ",", decimal: str = ".") -> pd.DataFrame:
    """Load a benchmark CSV into a DataFrame and normalise types."""
    df = pd.read_csv(path, sep=sep, decimal=decimal)
    if "model" not in df.columns and any(col.lower().startswith("column") for col in df.columns):
        df = pd.read_csv(path, sep=sep, decimal=decimal, header=1)
    expected_columns = {"model", "map", "map_run_id"} | set(METRICS)
    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing_columns)}")

    df["map"] = df["map"].astype(str)
    df["model"] = df["model"].astype(str)
    return df





def calculate_mean_values(data: pd.DataFrame) -> Tuple[float, float]:
    """Calculate mean path lengths for final coverage and 90% coverage."""
    mean_final = data["path_length_end"].mean()
    data_90 = data.dropna(subset=["path_length_90_pct"])
    mean_90 = data_90["path_length_90_pct"].mean() if not data_90.empty else None
    return mean_final, mean_90


def calculate_performance_indicator(ours_data: pd.DataFrame, sota_data: pd.DataFrame) -> str:
    """Calculate performance improvement/worsening and return formatted indicator."""
    if ours_data.empty or sota_data.empty:
        return ""
    
    # Use final path length as the main metric (lower is better)
    ours_mean = ours_data["path_length_end"].mean()
    sota_mean = sota_data["path_length_end"].mean()
    
    if pd.isna(ours_mean) or pd.isna(sota_mean) or sota_mean == 0:
        return ""
    
    # Calculate percentage improvement (negative means we're worse, positive means we're better)
    improvement_pct = ((sota_mean - ours_mean) / sota_mean) * 100
    
    if improvement_pct > 0:
        return f" - 🟢 avg improved by {improvement_pct:.2f}%"
    else:
        return f" - 🔴 avg worsened by {abs(improvement_pct):.2f}%"


def add_scatter_traces(
    fig: go.Figure,
    *,
    row: int,
    data: pd.DataFrame,
    label: str,
    color: str,
    showlegend: bool,
    y_range: list = None,
) -> None:
    # Calculate mean values
    mean_final, mean_90 = calculate_mean_values(data)
    
    # Create converted map names for hover data
    converted_map_names = [convert_map_name(map_name) for map_name in data["map"]]
    
    # Add main scatter points (final coverage)
    fig.add_trace(
        go.Scatter(
            x=data["path_length_end"],
            y=data["coverage_end"],
            mode="markers",
            name=f"{label} (Final)",
            legendgroup=label,
            marker=dict(color=color, size=10, line=dict(width=1, color="#000"), opacity=0.85),
            hovertemplate=(
                f"{label} (Final)" +
                "<br>map: %{customdata[0]}" +
                "<br>run: %{customdata[1]}" +
                "<br>path_end: %{x:.3f}" +
                "<br>coverage: %{y:.3f}" +
                "<extra></extra>"
            ),
            customdata=np.column_stack((converted_map_names, data["map_run_id"])),
            showlegend=showlegend,
        ),
        row=row,
        col=1,
    )
    
    # Add mean line for final coverage (solid horizontal line)
    if y_range and not np.isnan(mean_final):
        fig.add_trace(
            go.Scatter(
                x=[mean_final, mean_final],
                y=y_range,
                mode="lines",
                name=f"{label} Mean (Final)",
                legendgroup=f"{label}_mean",
                line=dict(color=color, width=2, dash="solid"),
                hovertemplate=f"{label} Mean Final Path Length: {mean_final:.3f}<extra></extra>",
                showlegend=showlegend,
            ),
            row=row,
            col=1,
        )
    
    # Add 90% coverage points
    data_90 = data.dropna(subset=["path_length_90_pct"])
    if not data_90.empty:
        # Create converted map names for 90% data hover
        converted_map_names_90 = [convert_map_name(map_name) for map_name in data_90["map"]]
        
        fig.add_trace(
            go.Scatter(
                x=data_90["path_length_90_pct"],
                y=[0.90] * len(data_90),  # All points at 90% coverage line
                mode="markers",
                name=f"{label} (90%)",
                legendgroup=f"{label}_90",
                marker=dict(
                    color=color, 
                    size=8, 
                    symbol="diamond",
                    line=dict(width=2, color="#fff"), 
                    opacity=0.9
                ),
                hovertemplate=(
                    f"{label} (90% Coverage)" +
                    "<br>map: %{customdata[0]}" +
                    "<br>run: %{customdata[1]}" +
                    "<br>path_90pct: %{x:.3f}" +
                    "<br>coverage: 0.900" +
                    "<extra></extra>"
                ),
                customdata=np.column_stack((converted_map_names_90, data_90["map_run_id"])),
                showlegend=showlegend,
            ),
            row=row,
            col=1,
        )
        
        # Add mean line for 90% coverage (dotted vertical line at 90% coverage level)
        if mean_90 is not None and not np.isnan(mean_90):
            fig.add_trace(
                go.Scatter(
                    x=[mean_90, mean_90],
                    y=[0.0, 1.0],  # Small range around 90% line
                    mode="lines",
                    name=f"{label} Mean (90%)",
                    legendgroup=f"{label}_mean_90",
                    line=dict(color=color, width=2, dash="dot"),
                    hovertemplate=f"{label} Mean 90% Path Length: {mean_90:.3f}<extra></extra>",
                    showlegend=showlegend,
                ),
                row=row,
                col=1,
            )


def build_scatter_figure(ours: pd.DataFrame, sota: pd.DataFrame, *, maps: Iterable[str]) -> go.Figure:
    maps = list(maps)
    
    # Create subplot titles for scatter plots with performance indicators
    subplot_titles = []
    for map_name in ["Overall"] + maps:
        display_name = convert_map_name(map_name) if map_name != "Overall" else map_name
        
        # Get data for this map (or all data for "Overall")
        if map_name == "Overall":
            ours_subset = ours
            sota_subset = sota
        else:
            ours_subset = ours[ours["map"] == map_name]
            sota_subset = sota[sota["map"] == map_name]
        
        # Calculate performance indicator
        performance_indicator = calculate_performance_indicator(ours_subset, sota_subset)
        
        # Create title with performance indicator
        title = f"{display_name}{performance_indicator}"
        subplot_titles.append(title)
    
    # Create subplots with 1 column: just scatter plots
    fig = make_subplots(
        rows=len(maps) + 1,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,  # Normal spacing without metrics
        specs=[[{"secondary_y": False}] for _ in range(len(maps) + 1)]
    )

    # Calculate axis ranges
    y_values = pd.concat([ours["coverage_end"], sota["coverage_end"]], ignore_index=True).astype(float)
    y_min = float(y_values.min()) if not y_values.empty else 0.0
    y_max = float(y_values.max()) if not y_values.empty else 1.0
    padding = max((y_max - y_min) * 0.05, 0.05)
    y_range = [y_min - padding, min(1.05, y_max + padding)]  # Cap at 105% for better visualization

    # Add traces for overall comparison (row 1)
    add_scatter_traces(
        fig,
        row=1,
        data=sota,
        label="State Of The Art",
        color=MODEL_COLORS["State Of The Art"],
        showlegend=True,
        y_range=y_range,
    )
    add_scatter_traces(
        fig,
        row=1,
        data=ours,
        label="Ours",
        color=MODEL_COLORS["Ours"],
        showlegend=True,
        y_range=y_range,
    )

    # Add traces for individual maps
    for idx, map_name in enumerate(maps, start=2):
        sota_map = sota[sota["map"] == map_name]
        ours_map = ours[ours["map"] == map_name]
        
        add_scatter_traces(
            fig,
            row=idx,
            data=sota_map,
            label="State Of The Art",
            color=MODEL_COLORS["State Of The Art"],
            showlegend=False,
            y_range=y_range,
        )
        add_scatter_traces(
            fig,
            row=idx,
            data=ours_map,
            label="Ours",
            color=MODEL_COLORS["Ours"],
            showlegend=False,
            y_range=y_range,
        )

    # Update axes for scatter plots
    for row_index in range(1, len(maps) + 2):
        fig.update_xaxes(title_text="Path Length", row=row_index, col=1)
        fig.update_yaxes(title_text="Coverage", row=row_index, col=1, range=y_range)

    fig.update_layout(
        title="Path Length vs. Coverage with 90% Coverage Indicators",
        height=350 * (len(maps) + 1),  # Reduced height since no metrics
        hovermode="closest",
        showlegend=True,
    )
    return fig





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Plotly comparison of benchmark results.")
    parser.add_argument(
        "--ours",
        type=Path,
        required=True,
        help="CSV file containing our benchmark results.",
    )
    parser.add_argument(
        "--sota",
        type=Path,
        default=Path("misc/logs/benchmark_results/benchmark_results_SOTA.csv"),
        help="CSV file containing the state-of-the-art benchmark results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_comparison.html"),
        help="Output HTML file for the interactive figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ours = read_results(args.ours)
    sota = read_results(args.sota, sep=";")

    common_maps = set(ours["map"]) & set(sota["map"])
    if not common_maps:
        raise RuntimeError("No overlapping maps between the provided result files.")
    
    # Sort maps numerically by extracting the number from eval_mowing_X format
    def sort_key(map_name: str) -> int:
        if map_name.startswith("eval_mowing_"):
            try:
                return int(map_name.replace("eval_mowing_", ""))
            except ValueError:
                pass
        return 999  # Put non-standard names at the end
    
    common_maps = sorted(common_maps, key=sort_key)

    ours = ours[ours["map"].isin(common_maps)].reset_index(drop=True)
    sota = sota[sota["map"].isin(common_maps)].reset_index(drop=True)

    scatter_fig = build_scatter_figure(ours, sota, maps=common_maps)

    import plotly.io as pio

    scatter_html = pio.to_html(scatter_fig, include_plotlyjs="cdn", full_html=False)

    html_document = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>Benchmark Comparison</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 24px; background: #0f172a; color: #e2e8f0; }}
        h1 {{ margin-top: 0; font-size: 32px; color: #38bdf8; text-align: center; }}
        h2 {{ color: #f8fafc; margin-top: 48px; font-size: 24px; }}
        section {{ margin-bottom: 56px; background: rgba(15, 23, 42, 0.85); border-radius: 12px; padding: 24px; box-shadow: 0 12px 24px rgba(15, 23, 42, 0.45); }}
        .description {{ font-size: 16px; line-height: 1.6; margin-bottom: 20px; }}
        .legend-info {{ background: rgba(56, 189, 248, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; }}
        a {{ color: #38bdf8; }}
    </style>
</head>
<body>
    <h1>Benchmark Comparison Dashboard</h1>
    <section>
        <div class=\"legend-info\">
            <strong>Legend:</strong> 
            <span style=\"color: {MODEL_COLORS['Ours']}\">● Our Method (Final Coverage)</span> |
            <span style=\"color: {MODEL_COLORS['State Of The Art']}\">● State of the Art (Final Coverage)</span> |
            ♦ 90% Coverage Achievement Points |
            — Mean Lines (Solid: Final, Dotted: 90%)
        </div>
        {scatter_html}
    </section>
</body>
</html>"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_document, encoding="utf-8")
    print(f"Saved interactive comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
