import sys
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import ensure_dir


# -----------------------------
# Unified plotting style
# -----------------------------
FIGSIZE = (6.5, 4.2)

LINE_WIDTH = 2.0
SHADE_ALPHA = 0.15
GRID_ALPHA = 0.25

MAX_COLOR = "C2"
SIM_COLOR = "C0"

METHOD_ORDER = [
    "Maximum-based",
    "Similarity-aware",
]

PLOT_STYLES = {
    "Maximum-based": {
        "color": MAX_COLOR,
        "linestyle": "-",
        "marker": "o",
    },
    "Similarity-aware": {
        "color": SIM_COLOR,
        "linestyle": "-",
        "marker": "o",
    },
}

def set_integer_yticks_keep_limits(ax):
    """
    Use integer y-axis ticks without changing the current y-axis limits.
    """
    ymin, ymax = ax.get_ylim()

    ticks = np.arange(
        np.ceil(ymin),
        np.floor(ymax) + 1,
        1.0,
    )

    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    # Restore the original limits so that the plot range is unchanged.
    ax.set_ylim(ymin, ymax)

def mean_and_sem(x, axis=0):
    x = np.asarray(x, dtype=float)
    mean = np.mean(x, axis=axis)
    sem = np.std(x, axis=axis, ddof=1) / np.sqrt(x.shape[axis])
    return mean, sem


def main():
    result_dir = PROJECT_ROOT / "results" / "exp3"
    figure_dir = PROJECT_ROOT / "figures" / "exp3"
    ensure_dir(figure_dir)

    result_path = result_dir / "exp3_results.csv"
    df = pd.read_csv(result_path)

    # Keep only final performance rows.
    perf = df[df["method"].isin(METHOD_ORDER)].copy()

    if perf.empty:
        raise ValueError(
            "No matching performance rows found. "
            "Please make sure exp3_results.csv contains "
            "Maximum-based and Similarity-aware."
        )

    bias_levels = np.sort(perf["bias_level"].unique().astype(float))

    method_tables = {}

    for method_name in METHOD_ORDER:
        method_df = perf[perf["method"] == method_name]

        if method_df.empty:
            raise ValueError(f"Missing method in results: {method_name}")

        table = (
            method_df
            .pivot(index="seed", columns="bias_level", values="normalized_performance")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )

        method_tables[method_name] = table

    # Use only bias levels shared by all methods.
    common_bias_levels = None

    for method_name in METHOD_ORDER:
        levels = method_tables[method_name].columns.to_numpy(dtype=float)

        if common_bias_levels is None:
            common_bias_levels = levels
        else:
            common_bias_levels = np.intersect1d(common_bias_levels, levels)

    curve_stats = {}

    for method_name in METHOD_ORDER:
        curves = method_tables[method_name][common_bias_levels].to_numpy()
        mean_curve, sem_curve = mean_and_sem(curves, axis=0)

        curve_stats[method_name] = {
            "mean": mean_curve,
            "sem": sem_curve,
        }

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method_name in METHOD_ORDER:
        style = PLOT_STYLES[method_name]

        # Convert normalized fraction to percentage values.
        mean_curve = 100.0 * curve_stats[method_name]["mean"]
        sem_curve = 100.0 * curve_stats[method_name]["sem"]

        ax.plot(
            common_bias_levels,
            mean_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4,
            linewidth=LINE_WIDTH,
            label=method_name,
        )

        ax.fill_between(
            common_bias_levels,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=style["color"],
            alpha=SHADE_ALPHA,
            linewidth=0.0,
        )

    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\nu(T)$ (%)")

    ax.set_xticks(common_bias_levels)
    ax.set_xticklabels([f"{d:.1f}" for d in common_bias_levels])

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    ax.legend(loc="lower left")

    set_integer_yticks_keep_limits(ax)

    fig.tight_layout()

    fig.savefig(figure_dir / "exp3_bias_sensitivity.pdf")
    fig.savefig(figure_dir / "exp3_bias_sensitivity.png", dpi=300)

    print("Experiment 3 figure regenerated from saved results.")
    print(f"Read results from: {result_path}")
    print(f"Saved figure to: {figure_dir / 'exp3_bias_sensitivity.pdf'}")


if __name__ == "__main__":
    main()