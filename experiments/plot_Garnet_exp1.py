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

METHOD_ORDER = [
    "Maximum-based",
    "Similarity-aware",
    "Uniform",
    # "Non-robust DR",  # Temporarily disabled.
]

PLOT_STYLES = {
    "Maximum-based": {
        "color": "C2",
        "linestyle": "-",
    },
    "Similarity-aware": {
        "color": "C0",
        "linestyle": "-",
    },
    "Uniform": {
        "color": "C1",
        "linestyle": "-",
    },
    "Non-robust DR": {
        "color": "C4",
        "linestyle": "-",
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
    result_dir = PROJECT_ROOT / "results" / "Garnet_exp1"
    figure_dir = PROJECT_ROOT / "figures" / "Garnet_exp1"
    ensure_dir(figure_dir)

    result_path = result_dir / "Garnet_exp1.csv"
    df = pd.read_csv(result_path)

    # Keep only the plotted methods.
    perf = df[df["method"].isin(METHOD_ORDER)].copy()

    if perf.empty:
        raise ValueError(
            "No matching performance rows found. "
            "Please make sure Garnet_exp1.csv contains "
            "Maximum-based, Similarity-aware, and Uniform."
        )

    method_tables = {}

    for method_name in METHOD_ORDER:
        method_df = perf[perf["method"] == method_name]

        if method_df.empty:
            raise ValueError(f"Missing method in results: {method_name}")

        table = (
            method_df
            .pivot(index="seed", columns="iteration", values="normalized_performance")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )

        method_tables[method_name] = table

    # Use only iterations shared by all methods.
    common_iterations = None

    for method_name in METHOD_ORDER:
        iterations = method_tables[method_name].columns.to_numpy()

        if common_iterations is None:
            common_iterations = iterations
        else:
            common_iterations = np.intersect1d(common_iterations, iterations)

    curve_stats = {}

    for method_name in METHOD_ORDER:
        curves = method_tables[method_name][common_iterations].to_numpy()
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
            common_iterations,
            mean_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=LINE_WIDTH,
            label=method_name,
        )

        ax.fill_between(
            common_iterations,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=style["color"],
            alpha=SHADE_ALPHA,
            linewidth=0.0,
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\nu(t)$ (%)")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    ax.legend(loc="lower right")

    # Match Garnet_exp1.py: show 100 as the highest integer tick and leave
    # roughly 5% visual headroom above it.
    ymin, _ = ax.get_ylim()
    headroom = min(0.05 * max(100.0 - ymin, 1.0), 0.95)
    ax.set_ylim(ymin, 100.0 + headroom)

    set_integer_yticks_keep_limits(ax)

    fig.tight_layout()

    figure_pdf_path = figure_dir / "Garnet_exp1.pdf"
    figure_png_path = figure_dir / "Garnet_exp1.png"

    fig.savefig(figure_pdf_path)
    fig.savefig(figure_png_path, dpi=300)
    plt.close(fig)

    print("Garnet Experiment 1 figure regenerated from saved results.")
    print(f"Read results from: {result_path}")
    print(f"Saved figure to: {figure_pdf_path}")


if __name__ == "__main__":
    main()