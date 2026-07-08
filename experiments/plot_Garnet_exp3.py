import sys
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------
RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "Garnet_exp3"
    / "Garnet_exp3.csv"
)

FIGURE_DIR = PROJECT_ROOT / "figures" / "Garnet_exp3"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


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
        "marker": "s",
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

    if x.shape[axis] <= 1:
        sem = np.zeros_like(mean)
    else:
        sem = np.std(x, axis=axis, ddof=1) / np.sqrt(x.shape[axis])

    return mean, sem


def main():
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Result file not found: {RESULT_PATH}")

    df = pd.read_csv(RESULT_PATH)

    required_cols = {
        "bias_level",
        "method",
        "normalized_performance",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove metadata rows and keep only plotted methods.
    perf_df = df[df["method"].isin(METHOD_ORDER)].copy()

    # Exclude metadata bias_level = -1 if present.
    perf_df = perf_df[perf_df["bias_level"] >= 0].copy()

    bias_levels = np.sort(perf_df["bias_level"].unique().astype(float))

    curve_stats = {}

    for method_name in METHOD_ORDER:
        method_df = perf_df[perf_df["method"] == method_name].copy()

        curves = []

        for delta in bias_levels:
            values = method_df[
                np.isclose(method_df["bias_level"].astype(float), delta)
            ]["normalized_performance"].to_numpy(dtype=float)

            if len(values) == 0:
                raise ValueError(
                    f"No data found for method={method_name}, delta={delta}."
                )

            curves.append(values)

        # Shape: (num_seeds, num_bias_levels)
        curves = np.asarray(curves, dtype=float).T

        mean_curve, sem_curve = mean_and_sem(curves, axis=0)

        curve_stats[method_name] = {
            "mean": mean_curve,
            "sem": sem_curve,
            "num_runs": curves.shape[0],
        }

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method_name in METHOD_ORDER:
        style = PLOT_STYLES[method_name]

        mean_curve = 100.0 * curve_stats[method_name]["mean"]
        sem_curve = 100.0 * curve_stats[method_name]["sem"]

        ax.plot(
            bias_levels,
            mean_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4,
            linewidth=LINE_WIDTH,
            label=method_name,
        )

        ax.fill_between(
            bias_levels,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=style["color"],
            alpha=SHADE_ALPHA,
            linewidth=0.0,
        )

    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\nu(T)$ (%)")

    ax.set_xticks(bias_levels)
    ax.set_xticklabels([f"{d:.1f}" for d in bias_levels])

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    ax.legend(loc="lower left")

    set_integer_yticks_keep_limits(ax)

    fig.tight_layout()

    figure_pdf_path = FIGURE_DIR / "Garnet_exp3.pdf"
    figure_png_path = FIGURE_DIR / "Garnet_exp3.png"

    fig.savefig(figure_pdf_path)
    fig.savefig(figure_png_path, dpi=300)

    print("Garnet Exp3 plot regenerated from saved results.")
    print(f"Loaded results from: {RESULT_PATH}")
    print(f"PDF saved to: {figure_pdf_path}")
    print(f"PNG saved to: {figure_png_path}")

    for method_name in METHOD_ORDER:
        print(
            f"{method_name}: "
            f"{curve_stats[method_name]['num_runs']} runs per delta"
        )


if __name__ == "__main__":
    main()