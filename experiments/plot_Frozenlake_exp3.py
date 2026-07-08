import sys
from pathlib import Path
from matplotlib.ticker import MaxNLocator

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
    / "Frozenlake_exp3"
    / "Frozenlake_exp3.csv"
)

FIGURE_DIR = PROJECT_ROOT / "figures" / "Frozenlake_exp3"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Plot style
# -----------------------------
FIGSIZE = (6.5, 4.2)

LINE_WIDTH = 2.0
MARKER_SIZE = 4.5
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


def set_clean_yticks_keep_limits(ax, nbins=6):
    """
    Use clean y-axis ticks without changing the current y-axis limits.
    """
    ymin, ymax = ax.get_ylim()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
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
        "method",
        "normalized_performance",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Prefer noise_level. Fall back to bias_level for compatibility.
    if "noise_level" in df.columns:
        x_col = "noise_level"
    elif "bias_level" in df.columns:
        x_col = "bias_level"
    else:
        raise ValueError("Missing noise_level or bias_level column.")

    # Keep only performance rows.
    perf = df[df["method"].isin(METHOD_ORDER)].copy()
    perf = perf.dropna(subset=[x_col, "normalized_performance"])

    # Remove metadata rows if they use negative placeholders.
    perf = perf[perf[x_col].astype(float) >= 0.0].copy()

    noise_levels = np.sort(perf[x_col].unique().astype(float))

    # -----------------------------
    # Aggregate mean ± SEM over runs
    # -----------------------------
    curve_stats = {}

    for method_name in METHOD_ORDER:
        method_df = perf[perf["method"] == method_name].copy()

        mean_curve = []
        sem_curve = []
        n_runs_curve = []

        for delta in noise_levels:
            values = method_df[
                np.isclose(
                    method_df[x_col].astype(float),
                    float(delta),
                )
            ]["normalized_performance"].to_numpy(dtype=float)

            values = 100.0 * values

            mean_y, sem_y = mean_and_sem(values, axis=0)

            mean_curve.append(float(mean_y))
            sem_curve.append(float(sem_y))
            n_runs_curve.append(len(values))

        curve_stats[method_name] = {
            "mean": np.asarray(mean_curve, dtype=float),
            "sem": np.asarray(sem_curve, dtype=float),
            "n_runs": np.asarray(n_runs_curve, dtype=int),
        }

    # -----------------------------
    # Plot normalized target performance
    # -----------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method_name in METHOD_ORDER:
        style = PLOT_STYLES[method_name]

        mean_curve = curve_stats[method_name]["mean"]
        sem_curve = curve_stats[method_name]["sem"]

        ax.plot(
            noise_levels,
            mean_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=method_name,
        )

        ax.fill_between(
            noise_levels,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=style["color"],
            alpha=SHADE_ALPHA,
            linewidth=0.0,
        )

    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\nu(T)$ (%)")

    ax.set_xticks(noise_levels)
    ax.set_xticklabels([f"{d:.3f}" for d in noise_levels])

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    ax.legend(loc="lower left")

    # -----------------------------
    # Make y-axis range taller
    # -----------------------------
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    ax.set_ylim(
        max(0.0, ymin),
        max(ymax + 0.05 * yrange, 100.0 + 0.02 * yrange),
    )

    set_clean_yticks_keep_limits(ax, nbins=6)

    fig.tight_layout()

    figure_pdf_path = FIGURE_DIR / "Frozenlake_exp3.pdf"
    figure_png_path = FIGURE_DIR / "Frozenlake_exp3.png"

    fig.savefig(figure_pdf_path)
    fig.savefig(figure_png_path, dpi=300)

    print("FrozenLake Exp3 plot regenerated from saved results.")
    print(f"Loaded results from: {RESULT_PATH}")
    print(f"PDF saved to: {figure_pdf_path}")
    print(f"PNG saved to: {figure_png_path}")

    for delta in noise_levels:
        print(f"delta={delta:.3f}")

        for method_name in METHOD_ORDER:
            method_df = perf[
                (perf["method"] == method_name)
                & np.isclose(
                    perf[x_col].astype(float),
                    float(delta),
                )
            ]

            values = 100.0 * method_df["normalized_performance"].to_numpy(
                dtype=float
            )

            if len(values) > 0:
                print(
                    f"  {method_name}: "
                    f"n={len(values)}, "
                    f"mean={np.mean(values):.2f}, "
                    f"sem={np.std(values, ddof=1) / np.sqrt(len(values)):.2f}"
                    if len(values) > 1
                    else f"  {method_name}: n={len(values)}, mean={np.mean(values):.2f}, sem=0.00"
                )


if __name__ == "__main__":
    main()