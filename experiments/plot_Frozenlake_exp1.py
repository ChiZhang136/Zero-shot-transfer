import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# -----------------------------
# Paths
# -----------------------------
RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "Frozenlake_exp1"
    / "Frozenlake_exp1_results.csv"
)

FIGURE_DIR = PROJECT_ROOT / "figures" / "Frozenlake_exp1"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Plot settings
# -----------------------------
FIGSIZE = (6.5, 4.2)

LINE_WIDTH = 2.0
SHADE_ALPHA = 0.15
GRID_ALPHA = 0.25

METHOD_ORDER = [
    "Maximum-based",
    "Similarity-aware",
    "Uniform",
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
}

PLOT_METRIC = "target_performance"
Y_LABEL = r"$V_{P_0}^{\pi_t}(s_0)$"


def set_clean_yticks_keep_limits(ax, nbins=6):
    """
    Use a clean number of y-axis ticks without changing axis limits.
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

    # Keep only actual learning curves.
    curve_df = df[
        (df["iteration"] >= 0)
        & (df["method"].isin(METHOD_ORDER))
    ].copy()

    if PLOT_METRIC not in curve_df.columns:
        raise KeyError(
            f"Column '{PLOT_METRIC}' not found in result file. "
            f"Available columns are: {list(curve_df.columns)}"
        )

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method_name in METHOD_ORDER:
        method_df = curve_df[curve_df["method"] == method_name].copy()

        # Use run_id if available; otherwise fall back to seed.
        if "run_id" in method_df.columns:
            group_col = "run_id"
        elif "seed" in method_df.columns:
            group_col = "seed"
        else:
            method_df["_single_run"] = 0
            group_col = "_single_run"

        pivot = method_df.pivot_table(
            index=group_col,
            columns="iteration",
            values=PLOT_METRIC,
            aggfunc="mean",
        )

        pivot = pivot.sort_index(axis=1)

        iterations = pivot.columns.to_numpy(dtype=int)
        curves = pivot.to_numpy(dtype=float)

        mean_curve, sem_curve = mean_and_sem(curves, axis=0)

        style = PLOT_STYLES[method_name]

        ax.plot(
            iterations,
            mean_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=LINE_WIDTH,
            label=method_name,
        )

        ax.fill_between(
            iterations,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=style["color"],
            alpha=SHADE_ALPHA,
            linewidth=0.0,
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(Y_LABEL)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    ax.legend(loc="lower right")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(0.0, ymin), ymax)
    set_clean_yticks_keep_limits(ax, nbins=6)

    fig.tight_layout()

    pdf_path = FIGURE_DIR / "Frozenlake_exp1.pdf"
    png_path = FIGURE_DIR / "Frozenlake_exp1.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)

    print("Plot finished.")
    print(f"Result file: {RESULT_PATH}")
    print(f"PDF saved to: {pdf_path}")
    print(f"PNG saved to: {png_path}")

    # Optional: print source metadata for checking similarity weights.
    meta_df = df[df["method"] == "source_metadata"].copy()

    if len(meta_df) > 0:
        cols = [
            "source_index",
            "perturb_epsilon",
            "empirical_gamma_l1",
            "mean_local_l1_radius",
            "uniform_weight",
            "similarity_weight",
        ]
        cols = [c for c in cols if c in meta_df.columns]

        print("\nSource metadata:")
        print(meta_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()