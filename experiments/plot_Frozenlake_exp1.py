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
    / "Frozenlake_exp1"
    / "Frozenlake_exp1_results.csv"
)

FIGURE_DIR = PROJECT_ROOT / "figures" / "Frozenlake_exp1"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Plot style
# -----------------------------
FIGSIZE = (6.5, 4.2)
LINE_WIDTH = 2.0
SHADE_ALPHA = 0.15
GRID_ALPHA = 0.25

EXPECTED_EPS = np.array([0.10, 0.20, 0.30, 0.35])
EXPECTED_UNCERTAINTY_DISTANCE = "support_restricted_tv_l1"
EXPECTED_ROBUST_BACKUP_TYPE = "exact_support_restricted_l1"

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
        "run_id",
        "iteration",
        "method",
        "normalized_performance",
        "uncertainty_distance",
        "robust_backup_type",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    uncertainty_types = set(df["uncertainty_distance"].dropna().unique())
    backup_types = set(df["robust_backup_type"].dropna().unique())
    metadata_eps = np.sort(
        df.loc[df["method"] == "source_metadata", "perturb_epsilon"]
        .dropna()
        .unique()
        .astype(float)
    )

    if uncertainty_types != {EXPECTED_UNCERTAINTY_DISTANCE}:
        raise ValueError(f"Unexpected uncertainty distance: {uncertainty_types}")
    if backup_types != {EXPECTED_ROBUST_BACKUP_TYPE}:
        raise ValueError(f"Unexpected robust backup: {backup_types}")
    if not np.allclose(metadata_eps, EXPECTED_EPS):
        raise ValueError(
            f"Unexpected perturb eps: {metadata_eps}; expected {EXPECTED_EPS}."
        )

    # Keep only performance rows.
    perf = df[df["method"].isin(METHOD_ORDER)].copy()
    perf = perf.dropna(subset=["iteration", "normalized_performance"])
    perf = perf[perf["iteration"].astype(int) >= 0].copy()

    iterations = np.sort(perf["iteration"].unique().astype(int))

    # -----------------------------
    # Plot: normalized target performance mean ± SEM
    # -----------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method_name in METHOD_ORDER:
        method_df = perf[perf["method"] == method_name].copy()

        mean_curve = []
        sem_curve = []

        for t in iterations:
            values = method_df[
                method_df["iteration"].astype(int) == int(t)
            ]["normalized_performance"].to_numpy(dtype=float)

            values = 100.0 * values

            mean_y, sem_y = mean_and_sem(values, axis=0)

            mean_curve.append(float(mean_y))
            sem_curve.append(float(sem_y))

        mean_curve = np.asarray(mean_curve, dtype=float)
        sem_curve = np.asarray(sem_curve, dtype=float)

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
    ax.set_ylabel(r"$\nu(t)$ (%)")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    ax.legend(loc="lower right")

    # -----------------------------
    # Keep a clean y-axis range
    # -----------------------------
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    ax.set_ylim(
        max(0.0, ymin - 0.02 * yrange),
        ymax + 0.05 * yrange,
    )

    set_clean_yticks_keep_limits(ax, nbins=6)

    fig.tight_layout()

    pdf_path = FIGURE_DIR / "Frozenlake_exp1.pdf"
    png_path = FIGURE_DIR / "Frozenlake_exp1.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)

    print("FrozenLake Exp1 plot regenerated from saved results.")
    print(f"Loaded results from: {RESULT_PATH}")
    print(f"PDF saved to: {pdf_path}")
    print(f"PNG saved to: {png_path}")

    print("\nSummary:")
    final_t = int(iterations[-1])

    for method_name in METHOD_ORDER:
        values = perf[
            (perf["method"] == method_name)
            & (perf["iteration"].astype(int) == final_t)
        ]["normalized_performance"].to_numpy(dtype=float)

        values = 100.0 * values

        mean_y, sem_y = mean_and_sem(values, axis=0)

        print(
            f"{method_name}: "
            f"t={final_t}, "
            f"mean={mean_y:.2f}, "
            f"sem={sem_y:.2f}, "
            f"n={len(values)}"
        )


if __name__ == "__main__":
    main()