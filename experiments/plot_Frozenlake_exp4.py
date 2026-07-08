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
    / "Frozenlake_exp4"
    / "Frozenlake_exp4.csv"
)

FIGURE_DIR = PROJECT_ROOT / "figures" / "Frozenlake_exp4"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Plot style
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
    },
    "Similarity-aware": {
        "color": SIM_COLOR,
        "linestyle": "-",
    },
}

# Main plotted metric.
# Options:
#   "signed_selection_bias"
#   "selection_bias_inf_norm"
#   "signed_selection_bias_percent"
#   "selection_bias_inf_norm_percent"
PLOT_METRIC = "signed_selection_bias"


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


def get_ylabel(metric_name):
    if metric_name == "signed_selection_bias":
        return r"$\mu(t)$"

    if metric_name == "selection_bias_inf_norm":
        return r"Selection bias magnitude"

    if metric_name == "signed_selection_bias_percent":
        return r"$\mu(t)$ (%)"

    if metric_name == "selection_bias_inf_norm_percent":
        return r"Selection bias magnitude (%)"

    return metric_name


def main():
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Result file not found: {RESULT_PATH}")

    df = pd.read_csv(RESULT_PATH)

    required_cols = {
        "iteration",
        "method",
        PLOT_METRIC,
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Keep only performance / diagnostic rows.
    perf = df[df["method"].isin(METHOD_ORDER)].copy()
    perf = perf.dropna(subset=["iteration", PLOT_METRIC])
    perf = perf[perf["iteration"].astype(int) >= 0].copy()

    if perf.empty:
        raise ValueError("No valid diagnostic rows found.")

    iterations = np.sort(perf["iteration"].unique().astype(int))

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method_name in METHOD_ORDER:
        method_df = perf[perf["method"] == method_name].copy()

        mean_curve = []
        sem_curve = []

        for t in iterations:
            values = method_df[
                method_df["iteration"].astype(int) == int(t)
            ][PLOT_METRIC].to_numpy(dtype=float)

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
    ax.set_ylabel(get_ylabel(PLOT_METRIC))

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    ax.legend(loc="upper right")

    # -----------------------------
    # Make y-axis range slightly taller
    # -----------------------------
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    if yrange <= 1e-12:
        yrange = max(abs(ymax), 1.0)

    ax.set_ylim(
        ymin - 0.05 * yrange,
        ymax + 0.10 * yrange,
    )

    set_clean_yticks_keep_limits(ax, nbins=6)

    fig.tight_layout()

    pdf_path = FIGURE_DIR / "Frozenlake_exp4.pdf"
    png_path = FIGURE_DIR / "Frozenlake_exp4.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)

    print("FrozenLake Exp4 plot regenerated from saved results.")
    print(f"Loaded results from: {RESULT_PATH}")
    print(f"PDF saved to: {pdf_path}")
    print(f"PNG saved to: {png_path}")

    print("\nFinal-time summary:")
    final_t = int(iterations[-1])

    for method_name in METHOD_ORDER:
        values = perf[
            (perf["method"] == method_name)
            & (perf["iteration"].astype(int) == final_t)
        ][PLOT_METRIC].to_numpy(dtype=float)

        mean_y, sem_y = mean_and_sem(values, axis=0)

        print(
            f"{method_name}: "
            f"t={final_t}, "
            f"{PLOT_METRIC} mean={mean_y:.8f}, "
            f"sem={sem_y:.8f}, "
            f"n={len(values)}"
        )


if __name__ == "__main__":
    main()