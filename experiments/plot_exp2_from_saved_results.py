import sys
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import similarity_weights, ensure_dir


# -----------------------------
# Unified plotting style
# -----------------------------
FIGSIZE = (6.5, 4.2)

LINE_WIDTH = 2.0
INSET_LINE_WIDTH = 1.1
BOX_LINE_WIDTH = 1.6
MEDIAN_LINE_WIDTH = 2.0

BOX_ALPHA = 0.35
GRID_ALPHA = 0.25

SIM_COLOR = "C0"
UNI_COLOR = "C1"

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

def style_boxplot(boxplot_dict, color):
    """Apply a unified style to matplotlib boxplot objects."""
    for box in boxplot_dict["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(BOX_ALPHA)
        box.set_edgecolor(color)
        box.set_linewidth(BOX_LINE_WIDTH)

    for whisker in boxplot_dict["whiskers"]:
        whisker.set_color(color)
        whisker.set_linewidth(BOX_LINE_WIDTH)

    for cap in boxplot_dict["caps"]:
        cap.set_color(color)
        cap.set_linewidth(BOX_LINE_WIDTH)

    for median in boxplot_dict["medians"]:
        median.set_color(color)
        median.set_linewidth(MEDIAN_LINE_WIDTH)

    for flier in boxplot_dict["fliers"]:
        flier.set_markeredgecolor(color)
        flier.set_markerfacecolor("none")
        flier.set_markersize(5)


def main():
    result_dir = PROJECT_ROOT / "results" / "exp2"
    figure_dir = PROJECT_ROOT / "figures" / "exp2"
    ensure_dir(figure_dir)

    result_path = result_dir / "exp2_results.csv"
    df = pd.read_csv(result_path)

    # Keep only final performance rows.
    perf = df[df["method"].isin(["Similarity-aware", "Uniform"])].copy()

    if perf.empty:
        raise ValueError(
            "No matching performance rows found. "
            "Please make sure exp2_results.csv contains "
            "Similarity-aware and Uniform."
        )

    x = np.sort(perf["bad_source_gamma"].unique().astype(float))
    num_settings = len(x)
    base_positions = np.arange(num_settings)

    sim_values = []
    uni_values = []

    for bad_gamma in x:
        sim_values.append(
            100.0
            * perf[
                (perf["method"] == "Similarity-aware")
                & (perf["bad_source_gamma"] == bad_gamma)
            ]["normalized_performance"].to_numpy()
        )

        uni_values.append(
            100.0
            * perf[
                (perf["method"] == "Uniform")
                & (perf["bad_source_gamma"] == bad_gamma)
            ]["normalized_performance"].to_numpy()
        )

    # -----------------------------
    # Main plot: boxplot of final target performance
    # -----------------------------
    offset = 0.18
    box_width = 0.28

    sim_positions = base_positions - offset
    uni_positions = base_positions + offset

    fig, ax = plt.subplots(figsize=FIGSIZE)

    box_sim = ax.boxplot(
        sim_values,
        positions=sim_positions,
        widths=box_width,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
    )

    box_uni = ax.boxplot(
        uni_values,
        positions=uni_positions,
        widths=box_width,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
    )

    style_boxplot(box_sim, SIM_COLOR)
    style_boxplot(box_uni, UNI_COLOR)

    ax.set_xticks(base_positions)
    ax.set_xticklabels([f"{g:.1f}" for g in x])
    ax.set_xlabel(r"$\Gamma_b$")
    ax.set_ylabel(r"$\nu(T)$ (%)")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    legend_handles = [
        Patch(
            facecolor=SIM_COLOR,
            edgecolor=SIM_COLOR,
            alpha=BOX_ALPHA,
            label="Similarity-aware",
        ),
        Patch(
            facecolor=UNI_COLOR,
            edgecolor=UNI_COLOR,
            alpha=BOX_ALPHA,
            label="Uniform",
        ),
    ]

    ax.legend(handles=legend_handles, loc="lower left")
    
    set_integer_yticks_keep_limits(ax)

    # -----------------------------
    # Inset: deterministic similarity-aware bad-source weight
    # -----------------------------
    metadata = df[df["method"] == "metadata"].copy()

    good_gammas = (
        metadata[metadata["source_type"] == "good"]
        .drop_duplicates("source_index")
        .sort_values("source_index")["prescribed_source_gamma"]
        .to_numpy()
    )

    if good_gammas.size == 0:
        raise ValueError(
            "Could not recover good-source gammas from metadata rows."
        )

    sim_bad_weight = []

    for bad_gamma in x:
        prescribed_gammas = np.concatenate(
            [good_gammas, np.array([bad_gamma])]
        )

        w_sim = similarity_weights(
            prescribed_gammas,
            eps=1e-6,
            power=1.0,
        )

        sim_bad_weight.append(w_sim[-1])

    sim_bad_weight = np.asarray(sim_bad_weight)

    # Smaller inset placed in the lower-middle/right blank region.
    # Format: [left, bottom, width, height] in axes coordinates.
    axins = ax.inset_axes([0.54, 0.17, 0.22, 0.18])

    axins.plot(
        x,
        sim_bad_weight,
        marker="o",
        markersize=3,
        color=SIM_COLOR,
        linewidth=INSET_LINE_WIDTH,
    )

    axins.set_xlabel(r"$\Gamma_b$", fontsize=8)
    axins.set_ylabel(r"$w_b$", fontsize=8)

    axins.set_xticks(x)
    axins.set_xticklabels([f"{g:.1f}" for g in x], fontsize=7)

    axins.tick_params(axis="both", labelsize=7)
    axins.set_axisbelow(True)
    axins.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    fig.tight_layout()

    fig.savefig(figure_dir / "exp2_bad_source_stress_test.pdf")
    fig.savefig(figure_dir / "exp2_bad_source_stress_test.png", dpi=300)

    print("Experiment 2 figure regenerated from saved results.")
    print(f"Read results from: {result_path}")
    print(f"Saved figure to: {figure_dir / 'exp2_bad_source_stress_test.pdf'}")


if __name__ == "__main__":
    main()