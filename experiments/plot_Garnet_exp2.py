import sys
from pathlib import Path
from matplotlib.patches import Patch
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
    / "Garnet_exp2"
    / "Garnet_exp2.csv"
)

FIGURE_DIR = PROJECT_ROOT / "figures" / "Garnet_exp2"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Unified plotting style
# -----------------------------
FIGSIZE = (6.5, 4.2)

INSET_LINE_WIDTH = 1.1
BOX_LINE_WIDTH = 1.6
MEDIAN_LINE_WIDTH = 2.0

BOX_ALPHA = 0.35
GRID_ALPHA = 0.25

SIM_COLOR = "C0"
UNI_COLOR = "C1"

METHOD_ORDER = [
    "Similarity-aware",
    "Uniform",
]


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


def similarity_weights(discrepancies, eps=1e-6, power=1.0):
    discrepancies = np.asarray(discrepancies, dtype=float)
    scores = 1.0 / np.power(discrepancies + eps, power)
    return scores / np.sum(scores)


def get_similarity_bad_source_weights(df, x):
    """
    Return similarity-aware bad-source weights for each bad-source gamma.

    Preferred source:
        performance rows with method == "Similarity-aware" and column
        bad_source_weight.

    Fallback:
        metadata rows with prescribed_source_gamma / similarity_weight.

    Final fallback:
        recompute from prescribed good/bad gammas if metadata is available.
    """
    sim_bad_weight = []

    perf_df = df[df["method"] == "Similarity-aware"].copy()

    if "bad_source_weight" in perf_df.columns:
        for bad_gamma in x:
            values = perf_df[
                np.isclose(
                    perf_df["bad_source_gamma"].astype(float),
                    float(bad_gamma),
                )
            ]["bad_source_weight"].dropna().to_numpy(dtype=float)

            if len(values) > 0:
                sim_bad_weight.append(float(np.mean(values)))
            else:
                sim_bad_weight.append(np.nan)

        sim_bad_weight = np.asarray(sim_bad_weight, dtype=float)

        if not np.any(np.isnan(sim_bad_weight)):
            return sim_bad_weight

    metadata_df = df[df["method"] == "metadata"].copy()

    if {
        "bad_source_gamma",
        "source_type",
        "similarity_weight",
    }.issubset(metadata_df.columns):
        for bad_gamma in x:
            bad_meta = metadata_df[
                np.isclose(
                    metadata_df["bad_source_gamma"].astype(float),
                    float(bad_gamma),
                )
                & (metadata_df["source_type"] == "bad")
            ]

            values = bad_meta["similarity_weight"].dropna().to_numpy(dtype=float)

            if len(values) > 0:
                sim_bad_weight.append(float(np.mean(values)))
            else:
                sim_bad_weight.append(np.nan)

        sim_bad_weight = np.asarray(sim_bad_weight, dtype=float)

        if not np.any(np.isnan(sim_bad_weight)):
            return sim_bad_weight

    if {
        "bad_source_gamma",
        "source_type",
        "prescribed_source_gamma",
    }.issubset(metadata_df.columns):
        first_bad_gamma = float(x[0])

        good_meta = metadata_df[
            np.isclose(
                metadata_df["bad_source_gamma"].astype(float),
                first_bad_gamma,
            )
            & (metadata_df["source_type"] == "good")
        ].copy()

        good_meta = good_meta.sort_values("source_index")
        good_gammas = good_meta["prescribed_source_gamma"].to_numpy(dtype=float)

        if len(good_gammas) == 0:
            raise ValueError(
                "Cannot infer good-source gammas from metadata rows."
            )

        sim_bad_weight = []

        for bad_gamma in x:
            prescribed_gammas = np.concatenate(
                [good_gammas, np.array([float(bad_gamma)])]
            )

            w_sim = similarity_weights(
                prescribed_gammas,
                eps=1e-6,
                power=1.0,
            )

            sim_bad_weight.append(float(w_sim[-1]))

        return np.asarray(sim_bad_weight, dtype=float)

    raise ValueError(
        "Cannot recover similarity-aware bad-source weights from the CSV."
    )


def main():
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Result file not found: {RESULT_PATH}")

    df = pd.read_csv(RESULT_PATH)

    required_cols = {
        "bad_source_gamma",
        "method",
        "normalized_performance",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Keep only performance rows.
    perf = df[df["method"].isin(METHOD_ORDER)].copy()
    perf = perf.dropna(subset=["bad_source_gamma", "normalized_performance"])

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
                & np.isclose(
                    perf["bad_source_gamma"].astype(float),
                    float(bad_gamma),
                )
            ]["normalized_performance"].to_numpy(dtype=float)
        )

        uni_values.append(
            100.0
            * perf[
                (perf["method"] == "Uniform")
                & np.isclose(
                    perf["bad_source_gamma"].astype(float),
                    float(bad_gamma),
                )
            ]["normalized_performance"].to_numpy(dtype=float)
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

    # -----------------------------
    # Make y-axis range taller
    # -----------------------------
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    ax.set_ylim(
        max(0.0, ymin - 0.50 * yrange),
        ymax,
    )

    set_integer_yticks_keep_limits(ax)

    # -----------------------------
    # Inset: similarity-aware bad-source weight
    # -----------------------------
    sim_bad_weight = get_similarity_bad_source_weights(df, x)

    # Format: [left, bottom, width, height] in axes coordinates.
    # Change this line to move or resize the weight inset.
    axins = ax.inset_axes([0.60, 0.17, 0.22, 0.18])

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

    figure_pdf_path = FIGURE_DIR / "Garnet_exp2.pdf"
    figure_png_path = FIGURE_DIR / "Garnet_exp2.png"

    fig.savefig(figure_pdf_path)
    fig.savefig(figure_png_path, dpi=300)

    print("Garnet Exp2 plot regenerated from saved results.")
    print(f"Loaded results from: {RESULT_PATH}")
    print(f"PDF saved to: {figure_pdf_path}")
    print(f"PNG saved to: {figure_png_path}")

    for bad_gamma, sim_data, uni_data in zip(x, sim_values, uni_values):
        print(
            f"Gamma_b={bad_gamma:.1f}: "
            f"Similarity-aware n={len(sim_data)}, "
            f"Uniform n={len(uni_data)}"
        )


if __name__ == "__main__":
    main()