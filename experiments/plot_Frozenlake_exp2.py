import sys
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------
RESULT_DIR = PROJECT_ROOT / "results" / "Frozenlake_exp2"

FINAL_RESULT_PATH = RESULT_DIR / "Frozenlake_exp2_final_results.csv"
FULL_RESULT_PATH = RESULT_DIR / "Frozenlake_exp2_results.csv"

FIGURE_DIR = PROJECT_ROOT / "figures" / "Frozenlake_exp2"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Plot style
# -----------------------------
FIGSIZE = (6.5, 4.2)

SHADE_ALPHA = 0.15
GRID_ALPHA = 0.25

INSET_LINE_WIDTH = 1.1
BOX_LINE_WIDTH = 1.6
MEDIAN_LINE_WIDTH = 2.0
BOX_ALPHA = 0.35

SIM_COLOR = "C0"
UNI_COLOR = "C1"

METHOD_ORDER = [
    "Similarity-aware",
    "Uniform",
]


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


def load_final_results():
    """
    Load final results.

    Preferred:
        Frozenlake_exp2_final_results.csv

    Fallback:
        Reconstruct final rows from Frozenlake_exp2_results.csv.
    """
    if FINAL_RESULT_PATH.exists():
        final_df = pd.read_csv(FINAL_RESULT_PATH)
        print(f"Loaded final results from: {FINAL_RESULT_PATH}")
        return final_df

    if not FULL_RESULT_PATH.exists():
        raise FileNotFoundError(
            f"Neither final nor full result file exists:\n"
            f"  {FINAL_RESULT_PATH}\n"
            f"  {FULL_RESULT_PATH}"
        )

    df = pd.read_csv(FULL_RESULT_PATH)
    print(f"Loaded full results from: {FULL_RESULT_PATH}")

    required_cols = {
        "run_id",
        "bad_source_epsilon",
        "iteration",
        "method",
        "target_performance",
        "normalized_performance",
        "policy_error_rate",
        "bad_source_empirical_gamma_l1",
        "bad_source_mean_local_l1_radius",
        "bad_source_similarity_weight",
        "bad_source_uniform_weight",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in full results: {missing_cols}")

    perf = df[df["method"].isin(METHOD_ORDER)].copy()
    perf = perf[perf["iteration"].astype(int) >= 0].copy()

    final_rows = []

    group_cols = [
        "run_id",
        "bad_source_epsilon",
        "method",
    ]

    for _, group in perf.groupby(group_cols):
        group = group.sort_values("iteration")
        row = group.iloc[-1]

        final_rows.append(
            {
                "run_id": int(row["run_id"]),
                "good_source_seed": (
                    int(row["good_source_seed"])
                    if "good_source_seed" in row and not pd.isna(row["good_source_seed"])
                    else np.nan
                ),
                "bad_source_seed": (
                    int(row["bad_source_seed"])
                    if "bad_source_seed" in row and not pd.isna(row["bad_source_seed"])
                    else np.nan
                ),
                "bad_source_epsilon": float(row["bad_source_epsilon"]),
                "bad_source_empirical_gamma_l1": float(
                    row["bad_source_empirical_gamma_l1"]
                ),
                "bad_source_mean_local_l1_radius": float(
                    row["bad_source_mean_local_l1_radius"]
                ),
                "bad_source_similarity_weight": float(
                    row["bad_source_similarity_weight"]
                ),
                "bad_source_uniform_weight": float(
                    row["bad_source_uniform_weight"]
                ),
                "method": row["method"],
                "final_iteration": int(row["iteration"]),
                "final_target_performance": float(row["target_performance"]),
                "final_normalized_performance": float(
                    row["normalized_performance"]
                ),
                "final_policy_error_rate": float(row["policy_error_rate"]),
                "oracle_performance": float(row["oracle_performance"]),
            }
        )

    final_df = pd.DataFrame(final_rows)
    return final_df


def main():
    final_df = load_final_results()

    required_cols = {
        "run_id",
        "bad_source_epsilon",
        "method",
        "final_normalized_performance",
        "bad_source_empirical_gamma_l1",
        "bad_source_similarity_weight",
    }

    missing_cols = required_cols - set(final_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in final results: {missing_cols}")

    final_df = final_df[final_df["method"].isin(METHOD_ORDER)].copy()
    final_df = final_df.dropna(
        subset=[
            "bad_source_epsilon",
            "final_normalized_performance",
            "bad_source_empirical_gamma_l1",
        ]
    )

    # -----------------------------
    # Plot: boxplot of final normalized target performance
    # -----------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    bad_eps_values = np.sort(
        final_df["bad_source_epsilon"].unique().astype(float)
    )

    num_settings = len(bad_eps_values)
    base_positions = np.arange(num_settings)

    gamma_tick_labels = []
    sim_values = []
    uni_values = []

    for bad_source_epsilon in bad_eps_values:
        setting_df = final_df[
            np.isclose(
                final_df["bad_source_epsilon"].astype(float),
                float(bad_source_epsilon),
            )
        ].copy()

        gamma_mean = float(
            setting_df
            .drop_duplicates(subset=["run_id", "bad_source_epsilon"])
            ["bad_source_empirical_gamma_l1"]
            .mean()
        )

        gamma_tick_labels.append(gamma_mean)

        sim_values.append(
            100.0
            * setting_df[
                setting_df["method"] == "Similarity-aware"
            ]["final_normalized_performance"].to_numpy(dtype=float)
        )

        uni_values.append(
            100.0
            * setting_df[
                setting_df["method"] == "Uniform"
            ]["final_normalized_performance"].to_numpy(dtype=float)
        )

    gamma_tick_labels = np.asarray(gamma_tick_labels, dtype=float)

    offset = 0.18
    box_width = 0.28

    sim_positions = base_positions - offset
    uni_positions = base_positions + offset

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
    ax.set_xticklabels([f"{g:.3f}" for g in gamma_tick_labels])

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
    # Make y-axis range slightly taller
    # -----------------------------
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    ax.set_ylim(
        max(0.0, ymin - 0.10 * yrange),
        ymax + 0.05 * yrange,
    )

    set_clean_yticks_keep_limits(ax, nbins=6)

    # -----------------------------
    # Inset: mean ± SEM similarity-aware bad-source weight
    # -----------------------------
    weight_mean = []
    weight_sem = []

    for bad_source_epsilon in bad_eps_values:
        weights = final_df[
            (final_df["method"] == "Similarity-aware")
            & np.isclose(
                final_df["bad_source_epsilon"].astype(float),
                float(bad_source_epsilon),
            )
        ]["bad_source_similarity_weight"].to_numpy(dtype=float)

        m, s = mean_and_sem(weights, axis=0)
        weight_mean.append(float(m))
        weight_sem.append(float(s))

    weight_mean = np.asarray(weight_mean, dtype=float)
    weight_sem = np.asarray(weight_sem, dtype=float)

    axins = ax.inset_axes([0.32, 0.30, 0.22, 0.18])

    axins.plot(
        gamma_tick_labels,
        weight_mean,
        color=SIM_COLOR,
        linestyle="-",
        marker="o",
        markerfacecolor=SIM_COLOR,
        markeredgecolor=SIM_COLOR,
        linewidth=INSET_LINE_WIDTH,
        markersize=3.2,
    )

    axins.fill_between(
        gamma_tick_labels,
        weight_mean - weight_sem,
        weight_mean + weight_sem,
        color=SIM_COLOR,
        alpha=SHADE_ALPHA,
        linewidth=0.0,
    )

    axins.set_xlabel(r"$\Gamma_b$", fontsize=8)
    axins.set_ylabel(r"$w_b$", fontsize=8)

    if len(gamma_tick_labels) > 4:
        axins.set_xticks(gamma_tick_labels[::2])
    else:
        axins.set_xticks(gamma_tick_labels)

    axins.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axins.tick_params(axis="both", labelsize=7)

    axins.set_axisbelow(True)
    axins.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    fig.tight_layout()

    pdf_path = FIGURE_DIR / "Frozenlake_exp2.pdf"
    png_path = FIGURE_DIR / "Frozenlake_exp2.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)

    print("FrozenLake Exp2 plot regenerated from saved results.")
    print(f"PDF saved to: {pdf_path}")
    print(f"PNG saved to: {png_path}")

    print("\nSummary:")
    for bad_eps, gamma_label, sim_data, uni_data in zip(
        bad_eps_values,
        gamma_tick_labels,
        sim_values,
        uni_values,
    ):
        print(
            f"bad_eps={bad_eps:.3f}, "
            f"Gamma_b={gamma_label:.3f}: "
            f"Similarity-aware n={len(sim_data)}, "
            f"mean={np.mean(sim_data):.2f}; "
            f"Uniform n={len(uni_data)}, "
            f"mean={np.mean(uni_data):.2f}"
        )


if __name__ == "__main__":
    main()