import sys
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.garnet import (
    generate_target_garnet,
    generate_sources_from_target,
    random_garnet_kernel,
    max_lp_distance,
)
from src.algorithms import run_periodic_learning
from src.evaluation import greedy_policy, evaluate_policy, target_value_iteration
from src.utils import similarity_weights, uniform_weights, ensure_dir


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

def construct_source_with_prescribed_gamma(
    P0,
    P_candidate,
    source_gamma,
    p_norm=1,
):
    """
    Given target kernel P0 and a candidate kernel P_candidate, construct

        P_source = (1 - rho) P0 + rho P_candidate

    such that

        max_{s,a} ||P_source(.|s,a) - P0(.|s,a)||_p = source_gamma.

    This requires the candidate distance to be at least source_gamma.
    """
    base_distance = max_lp_distance(P_candidate, P0, p_norm=p_norm)

    if base_distance < source_gamma or base_distance <= 0:
        return None, None, None

    rho = source_gamma / base_distance
    P_source = (1.0 - rho) * P0 + rho * P_candidate
    actual_gamma = max_lp_distance(P_source, P0, p_norm=p_norm)

    return P_source, actual_gamma, rho


def select_bad_source_with_gamma(
    target_mdp,
    source_gamma,
    branching_factor,
    p_norm,
    num_candidates,
    seed,
):
    """
    Generate a bad source domain with a prescribed source-target distance.

    We sample several candidate source kernels at the same prescribed distance
    and select the one whose nominal source-optimal policy has the lowest
    target-domain performance.

    This makes the experiment a stress test: the bad source is not only far
    from the target, but also policy-mismatched with the target.
    """
    rng = np.random.default_rng(seed)

    P0 = target_mdp.transitions
    rewards = target_mdp.rewards
    discount = target_mdp.discount

    S = target_mdp.S
    A = target_mdp.A

    best_bad_source = None
    best_actual_gamma = None
    best_rho = None
    worst_transfer_perf = np.inf

    num_valid_candidates = 0

    while num_valid_candidates < num_candidates:
        P_candidate = random_garnet_kernel(
            num_states=S,
            num_actions=A,
            branching_factor=branching_factor,
            rng=rng,
        )

        P_source, actual_gamma, rho = construct_source_with_prescribed_gamma(
            P0=P0,
            P_candidate=P_candidate,
            source_gamma=source_gamma,
            p_norm=p_norm,
        )

        if P_source is None:
            continue

        num_valid_candidates += 1

        # Compute the nominal source-optimal policy.
        Q_source_star, _, _ = target_value_iteration(
            P=P_source,
            rewards=rewards,
            discount=discount,
        )
        pi_source_star = greedy_policy(Q_source_star)

        # Evaluate this source-optimal policy on the target domain.
        _, transfer_perf = evaluate_policy(
            P=P0,
            rewards=rewards,
            policy=pi_source_star,
            discount=discount,
        )

        # Select the source whose source-optimal policy transfers worst.
        if transfer_perf < worst_transfer_perf:
            worst_transfer_perf = transfer_perf
            best_bad_source = P_source
            best_actual_gamma = actual_gamma
            best_rho = rho

    return best_bad_source, best_actual_gamma, best_rho, worst_transfer_perf


def main():
    # -----------------------------
    # Experiment parameters
    # -----------------------------
    num_states = 30
    num_actions = 4
    branching_factor = 3
    reward_range = (0.0, 1.0)
    discount = 0.95

    # Robust geometry:
    # p_norm = 1 means Gamma_k is computed by L1 transition distance.
    # robust_q = "inf" means kappa_q(V) = (max V - min V) / 2.
    p_norm = 1
    robust_q = "inf"

    # Three reliable / relevant source domains.
    good_source_gammas = np.array([0.10, 0.20, 0.30])

    # One bad source domain whose distance increases.
    bad_source_gammas = np.array([0.90, 1.20, 1.50, 1.80])

    total_iterations = 500
    eval_every = 5
    sync_period = 1
    stepsize = 0.01

    num_seeds = 20

    # Number of random candidates used to select a policy-mismatched bad source.
    num_bad_candidates = 20

    result_dir = PROJECT_ROOT / "results" / "exp2"
    figure_dir = PROJECT_ROOT / "figures" / "exp2"
    ensure_dir(result_dir)
    ensure_dir(figure_dir)

    all_rows = []

    sim_by_bad_gamma = {float(g): [] for g in bad_source_gammas}
    uni_by_bad_gamma = {float(g): [] for g in bad_source_gammas}

    for seed in tqdm(range(num_seeds), desc="Experiment 2 seeds"):
        # -----------------------------
        # Generate target Garnet MDP
        # -----------------------------
        target_mdp = generate_target_garnet(
            num_states=num_states,
            num_actions=num_actions,
            branching_factor=branching_factor,
            reward_range=reward_range,
            discount=discount,
            seed=1000 + seed,
        )

        rewards = target_mdp.rewards
        P0 = target_mdp.transitions

        # -----------------------------
        # Target oracle
        # -----------------------------
        Q_star, _, _ = target_value_iteration(
            P=P0,
            rewards=rewards,
            discount=discount,
        )
        pi_star = greedy_policy(Q_star)
        _, oracle_perf = evaluate_policy(
            P=P0,
            rewards=rewards,
            policy=pi_star,
            discount=discount,
        )

        # -----------------------------
        # Generate fixed good sources
        # -----------------------------
        P_good_sources, actual_good_gammas, good_rhos = generate_sources_from_target(
            target_mdp=target_mdp,
            source_gammas=good_source_gammas,
            branching_factor=branching_factor,
            seed=2000 + seed,
            p_norm=p_norm,
        )

        for bad_idx, bad_gamma in enumerate(bad_source_gammas):
            bad_gamma = float(bad_gamma)

            # -----------------------------
            # Generate selected bad source
            # -----------------------------
            (
                P_bad,
                actual_bad_gamma,
                bad_rho,
                bad_source_nominal_transfer_perf,
            ) = select_bad_source_with_gamma(
                target_mdp=target_mdp,
                source_gamma=bad_gamma,
                branching_factor=branching_factor,
                p_norm=p_norm,
                num_candidates=num_bad_candidates,
                seed=3000 + 100 * seed + bad_idx,
            )

            P_sources = np.concatenate(
                [P_good_sources, P_bad[None, :, :, :]],
                axis=0,
            )

            source_gammas = np.concatenate(
                [good_source_gammas, np.array([bad_gamma])],
                axis=0,
            )

            actual_source_gammas = np.concatenate(
                [actual_good_gammas, np.array([actual_bad_gamma])],
                axis=0,
            )

            K = len(source_gammas)

            # -----------------------------
            # Weights
            # -----------------------------
            w_uni = uniform_weights(K)
            w_sim = similarity_weights(source_gammas, eps=1e-6, power=1.0)

            # -----------------------------
            # Uniform aggregation
            # -----------------------------
            Q_uni, _ = run_periodic_learning(
                P_sources=P_sources,
                rewards=rewards,
                gammas=source_gammas,
                weights=w_uni,
                discount=discount,
                target_P=None,
                total_iterations=total_iterations,
                stepsize=stepsize,
                sync_period=sync_period,
                eval_every=eval_every,
                q=robust_q,
                aggregation_type="weighted",
            )

            pi_uni = greedy_policy(Q_uni)
            _, uni_perf = evaluate_policy(
                P=P0,
                rewards=rewards,
                policy=pi_uni,
                discount=discount,
            )
            uni_norm = uni_perf / oracle_perf

            # -----------------------------
            # Similarity-aware aggregation
            # -----------------------------
            Q_sim, _ = run_periodic_learning(
                P_sources=P_sources,
                rewards=rewards,
                gammas=source_gammas,
                weights=w_sim,
                discount=discount,
                target_P=None,
                total_iterations=total_iterations,
                stepsize=stepsize,
                sync_period=sync_period,
                eval_every=eval_every,
                q=robust_q,
                aggregation_type="weighted",
            )

            pi_sim = greedy_policy(Q_sim)
            _, sim_perf = evaluate_policy(
                P=P0,
                rewards=rewards,
                policy=pi_sim,
                discount=discount,
            )
            sim_norm = sim_perf / oracle_perf

            sim_by_bad_gamma[bad_gamma].append(sim_norm)
            uni_by_bad_gamma[bad_gamma].append(uni_norm)

            # -----------------------------
            # Save rows
            # -----------------------------
            all_rows.append(
                {
                    "seed": seed,
                    "bad_source_gamma": bad_gamma,
                    "method": "Uniform",
                    "normalized_performance": float(uni_norm),
                    "target_performance": float(uni_perf),
                    "oracle_performance": float(oracle_perf),
                    "bad_source_weight": float(w_uni[-1]),
                    "actual_bad_source_gamma": float(actual_bad_gamma),
                    "bad_source_mixing_coefficient": float(bad_rho),
                    "bad_source_nominal_transfer_performance": float(
                        bad_source_nominal_transfer_perf
                    ),
                }
            )

            all_rows.append(
                {
                    "seed": seed,
                    "bad_source_gamma": bad_gamma,
                    "method": "Similarity-aware",
                    "normalized_performance": float(sim_norm),
                    "target_performance": float(sim_perf),
                    "oracle_performance": float(oracle_perf),
                    "bad_source_weight": float(w_sim[-1]),
                    "actual_bad_source_gamma": float(actual_bad_gamma),
                    "bad_source_mixing_coefficient": float(bad_rho),
                    "bad_source_nominal_transfer_performance": float(
                        bad_source_nominal_transfer_perf
                    ),
                }
            )

            for k in range(K):
                source_type = "bad" if k == K - 1 else "good"
                all_rows.append(
                    {
                        "seed": seed,
                        "bad_source_gamma": bad_gamma,
                        "method": "metadata",
                        "source_index": k,
                        "source_type": source_type,
                        "prescribed_source_gamma": float(source_gammas[k]),
                        "actual_source_gamma": float(actual_source_gammas[k]),
                        "uniform_weight": float(w_uni[k]),
                        "similarity_weight": float(w_sim[k]),
                    }
                )

    # -----------------------------
    # Save raw results
    # -----------------------------
    df = pd.DataFrame(all_rows)
    result_path = result_dir / "exp2_results.csv"
    df.to_csv(result_path, index=False)

    # -----------------------------
    # Plot from generated data
    # -----------------------------
    perf = df[df["method"].isin(["Similarity-aware", "Uniform"])].copy()

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
    sim_bad_weight = []

    for bad_gamma in x:
        prescribed_gammas = np.concatenate(
            [good_source_gammas, np.array([bad_gamma])]
        )

        w_sim_prescribed = similarity_weights(
            prescribed_gammas,
            eps=1e-6,
            power=1.0,
        )

        sim_bad_weight.append(w_sim_prescribed[-1])

    sim_bad_weight = np.asarray(sim_bad_weight)

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

    print("Experiment 2 finished.")
    print(f"Results saved to: {result_path}")
    print(f"Figure saved to: {figure_dir / 'exp2_bad_source_stress_test.pdf'}")


if __name__ == "__main__":
    main()