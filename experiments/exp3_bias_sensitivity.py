import sys
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.garnet import generate_target_garnet, generate_sources_from_target
from src.algorithms import (
    robust_bellman_optimal_source_radius,
    aggregate_q_tables,
)
from src.evaluation import greedy_policy, evaluate_policy, target_value_iteration
from src.utils import similarity_weights, ensure_dir


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


def mean_and_sem(x, axis=0):
    x = np.asarray(x, dtype=float)
    mean = np.mean(x, axis=axis)
    sem = np.std(x, axis=axis, ddof=1) / np.sqrt(x.shape[axis])
    return mean, sem


def run_periodic_learning_with_unbiased_operator_estimates(
    P_sources,
    rewards,
    gammas,
    operator_noise,
    bias_level,
    weights=None,
    discount=0.95,
    target_P=None,
    total_iterations=400,
    stepsize=0.05,
    sync_period=5,
    q="inf",
    aggregation_type="weighted",
):
    """
    Periodic multi-source robust learning with unbiased stochastic local
    Bellman operator estimates.

    At each local update, the source-k backup is perturbed by

        T_hat_{k,t} Q(s,a)
        =
        T_k Q(s,a) + delta * xi_{k,t}(s,a),

    where xi_{k,t}(s,a) has zero mean.

    operator_noise has shape:
        (total_iterations, K, S, A)

    The same operator_noise array should be passed to different aggregation
    methods to make the comparison fair.
    """
    K, S, A, _ = P_sources.shape

    gammas = np.asarray(gammas, dtype=float)

    if operator_noise.shape != (total_iterations, K, S, A):
        raise ValueError(
            "operator_noise must have shape "
            "(total_iterations, K, S, A)."
        )

    if aggregation_type == "weighted":
        if weights is None:
            raise ValueError("weights must be provided for weighted aggregation.")

        weights = np.asarray(weights, dtype=float)

        if weights.shape[0] != K:
            raise ValueError("weights must have length K.")

    Q_locals = np.zeros((K, S, A))

    for t in range(total_iterations):
        # -----------------------------
        # Local stochastic Bellman updates
        # -----------------------------
        for k in range(K):
            if gammas.ndim == 1:
                radius_k = gammas[k]
            elif gammas.ndim == 3:
                radius_k = gammas[k]
            else:
                raise ValueError("gammas must have shape (K,) or (K, S, A).")

            TQ = robust_bellman_optimal_source_radius(
                P=P_sources[k],
                rewards=rewards,
                radius=radius_k,
                discount=discount,
                Q=Q_locals[k],
                q=q,
            )

            # Unbiased local operator estimate:
            # E[xi_{k,t}(s,a)] = 0.
            TQ_hat = TQ + bias_level * operator_noise[t, k]

            Q_locals[k] = (1.0 - stepsize) * Q_locals[k] + stepsize * TQ_hat

        # -----------------------------
        # Synchronization
        # -----------------------------
        if (t + 1) % sync_period == 0:
            Q_agg = aggregate_q_tables(
                Q_locals,
                weights=weights,
                aggregation_type=aggregation_type,
            )
            Q_locals[:] = Q_agg[None, :, :]

    Q_final = aggregate_q_tables(
        Q_locals,
        weights=weights,
        aggregation_type=aggregation_type,
    )

    if target_P is not None:
        policy = greedy_policy(Q_final)
        _, perf = evaluate_policy(
            P=target_P,
            rewards=rewards,
            policy=policy,
            discount=discount,
        )
    else:
        perf = None

    return Q_final, perf


def main():
    # -----------------------------
    # Experiment parameters
    # -----------------------------
    num_states = 30
    num_actions = 4
    branching_factor = 3
    reward_range = (0.0, 1.0)
    discount = 0.95

    # Same heterogeneous source configuration as Experiment 1.
    source_gammas = np.array([0.10, 0.20, 0.40, 0.80, 1.60])
    K = len(source_gammas)

    # Robust geometry:
    # p_norm = 1 means Gamma_k is computed by L1 transition distance.
    # robust_q = "inf" means kappa_q(V) = (max V - min V) / 2.
    p_norm = 1
    robust_q = "inf"

    # Same practical learning parameters as Experiment 1.
    total_iterations = 400
    sync_period = 5
    stepsize = 0.05

    num_seeds = 20

    # Magnitude of unbiased local Bellman backup estimation error.
    # Since rewards are in [0, 1], delta is measured in reward units.
    bias_levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    result_dir = PROJECT_ROOT / "results" / "exp3"
    figure_dir = PROJECT_ROOT / "figures" / "exp3"
    ensure_dir(result_dir)
    ensure_dir(figure_dir)

    all_rows = []

    final_perf_by_method = {
        method: {float(delta): [] for delta in bias_levels}
        for method in METHOD_ORDER
    }

    for seed in tqdm(range(num_seeds), desc="Experiment 3 seeds"):
        # -----------------------------
        # Generate target and sources
        # -----------------------------
        target_mdp = generate_target_garnet(
            num_states=num_states,
            num_actions=num_actions,
            branching_factor=branching_factor,
            reward_range=reward_range,
            discount=discount,
            seed=1000 + seed,
        )

        P_sources, actual_source_gammas, rhos = generate_sources_from_target(
            target_mdp=target_mdp,
            source_gammas=source_gammas,
            branching_factor=branching_factor,
            seed=2000 + seed,
            p_norm=p_norm,
        )

        rewards = target_mdp.rewards
        P0 = target_mdp.transitions

        # -----------------------------
        # Similarity-aware weights
        # -----------------------------
        w_sim = similarity_weights(actual_source_gammas, eps=1e-6, power=1.0)

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

        # Base zero-mean operator-noise sequence for this seed.
        # The same sequence is reused for all delta values and both methods.
        rng = np.random.default_rng(3000 + seed)
        base_operator_noise = rng.uniform(
            low=-1.0,
            high=1.0,
            size=(total_iterations, K, num_states, num_actions),
        )

        for delta in bias_levels:
            delta = float(delta)

            # -----------------------------
            # Maximum-based aggregation
            # -----------------------------
            Q_max, max_perf = run_periodic_learning_with_unbiased_operator_estimates(
                P_sources=P_sources,
                rewards=rewards,
                gammas=actual_source_gammas,
                operator_noise=base_operator_noise,
                bias_level=delta,
                weights=None,
                discount=discount,
                target_P=P0,
                total_iterations=total_iterations,
                stepsize=stepsize,
                sync_period=sync_period,
                q=robust_q,
                aggregation_type="max",
            )

            max_norm = max_perf / oracle_perf
            final_perf_by_method["Maximum-based"][delta].append(max_norm)

            all_rows.append(
                {
                    "seed": seed,
                    "bias_level": delta,
                    "method": "Maximum-based",
                    "normalized_performance": float(max_norm),
                    "target_performance": float(max_perf),
                    "oracle_performance": float(oracle_perf),
                }
            )

            # -----------------------------
            # Similarity-aware aggregation
            # -----------------------------
            Q_sim, sim_perf = run_periodic_learning_with_unbiased_operator_estimates(
                P_sources=P_sources,
                rewards=rewards,
                gammas=actual_source_gammas,
                operator_noise=base_operator_noise,
                bias_level=delta,
                weights=w_sim,
                discount=discount,
                target_P=P0,
                total_iterations=total_iterations,
                stepsize=stepsize,
                sync_period=sync_period,
                q=robust_q,
                aggregation_type="weighted",
            )

            sim_norm = sim_perf / oracle_perf
            final_perf_by_method["Similarity-aware"][delta].append(sim_norm)

            all_rows.append(
                {
                    "seed": seed,
                    "bias_level": delta,
                    "method": "Similarity-aware",
                    "normalized_performance": float(sim_norm),
                    "target_performance": float(sim_perf),
                    "oracle_performance": float(oracle_perf),
                }
            )

        # Save Gamma and weights for inspection.
        for k in range(K):
            all_rows.append(
                {
                    "seed": seed,
                    "bias_level": -1,
                    "method": "metadata",
                    "source_index": k,
                    "source_gamma": float(source_gammas[k]),
                    "actual_source_gamma": float(actual_source_gammas[k]),
                    "mixing_coefficient": float(rhos[k]),
                    "similarity_weight": float(w_sim[k]),
                }
            )

    # -----------------------------
    # Save raw results
    # -----------------------------
    df = pd.DataFrame(all_rows)
    result_path = result_dir / "exp3_results.csv"
    df.to_csv(result_path, index=False)

    # -----------------------------
    # Aggregate curves
    # -----------------------------
    curve_stats = {}

    for method_name in METHOD_ORDER:
        curves = []

        for delta in bias_levels:
            values = final_perf_by_method[method_name][float(delta)]
            curves.append(values)

        curves = np.asarray(curves).T  # shape: (num_seeds, num_bias_levels)
        mean_curve, sem_curve = mean_and_sem(curves, axis=0)

        curve_stats[method_name] = {
            "mean": mean_curve,
            "sem": sem_curve,
        }

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    plot_styles = {
        "Maximum-based": {
            "color": MAX_COLOR,
            "linestyle": "-",
            "marker": "o",
        },
        "Similarity-aware": {
            "color": SIM_COLOR,
            "linestyle": "-",
            "marker": "o",
        },
    }

    for method_name in METHOD_ORDER:
        style = plot_styles[method_name]

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

    fig.tight_layout()

    fig.savefig(figure_dir / "exp3_bias_sensitivity.pdf")
    fig.savefig(figure_dir / "exp3_bias_sensitivity.png", dpi=300)

    print("Experiment 3 finished.")
    print(f"Results saved to: {result_path}")
    print(f"Figure saved to: {figure_dir / 'exp3_bias_sensitivity.pdf'}")


if __name__ == "__main__":
    main()