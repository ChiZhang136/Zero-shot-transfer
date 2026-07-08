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
from src.algorithms import run_periodic_learning
from src.evaluation import greedy_policy, evaluate_policy, target_value_iteration
from src.utils import similarity_weights, uniform_weights, mean_and_sem, ensure_dir


def compute_local_transition_discrepancies(P_sources, P0, p_norm=1):
    """
    Compute state-action-wise transition discrepancies:

        R_k(s,a) = || P_k(.|s,a) - P_0(.|s,a) ||_p.

    Parameters
    ----------
    P_sources : np.ndarray
        Shape: (K, S, A, S)
    P0 : np.ndarray
        Shape: (S, A, S)
    p_norm : int, float, or str
        Norm used over next-state distributions.

    Returns
    -------
    local_gammas : np.ndarray
        Shape: (K, S, A)
    """
    P_sources = np.asarray(P_sources, dtype=float)
    P0 = np.asarray(P0, dtype=float)

    diff = P_sources - P0[None, :, :, :]

    if p_norm == 1:
        local_gammas = np.sum(np.abs(diff), axis=-1)
    elif p_norm == 2:
        local_gammas = np.sqrt(np.sum(diff ** 2, axis=-1))
    elif p_norm == np.inf or p_norm == "inf":
        local_gammas = np.max(np.abs(diff), axis=-1)
    else:
        local_gammas = np.linalg.norm(diff, ord=p_norm, axis=-1)

    return local_gammas


def main():
    # -----------------------------
    # Experiment parameters
    # -----------------------------
    num_states = 30
    num_actions = 4
    branching_factor = 3
    reward_range = (0.0, 1.0)
    discount = 0.95

    # Source heterogeneity levels.
    # Gamma_k is still used as a source-level discrepancy for generating
    # source domains and for computing similarity weights.
    source_gammas = np.array([0.10, 0.20, 0.40, 0.80, 1.60])
    K = len(source_gammas)

    # Robust geometry:
    # p_norm = 1 means local R_k(s,a) is computed by L1 transition distance.
    # robust_q = "inf" means kappa_q(V) = (max V - min V) / 2.
    p_norm = 1
    robust_q = "inf"

    total_iterations = 200
    eval_every = 5
    sync_period = 5
    stepsize = 0.05

    num_seeds = 20

    result_dir = PROJECT_ROOT / "results" / "Garnet_exp1"
    figure_dir = PROJECT_ROOT / "figures" / "Garnet_exp1"
    ensure_dir(result_dir)
    ensure_dir(figure_dir)

    method_order = [
        "Maximum-based",
        "Similarity-aware",
        "Uniform",
    ]

    all_rows = []
    method_curves = {method: [] for method in method_order}

    saved_iterations = None

    for seed in tqdm(range(num_seeds), desc="Experiment 1 seeds"):
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
        # State-action-wise local radii
        # -----------------------------
        local_source_gammas = compute_local_transition_discrepancies(
            P_sources=P_sources,
            P0=P0,
            p_norm=p_norm,
        )

        local_gamma_max = np.max(local_source_gammas, axis=(1, 2))
        local_gamma_mean = np.mean(local_source_gammas, axis=(1, 2))

        # -----------------------------
        # Weights
        # -----------------------------
        # We keep source-level discrepancies for weighting.
        # The local (s,a)-wise radii are used only in the robust penalty.
        w_sim = similarity_weights(actual_source_gammas, eps=1e-6, power=1.0)
        w_uni = uniform_weights(K)

        # -----------------------------
        # Target oracle
        # -----------------------------
        Q_star, _, _ = target_value_iteration(P0, rewards, discount)
        pi_star = greedy_policy(Q_star)
        _, oracle_perf = evaluate_policy(P0, rewards, pi_star, discount)

        # -----------------------------
        # Method configurations
        # -----------------------------
        method_configs = {
            "Maximum-based": {
                "weights": None,
                "aggregation_type": "max",
            },
            "Similarity-aware": {
                "weights": w_sim,
                "aggregation_type": "weighted",
            },
            "Uniform": {
                "weights": w_uni,
                "aggregation_type": "weighted",
            },
        }

        for method_name in method_order:
            config = method_configs[method_name]

            _, hist = run_periodic_learning(
                P_sources=P_sources,
                rewards=rewards,
                # Important change:
                # use R_k(s,a), shape (K, S, A), instead of source-level Gamma_k.
                gammas=local_source_gammas,
                weights=config["weights"],
                discount=discount,
                target_P=P0,
                total_iterations=total_iterations,
                stepsize=stepsize,
                sync_period=sync_period,
                eval_every=eval_every,
                q=robust_q,
                aggregation_type=config["aggregation_type"],
            )

            iterations = np.asarray(hist["iterations"])
            saved_iterations = iterations

            perf = np.asarray(hist["target_performance"])
            norm_perf = perf / oracle_perf

            method_curves[method_name].append(norm_perf)

            for i, t in enumerate(iterations):
                all_rows.append(
                    {
                        "seed": seed,
                        "iteration": int(t),
                        "method": method_name,
                        "target_performance": float(perf[i]),
                        "normalized_performance": float(norm_perf[i]),
                        "oracle_performance": float(oracle_perf),
                        "penalty_type": "state_action_local",
                        "p_norm": str(p_norm),
                        "robust_q": robust_q,
                        "sync_period": int(sync_period),
                        "stepsize": float(stepsize),
                    }
                )

        # Save Gamma and weights for inspection.
        for k in range(K):
            all_rows.append(
                {
                    "seed": seed,
                    "iteration": -1,
                    "method": "metadata",
                    "source_index": k,
                    "source_gamma": float(source_gammas[k]),
                    "actual_source_gamma": float(actual_source_gammas[k]),
                    "local_gamma_max": float(local_gamma_max[k]),
                    "local_gamma_mean": float(local_gamma_mean[k]),
                    "mixing_coefficient": float(rhos[k]),
                    "uniform_weight": float(w_uni[k]),
                    "similarity_weight": float(w_sim[k]),
                    "penalty_type": "state_action_local",
                    "p_norm": str(p_norm),
                    "robust_q": robust_q,
                    "sync_period": int(sync_period),
                    "stepsize": float(stepsize),
                }
            )

    # -----------------------------
    # Save raw results
    # -----------------------------
    df = pd.DataFrame(all_rows)
    result_path = result_dir / "Garnet_exp1.csv"
    df.to_csv(result_path, index=False)

    # -----------------------------
    # Aggregate curves
    # -----------------------------
    curve_stats = {}

    for method_name in method_order:
        curves = np.asarray(method_curves[method_name])
        mean_curve, sem_curve = mean_and_sem(curves, axis=0)

        curve_stats[method_name] = {
            "mean": mean_curve,
            "sem": sem_curve,
        }

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    plot_styles = {
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

    for method_name in method_order:
        mean_curve = 100.0 * curve_stats[method_name]["mean"]
        sem_curve = 100.0 * curve_stats[method_name]["sem"]
        style = plot_styles[method_name]

        ax.plot(
            saved_iterations,
            mean_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            label=method_name,
        )

        ax.fill_between(
            saved_iterations,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=style["color"],
            alpha=0.15,
            linewidth=0.0,
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\nu(t)$ (%)")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=0.25)

    ax.legend(loc="lower right")

    set_integer_yticks_keep_limits(ax)

    fig.tight_layout()

    figure_pdf_path = figure_dir / "Garnet_exp1.pdf"
    figure_png_path = figure_dir / "Garnet_exp1.png"

    fig.savefig(figure_pdf_path)
    fig.savefig(figure_png_path, dpi=300)

    print("Experiment 1 finished.")
    print("Penalty type: state-action local radii R_k(s,a)")
    print(f"Results saved to: {result_path}")
    print(f"Figure saved to: {figure_pdf_path}")


if __name__ == "__main__":
    main()