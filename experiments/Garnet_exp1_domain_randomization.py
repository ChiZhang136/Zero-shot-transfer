"""Run centralized model-based domain randomization for Garnet Exp1.

The known source transition models are uniformly averaged into one mixture
model, and one Q-table is updated by nominal model-based Bellman iteration.
There is no source-model sampling, local Q-table aggregation, or federated
synchronization.

Results use the same row format as ``Garnet_exp1.csv`` so that
``plot_Garnet_exp1.py`` can combine the result files without rerunning the
existing three methods.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms import run_periodic_learning
from src.evaluation import evaluate_policy, greedy_policy, target_value_iteration
from src.garnet import generate_sources_from_target, generate_target_garnet
from src.utils import ensure_dir, similarity_weights, uniform_weights


METHOD_NAME = "Non-robust DR"
RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "Garnet_exp1"
    / "Garnet_exp1_domain_randomization.csv"
)


def compute_local_transition_discrepancies(P_sources, P0, p_norm=1):
    """Compute source/target transition discrepancies for metadata output."""
    P_sources = np.asarray(P_sources, dtype=float)
    P0 = np.asarray(P0, dtype=float)
    diff = P_sources - P0[None, :, :, :]

    if p_norm == 1:
        return np.sum(np.abs(diff), axis=-1)
    if p_norm == 2:
        return np.sqrt(np.sum(diff**2, axis=-1))
    if p_norm == np.inf or p_norm == "inf":
        return np.max(np.abs(diff), axis=-1)
    return np.linalg.norm(diff, ord=p_norm, axis=-1)


def build_uniform_mixture_model(P_sources, rewards, weights):
    """Construct one uniformly mixed Garnet source model."""
    P_sources = np.asarray(P_sources, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    P_mix = np.tensordot(weights, P_sources, axes=(0, 0))
    R_stack = np.broadcast_to(rewards, (P_sources.shape[0],) + rewards.shape)
    R_mix = np.tensordot(weights, R_stack, axes=(0, 0))
    return P_mix, R_mix


def nominal_bellman_update(P_mix, R_mix, Q, discount, stepsize):
    """Apply one nominal model-based Bellman update to the mixed model."""
    V = np.max(Q, axis=1)
    Q_backup = R_mix + discount * np.einsum("sat,t->sa", P_mix, V)
    return (1.0 - stepsize) * Q + stepsize * Q_backup


def main():
    # Keep these settings identical to Garnet_exp1.py.
    num_states = 30
    num_actions = 4
    branching_factor = 3
    reward_range = (0.0, 1.0)
    discount = 0.95

    source_gammas = np.array([0.30, 0.60, 0.90, 1.20, 1.50, 1.80])
    num_sources = len(source_gammas)

    p_norm = 1
    robust_q = "inf"
    total_iterations = 300
    eval_every = 5
    stepsize = 0.05
    num_seeds = 20
    # Centralized mixture DR has no local synchronization. Keep 1 only as a
    # schema-compatible metadata value for one global update per iteration.
    sync_period = 1
    aggregation_type = "uniform_model_mixture"

    result_dir = PROJECT_ROOT / "results" / "Garnet_exp1"
    ensure_dir(result_dir)

    all_rows = []

    progress = tqdm(
        total=num_seeds,
        desc="Garnet Exp1 (centralized mixture-model DR)",
        unit="run",
        dynamic_ncols=True,
        mininterval=0.1,
        file=sys.stdout,
    )

    for seed in range(num_seeds):
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
        w_uni = uniform_weights(num_sources)
        w_sim = similarity_weights(actual_source_gammas, eps=1e-6, power=1.0)

        local_source_gammas = compute_local_transition_discrepancies(
            P_sources=P_sources,
            P0=P0,
            p_norm=p_norm,
        )
        local_gamma_max = np.max(local_source_gammas, axis=(1, 2))
        local_gamma_mean = np.mean(local_source_gammas, axis=(1, 2))

        Q_star, _, _ = target_value_iteration(P0, rewards, discount)
        pi_star = greedy_policy(Q_star)
        _, oracle_perf = evaluate_policy(P0, rewards, pi_star, discount)

        P_mix, R_mix = build_uniform_mixture_model(
            P_sources=P_sources,
            rewards=rewards,
            weights=w_uni,
        )
        Q = np.zeros((num_states, num_actions), dtype=float)

        iterations = []
        performances = []
        pending_progress_updates = 0

        for iteration in range(total_iterations + 1):
            if iteration % eval_every == 0:
                policy = greedy_policy(Q)
                _, perf = evaluate_policy(
                    P0,
                    rewards,
                    policy,
                    discount,
                )
                iterations.append(iteration)
                performances.append(perf)

            if iteration == total_iterations:
                break

            Q = nominal_bellman_update(
                P_mix=P_mix,
                R_mix=R_mix,
                Q=Q,
                discount=discount,
                stepsize=stepsize,
            )

            pending_progress_updates += 1

        perf = np.asarray(performances)
        iterations = np.asarray(iterations)
        norm_perf = perf / oracle_perf

        for i, t in enumerate(iterations):
            all_rows.append(
                {
                    "seed": seed,
                    "iteration": int(t),
                    "method": METHOD_NAME,
                    "target_performance": float(perf[i]),
                    "normalized_performance": float(norm_perf[i]),
                    "oracle_performance": float(oracle_perf),
                    "uses_robust_term": False,
                    "penalty_type": "none",
                    "p_norm": str(p_norm),
                    "robust_q": robust_q,
                    "sync_period": int(sync_period),
                    "stepsize": float(stepsize),
                    "aggregation": aggregation_type,
                }
            )

        # Keep the source metadata fields aligned with Garnet_exp1.csv.
        for k in range(num_sources):
            all_rows.append(
                {
                    "seed": seed,
                    "iteration": -1,
                    "method": "metadata",
                    "source_index": k,
                    "source_gamma": float(source_gammas[k]),
                    "max_selected_count": 0,
                    "max_selected_fraction": 0.0,
                    "actual_source_gamma": float(actual_source_gammas[k]),
                    "local_gamma_max": float(local_gamma_max[k]),
                    "local_gamma_mean": float(local_gamma_mean[k]),
                    "mixing_coefficient": float(rhos[k]),
                    "uniform_weight": float(w_uni[k]),
                    "similarity_weight": float(w_sim[k]),
                    "penalty_type": "none",
                    "p_norm": str(p_norm),
                    "robust_q": robust_q,
                    "sync_period": int(sync_period),
                    "stepsize": float(stepsize),
                    "aggregation": aggregation_type,
                }
            )

        progress.update(1)

    progress.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULT_PATH, index=False)

    print("Garnet Exp1 centralized mixture-model DR baseline finished.")
    print("All source transition models were averaged deterministically.")
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
