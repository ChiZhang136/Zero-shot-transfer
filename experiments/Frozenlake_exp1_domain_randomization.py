"""Run centralized model-based domain randomization on FrozenLake.

The four known source transition models are uniformly averaged into one
mixture model, and one Q-table is updated by nominal model-based Bellman
iteration. There is no environment sampling, source-model sampling, local
Q-table aggregation, or federated synchronization.

The output uses the same CSV schema as ``Frozenlake_exp1.py`` so that
``plot_Frozenlake_exp1.py`` can combine this baseline with the other methods.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from Frozenlake_exp1 import (  # noqa: E402
    DISCOUNT,
    EVAL_EVERY,
    ITERATIONS,
    MAP_NAME,
    NUM_RUNS,
    PERTURB_EPS_LIST,
    PROGRESS_UPDATE_EVERY,
    ROBUST_BACKUP_TYPE,
    SIMILARITY_EPS,
    SIMILARITY_POWER,
    SOURCE_DOMAIN_BASE_SEED,
    STEPSIZE,
    SYNC_PERIOD,
    UNCERTAINTY_DISTANCE,
    aggregate_q_tables,
    evaluate_policy_exact,
    generate_sources_with_seed,
    get_env_transition_dict,
    greedy_policy,
    gym_transition_dict_to_arrays,
    make_frozenlake_env,
    robust_bellman_update_gym,
    similarity_weights,
    target_value_iteration,
    uniform_weights,
)


METHOD_NAME = "Non-robust DR"
# Use the same nominal update step as the other FrozenLake methods.
DR_STEPSIZE = 0.4
# Training discount for the mixture-model DR only; target evaluation remains 0.99.
DR_DISCOUNT = 0.98
# Centralized mixture DR has no local synchronization; retain 1 as a
# schema-compatible metadata value for one global update per iteration.
DR_SYNC_PERIOD = 1
DR_AGGREGATION = "uniform_model_mixture"
DR_UNCERTAINTY_DISTANCE = "none"
DR_ROBUST_BACKUP_TYPE = "nominal_mixture_bellman"
RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "Frozenlake_exp1"
    / "Frozenlake_exp1_domain_randomization.csv"
)


def build_uniform_mixture_model(P_sources, n_states, n_actions):
    """Construct the uniform mixture of all known source models."""
    source_arrays = [
        gym_transition_dict_to_arrays(
            P_source,
            n_states=n_states,
            n_actions=n_actions,
        )
        for P_source in P_sources
    ]
    P_stack = np.stack([arrays[0] for arrays in source_arrays], axis=0)
    R_stack = np.stack([arrays[1] for arrays in source_arrays], axis=0)
    return np.mean(P_stack, axis=0), np.mean(R_stack, axis=0)


def nominal_bellman_update_arrays(Q, P_model, R_model, discount, stepsize):
    """Apply one nominal model-based Bellman update to an array model."""
    V = np.max(Q, axis=1)
    Q_backup = R_model + discount * np.einsum("sat,t->sa", P_model, V)
    return (1.0 - stepsize) * Q + stepsize * Q_backup


def train_domain_randomization_once(
    P_sources,
    local_l1_radii,
    P_true_arr,
    R_true_arr,
    oracle_policy,
    oracle_performance,
    n_states,
    n_actions,
    progress=None,
):
    """Train one centralized Q-table on the uniformly mixed source model."""
    del local_l1_radii
    P_mix_arr, R_mix_arr = build_uniform_mixture_model(
        P_sources=P_sources,
        n_states=n_states,
        n_actions=n_actions,
    )
    Q = np.zeros((n_states, n_actions), dtype=float)

    iterations = []
    target_performances = []
    normalized_performances = []
    policy_error_rates = []
    pending_progress_updates = 0

    for iteration in range(ITERATIONS + 1):
        if iteration % EVAL_EVERY == 0:
            policy = greedy_policy(Q)
            _, perf = evaluate_policy_exact(
                P_arr=P_true_arr,
                R_arr=R_true_arr,
                policy=policy,
                discount=DISCOUNT,
                start_state=0,
            )
            iterations.append(iteration)
            target_performances.append(perf)
            normalized_performances.append(perf / oracle_performance)
            policy_error_rates.append(float(np.mean(policy != oracle_policy)))

        if iteration == ITERATIONS:
            break

        Q = nominal_bellman_update_arrays(
            Q=Q,
            P_model=P_mix_arr,
            R_model=R_mix_arr,
            discount=DR_DISCOUNT,
            stepsize=DR_STEPSIZE,
        )

        pending_progress_updates += 1
        if (
            progress is not None
            and pending_progress_updates >= PROGRESS_UPDATE_EVERY
        ):
            progress.update(pending_progress_updates)
            pending_progress_updates = 0

    if progress is not None and pending_progress_updates > 0:
        progress.update(pending_progress_updates)

    return {
        "iterations": np.asarray(iterations),
        "target_performance": np.asarray(target_performances),
        "normalized_performance": np.asarray(normalized_performances),
        "policy_error_rate": np.asarray(policy_error_rates),
    }


def main():
    env_true = make_frozenlake_env()
    n_states = env_true.observation_space.n
    n_actions = env_true.action_space.n
    P_true = get_env_transition_dict(env_true)

    P_true_arr, R_true_arr = gym_transition_dict_to_arrays(
        P_true,
        n_states=n_states,
        n_actions=n_actions,
    )
    env_true.close()

    Q_star, _, _ = target_value_iteration(
        P_arr=P_true_arr,
        R_arr=R_true_arr,
        discount=DISCOUNT,
    )
    oracle_policy = greedy_policy(Q_star)
    _, oracle_performance = evaluate_policy_exact(
        P_arr=P_true_arr,
        R_arr=R_true_arr,
        policy=oracle_policy,
        discount=DISCOUNT,
        start_state=0,
    )

    all_rows = []
    progress = tqdm(
        total=NUM_RUNS * ITERATIONS,
        desc="FrozenLake Exp1 (centralized mixture-model DR)",
        unit="iteration",
        dynamic_ncols=True,
    )

    for run_id in range(NUM_RUNS):
        source_seed = SOURCE_DOMAIN_BASE_SEED + run_id
        (
            P_sources,
            local_l1_radii,
            empirical_gammas,
            mean_local_radii,
        ) = generate_sources_with_seed(
            P_true=P_true,
            perturb_eps_list=PERTURB_EPS_LIST,
            n_states=n_states,
            n_actions=n_actions,
            seed=source_seed,
        )

        w_uniform = uniform_weights(len(P_sources))
        w_similarity = similarity_weights(
            empirical_gammas,
            eps=SIMILARITY_EPS,
            power=SIMILARITY_POWER,
        )

        curve = train_domain_randomization_once(
            P_sources=P_sources,
            local_l1_radii=local_l1_radii,
            P_true_arr=P_true_arr,
            R_true_arr=R_true_arr,
            oracle_policy=oracle_policy,
            oracle_performance=oracle_performance,
            n_states=n_states,
            n_actions=n_actions,
            progress=progress,
        )

        for i, iteration in enumerate(curve["iterations"]):
            all_rows.append(
                {
                    "run_id": run_id,
                    "source_domain_seed": int(source_seed),
                    "iteration": int(iteration),
                    "method": METHOD_NAME,
                    "target_performance": float(curve["target_performance"][i]),
                    "normalized_performance": float(
                        curve["normalized_performance"][i]
                    ),
                    "policy_error_rate": float(curve["policy_error_rate"][i]),
                    "oracle_performance": float(oracle_performance),
                    "map_name": MAP_NAME,
                    "discount": float(DR_DISCOUNT),
                    "evaluation_discount": float(DISCOUNT),
                    "stepsize": float(DR_STEPSIZE),
                    "sync_period": int(DR_SYNC_PERIOD),
                    "aggregation": DR_AGGREGATION,
                    "evaluation_type": "exact",
                    "target_domain": "fixed",
                    "source_randomness": "perturbation_seed",
                    "uncertainty_distance": DR_UNCERTAINTY_DISTANCE,
                    "robust_backup_type": DR_ROBUST_BACKUP_TYPE,
                }
            )

        for source_index, epsilon in enumerate(PERTURB_EPS_LIST):
            all_rows.append(
                {
                    "run_id": run_id,
                    "source_domain_seed": int(source_seed),
                    "iteration": -1,
                    "method": "source_metadata",
                    "source_index": source_index,
                    "perturb_epsilon": float(epsilon),
                    "empirical_gamma_l1": float(empirical_gammas[source_index]),
                    "mean_local_l1_radius": float(
                        mean_local_radii[source_index]
                    ),
                    "uniform_weight": float(w_uniform[source_index]),
                    "similarity_weight": float(w_similarity[source_index]),
                    "oracle_performance": float(oracle_performance),
                    "map_name": MAP_NAME,
                    "discount": float(DR_DISCOUNT),
                    "evaluation_discount": float(DISCOUNT),
                    "stepsize": float(DR_STEPSIZE),
                    "sync_period": int(DR_SYNC_PERIOD),
                    "aggregation": DR_AGGREGATION,
                    "evaluation_type": "exact",
                    "target_domain": "fixed",
                    "source_randomness": "perturbation_seed",
                    "uncertainty_distance": DR_UNCERTAINTY_DISTANCE,
                    "robust_backup_type": DR_ROBUST_BACKUP_TYPE,
                }
            )

    progress.close()

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(RESULT_PATH, index=False)

    print("FrozenLake centralized mixture-model DR baseline finished.")
    print("All source models were averaged deterministically; source-model sampling: none")
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
