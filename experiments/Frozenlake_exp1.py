import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

try:
    import gymnasium as gym
except ImportError:
    import gym


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_ROOT / "results" / "Frozenlake_exp1"
FIGURE_DIR = PROJECT_ROOT / "figures" / "Frozenlake_exp1"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Experiment parameters
# -----------------------------
MAP_NAME = "8x8"
IS_SLIPPERY = True

ITERATIONS = 300
EVAL_EVERY = 5

DISCOUNT = 0.99

# Four fixed source domains.
# Source discrepancy is represented by perturbation magnitude.
# Do not change this list.
PERTURB_EPS_LIST = np.array([0.010, 0.015, 0.020, 0.025])

# With fixed source domains and exact target evaluation, one run is enough.
NUM_RUNS = 1

# Fixed source domains are generated once with this seed.
SOURCE_DOMAIN_SEED = 2026

# Exact model-based Bellman iteration.
STEPSIZE = 1.0

SIMILARITY_POWER = 1.0
SIMILARITY_EPS = 1e-6


# -----------------------------
# Plot style
# -----------------------------
FIGSIZE = (6.5, 4.2)
LINE_WIDTH = 2.0
SHADE_ALPHA = 0.15
GRID_ALPHA = 0.25

METHOD_ORDER = [
    "Maximum-based",
    "Similarity-aware",
    "Uniform",
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
    ymin, ymax = ax.get_ylim()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.set_ylim(ymin, ymax)


# -----------------------------
# Gym helpers
# -----------------------------
def make_frozenlake_env():
    return gym.make(
        "FrozenLake-v1",
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
    )


def get_env_transition_dict(env):
    return env.unwrapped.P


def transition_list_to_vector(transitions, n_states):
    """
    Convert Gym transition list into a full next-state probability vector.

    Gym FrozenLake can contain repeated next states because of boundary effects.
    Therefore, we aggregate probabilities by next state before computing L1.
    """
    vec = np.zeros(n_states, dtype=float)

    for p, s_next, reward, done in transitions:
        vec[int(s_next)] += float(p)

    return vec


def gym_transition_dict_to_arrays(P_dict, n_states, n_actions):
    """
    Convert Gym env.P dictionary into transition and expected-reward arrays.

        P_arr[s, a, s'] = P(s' | s, a)
        R_arr[s, a]     = E[r | s, a]
    """
    P_arr = np.zeros((n_states, n_actions, n_states), dtype=float)
    R_arr = np.zeros((n_states, n_actions), dtype=float)

    for s in range(n_states):
        for a in range(n_actions):
            for p, s_next, reward, done in P_dict[s][a]:
                P_arr[s, a, int(s_next)] += float(p)
                R_arr[s, a] += float(p) * float(reward)

    return P_arr, R_arr


# -----------------------------
# Source perturbation construction
# -----------------------------
def perturb_kernel(P_base, epsilon, n_states, n_actions, rng):
    """
    Perturb each transition row of the target kernel.

    The local radius is computed as L1 distance:

        R(s,a) = ||P_perturbed(.|s,a) - P_base(.|s,a)||_1.

    This matches the penalty form:

        R(s,a) * span(V) / 2.
    """
    P_perturbed = copy.deepcopy(P_base)
    local_l1_radii = np.zeros((n_states, n_actions), dtype=float)

    for s in range(n_states):
        for a in range(n_actions):
            transitions = P_base[s][a]

            probs = np.array(
                [float(item[0]) for item in transitions],
                dtype=float,
            )

            delta = rng.uniform(
                low=-epsilon,
                high=epsilon,
                size=len(probs),
            )

            probs_perturbed = probs + delta
            probs_perturbed = np.clip(probs_perturbed, 0.0, None)

            if probs_perturbed.sum() <= 1e-12:
                probs_perturbed = np.ones_like(probs_perturbed) / len(
                    probs_perturbed
                )
            else:
                probs_perturbed = probs_perturbed / probs_perturbed.sum()

            new_transitions = []

            for i, transition in enumerate(transitions):
                _, s_next, reward, done = transition

                new_transitions.append(
                    (
                        float(probs_perturbed[i]),
                        int(s_next),
                        float(reward),
                        bool(done),
                    )
                )

            P_perturbed[s][a] = new_transitions

            base_vec = transition_list_to_vector(transitions, n_states)
            perturbed_vec = transition_list_to_vector(new_transitions, n_states)

            local_l1_radii[s, a] = np.sum(np.abs(base_vec - perturbed_vec))

    empirical_gamma = float(np.max(local_l1_radii))
    mean_local_radius = float(np.mean(local_l1_radii))

    return P_perturbed, local_l1_radii, empirical_gamma, mean_local_radius


def generate_fixed_sources(P_true, perturb_eps_list, n_states, n_actions):
    """
    Generate fixed source domains once.

    These source domains are shared by all methods and all runs.
    """
    rng = np.random.default_rng(SOURCE_DOMAIN_SEED)

    P_sources = []
    local_l1_radii = []
    empirical_gammas = []
    mean_local_radii = []

    for epsilon in perturb_eps_list:
        P_i, R_i, gamma_i, mean_radius_i = perturb_kernel(
            P_base=P_true,
            epsilon=float(epsilon),
            n_states=n_states,
            n_actions=n_actions,
            rng=rng,
        )

        P_sources.append(P_i)
        local_l1_radii.append(R_i)
        empirical_gammas.append(gamma_i)
        mean_local_radii.append(mean_radius_i)

    return (
        P_sources,
        np.asarray(local_l1_radii),
        np.asarray(empirical_gammas),
        np.asarray(mean_local_radii),
    )


# -----------------------------
# Weights and aggregation
# -----------------------------
def similarity_weights(discrepancies, eps=1e-6, power=1.0):
    discrepancies = np.asarray(discrepancies, dtype=float)
    scores = 1.0 / np.power(discrepancies + eps, power)
    return scores / np.sum(scores)


def uniform_weights(num_sources):
    return np.ones(num_sources, dtype=float) / num_sources


def aggregate_q_tables(Q_tables, method, weights=None):
    Q_stack = np.asarray(Q_tables)

    if method == "Maximum-based":
        return np.max(Q_stack, axis=0)

    if method == "Uniform":
        return np.mean(Q_stack, axis=0)

    if method == "Similarity-aware":
        if weights is None:
            raise ValueError("weights must be provided for Similarity-aware.")
        return np.tensordot(weights, Q_stack, axes=(0, 0))

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# Robust Bellman update
# -----------------------------
def robust_bellman_update_gym(
    Q,
    P_source,
    local_l1_radius,
    n_states,
    n_actions,
    discount,
    stepsize=1.0,
):
    """
    Robust Bellman update.

        V(s) = max_a Q(s,a)
        kappa(V) = (max_s V(s) - min_s V(s)) / 2

        TQ(s,a)
        =
        sum_{s'} P_k(s'|s,a) [r + gamma V(s')]
        - gamma * R_k(s,a) * kappa(V)

    where

        R_k(s,a) = ||P_k(.|s,a) - P_0(.|s,a)||_1.
    """
    V = np.max(Q, axis=1)
    kappa = 0.5 * (np.max(V) - np.min(V))

    Q_backup = np.zeros_like(Q)

    for s in range(n_states):
        for a in range(n_actions):
            q_value = 0.0

            for p, s_next, reward, done in P_source[s][a]:
                q_value += float(p) * (
                    float(reward) + discount * V[int(s_next)]
                )

            penalty = discount * local_l1_radius[s, a] * kappa
            q_value -= penalty

            Q_backup[s, a] = q_value

    return (1.0 - stepsize) * Q + stepsize * Q_backup


# -----------------------------
# Exact target-domain evaluation
# -----------------------------
def greedy_policy(Q):
    return np.argmax(Q, axis=1)


def evaluate_policy_exact(P_arr, R_arr, policy, discount, start_state=0):
    """
    Exact expected discounted return of a deterministic policy on the target MDP.

        V^pi = (I - gamma P_pi)^(-1) r_pi.
    """
    n_states = P_arr.shape[0]

    P_pi = P_arr[np.arange(n_states), policy, :]
    r_pi = R_arr[np.arange(n_states), policy]

    V = np.linalg.solve(
        np.eye(n_states) - discount * P_pi,
        r_pi,
    )

    return V, float(V[start_state])


def target_value_iteration(P_arr, R_arr, discount, tol=1e-12, max_iter=10000):
    """
    Compute the target optimal Q function for oracle normalization/diagnostics.
    """
    n_states, n_actions, _ = P_arr.shape

    Q = np.zeros((n_states, n_actions), dtype=float)

    for it in range(max_iter):
        V = np.max(Q, axis=1)
        Q_new = R_arr + discount * np.einsum("sat,t->sa", P_arr, V)

        diff = float(np.max(np.abs(Q_new - Q)))
        Q = Q_new

        if diff < tol:
            return Q, it + 1, diff

    return Q, max_iter, diff


# -----------------------------
# Training
# -----------------------------
def train_once(
    run_id,
    P_sources,
    local_l1_radii,
    empirical_gammas,
    P_true_arr,
    R_true_arr,
    oracle_policy,
    oracle_performance,
    n_states,
    n_actions,
):
    """
    Train all methods with fixed source domains and exact target evaluation.
    """
    num_sources = len(P_sources)

    w_uniform = uniform_weights(num_sources)
    w_similarity = similarity_weights(
        empirical_gammas,
        eps=SIMILARITY_EPS,
        power=SIMILARITY_POWER,
    )

    method_weight_map = {
        "Maximum-based": None,
        "Similarity-aware": w_similarity,
        "Uniform": w_uniform,
    }

    results = {}

    for method_name in METHOD_ORDER:
        Q_tables = [
            np.zeros((n_states, n_actions), dtype=float)
            for _ in range(num_sources)
        ]

        iterations = []
        target_performances = []
        normalized_performances = []
        policy_error_rates = []

        for iteration in range(ITERATIONS + 1):
            if iteration % EVAL_EVERY == 0:
                Q_shared = aggregate_q_tables(
                    Q_tables,
                    method=method_name,
                    weights=method_weight_map[method_name],
                )

                policy = greedy_policy(Q_shared)

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

            new_q_tables = []

            for k in range(num_sources):
                Q_new = robust_bellman_update_gym(
                    Q=Q_tables[k],
                    P_source=P_sources[k],
                    local_l1_radius=local_l1_radii[k],
                    n_states=n_states,
                    n_actions=n_actions,
                    discount=DISCOUNT,
                    stepsize=STEPSIZE,
                )

                new_q_tables.append(Q_new)

            # Old-code style: aggregate every iteration.
            Q_shared = aggregate_q_tables(
                new_q_tables,
                method=method_name,
                weights=method_weight_map[method_name],
            )

            Q_tables = [
                copy.deepcopy(Q_shared)
                for _ in range(num_sources)
            ]

        results[method_name] = {
            "iterations": np.asarray(iterations),
            "target_performance": np.asarray(target_performances),
            "normalized_performance": np.asarray(normalized_performances),
            "policy_error_rate": np.asarray(policy_error_rates),
        }

    metadata = {
        "empirical_gammas": empirical_gammas,
        "uniform_weights": w_uniform,
        "similarity_weights": w_similarity,
        "oracle_performance": oracle_performance,
    }

    return results, metadata


def mean_and_sem(x, axis=0):
    x = np.asarray(x, dtype=float)
    mean = np.mean(x, axis=axis)

    if x.shape[axis] <= 1:
        sem = np.zeros_like(mean)
    else:
        sem = np.std(x, axis=axis, ddof=1) / np.sqrt(x.shape[axis])

    return mean, sem


# -----------------------------
# Main
# -----------------------------
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

    (
        P_sources,
        local_l1_radii,
        empirical_gammas,
        mean_local_radii,
    ) = generate_fixed_sources(
        P_true=P_true,
        perturb_eps_list=PERTURB_EPS_LIST,
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

    print("Fixed source domains generated once.")
    print("Perturb eps:", PERTURB_EPS_LIST)
    print("Empirical L1 gammas:", empirical_gammas)
    print("Mean local L1 radii:", mean_local_radii)
    print("Similarity weights:", similarity_weights(empirical_gammas))
    print("Uniform weights:", uniform_weights(len(P_sources)))
    print("Target oracle performance:", oracle_performance)

    all_rows = []
    all_curves = {
        method: []
        for method in METHOD_ORDER
    }

    saved_iterations = None
    saved_metadata = None

    for run_id in tqdm(range(NUM_RUNS), desc="FrozenLake exact evaluation runs"):
        run_curves, metadata = train_once(
            run_id=run_id,
            P_sources=P_sources,
            local_l1_radii=local_l1_radii,
            empirical_gammas=empirical_gammas,
            P_true_arr=P_true_arr,
            R_true_arr=R_true_arr,
            oracle_policy=oracle_policy,
            oracle_performance=oracle_performance,
            n_states=n_states,
            n_actions=n_actions,
        )

        saved_metadata = metadata

        for method_name in METHOD_ORDER:
            curve = run_curves[method_name]

            iterations = curve["iterations"]
            saved_iterations = iterations

            target_perf = curve["target_performance"]
            normalized_perf = curve["normalized_performance"]
            policy_error_rate = curve["policy_error_rate"]

            all_curves[method_name].append(target_perf)

            for i, t in enumerate(iterations):
                all_rows.append(
                    {
                        "run_id": run_id,
                        "iteration": int(t),
                        "method": method_name,
                        "target_performance": float(target_perf[i]),
                        "normalized_performance": float(normalized_perf[i]),
                        "policy_error_rate": float(policy_error_rate[i]),
                        "oracle_performance": float(oracle_performance),
                        "map_name": MAP_NAME,
                        "discount": float(DISCOUNT),
                        "stepsize": float(STEPSIZE),
                        "source_domain_seed": int(SOURCE_DOMAIN_SEED),
                        "evaluation_type": "exact",
                    }
                )

    for k, eps in enumerate(PERTURB_EPS_LIST):
        all_rows.append(
            {
                "run_id": -1,
                "iteration": -1,
                "method": "source_metadata",
                "source_index": k,
                "perturb_epsilon": float(eps),
                "empirical_gamma_l1": float(empirical_gammas[k]),
                "mean_local_l1_radius": float(mean_local_radii[k]),
                "uniform_weight": float(saved_metadata["uniform_weights"][k]),
                "similarity_weight": float(saved_metadata["similarity_weights"][k]),
                "oracle_performance": float(oracle_performance),
                "map_name": MAP_NAME,
                "discount": float(DISCOUNT),
                "stepsize": float(STEPSIZE),
                "source_domain_seed": int(SOURCE_DOMAIN_SEED),
                "evaluation_type": "exact",
            }
        )

    df = pd.DataFrame(all_rows)

    csv_path = RESULT_DIR / "Frozenlake_exp1_results.csv"
    df.to_csv(csv_path, index=False)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method_name in METHOD_ORDER:
        curves = np.asarray(all_curves[method_name])
        mean_curve, sem_curve = mean_and_sem(curves, axis=0)

        style = PLOT_STYLES[method_name]

        ax.plot(
            saved_iterations,
            mean_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=LINE_WIDTH,
            label=method_name,
        )

        ax.fill_between(
            saved_iterations,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=style["color"],
            alpha=SHADE_ALPHA,
            linewidth=0.0,
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$V_{P_0}^{\pi_t}(s_0)$")

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    ax.legend(loc="lower right")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(0.0, ymin), ymax)
    set_clean_yticks_keep_limits(ax, nbins=6)

    fig.tight_layout()

    pdf_path = FIGURE_DIR / "Frozenlake_exp1.pdf"
    png_path = FIGURE_DIR / "Frozenlake_exp1.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)

    print("FrozenLake Gym perturbation experiment finished.")
    print(f"CSV saved to: {csv_path}")
    print(f"PDF saved to: {pdf_path}")
    print(f"PNG saved to: {png_path}")


if __name__ == "__main__":
    main()