import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from tqdm import tqdm

try:
    import gymnasium as gym
except ImportError:
    import gym


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_ROOT / "results" / "Frozenlake_exp2"
FIGURE_DIR = PROJECT_ROOT / "figures" / "Frozenlake_exp2"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Experiment parameters
# -----------------------------
MAP_NAME = "8x8"
IS_SLIPPERY = True

ITERATIONS = 500
EVAL_EVERY = 5

DISCOUNT = 0.99

# Three reliable / relevant source domains.
GOOD_SOURCE_EPS_LIST = np.array([0.010, 0.015, 0.020])

# One bad source domain whose perturbation magnitude increases.
BAD_SOURCE_EPS_LIST = np.array([
    0.030,
    0.035,
    0.040,
    0.045,
])

# Multiple random source perturbation seeds under the same fixed target domain.
NUM_RUNS = 10

# Run-specific seeds:
# good_seed = GOOD_SOURCE_BASE_SEED + run_id
# bad_seed  = BAD_SOURCE_BASE_SEED + 1000 * run_id + bad_idx
GOOD_SOURCE_BASE_SEED = 2026
BAD_SOURCE_BASE_SEED = 3026

# Exact model-based Bellman iteration.
STEPSIZE = 0.5
SYNC_PERIOD = 5

SIMILARITY_POWER = 1.0
SIMILARITY_EPS = 1e-6

PROGRESS_UPDATE_EVERY = 10


# -----------------------------
# Plot style
# -----------------------------
FIGSIZE = (6.5, 4.2)

LINE_WIDTH = 2.0
MARKER_SIZE = 5.5
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

PLOT_STYLES = {
    "Similarity-aware": {
        "color": SIM_COLOR,
        "linestyle": "-",
        "marker": "o",
    },
    "Uniform": {
        "color": UNI_COLOR,
        "linestyle": "-",
        "marker": "^",
    },
}


def set_clean_yticks_keep_limits(ax, nbins=6):
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
        R_arr[s, a]     = E[r | s, a)
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
    Perturb each transition row of the fixed target kernel.

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


def generate_sources_from_eps_list(
    P_true,
    perturb_eps_list,
    n_states,
    n_actions,
    seed,
):
    """
    Generate one random realization of source domains for a given
    perturbation-epsilon list.
    """
    rng = np.random.default_rng(seed)

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


def generate_bad_source(
    P_true,
    bad_source_epsilon,
    n_states,
    n_actions,
    seed,
):
    """
    Generate one bad source domain with a given perturbation magnitude.
    """
    rng = np.random.default_rng(seed)

    return perturb_kernel(
        P_base=P_true,
        epsilon=float(bad_source_epsilon),
        n_states=n_states,
        n_actions=n_actions,
        rng=rng,
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
    stepsize,
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
    Exact expected discounted return of a deterministic policy on the fixed
    target MDP.

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
    bad_source_epsilon,
    P_sources,
    local_l1_radii,
    empirical_gammas,
    P_true_arr,
    R_true_arr,
    oracle_policy,
    oracle_performance,
    n_states,
    n_actions,
    pbar=None,
    progress_update_every=10,
):
    """
    Train all methods with one random source-domain realization and exact
    target evaluation.
    """
    num_sources = len(P_sources)

    w_uniform = uniform_weights(num_sources)
    w_similarity = similarity_weights(
        empirical_gammas,
        eps=SIMILARITY_EPS,
        power=SIMILARITY_POWER,
    )

    method_weight_map = {
        "Similarity-aware": w_similarity,
        "Uniform": w_uniform,
    }

    results = {}

    pending_progress_updates = 0

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

            # Synchronize only every SYNC_PERIOD local updates.
            if (iteration + 1) % SYNC_PERIOD == 0:
                Q_shared = aggregate_q_tables(
                    new_q_tables,
                    method=method_name,
                    weights=method_weight_map[method_name],
                )

                Q_tables = [
                    copy.deepcopy(Q_shared)
                    for _ in range(num_sources)
                ]
            else:
                Q_tables = new_q_tables

            pending_progress_updates += 1

            if (
                pbar is not None
                and pending_progress_updates >= progress_update_every
            ):
                pbar.update(pending_progress_updates)
                pending_progress_updates = 0

        results[method_name] = {
            "iterations": np.asarray(iterations),
            "target_performance": np.asarray(target_performances),
            "normalized_performance": np.asarray(normalized_performances),
            "policy_error_rate": np.asarray(policy_error_rates),
        }

    if pbar is not None and pending_progress_updates > 0:
        pbar.update(pending_progress_updates)

    metadata = {
        "bad_source_epsilon": float(bad_source_epsilon),
        "empirical_gammas": empirical_gammas,
        "uniform_weights": w_uniform,
        "similarity_weights": w_similarity,
        "oracle_performance": oracle_performance,
    }

    return results, metadata


# -----------------------------
# Main
# -----------------------------
def main():
    # -----------------------------
    # Fixed target FrozenLake domain
    # -----------------------------
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

    print("FrozenLake Experiment 2: fixed target + random source perturbation seeds")
    print("Map:", MAP_NAME)
    print("Is slippery:", IS_SLIPPERY)
    print("Good source perturb eps:", GOOD_SOURCE_EPS_LIST)
    print("Bad source perturb eps list:", BAD_SOURCE_EPS_LIST)
    print("Number of source perturbation seeds:", NUM_RUNS)
    print("Good source base seed:", GOOD_SOURCE_BASE_SEED)
    print("Bad source base seed:", BAD_SOURCE_BASE_SEED)
    print("Target oracle performance:", oracle_performance)
    print("Synchronization period:", SYNC_PERIOD)

    all_rows = []
    all_final_rows = []

    total_steps = (
        NUM_RUNS
        * len(BAD_SOURCE_EPS_LIST)
        * len(METHOD_ORDER)
        * ITERATIONS
    )

    with tqdm(
        total=total_steps,
        desc="FrozenLake Exp2 training iterations",
    ) as pbar:
        for run_id in range(NUM_RUNS):
            good_seed = GOOD_SOURCE_BASE_SEED + run_id

            (
                P_good_sources,
                good_local_l1_radii,
                good_empirical_gammas,
                good_mean_local_radii,
            ) = generate_sources_from_eps_list(
                P_true=P_true,
                perturb_eps_list=GOOD_SOURCE_EPS_LIST,
                n_states=n_states,
                n_actions=n_actions,
                seed=good_seed,
            )

            for bad_idx, bad_source_epsilon in enumerate(BAD_SOURCE_EPS_LIST):
                bad_source_epsilon = float(bad_source_epsilon)
                bad_seed = BAD_SOURCE_BASE_SEED + 1000 * run_id + bad_idx

                P_bad, R_bad, gamma_bad, mean_radius_bad = generate_bad_source(
                    P_true=P_true,
                    bad_source_epsilon=bad_source_epsilon,
                    n_states=n_states,
                    n_actions=n_actions,
                    seed=bad_seed,
                )

                P_sources = list(P_good_sources) + [P_bad]

                local_l1_radii = np.concatenate(
                    [
                        good_local_l1_radii,
                        R_bad[None, :, :],
                    ],
                    axis=0,
                )

                empirical_gammas = np.concatenate(
                    [
                        good_empirical_gammas,
                        np.array([gamma_bad]),
                    ],
                    axis=0,
                )

                mean_local_radii = np.concatenate(
                    [
                        good_mean_local_radii,
                        np.array([mean_radius_bad]),
                    ],
                    axis=0,
                )

                prescribed_eps = np.concatenate(
                    [
                        GOOD_SOURCE_EPS_LIST,
                        np.array([bad_source_epsilon]),
                    ],
                    axis=0,
                )

                run_curves, metadata = train_once(
                    run_id=run_id,
                    bad_source_epsilon=bad_source_epsilon,
                    P_sources=P_sources,
                    local_l1_radii=local_l1_radii,
                    empirical_gammas=empirical_gammas,
                    P_true_arr=P_true_arr,
                    R_true_arr=R_true_arr,
                    oracle_policy=oracle_policy,
                    oracle_performance=oracle_performance,
                    n_states=n_states,
                    n_actions=n_actions,
                    pbar=pbar,
                    progress_update_every=PROGRESS_UPDATE_EVERY,
                )

                for method_name in METHOD_ORDER:
                    curve = run_curves[method_name]

                    iterations = curve["iterations"]
                    target_perf = curve["target_performance"]
                    normalized_perf = curve["normalized_performance"]
                    policy_error_rate = curve["policy_error_rate"]

                    for i, t in enumerate(iterations):
                        all_rows.append(
                            {
                                "run_id": run_id,
                                "good_source_seed": int(good_seed),
                                "bad_source_seed": int(bad_seed),
                                "bad_source_epsilon": bad_source_epsilon,
                                "iteration": int(t),
                                "method": method_name,
                                "target_performance": float(target_perf[i]),
                                "normalized_performance": float(normalized_perf[i]),
                                "policy_error_rate": float(policy_error_rate[i]),
                                "oracle_performance": float(oracle_performance),
                                "bad_source_empirical_gamma_l1": float(gamma_bad),
                                "bad_source_mean_local_l1_radius": float(
                                    mean_radius_bad
                                ),
                                "bad_source_similarity_weight": float(
                                    metadata["similarity_weights"][-1]
                                ),
                                "bad_source_uniform_weight": float(
                                    metadata["uniform_weights"][-1]
                                ),
                                "map_name": MAP_NAME,
                                "discount": float(DISCOUNT),
                                "stepsize": float(STEPSIZE),
                                "sync_period": int(SYNC_PERIOD),
                                "evaluation_type": "exact",
                                "target_domain": "fixed",
                                "source_randomness": "perturbation_seed",
                            }
                        )

                    all_final_rows.append(
                        {
                            "run_id": run_id,
                            "good_source_seed": int(good_seed),
                            "bad_source_seed": int(bad_seed),
                            "bad_source_epsilon": bad_source_epsilon,
                            "bad_source_empirical_gamma_l1": float(gamma_bad),
                            "bad_source_mean_local_l1_radius": float(
                                mean_radius_bad
                            ),
                            "bad_source_similarity_weight": float(
                                metadata["similarity_weights"][-1]
                            ),
                            "bad_source_uniform_weight": float(
                                metadata["uniform_weights"][-1]
                            ),
                            "method": method_name,
                            "final_iteration": int(iterations[-1]),
                            "final_target_performance": float(target_perf[-1]),
                            "final_normalized_performance": float(
                                normalized_perf[-1]
                            ),
                            "final_policy_error_rate": float(
                                policy_error_rate[-1]
                            ),
                            "oracle_performance": float(oracle_performance),
                        }
                    )

                # Save source metadata for this run and bad-source setting.
                for k, eps in enumerate(prescribed_eps):
                    source_type = "bad" if k == len(prescribed_eps) - 1 else "good"

                    all_rows.append(
                        {
                            "run_id": run_id,
                            "good_source_seed": int(good_seed),
                            "bad_source_seed": int(bad_seed),
                            "bad_source_epsilon": bad_source_epsilon,
                            "iteration": -1,
                            "method": "source_metadata",
                            "source_index": k,
                            "source_type": source_type,
                            "perturb_epsilon": float(eps),
                            "empirical_gamma_l1": float(empirical_gammas[k]),
                            "mean_local_l1_radius": float(mean_local_radii[k]),
                            "uniform_weight": float(metadata["uniform_weights"][k]),
                            "similarity_weight": float(
                                metadata["similarity_weights"][k]
                            ),
                            "oracle_performance": float(oracle_performance),
                            "map_name": MAP_NAME,
                            "discount": float(DISCOUNT),
                            "stepsize": float(STEPSIZE),
                            "sync_period": int(SYNC_PERIOD),
                            "evaluation_type": "exact",
                            "target_domain": "fixed",
                            "source_randomness": "perturbation_seed",
                        }
                    )

    df = pd.DataFrame(all_rows)
    final_df = pd.DataFrame(all_final_rows)

    csv_path = RESULT_DIR / "Frozenlake_exp2_results.csv"
    final_csv_path = RESULT_DIR / "Frozenlake_exp2_final_results.csv"

    df.to_csv(csv_path, index=False)
    final_df.to_csv(final_csv_path, index=False)

    # -----------------------------
    # Plot: boxplot of final normalized target performance
    # -----------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    bad_eps_values = np.asarray(BAD_SOURCE_EPS_LIST, dtype=float)
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

    axins = ax.inset_axes([0.60, 0.17, 0.22, 0.18])

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

    print("FrozenLake Experiment 2 finished.")
    print(f"Full CSV saved to: {csv_path}")
    print(f"Final CSV saved to: {final_csv_path}")
    print(f"PDF saved to: {pdf_path}")
    print(f"PNG saved to: {png_path}")


if __name__ == "__main__":
    main()