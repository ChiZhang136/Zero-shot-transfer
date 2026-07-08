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

RESULT_DIR = PROJECT_ROOT / "results" / "Frozenlake_exp3"
FIGURE_DIR = PROJECT_ROOT / "figures" / "Frozenlake_exp3"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Experiment parameters
# -----------------------------
MAP_NAME = "8x8"
IS_SLIPPERY = True

# Same main learning parameters as Frozenlake_exp1.
ITERATIONS = 500
DISCOUNT = 0.99

# Same heterogeneous source configuration as Frozenlake_exp1.
PERTURB_EPS_LIST = np.array([0.010, 0.015, 0.020, 0.030])

# Multiple random source perturbation seeds under the same fixed target domain.
NUM_RUNS = 10

# Base seed for source-domain perturbations.
# Run-specific seed is SOURCE_DOMAIN_BASE_SEED + run_id.
SOURCE_DOMAIN_BASE_SEED = 2026

# Same learning and synchronization parameters as Frozenlake_exp1.
STEPSIZE = 0.5
SYNC_PERIOD = 5

SIMILARITY_POWER = 1.0
SIMILARITY_EPS = 1e-6

# Progress bar update frequency.
PROGRESS_UPDATE_EVERY = 10

# Magnitude of zero-mean noise added to local Bellman backups.
# This is Bellman-target noise, not aggregation-observation noise.
NOISE_LEVELS = np.array([0.000, 0.002, 0.004, 0.006, 0.008])

# Keep this alias so old plotting/CSV naming remains familiar.
BIAS_LEVELS = NOISE_LEVELS

# Bellman backup noise is Uniform[-1, 1].
NOISE_LOW = -1.0
NOISE_HIGH = 1.0

# Base seed for Bellman-noise realizations.
BELLMAN_NOISE_BASE_SEED = 3000


# -----------------------------
# Plot style
# -----------------------------
FIGSIZE = (6.5, 4.2)

LINE_WIDTH = 2.0
MARKER_SIZE = 4.5
SHADE_ALPHA = 0.15
GRID_ALPHA = 0.25

MAX_COLOR = "C2"
SIM_COLOR = "C0"

METHOD_ORDER = [
    "Maximum-based",
    "Similarity-aware",
]

PLOT_STYLES = {
    "Maximum-based": {
        "color": MAX_COLOR,
        "linestyle": "-",
        "marker": "s",
    },
    "Similarity-aware": {
        "color": SIM_COLOR,
        "linestyle": "-",
        "marker": "o",
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


def generate_sources_with_seed(
    P_true,
    perturb_eps_list,
    n_states,
    n_actions,
    seed,
):
    """
    Generate one random realization of source domains under the same fixed
    target FrozenLake domain.
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


# -----------------------------
# Weights and aggregation
# -----------------------------
def similarity_weights(discrepancies, eps=1e-6, power=1.0):
    discrepancies = np.asarray(discrepancies, dtype=float)
    scores = 1.0 / np.power(discrepancies + eps, power)
    return scores / np.sum(scores)


def aggregate_q_tables(Q_tables, method, weights=None):
    Q_stack = np.asarray(Q_tables)

    if method == "Maximum-based":
        return np.max(Q_stack, axis=0)

    if method == "Similarity-aware":
        if weights is None:
            raise ValueError("weights must be provided for Similarity-aware.")
        return np.tensordot(weights, Q_stack, axes=(0, 0))

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# Robust Bellman backup
# -----------------------------
def robust_bellman_backup_gym(
    Q,
    P_source,
    local_l1_radius,
    n_states,
    n_actions,
    discount,
):
    """
    Exact robust Bellman backup:

        TQ(s,a)
        =
        sum_{s'} P_k(s'|s,a) [r + gamma max_a' Q(s',a')]
        - gamma * R_k(s,a) * kappa(V).

    The model-free-style noise is added outside this function.
    """
    V = np.max(Q, axis=1)
    kappa = 0.5 * (np.max(V) - np.min(V))

    TQ = np.zeros_like(Q)

    for s in range(n_states):
        for a in range(n_actions):
            q_value = 0.0

            for p, s_next, reward, done in P_source[s][a]:
                q_value += float(p) * (
                    float(reward) + discount * V[int(s_next)]
                )

            penalty = discount * local_l1_radius[s, a] * kappa
            q_value -= penalty

            TQ[s, a] = q_value

    return TQ


# -----------------------------
# Exact target-domain evaluation
# -----------------------------
def greedy_policy(Q):
    return np.argmax(Q, axis=1)


def evaluate_policy_exact(P_arr, R_arr, policy, discount, start_state=0):
    """
    Exact expected discounted return of a deterministic policy on the fixed
    target MDP:

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
    Compute the target optimal Q function for oracle diagnostics.
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
# Model-free-style Bellman-noise learning
# -----------------------------
def run_learning_with_noisy_bellman_backups(
    P_sources,
    local_l1_radii,
    bellman_noise,
    noise_level,
    weights,
    method,
    discount,
    total_iterations,
    stepsize,
    sync_period,
    n_states,
    n_actions,
    pbar=None,
    progress_update_every=10,
):
    """
    Multi-source robust learning with model-free-style local Bellman noise.

    We add zero-mean noise to the local Bellman backups:

        T_hat_k Q_k = T_k Q_k + delta * xi_k,

    and then update:

        Q_k <- (1 - eta) Q_k + eta * T_hat_k Q_k.

    Synchronization uses clean local Q-tables directly:

        Q_agg = Agg(Q_1, ..., Q_K).

    No extra noisy observation is used at aggregation or final output.
    """
    K = len(P_sources)

    if bellman_noise.shape != (total_iterations, K, n_states, n_actions):
        raise ValueError(
            "bellman_noise must have shape "
            "(total_iterations, K, n_states, n_actions)."
        )

    if method == "Similarity-aware":
        if weights is None:
            raise ValueError("weights must be provided for Similarity-aware.")
    elif method == "Maximum-based":
        weights = None
    else:
        raise ValueError(f"Unknown method: {method}")

    Q_locals = np.zeros((K, n_states, n_actions), dtype=float)

    pending_progress_updates = 0

    for t in range(total_iterations):
        # -----------------------------
        # Local stochastic Bellman updates
        # -----------------------------
        for k in range(K):
            TQ = robust_bellman_backup_gym(
                Q=Q_locals[k],
                P_source=P_sources[k],
                local_l1_radius=local_l1_radii[k],
                n_states=n_states,
                n_actions=n_actions,
                discount=discount,
            )

            TQ_noisy = TQ + noise_level * bellman_noise[t, k]

            Q_locals[k] = (
                (1.0 - stepsize) * Q_locals[k]
                + stepsize * TQ_noisy
            )

        # -----------------------------
        # Clean aggregation / synchronization
        # -----------------------------
        if (t + 1) % sync_period == 0:
            Q_agg = aggregate_q_tables(
                Q_locals,
                method=method,
                weights=weights,
            )

            Q_locals[:] = Q_agg[None, :, :]

        # -----------------------------
        # Progress bar update
        # -----------------------------
        pending_progress_updates += 1

        if (
            pbar is not None
            and pending_progress_updates >= progress_update_every
        ):
            pbar.update(pending_progress_updates)
            pending_progress_updates = 0

    if pbar is not None and pending_progress_updates > 0:
        pbar.update(pending_progress_updates)

    # Final aggregation uses clean local Q-tables.
    Q_final = aggregate_q_tables(
        Q_locals,
        method=method,
        weights=weights,
    )

    return Q_final


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

    print("FrozenLake Experiment 3: fixed target + random source seeds + Bellman noise")
    print("Map:", MAP_NAME)
    print("Is slippery:", IS_SLIPPERY)
    print("Perturb eps:", PERTURB_EPS_LIST)
    print("Number of source perturbation seeds:", NUM_RUNS)
    print("Source domain base seed:", SOURCE_DOMAIN_BASE_SEED)
    print("Bellman noise base seed:", BELLMAN_NOISE_BASE_SEED)
    print("Target oracle performance:", oracle_performance)
    print("Synchronization period:", SYNC_PERIOD)
    print("Stepsize:", STEPSIZE)
    print("Noise location: Bellman backup only")

    all_rows = []

    final_perf_by_method = {
        method: {float(delta): [] for delta in NOISE_LEVELS}
        for method in METHOD_ORDER
    }

    total_training_steps = (
        NUM_RUNS
        * len(NOISE_LEVELS)
        * len(METHOD_ORDER)
        * ITERATIONS
    )

    with tqdm(
        total=total_training_steps,
        desc="FrozenLake Exp3 Bellman-noise iterations",
    ) as pbar:
        for run_id in range(NUM_RUNS):
            source_seed = SOURCE_DOMAIN_BASE_SEED + run_id
            bellman_noise_seed = BELLMAN_NOISE_BASE_SEED + run_id

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

            w_sim = similarity_weights(
                empirical_gammas,
                eps=SIMILARITY_EPS,
                power=SIMILARITY_POWER,
            )

            K = len(P_sources)

            rng = np.random.default_rng(bellman_noise_seed)

            # Base zero-mean Bellman backup noise for this run.
            # Reused for all noise levels and both methods.
            base_bellman_noise = rng.uniform(
                low=NOISE_LOW,
                high=NOISE_HIGH,
                size=(ITERATIONS, K, n_states, n_actions),
            )

            for delta in NOISE_LEVELS:
                delta = float(delta)

                for method_name in METHOD_ORDER:
                    pbar.set_postfix(
                        run=run_id,
                        delta=f"{delta:.3f}",
                        method=method_name,
                    )

                    if method_name == "Similarity-aware":
                        weights = w_sim
                    else:
                        weights = None

                    Q_final = run_learning_with_noisy_bellman_backups(
                        P_sources=P_sources,
                        local_l1_radii=local_l1_radii,
                        bellman_noise=base_bellman_noise,
                        noise_level=delta,
                        weights=weights,
                        method=method_name,
                        discount=DISCOUNT,
                        total_iterations=ITERATIONS,
                        stepsize=STEPSIZE,
                        sync_period=SYNC_PERIOD,
                        n_states=n_states,
                        n_actions=n_actions,
                        pbar=pbar,
                        progress_update_every=PROGRESS_UPDATE_EVERY,
                    )

                    policy = greedy_policy(Q_final)

                    _, target_perf = evaluate_policy_exact(
                        P_arr=P_true_arr,
                        R_arr=R_true_arr,
                        policy=policy,
                        discount=DISCOUNT,
                        start_state=0,
                    )

                    normalized_perf = target_perf / oracle_performance
                    policy_error_rate = float(np.mean(policy != oracle_policy))

                    final_perf_by_method[method_name][delta].append(
                        normalized_perf
                    )

                    all_rows.append(
                        {
                            "run_id": run_id,
                            "source_domain_seed": int(source_seed),
                            "bellman_noise_seed": int(bellman_noise_seed),
                            "bias_level": delta,
                            "noise_level": delta,
                            "bellman_noise_level": delta,
                            "method": method_name,
                            "target_performance": float(target_perf),
                            "normalized_performance": float(normalized_perf),
                            "oracle_performance": float(oracle_performance),
                            "policy_error_rate": float(policy_error_rate),
                            "map_name": MAP_NAME,
                            "discount": float(DISCOUNT),
                            "stepsize": float(STEPSIZE),
                            "sync_period": int(SYNC_PERIOD),
                            "iterations": int(ITERATIONS),
                            "source_domain_base_seed": int(
                                SOURCE_DOMAIN_BASE_SEED
                            ),
                            "bellman_noise_base_seed": int(
                                BELLMAN_NOISE_BASE_SEED
                            ),
                            "aggregation_noise_seed": -1,
                            "evaluation_type": "exact",
                            "target_domain": "fixed",
                            "source_randomness": "perturbation_seed",
                            "noise_location": "bellman_backup_only",
                        }
                    )

            # Save source metadata for inspection.
            for k in range(K):
                all_rows.append(
                    {
                        "run_id": run_id,
                        "source_domain_seed": int(source_seed),
                        "bellman_noise_seed": int(bellman_noise_seed),
                        "bias_level": -1,
                        "noise_level": -1,
                        "bellman_noise_level": -1,
                        "method": "metadata",
                        "source_index": k,
                        "perturb_epsilon": float(PERTURB_EPS_LIST[k]),
                        "empirical_gamma_l1": float(empirical_gammas[k]),
                        "mean_local_l1_radius": float(mean_local_radii[k]),
                        "similarity_weight": float(w_sim[k]),
                        "map_name": MAP_NAME,
                        "discount": float(DISCOUNT),
                        "stepsize": float(STEPSIZE),
                        "sync_period": int(SYNC_PERIOD),
                        "iterations": int(ITERATIONS),
                        "source_domain_base_seed": int(
                            SOURCE_DOMAIN_BASE_SEED
                        ),
                        "bellman_noise_base_seed": int(
                            BELLMAN_NOISE_BASE_SEED
                        ),
                        "aggregation_noise_seed": -1,
                        "evaluation_type": "exact",
                        "target_domain": "fixed",
                        "source_randomness": "perturbation_seed",
                        "noise_location": "bellman_backup_only",
                    }
                )

    # -----------------------------
    # Save raw results
    # -----------------------------
    df = pd.DataFrame(all_rows)
    result_path = RESULT_DIR / "Frozenlake_exp3.csv"
    df.to_csv(result_path, index=False)

    # -----------------------------
    # Aggregate curves
    # -----------------------------
    curve_stats = {}

    for method_name in METHOD_ORDER:
        curves = []

        for delta in NOISE_LEVELS:
            values = final_perf_by_method[method_name][float(delta)]
            curves.append(values)

        curves = np.asarray(curves).T  # shape: (NUM_RUNS, num_noise_levels)
        mean_curve, sem_curve = mean_and_sem(curves, axis=0)

        curve_stats[method_name] = {
            "mean": mean_curve,
            "sem": sem_curve,
        }

    # -----------------------------
    # Plot normalized target performance
    # -----------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method_name in METHOD_ORDER:
        style = PLOT_STYLES[method_name]

        mean_curve = 100.0 * curve_stats[method_name]["mean"]
        sem_curve = 100.0 * curve_stats[method_name]["sem"]

        ax.plot(
            NOISE_LEVELS,
            mean_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=method_name,
        )

        ax.fill_between(
            NOISE_LEVELS,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=style["color"],
            alpha=SHADE_ALPHA,
            linewidth=0.0,
        )

    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\nu(T)$ (%)")

    ax.set_xticks(NOISE_LEVELS)
    ax.set_xticklabels([f"{d:.3f}" for d in NOISE_LEVELS])

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    ax.legend(loc="lower left")

    # -----------------------------
    # Make y-axis range taller
    # -----------------------------
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    ax.set_ylim(
        max(0.0, ymin),
        max(ymax + 0.05 * yrange, 100.0 + 0.02 * yrange),
    )

    set_clean_yticks_keep_limits(ax, nbins=6)

    fig.tight_layout()

    figure_pdf_path = FIGURE_DIR / "Frozenlake_exp3.pdf"
    figure_png_path = FIGURE_DIR / "Frozenlake_exp3.png"

    fig.savefig(figure_pdf_path)
    fig.savefig(figure_png_path, dpi=300)

    print("FrozenLake Experiment 3 finished.")
    print(f"Results saved to: {result_path}")
    print(f"Figure saved to: {figure_pdf_path}")
    print(f"PNG saved to: {figure_png_path}")


if __name__ == "__main__":
    main()