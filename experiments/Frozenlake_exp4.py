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

RESULT_DIR = PROJECT_ROOT / "results" / "Frozenlake_exp4"
FIGURE_DIR = PROJECT_ROOT / "figures" / "Frozenlake_exp4"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Experiment parameters
# -----------------------------
MAP_NAME = "8x8"
IS_SLIPPERY = True

ITERATIONS = 500
DISCOUNT = 0.99

PERTURB_EPS_LIST = np.array([0.010, 0.015, 0.020, 0.030])

NUM_RUNS = 10
SOURCE_DOMAIN_BASE_SEED = 2026

STEPSIZE = 0.5
SYNC_PERIOD = 5

SIMILARITY_POWER = 1.0
SIMILARITY_EPS = 1e-6

UNCERTAINTY_DISTANCE = "support_restricted_tv_l1"
ROBUST_BACKUP_TYPE = "exact_support_restricted_l1"

# Bellman backup noise level used for the trajectory diagnostic.
DIAGNOSTIC_NOISE_LEVEL = 0.008

# Number of Bellman-noise trajectories used to estimate E_xi.
# Must be even because antithetic noise is used.
NUM_NOISE_TRAJECTORIES = 60

BELLMAN_NOISE_BASE_SEED = 3000

NOISE_LOW = -1.0
NOISE_HIGH = 1.0


# -----------------------------
# Plot style
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

PLOT_STYLES = {
    "Maximum-based": {
        "color": MAX_COLOR,
        "linestyle": "-",
    },
    "Similarity-aware": {
        "color": SIM_COLOR,
        "linestyle": "-",
    },
}

# Main plotted metric.
# Options:
#   "signed_selection_bias"
#   "selection_bias_inf_norm"
#   "signed_selection_bias_percent"
#   "selection_bias_inf_norm_percent"
PLOT_METRIC = "signed_selection_bias"


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

    This radius defines a support-restricted L1 probability ball, equivalent
    to a total-variation ball with radius R(s,a) / 2.
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


def convert_source_dicts_to_arrays(P_sources, n_states, n_actions):
    P_arr_list = []
    R_arr_list = []
    support_mask_list = []

    for P_source in P_sources:
        P_arr, R_arr = gym_transition_dict_to_arrays(
            P_source,
            n_states=n_states,
            n_actions=n_actions,
        )
        P_arr_list.append(P_arr)
        R_arr_list.append(R_arr)

        support_mask = np.zeros(
            (n_states, n_actions, n_states),
            dtype=bool,
        )
        for s in range(n_states):
            for a in range(n_actions):
                for _, s_next, _, _ in P_source[s][a]:
                    support_mask[s, a, int(s_next)] = True
        support_mask_list.append(support_mask)

    return (
        np.asarray(P_arr_list),
        np.asarray(R_arr_list),
        np.asarray(support_mask_list),
    )


# -----------------------------
# Weights and aggregation
# -----------------------------
def similarity_weights(discrepancies, eps=1e-6, power=1.0):
    discrepancies = np.asarray(discrepancies, dtype=float)
    scores = 1.0 / np.power(discrepancies + eps, power)
    return scores / np.sum(scores)


def aggregate_q_stack(Q_stack, method, weights=None):
    """
    Aggregate one local Q stack.

    Q_stack shape:
        (K, S, A)
    """
    if method == "Maximum-based":
        return np.max(Q_stack, axis=0)

    if method == "Similarity-aware":
        if weights is None:
            raise ValueError("weights must be provided for Similarity-aware.")
        return np.tensordot(weights, Q_stack, axes=(0, 0))

    raise ValueError(f"Unknown method: {method}")


def aggregate_q_trajectories(Q_locals, method, weights=None):
    """
    Aggregate local Q estimates for multiple noise trajectories.

    Q_locals shape:
        (B, K, S, A)

    returns:
        agg_each shape (B, S, A)
    """
    if method == "Maximum-based":
        return np.max(Q_locals, axis=1)

    if method == "Similarity-aware":
        if weights is None:
            raise ValueError("weights must be provided for Similarity-aware.")
        return np.sum(
            Q_locals * weights[None, :, None, None],
            axis=1,
        )

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# Vectorized robust Bellman backup with Bellman noise
# -----------------------------
def exact_l1_worst_case_expectation_batch(
    nominal_probs,
    values,
    l1_radius,
):
    """Exact support-restricted L1 worst-case expectations for a batch."""
    nominal_probs = np.asarray(nominal_probs, dtype=float)
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[1] != len(nominal_probs):
        raise ValueError("values must have shape (B, len(nominal_probs)).")
    if len(nominal_probs) == 0:
        raise ValueError("At least one feasible successor is required.")

    total_probability = float(np.sum(nominal_probs))
    if total_probability <= 0.0:
        raise ValueError("nominal_probs must have positive total mass.")

    batch_size, support_size = values.shape
    q = np.broadcast_to(
        nominal_probs / total_probability,
        (batch_size, support_size),
    ).copy()
    remaining = np.full(
        batch_size,
        min(max(float(l1_radius), 0.0) / 2.0, 1.0),
    )

    low_order = np.argsort(values, axis=1)
    high_order = np.argsort(-values, axis=1)
    low_pointer = np.zeros(batch_size, dtype=int)
    high_pointer = np.zeros(batch_size, dtype=int)
    rows = np.arange(batch_size)
    tolerance = 1e-15

    for _ in range(2 * support_size):
        low = low_order[rows, low_pointer]
        high = high_order[rows, high_pointer]
        active = (
            (remaining > tolerance)
            & (values[rows, low] < values[rows, high] - tolerance)
        )
        if not np.any(active):
            break

        moved = np.minimum.reduce(
            [1.0 - q[rows, low], q[rows, high], remaining]
        )
        moved = np.where(active, moved, 0.0)
        q[rows, low] += moved
        q[rows, high] -= moved
        remaining -= moved

        low_done = active & (1.0 - q[rows, low] <= tolerance)
        high_done = active & (q[rows, high] <= tolerance)
        low_pointer = np.minimum(
            low_pointer + low_done.astype(int),
            support_size - 1,
        )
        high_pointer = np.minimum(
            high_pointer + high_done.astype(int),
            support_size - 1,
        )

    return np.sum(q * values, axis=1)


def vectorized_robust_bellman_backup(
    Q_locals,
    P_sources_arr,
    R_sources_arr,
    feasible_support_mask,
    local_l1_radii,
    discount,
):
    """Exact support-restricted TV/L1 backup for noise trajectories."""
    V = np.max(Q_locals, axis=3)
    batch_size, num_sources, n_states = V.shape
    n_actions = R_sources_arr.shape[2]
    TQ = np.empty(
        (batch_size, num_sources, n_states, n_actions),
        dtype=float,
    )

    for k in range(num_sources):
        for s in range(n_states):
            for a in range(n_actions):
                support = np.flatnonzero(feasible_support_mask[k, s, a])
                worst_future = exact_l1_worst_case_expectation_batch(
                    nominal_probs=P_sources_arr[k, s, a, support],
                    values=V[:, k, support],
                    l1_radius=local_l1_radii[k, s, a],
                )
                TQ[:, k, s, a] = (
                    R_sources_arr[k, s, a]
                    + discount * worst_future
                )

    return TQ


def generate_antithetic_noise(
    rng,
    num_trajectories,
    num_sources,
    n_states,
    n_actions,
):
    """
    Generate antithetic zero-mean Bellman noise across trajectories.

    Shape:
        (B, K, S, A)
    """
    if num_trajectories % 2 != 0:
        raise ValueError("NUM_NOISE_TRAJECTORIES must be even.")

    half = num_trajectories // 2

    noise_half = rng.uniform(
        low=NOISE_LOW,
        high=NOISE_HIGH,
        size=(half, num_sources, n_states, n_actions),
    )

    return np.concatenate([noise_half, -noise_half], axis=0)


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
    Compute the target optimal Q function for oracle normalization.
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
# Trajectory-level selection bias diagnostic
# -----------------------------
def run_selection_bias_trajectory(
    method,
    weights,
    P_sources_arr,
    R_sources_arr,
    feasible_support_mask,
    local_l1_radii,
    noise_level,
    rng,
    num_noise_trajectories,
    iterations,
    stepsize,
    sync_period,
    discount,
    oracle_performance,
    run_id,
    source_seed,
):
    """
    Run noisy Bellman-backup learning and record selection bias at each
    synchronization time.

    Bellman backup noise participates in the learning trajectory.

    At each synchronization time t, before writing the aggregated Q back to
    local domains, compute

        E[ Agg(Q_xi^t) ] - Agg( E[Q_xi^t] ).

    Then the actual per-trajectory aggregated Q is written back, so the
    aggregation rule affects subsequent local updates.
    """
    K, n_states, n_actions, _ = P_sources_arr.shape

    Q_locals = np.zeros(
        (num_noise_trajectories, K, n_states, n_actions),
        dtype=float,
    )

    rows = []

    for iteration in range(iterations):
        noise = generate_antithetic_noise(
            rng=rng,
            num_trajectories=num_noise_trajectories,
            num_sources=K,
            n_states=n_states,
            n_actions=n_actions,
        )

        TQ = vectorized_robust_bellman_backup(
            Q_locals=Q_locals,
            P_sources_arr=P_sources_arr,
            R_sources_arr=R_sources_arr,
            feasible_support_mask=feasible_support_mask,
            local_l1_radii=local_l1_radii,
            discount=discount,
        )

        Q_locals = (
            (1.0 - stepsize) * Q_locals
            + stepsize * (TQ + noise_level * noise)
        )

        if (iteration + 1) % sync_period == 0:
            sync_iteration = iteration + 1

            # E[ Agg(Q_xi^t) ]
            agg_each = aggregate_q_trajectories(
                Q_locals=Q_locals,
                method=method,
                weights=weights,
            )

            mean_agg_of_noisy = np.mean(agg_each, axis=0)

            # Agg( E[Q_xi^t] )
            mean_local_q = np.mean(Q_locals, axis=0)

            agg_of_mean_q = aggregate_q_stack(
                Q_stack=mean_local_q,
                method=method,
                weights=weights,
            )

            # Selection bias caused by the aggregation rule.
            selection_bias = mean_agg_of_noisy - agg_of_mean_q

            signed_selection_bias = float(np.mean(selection_bias))
            positive_selection_bias = float(
                np.mean(np.maximum(selection_bias, 0.0))
            )
            selection_bias_inf_norm = float(
                np.max(np.abs(selection_bias))
            )
            max_positive_selection_bias = float(
                np.max(selection_bias)
            )
            min_signed_selection_bias = float(
                np.min(selection_bias)
            )

            scale = max(abs(float(oracle_performance)), 1e-12)

            rows.append(
                {
                    "run_id": int(run_id),
                    "source_domain_seed": int(source_seed),
                    "iteration": int(sync_iteration),
                    "method": method,
                    "noise_level": float(noise_level),
                    "signed_selection_bias": signed_selection_bias,
                    "positive_selection_bias": positive_selection_bias,
                    "selection_bias_inf_norm": selection_bias_inf_norm,
                    "max_positive_selection_bias": max_positive_selection_bias,
                    "min_signed_selection_bias": min_signed_selection_bias,
                    "signed_selection_bias_percent": (
                        100.0 * signed_selection_bias / scale
                    ),
                    "positive_selection_bias_percent": (
                        100.0 * positive_selection_bias / scale
                    ),
                    "selection_bias_inf_norm_percent": (
                        100.0 * selection_bias_inf_norm / scale
                    ),
                    "max_positive_selection_bias_percent": (
                        100.0 * max_positive_selection_bias / scale
                    ),
                    "min_signed_selection_bias_percent": (
                        100.0 * min_signed_selection_bias / scale
                    ),
                    "oracle_performance": float(oracle_performance),
                    "num_noise_trajectories": int(num_noise_trajectories),
                    "sync_period": int(sync_period),
                    "stepsize": float(stepsize),
                    "discount": float(discount),
                    "selection_bias_definition": (
                        "E[Agg(Q_xi)] - Agg(E[Q_xi])"
                    ),
                    "noise_location": "bellman_backup",
                    "aggregation_is_written_back": True,
                }
            )

            # Actual synchronization:
            # Each noise trajectory writes back its own aggregated Q.
            Q_locals = np.repeat(
                agg_each[:, None, :, :],
                repeats=K,
                axis=1,
            )

    return rows


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

    print("FrozenLake trajectory-level aggregation bias diagnostic")
    print("Map:", MAP_NAME)
    print("Is slippery:", IS_SLIPPERY)
    print("Perturb eps:", PERTURB_EPS_LIST)
    print("Number of source perturbation seeds:", NUM_RUNS)
    print("Source domain base seed:", SOURCE_DOMAIN_BASE_SEED)
    print("Bellman noise base seed:", BELLMAN_NOISE_BASE_SEED)
    print("Diagnostic noise level:", DIAGNOSTIC_NOISE_LEVEL)
    print("Number of noise trajectories:", NUM_NOISE_TRAJECTORIES)
    print("Uncertainty distance:", UNCERTAINTY_DISTANCE)
    print("Robust backup:", ROBUST_BACKUP_TYPE)
    print("Target oracle performance:", oracle_performance)
    print("Selection bias: E[Agg(Q_xi)] - Agg(E[Q_xi])")

    all_rows = []

    total_tasks = NUM_RUNS * len(METHOD_ORDER)

    with tqdm(
        total=total_tasks,
        desc="Trajectory selection-bias diagnostic",
    ) as pbar:
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

            pbar.write(
                f"Run {run_id:02d} | "
                f"eps={np.array2string(PERTURB_EPS_LIST, precision=3)} | "
                f"empirical max L1 Gamma="
                f"{np.array2string(empirical_gammas, precision=4)} | "
                f"mean local L1 radius="
                f"{np.array2string(mean_local_radii, precision=4)}"
            )

            (
                P_sources_arr,
                R_sources_arr,
                feasible_support_mask,
            ) = convert_source_dicts_to_arrays(
                P_sources=P_sources,
                n_states=n_states,
                n_actions=n_actions,
            )

            # Similarity weights use empirical max L1 Gamma. Since TV = L1 / 2,
            # normalized inverse-distance weights are equivalent up to scaling.
            w_sim = similarity_weights(
                empirical_gammas,
                eps=SIMILARITY_EPS,
                power=SIMILARITY_POWER,
            )

            for method_name in METHOD_ORDER:
                if method_name == "Similarity-aware":
                    weights = w_sim
                else:
                    weights = None

                noise_seed = BELLMAN_NOISE_BASE_SEED + 1000 * run_id

                # Use the same Bellman-noise trajectories for both methods
                # under the same source seed.
                rng = np.random.default_rng(noise_seed)

                rows = run_selection_bias_trajectory(
                    method=method_name,
                    weights=weights,
                    P_sources_arr=P_sources_arr,
                    R_sources_arr=R_sources_arr,
                    feasible_support_mask=feasible_support_mask,
                    local_l1_radii=local_l1_radii,
                    noise_level=DIAGNOSTIC_NOISE_LEVEL,
                    rng=rng,
                    num_noise_trajectories=NUM_NOISE_TRAJECTORIES,
                    iterations=ITERATIONS,
                    stepsize=STEPSIZE,
                    sync_period=SYNC_PERIOD,
                    discount=DISCOUNT,
                    oracle_performance=oracle_performance,
                    run_id=run_id,
                    source_seed=source_seed,
                )

                for row in rows:
                    row.update(
                        {
                            "map_name": MAP_NAME,
                            "target_domain": "fixed",
                            "source_randomness": "perturbation_seed",
                            "bellman_noise_seed": int(noise_seed),
                            "source_domain_base_seed": int(
                                SOURCE_DOMAIN_BASE_SEED
                            ),
                            "bellman_noise_base_seed": int(
                                BELLMAN_NOISE_BASE_SEED
                            ),
                            "uncertainty_distance": UNCERTAINTY_DISTANCE,
                            "robust_backup_type": ROBUST_BACKUP_TYPE,
                        }
                    )

                all_rows.extend(rows)

                pbar.set_postfix(
                    run=run_id,
                    method=method_name,
                )
                pbar.update(1)

            # Source metadata.
            for k in range(len(P_sources)):
                all_rows.append(
                    {
                        "run_id": int(run_id),
                        "source_domain_seed": int(source_seed),
                        "iteration": -1,
                        "method": "metadata",
                        "source_index": int(k),
                        "perturb_epsilon": float(PERTURB_EPS_LIST[k]),
                        "empirical_gamma_l1": float(empirical_gammas[k]),
                        "mean_local_l1_radius": float(mean_local_radii[k]),
                        "similarity_weight": float(w_sim[k]),
                        "oracle_performance": float(oracle_performance),
                        "map_name": MAP_NAME,
                        "discount": float(DISCOUNT),
                        "stepsize": float(STEPSIZE),
                        "sync_period": int(SYNC_PERIOD),
                        "num_noise_trajectories": int(NUM_NOISE_TRAJECTORIES),
                        "target_domain": "fixed",
                        "source_randomness": "perturbation_seed",
                        "uncertainty_distance": UNCERTAINTY_DISTANCE,
                        "robust_backup_type": ROBUST_BACKUP_TYPE,
                    }
                )

    df = pd.DataFrame(all_rows)

    csv_path = RESULT_DIR / "Frozenlake_exp4.csv"
    df.to_csv(csv_path, index=False)

    # -----------------------------
    # Plot
    # -----------------------------
    perf = df[df["method"].isin(METHOD_ORDER)].copy()
    perf = perf[perf["iteration"].astype(int) >= 0].copy()

    iterations = np.sort(perf["iteration"].unique().astype(int))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for method_name in METHOD_ORDER:
        method_df = perf[perf["method"] == method_name].copy()

        mean_curve = []
        sem_curve = []

        for t in iterations:
            values = method_df[
                method_df["iteration"].astype(int) == int(t)
            ][PLOT_METRIC].to_numpy(dtype=float)

            mean_y, sem_y = mean_and_sem(values, axis=0)

            mean_curve.append(float(mean_y))
            sem_curve.append(float(sem_y))

        mean_curve = np.asarray(mean_curve, dtype=float)
        sem_curve = np.asarray(sem_curve, dtype=float)

        style = PLOT_STYLES[method_name]

        ax.plot(
            iterations,
            mean_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=LINE_WIDTH,
            label=method_name,
        )

        ax.fill_between(
            iterations,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=style["color"],
            alpha=SHADE_ALPHA,
            linewidth=0.0,
        )

    ax.set_xlabel(r"$t$")

    if PLOT_METRIC == "signed_selection_bias":
        ax.set_ylabel(r"$\mu(t)$")
    elif PLOT_METRIC == "selection_bias_inf_norm":
        ax.set_ylabel(r"Selection bias magnitude")
    elif PLOT_METRIC == "signed_selection_bias_percent":
        ax.set_ylabel(r"$\mu(t)$ (%)")
    elif PLOT_METRIC == "selection_bias_inf_norm_percent":
        ax.set_ylabel(r"Selection bias magnitude (%)")
    else:
        ax.set_ylabel(PLOT_METRIC)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)

    ax.legend(loc="upper right")

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    ax.set_ylim(
        ymin - 0.05 * yrange,
        ymax + 0.10 * yrange,
    )

    set_clean_yticks_keep_limits(ax, nbins=6)

    fig.tight_layout()

    pdf_path = FIGURE_DIR / "Frozenlake_exp4.pdf"
    png_path = FIGURE_DIR / "Frozenlake_exp4.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)

    print("FrozenLake trajectory-level aggregation bias diagnostic finished.")
    print(f"CSV saved to: {csv_path}")
    print(f"PDF saved to: {pdf_path}")
    print(f"PNG saved to: {png_path}")

    print("\nFinal-time summary:")
    final_t = int(iterations[-1])

    for method_name in METHOD_ORDER:
        values = perf[
            (perf["method"] == method_name)
            & (perf["iteration"].astype(int) == final_t)
        ][PLOT_METRIC].to_numpy(dtype=float)

        mean_y, sem_y = mean_and_sem(values, axis=0)

        print(
            f"{method_name}: "
            f"t={final_t}, "
            f"{PLOT_METRIC} mean={mean_y:.8f}, "
            f"sem={sem_y:.8f}, "
            f"n={len(values)}"
        )


if __name__ == "__main__":
    main()