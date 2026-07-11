import numpy as np

from .evaluation import greedy_policy, evaluate_policy


def _kappa_q(V, q="inf"):
    """
    Compute kappa_q(V) = min_c ||V - c 1||_q.

    In the current experiments, q = "inf", hence
        kappa_inf(V) = (max_s V(s) - min_s V(s)) / 2.
    """
    V = np.asarray(V, dtype=float)

    if q == "inf" or q == np.inf:
        return 0.5 * (np.max(V) - np.min(V))

    if q == 2 or q == "2":
        return np.linalg.norm(V - np.mean(V), ord=2)

    if q == 1 or q == "1":
        median = np.median(V)
        return np.sum(np.abs(V - median))

    raise ValueError("Only q in {'inf', 1, 2} is supported.")


def robust_bellman_optimal_source_radius(
    P,
    rewards,
    radius,
    discount,
    Q,
    q="inf",
):
    """
    Robust Bellman optimality backup for an sa-rectangular L_p uncertainty set.

    The current experiments use source-level radii:
        radius = Gamma_k,

    but this function also supports state-action dependent radii:
        radius.shape = (S, A).

    For q = inf and L1 transition uncertainty:
        T_k Q(s,a)
        =
        r(s,a)
        + gamma [P_k(.|s,a)^T V_Q - radius * kappa_inf(V_Q)].
    """
    V = np.max(Q, axis=1)
    kappa = _kappa_q(V, q=q)

    expected_value = np.einsum("sat,t->sa", P, V)

    return rewards + discount * (expected_value - radius * kappa)


def aggregate_q_tables(Q_locals, weights=None, aggregation_type="weighted"):
    """
    Aggregate local Q-tables.

    aggregation_type:
        "weighted": weighted average aggregation
        "max"     : element-wise maximum aggregation
    """
    if aggregation_type == "weighted":
        if weights is None:
            raise ValueError("weights must be provided for weighted aggregation.")

        weights = np.asarray(weights, dtype=float)
        return np.tensordot(weights, Q_locals, axes=(0, 0))

    if aggregation_type == "max":
        return np.max(Q_locals, axis=0)

    raise ValueError("aggregation_type must be either 'weighted' or 'max'.")


def run_periodic_learning(
    P_sources,
    rewards,
    gammas,
    weights=None,
    discount=0.95,
    target_P=None,
    total_iterations=500,
    stepsize=1.0,
    sync_period=1,
    eval_every=5,
    q="inf",
    aggregation_type="weighted",
):
    """
    Periodic multi-source robust learning.

    Each source domain performs local robust Bellman updates. Every E steps,
    the local Q-tables are aggregated and synchronized.

    aggregation_type:
        "weighted":
            Q <- sum_k w_k Q_k.
            This includes uniform and similarity-aware aggregation.

        "max":
            Q <- max_k Q_k element-wise.
            This corresponds to maximum-based aggregation.

    If target_P is provided, the greedy policy induced by the current
    aggregated Q-table is evaluated on the target domain.
    """
    K, S, A, _ = P_sources.shape

    gammas = np.asarray(gammas, dtype=float)

    if aggregation_type == "weighted":
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != K:
            raise ValueError("weights must have length K.")

    Q_locals = np.zeros((K, S, A))

    history = {
        "iterations": [],
        "target_performance": [],
        "max_selection_counts": np.zeros(K, dtype=int),
    }

    for t in range(total_iterations):
        # -----------------------------
        # Local robust Bellman updates
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

            Q_locals[k] = (1.0 - stepsize) * Q_locals[k] + stepsize * TQ

        # -----------------------------
        # Synchronization
        # -----------------------------
        if (t + 1) % sync_period == 0:
            if aggregation_type == "max":
                selected = np.argmax(Q_locals, axis=0).ravel()
                history["max_selection_counts"] += np.bincount(selected, minlength=K)
            Q_agg = aggregate_q_tables(
                Q_locals,
                weights=weights,
                aggregation_type=aggregation_type,
            )
            Q_locals[:] = Q_agg[None, :, :]
        else:
            # Virtual aggregate used only for evaluation.
            Q_agg = aggregate_q_tables(
                Q_locals,
                weights=weights,
                aggregation_type=aggregation_type,
            )

        # -----------------------------
        # Target-domain evaluation
        # -----------------------------
        if target_P is not None and ((t + 1) % eval_every == 0 or t == 0):
            policy = greedy_policy(Q_agg)
            _, perf = evaluate_policy(target_P, rewards, policy, discount)

            history["iterations"].append(t + 1)
            history["target_performance"].append(perf)

    Q_final = aggregate_q_tables(
        Q_locals,
        weights=weights,
        aggregation_type=aggregation_type,
    )

    return Q_final, history


def run_periodic_weighted_robust_learning(
    P_sources,
    rewards,
    gammas,
    weights,
    discount,
    target_P=None,
    total_iterations=500,
    stepsize=1.0,
    sync_period=1,
    eval_every=5,
    q="inf",
):
    """
    Backward-compatible wrapper for weighted robust aggregation.
    """
    return run_periodic_learning(
        P_sources=P_sources,
        rewards=rewards,
        gammas=gammas,
        weights=weights,
        discount=discount,
        target_P=target_P,
        total_iterations=total_iterations,
        stepsize=stepsize,
        sync_period=sync_period,
        eval_every=eval_every,
        q=q,
        aggregation_type="weighted",
    )