import numpy as np

from .robust_backup import lp_regularized_worst_case_expectation


def robust_bellman_optimal(P, rewards, radius, discount, Q, q="inf"):
    """
    Local robust Bellman optimality operator for one source domain:

        T_k Q(s,a)
        =
        r(s,a)
        +
        gamma * [ P_k(.|s,a)^T V_Q - beta_k * kappa_q(V_Q) ],

    where

        V_Q(s) = max_a Q(s,a).

    Here radius is beta_k / Gamma_k.
    """
    S, A = rewards.shape
    V = np.max(Q, axis=1)

    TQ = np.zeros_like(Q)

    for s in range(S):
        for a in range(A):
            sigma = lp_regularized_worst_case_expectation(
                p=P[s, a, :],
                V=V,
                beta=radius,
                q=q,
            )

            TQ[s, a] = rewards[s, a] + discount * sigma

    return TQ


def weighted_aggregate(Q_locals, weights):
    """
    Compute

        Q = sum_k w_k Q_k.

    Q_locals shape: (K, S, A)
    weights shape: (K,)
    """
    return np.tensordot(weights, Q_locals, axes=(0, 0))


def similarity_aware_operator(
    P_sources,
    rewards,
    gammas,
    weights,
    discount,
    Q,
    q="inf",
):
    """
    Similarity-aware weighted robust Bellman operator:

        T_S Q = sum_k w_k T_k Q.
    """
    K = P_sources.shape[0]

    TQ = np.zeros_like(Q)

    for k in range(K):
        TQ += weights[k] * robust_bellman_optimal(
            P=P_sources[k],
            rewards=rewards,
            radius=gammas[k],
            discount=discount,
            Q=Q,
            q=q,
        )

    return TQ


def compute_QS(
    P_sources,
    rewards,
    gammas,
    weights,
    discount,
    q="inf",
    tol=1e-10,
    max_iter=10000,
):
    """
    Compute the fixed point

        Q_S = T_S Q_S

    by centralized value iteration.

    This is used only as a reference solution in experiments.
    """
    S, A = rewards.shape
    Q = np.zeros((S, A))

    for it in range(max_iter):
        Q_new = similarity_aware_operator(
            P_sources=P_sources,
            rewards=rewards,
            gammas=gammas,
            weights=weights,
            discount=discount,
            Q=Q,
            q=q,
        )

        diff = np.max(np.abs(Q_new - Q))
        Q = Q_new

        if diff < tol:
            return Q, it + 1, diff

    return Q, max_iter, diff