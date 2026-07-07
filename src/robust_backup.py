import numpy as np


def kappa_inf(V):
    """
    kappa_infty(V) = min_omega ||V - omega 1||_infty
                   = (max_s V(s) - min_s V(s)) / 2.

    This corresponds to the q-variance term when the transition
    uncertainty is constrained by an L1 norm.
    """
    V = np.asarray(V, dtype=float)
    return 0.5 * (np.max(V) - np.min(V))


def kappa_2(V):
    """
    kappa_2(V) = min_omega ||V - omega 1||_2
               = ||V - mean(V) 1||_2.
    """
    V = np.asarray(V, dtype=float)
    return np.linalg.norm(V - np.mean(V), ord=2)


def kappa_1(V):
    """
    kappa_1(V) = min_omega ||V - omega 1||_1.

    A median minimizes the L1 distance.
    """
    V = np.asarray(V, dtype=float)
    med = np.median(V)
    return np.sum(np.abs(V - med))


def kappa_q(V, q="inf"):
    """
    Compute

        kappa_q(V) = min_omega ||V - omega 1||_q

    for q in {1, 2, infinity}.
    """
    if q == "inf" or q == np.inf:
        return kappa_inf(V)

    if q == 2 or q == "2":
        return kappa_2(V)

    if q == 1 or q == "1":
        return kappa_1(V)

    raise ValueError("Only q in {'inf', 1, 2} is currently implemented.")


def lp_regularized_worst_case_expectation(p, V, beta, q="inf"):
    """
    Robust expectation induced by the Lp-regularized robust MDP formulation:

        min_{kernel noise} (p + noise)^T V
        =
        p^T V - beta * kappa_q(V),

    where beta is the transition uncertainty radius, and q is the
    conjugate norm of the uncertainty norm p.

    In the default experiment, p_norm = 1 and q = infinity.
    """
    p = np.asarray(p, dtype=float)
    V = np.asarray(V, dtype=float)

    return float(np.dot(p, V) - beta * kappa_q(V, q=q))