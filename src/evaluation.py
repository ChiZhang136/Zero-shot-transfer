import numpy as np


def greedy_policy(Q):
    """
    Greedy deterministic policy induced by Q.
    """
    return np.argmax(Q, axis=1)


def evaluate_policy(P, rewards, policy, discount):
    """
    Exact policy evaluation under transition kernel P.

    Returns:
        V: state-value vector.
        avg_value: average value over all states.

    The scalar performance is

        (1 / |S|) sum_s V(s),

    which is used because Garnet MDPs do not prescribe a particular
    initial state.
    """
    S = P.shape[0]

    P_pi = P[np.arange(S), policy, :]
    r_pi = rewards[np.arange(S), policy]

    V = np.linalg.solve(np.eye(S) - discount * P_pi, r_pi)

    avg_value = float(np.mean(V))

    return V, avg_value


def target_value_iteration(P, rewards, discount, tol=1e-10, max_iter=10000):
    """
    Standard value iteration on the target MDP.

    Used only to compute the target oracle benchmark.
    """
    S, A = rewards.shape
    Q = np.zeros((S, A))

    for it in range(max_iter):
        V = np.max(Q, axis=1)

        Q_new = np.zeros_like(Q)

        for s in range(S):
            for a in range(A):
                Q_new[s, a] = rewards[s, a] + discount * np.dot(P[s, a, :], V)

        diff = np.max(np.abs(Q_new - Q))
        Q = Q_new

        if diff < tol:
            return Q, it + 1, diff

    return Q, max_iter, diff