from dataclasses import dataclass
import numpy as np


@dataclass
class GarnetMDP:
    transitions: np.ndarray  # shape: (S, A, S)
    rewards: np.ndarray      # shape: (S, A)
    discount: float

    @property
    def S(self):
        return self.transitions.shape[0]

    @property
    def A(self):
        return self.transitions.shape[1]


def random_garnet_kernel(num_states, num_actions, branching_factor, rng):
    """
    Generate a Garnet transition kernel.

    For each state-action pair (s,a), only B next states have nonzero
    probabilities.
    """
    S = num_states
    A = num_actions
    B = min(branching_factor, num_states)

    P = np.zeros((S, A, S))

    for s in range(S):
        for a in range(A):
            next_states = rng.choice(S, size=B, replace=False)
            probs = rng.dirichlet(np.ones(B))
            P[s, a, next_states] = probs

    return P


def generate_target_garnet(
    num_states,
    num_actions,
    branching_factor,
    reward_range=(0.0, 1.0),
    discount=0.95,
    seed=0,
):
    """
    Generate a target Garnet MDP.
    """
    rng = np.random.default_rng(seed)

    P0 = random_garnet_kernel(
        num_states=num_states,
        num_actions=num_actions,
        branching_factor=branching_factor,
        rng=rng,
    )

    rewards = rng.uniform(
        reward_range[0],
        reward_range[1],
        size=(num_states, num_actions),
    )

    return GarnetMDP(P0, rewards, discount)


def max_lp_distance(P, Q, p_norm=1):
    """
    Compute

        max_{s,a} ||P(.|s,a) - Q(.|s,a)||_p.

    This discrepancy is used as Gamma_k / beta_k in the Lp robust
    Bellman operator.

    p_norm = 1      corresponds to q = infinity.
    p_norm = 2      corresponds to q = 2.
    p_norm = np.inf corresponds to q = 1.
    """
    diff = P - Q

    if p_norm == 1 or p_norm == "1":
        distances = np.sum(np.abs(diff), axis=-1)
    elif p_norm == 2 or p_norm == "2":
        distances = np.linalg.norm(diff, ord=2, axis=-1)
    elif p_norm == np.inf or p_norm == "inf":
        distances = np.max(np.abs(diff), axis=-1)
    else:
        raise ValueError("Only p_norm in {1, 2, np.inf, 'inf'} is supported.")

    return float(np.max(distances))


def max_tv_distance(P, Q):
    """
    Compute

        max_{s,a} d_TV(P(.|s,a), Q(.|s,a)),

    where d_TV(p,q) = 0.5 * ||p-q||_1.

    This function is kept only for diagnostic purposes. In the current
    Lp-regularized robust backup, we use max_lp_distance instead.
    """
    return 0.5 * max_lp_distance(P, Q, p_norm=1)


def generate_sources_from_target(
    target_mdp,
    source_gammas,
    branching_factor,
    seed=1,
    p_norm=1,
    max_resample=100,
):
    """
    Generate source domains with prescribed source-target discrepancy levels.

    For each prescribed Gamma_k, we first sample a random Garnet kernel
    P_tilde_k, then construct

        P_k = (1 - rho_k) P_0 + rho_k P_tilde_k,

    where

        rho_k = Gamma_k / max_{s,a} ||P_tilde_k(.|s,a) - P_0(.|s,a)||_p.

    Therefore,

        max_{s,a} ||P_k(.|s,a) - P_0(.|s,a)||_p = Gamma_k.

    For the default q = infinity robust backup, use p_norm = 1.
    """
    rng = np.random.default_rng(seed)

    P0 = target_mdp.transitions
    S = target_mdp.S
    A = target_mdp.A

    sources = []
    actual_source_gammas = []

    mixing_coefficients = []

    for gamma_k in source_gammas:
        gamma_k = float(gamma_k)

        for _ in range(max_resample):
            P_rand = random_garnet_kernel(
                num_states=S,
                num_actions=A,
                branching_factor=branching_factor,
                rng=rng,
            )

            base_distance = max_lp_distance(P_rand, P0, p_norm=p_norm)

            if base_distance >= gamma_k and base_distance > 0:
                break
        else:
            raise RuntimeError(
                f"Could not generate a candidate source with distance at least "
                f"Gamma_k={gamma_k}. Try using a smaller Gamma_k or increasing "
                f"max_resample."
            )

        rho_k = gamma_k / base_distance

        Pk = (1.0 - rho_k) * P0 + rho_k * P_rand

        actual_gamma_k = max_lp_distance(Pk, P0, p_norm=p_norm)

        sources.append(Pk)
        actual_source_gammas.append(actual_gamma_k)
        mixing_coefficients.append(rho_k)

    return (
        np.asarray(sources),
        np.asarray(actual_source_gammas),
        np.asarray(mixing_coefficients),
    )