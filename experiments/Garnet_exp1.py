import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms import run_periodic_learning
from src.evaluation import evaluate_policy, greedy_policy, target_value_iteration
from src.garnet import generate_sources_from_target, generate_target_garnet
from src.utils import ensure_dir, mean_and_sem, similarity_weights, uniform_weights


METHOD_ORDER = [
    "Maximum-based",
    "Similarity-aware",
    "Uniform",
    "Non-robust DR",
]

PLOT_STYLES = {
    "Maximum-based": {"color": "C2", "linestyle": "-"},
    "Similarity-aware": {"color": "C0", "linestyle": "-"},
    "Uniform": {"color": "C1", "linestyle": "-"},
    "Non-robust DR": {"color": "C4", "linestyle": "-"},
}


def compute_local_transition_discrepancies(P_sources, P0, p_norm=1):
    """Compute ||P_k(.|s,a) - P_0(.|s,a)||_p for every source and (s,a)."""
    P_sources = np.asarray(P_sources, dtype=float)
    P0 = np.asarray(P0, dtype=float)
    diff = P_sources - P0[None, :, :, :]

    if p_norm == 1:
        return np.sum(np.abs(diff), axis=-1)
    if p_norm == 2:
        return np.sqrt(np.sum(diff**2, axis=-1))
    if p_norm == np.inf or p_norm == "inf":
        return np.max(np.abs(diff), axis=-1)
    return np.linalg.norm(diff, ord=p_norm, axis=-1)


def set_integer_yticks_keep_limits(ax):
    """Use integer y-axis ticks without changing the current limits."""
    ymin, ymax = ax.get_ylim()
    ticks = np.arange(np.ceil(ymin), np.floor(ymax) + 1, 1.0)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.set_ylim(ymin, ymax)


def main():
    # Experiment parameters (kept identical to Garnet_exp1.py).
    num_states = 30
    num_actions = 4
    branching_factor = 3
    reward_range = (0.0, 1.0)
    discount = 0.95

    source_gammas = np.array([0.30, 0.60, 0.90, 1.20, 1.50, 1.80])
    num_sources = len(source_gammas)

    p_norm = 1
    robust_q = "inf"
    total_iterations = 300
    eval_every = 5
    sync_period = 5
    stepsize = 0.05
    num_seeds = 20

    result_dir = PROJECT_ROOT / "results" / "Garnet_exp1"
    figure_dir = PROJECT_ROOT / "figures" / "Garnet_exp1"
    ensure_dir(result_dir)
    ensure_dir(figure_dir)

    all_rows = []
    method_curves = {method: [] for method in METHOD_ORDER}
    saved_iterations = None

    progress = tqdm(
        total=num_seeds * len(METHOD_ORDER),
        desc="Garnet Experiment 1 (4 algorithms)",
        unit="run",
        dynamic_ncols=True,
        mininterval=0.1,
        file=sys.stdout,
    )

    for seed in range(num_seeds):
        target_mdp = generate_target_garnet(
            num_states=num_states,
            num_actions=num_actions,
            branching_factor=branching_factor,
            reward_range=reward_range,
            discount=discount,
            seed=1000 + seed,
        )

        P_sources, actual_source_gammas, rhos = generate_sources_from_target(
            target_mdp=target_mdp,
            source_gammas=source_gammas,
            branching_factor=branching_factor,
            seed=2000 + seed,
            p_norm=p_norm,
        )

        rewards = target_mdp.rewards
        P0 = target_mdp.transitions

        local_source_gammas = compute_local_transition_discrepancies(
            P_sources=P_sources,
            P0=P0,
            p_norm=p_norm,
        )
        local_gamma_max = np.max(local_source_gammas, axis=(1, 2))
        local_gamma_mean = np.mean(local_source_gammas, axis=(1, 2))

        w_sim = similarity_weights(actual_source_gammas, eps=1e-6, power=1.0)
        w_uni = uniform_weights(num_sources)

        Q_star, _, _ = target_value_iteration(P0, rewards, discount)
        pi_star = greedy_policy(Q_star)
        _, oracle_perf = evaluate_policy(P0, rewards, pi_star, discount)

        method_configs = {
            "Maximum-based": {
                "weights": None,
                "aggregation_type": "max",
                "gammas": local_source_gammas,
                "uses_robust_term": True,
            },
            "Similarity-aware": {
                "weights": w_sim,
                "aggregation_type": "weighted",
                "gammas": local_source_gammas,
                "uses_robust_term": True,
            },
            "Uniform": {
                "weights": w_uni,
                "aggregation_type": "weighted",
                "gammas": local_source_gammas,
                "uses_robust_term": True,
            },
            "Non-robust DR": {
                "weights": w_uni,
                "aggregation_type": "weighted",
                # A zero radius removes the robust penalty while retaining
                # the same uniform aggregation and synchronization scheme.
                "gammas": np.zeros_like(local_source_gammas),
                "uses_robust_term": False,
            },
        }

        for method_name in METHOD_ORDER:
            config = method_configs[method_name]
            _, hist = run_periodic_learning(
                P_sources=P_sources,
                rewards=rewards,
                gammas=config["gammas"],
                weights=config["weights"],
                discount=discount,
                target_P=P0,
                total_iterations=total_iterations,
                stepsize=stepsize,
                sync_period=sync_period,
                eval_every=eval_every,
                q=robust_q,
                aggregation_type=config["aggregation_type"],
            )

            iterations = np.asarray(hist["iterations"])
            saved_iterations = iterations
            perf = np.asarray(hist["target_performance"])
            norm_perf = perf / oracle_perf
            method_curves[method_name].append(norm_perf)

            for i, t in enumerate(iterations):
                all_rows.append(
                    {
                        "seed": seed,
                        "iteration": int(t),
                        "method": method_name,
                        "target_performance": float(perf[i]),
                        "normalized_performance": float(norm_perf[i]),
                        "oracle_performance": float(oracle_perf),
                        "uses_robust_term": bool(config["uses_robust_term"]),
                        "penalty_type": (
                            "state_action_local"
                            if config["uses_robust_term"]
                            else "none"
                        ),
                        "p_norm": str(p_norm),
                        "robust_q": robust_q,
                        "sync_period": int(sync_period),
                        "stepsize": float(stepsize),
                    }
                )

            progress.set_postfix(seed=seed, method=method_name, refresh=False)
            progress.update(1)

        for k in range(num_sources):
            all_rows.append(
                {
                    "seed": seed,
                    "iteration": -1,
                    "method": "metadata",
                    "source_index": k,
                    "source_gamma": float(source_gammas[k]),
                    "actual_source_gamma": float(actual_source_gammas[k]),
                    "local_gamma_max": float(local_gamma_max[k]),
                    "local_gamma_mean": float(local_gamma_mean[k]),
                    "mixing_coefficient": float(rhos[k]),
                    "uniform_weight": float(w_uni[k]),
                    "similarity_weight": float(w_sim[k]),
                    "penalty_type": "state_action_local",
                    "p_norm": str(p_norm),
                    "robust_q": robust_q,
                    "sync_period": int(sync_period),
                    "stepsize": float(stepsize),
                }
            )

    progress.close()

    df = pd.DataFrame(all_rows)
    result_path = result_dir / "Garnet_exp1.csv"
    df.to_csv(result_path, index=False)

    curve_stats = {}
    for method_name in METHOD_ORDER:
        curves = np.asarray(method_curves[method_name])
        mean_curve, sem_curve = mean_and_sem(curves, axis=0)
        curve_stats[method_name] = {"mean": mean_curve, "sem": sem_curve}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method_name in METHOD_ORDER:
        mean_curve = 100.0 * curve_stats[method_name]["mean"]
        sem_curve = 100.0 * curve_stats[method_name]["sem"]
        style = PLOT_STYLES[method_name]

        ax.plot(
            saved_iterations,
            mean_curve,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            label=method_name,
        )
        ax.fill_between(
            saved_iterations,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=style["color"],
            alpha=0.15,
            linewidth=0.0,
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\nu(t)$ (%)")
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.legend(loc="lower right")
    set_integer_yticks_keep_limits(ax)
    fig.tight_layout()

    figure_pdf_path = figure_dir / "Garnet_exp1.pdf"
    figure_png_path = figure_dir / "Garnet_exp1.png"
    fig.savefig(figure_pdf_path)
    fig.savefig(figure_png_path, dpi=300)
    plt.close(fig)

    print("Garnet Experiment 1 (4 algorithms) finished.")
    print("Non-robust DR: uniform averaging with zero robust radius.")
    print(f"Results saved to: {result_path}")
    print(f"Figure saved to: {figure_pdf_path}")


if __name__ == "__main__":
    main()
