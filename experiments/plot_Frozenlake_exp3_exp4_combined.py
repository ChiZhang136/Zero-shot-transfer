"""Create a vertically stacked, paper-ready FrozenLake Exp3/Exp4 figure."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["Maximum-based", "Similarity-aware"]
STYLES = {"Maximum-based": {"color": "C2"}, "Similarity-aware": {"color": "C0"}}

def mean_sem(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)

def load_exp3():
    d = pd.read_csv(ROOT / "results/Frozenlake_exp3/Frozenlake_exp3.csv")
    xcol = "noise_level" if "noise_level" in d else "bias_level"
    d = d[d.method.isin(METHODS) & (d[xcol] >= 0)].copy()
    xs = np.sort(d[xcol].unique())
    return xs, d, xcol

def load_exp4():
    d = pd.read_csv(ROOT / "results/Frozenlake_exp4/Frozenlake_exp4.csv")
    d = d[d.method.isin(METHODS) & (d.iteration >= 0)].copy()
    ts = np.sort(d.iteration.unique().astype(int))
    return ts, d

def main():
    x, d3, xcol = load_exp3(); t, d4 = load_exp4()
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.5, 4.2), sharey=False)
    for method in METHODS:
        ys, es = [], []
        for z in x:
            m, e = mean_sem(100*d3[(d3.method == method) & np.isclose(d3[xcol], z)].normalized_performance)
            ys.append(m); es.append(e)
        a.plot(x, ys, color=STYLES[method]["color"], lw=2, marker="o", label=method)
        a.fill_between(x, np.array(ys)-es, np.array(ys)+es, color=STYLES[method]["color"], alpha=.15)
        ys, es = [], []
        for z in t:
            m, e = mean_sem(d4[(d4.method == method) & (d4.iteration == z)].signed_selection_bias)
            ys.append(m); es.append(e)
        b.plot(t, ys, color=STYLES[method]["color"], lw=2, label=method)
        b.fill_between(t, np.array(ys)-es, np.array(ys)+es, color=STYLES[method]["color"], alpha=.15)
    a.set_xlabel(r"$\delta$"); a.set_ylabel(r"$\nu(T)$ (%)")
    b.set_xlabel(r"$t$"); b.set_ylabel(r"$\mu(t)$")
    for ax in (a, b): ax.grid(alpha=.25); ax.set_axisbelow(True)
    a.legend(loc="lower left"); b.legend(loc="center right")
    fig.tight_layout(h_pad=1.2)
    out = ROOT / "figures/Frozenlake_exp3_exp4_combined"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "Frozenlake_exp3_exp4_combined.pdf", bbox_inches="tight"); fig.savefig(out / "Frozenlake_exp3_exp4_combined.png", dpi=300, bbox_inches="tight")
    print(f"Saved combined figure to {out}")

if __name__ == "__main__": main()
