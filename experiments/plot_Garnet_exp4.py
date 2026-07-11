from pathlib import Path
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "results" / "Garnet_exp4" / "Garnet_exp4.csv"
FIGURE_DIR = PROJECT_ROOT / "figures" / "Garnet_exp4"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FIGSIZE = (6.5, 4.2)
LINE_WIDTH = 2.0
SHADE_ALPHA = 0.15
GRID_ALPHA = 0.25
MAX_COLOR, SIM_COLOR = "C2", "C0"
METHOD_ORDER = ["Maximum-based", "Similarity-aware"]
PLOT_STYLES = {
    "Maximum-based": {"color": MAX_COLOR, "linestyle": "-"},
    "Similarity-aware": {"color": SIM_COLOR, "linestyle": "-"},
}
PLOT_METRIC = "signed_selection_bias"

def set_clean_yticks_keep_limits(ax, nbins=6):
    ymin, ymax = ax.get_ylim()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.set_ylim(ymin, ymax)

def mean_and_sem(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return float(np.mean(x)), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))

def get_ylabel(metric):
    return r"$\mu(t)$" if metric == "signed_selection_bias" else metric

def main():
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Result file not found: {RESULT_PATH}")
    df = pd.read_csv(RESULT_PATH)
    required = {"iteration", "method", PLOT_METRIC}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    perf = df[df["method"].isin(METHOD_ORDER)].dropna(subset=["iteration", PLOT_METRIC])
    perf = perf[perf["iteration"].astype(int) >= 0].copy()
    iterations = np.sort(perf["iteration"].unique().astype(int))
    if len(iterations) == 0:
        raise ValueError("No valid diagnostic rows found.")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for method in METHOD_ORDER:
        method_df = perf[perf["method"] == method]
        means, sems = [], []
        for t in iterations:
            values = method_df[method_df["iteration"].astype(int) == t][PLOT_METRIC].to_numpy(float)
            mean, sem = mean_and_sem(values)
            means.append(mean); sems.append(sem)
        means, sems = np.asarray(means), np.asarray(sems)
        style = PLOT_STYLES[method]
        ax.plot(iterations, means, color=style["color"], linestyle=style["linestyle"], linewidth=LINE_WIDTH, label=method)
        ax.fill_between(iterations, means-sems, means+sems, color=style["color"], alpha=SHADE_ALPHA, linewidth=0.0)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(get_ylabel(PLOT_METRIC))
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", alpha=GRID_ALPHA)
    ax.legend(loc="center right")
    ymin, ymax = ax.get_ylim()
    yrange = ymax-ymin if ymax-ymin > 1e-12 else max(abs(ymax), 1.0)
    ax.set_ylim(ymin-.05*yrange, ymax+.10*yrange)
    set_clean_yticks_keep_limits(ax, nbins=6)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "Garnet_exp4.pdf")
    fig.savefig(FIGURE_DIR / "Garnet_exp4.png", dpi=300)

if __name__ == "__main__":
    main()
