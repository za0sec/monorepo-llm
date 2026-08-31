"""
Gráficos del Experimento 1 -- lee los CSV que genera run_experiment1.py,
no recalcula ni reentrena nada (separación cómputo/gráficos, ver
CLAUDE.md).

Genera output/experiment1_curves.png: PR-AUC y ROC-AUC por época,
promediadas sobre las 3 semillas (± desvío estándar), train vs. valid --
para diagnosticar over/underfitting comparando ambas curvas.
"""
import glob
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "output"

# Paleta categórica (ver dataviz skill): slot 1 = train, slot 2 = valid.
COLOR_TRAIN = "#2a78d6"
COLOR_VALID = "#eb6834"
PREVALENCE = 0.1301  # tasa global de `bought`, ver ejercicio2/Notas.md -- referencia de un clasificador sin señal


def load_runs() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(f"{OUTPUT_DIR}/runs/exp1_seed*.csv")):
        seed = int(path.split("seed")[1].split(".")[0])
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_metric(ax, epochs_df, metric: str, ylabel: str, hline: Optional[float] = None):
    for split, color in [("train", COLOR_TRAIN), ("valid", COLOR_VALID)]:
        col = f"{split}_{metric}"
        mean = epochs_df.groupby("epoch")[col].mean()
        std = epochs_df.groupby("epoch")[col].std()
        ax.plot(mean.index, mean.values, color=color, linewidth=2, linestyle="-" if split == "train" else "--", label=split)
        ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    if hline is not None:
        ax.axhline(hline, color="#8a8a86", linewidth=1, linestyle=":", label="prevalencia (sin señal)")

    ax.set_xlabel("época")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    history = load_runs()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_metric(axes[0], history, "pr_auc", "PR-AUC", hline=PREVALENCE)
    plot_metric(axes[1], history, "roc_auc", "ROC-AUC", hline=0.5)
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Experimento 1 -- Transformer de texto puro (media ± desvío sobre 3 semillas)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/experiment1_curves.png", dpi=150)
    print(f"Guardado en {OUTPUT_DIR}/experiment1_curves.png")


if __name__ == "__main__":
    main()
