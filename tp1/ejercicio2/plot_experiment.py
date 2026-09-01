"""
Gráficos de un experimento -- lee los CSV que genera run_experiment<n>.py,
no recalcula ni reentrena nada (separación cómputo/gráficos, ver
CLAUDE.md). Genérico para cualquier experimento: mismo código de ploteo,
solo cambia qué runs lee y dónde guarda el resultado.

Uso: python3 plot_experiment.py <n>   (ej. `python3 plot_experiment.py 2`
lee output/runs/exp2_seed*.csv y guarda output/experiment2_curves.png)
"""
import argparse
import glob
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "output"

COLOR_TRAIN = "#2a78d6"
COLOR_VALID = "#eb6834"
PREVALENCE = 0.1301  # tasa global de `bought`, ver ejercicio2/Notas.md -- referencia de un clasificador sin señal


def load_runs(tag: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(f"{OUTPUT_DIR}/runs/{tag}_seed*.csv")):
        seed = int(path.split("seed")[1].split(".")[0])
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No se encontraron runs para '{tag}' en {OUTPUT_DIR}/runs/")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=int, help="número de experimento (ej. 1, 2)")
    args = parser.parse_args()

    tag = f"exp{args.experiment}"
    history = load_runs(tag)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_metric(axes[0], history, "pr_auc", "PR-AUC", hline=PREVALENCE)
    plot_metric(axes[1], history, "roc_auc", "ROC-AUC", hline=0.5)
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Experimento {args.experiment} (media ± desvío sobre {history['seed'].nunique()} semillas)")
    fig.tight_layout()

    out_path = f"{OUTPUT_DIR}/experiment{args.experiment}_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
