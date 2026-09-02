"""
Gráfico de la evaluación final -- lee output/runs/final_seed*.csv
(curvas de train/valid) y output/test_results.csv (resultado de test),
no recalcula ni reentrena nada (ver evaluate_test.py y CLAUDE.md).

Muestra las curvas de entrenamiento igual que plot_experiment.py, más una
línea horizontal con el resultado de test (una sola medición al final,
no una curva por época -- test no se evalúa epoch a epoch).
"""
import glob

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "output"
COLOR_TRAIN = "#2a78d6"
COLOR_VALID = "#eb6834"
COLOR_TEST = "#1baf7a"
PREVALENCE = 0.1301


def load_runs() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(f"{OUTPUT_DIR}/runs/final_seed*.csv")):
        seed = int(path.split("seed")[1].split(".")[0])
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_metric(ax, epochs_df, test_mean, test_std, metric: str, ylabel: str, hline=None):
    for split, color in [("train", COLOR_TRAIN), ("valid", COLOR_VALID)]:
        col = f"{split}_{metric}"
        mean = epochs_df.groupby("epoch")[col].mean()
        std = epochs_df.groupby("epoch")[col].std()
        ax.plot(mean.index, mean.values, color=color, linewidth=2, linestyle="-" if split == "train" else "--", label=split)
        ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    last_epoch = epochs_df["epoch"].max()
    ax.axhline(test_mean, color=COLOR_TEST, linewidth=2, linestyle=":", label="test (media)")
    ax.fill_between([1, last_epoch], test_mean - test_std, test_mean + test_std, color=COLOR_TEST, alpha=0.12, linewidth=0)

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
    test = pd.read_csv(f"{OUTPUT_DIR}/test_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_metric(
        axes[0], history, test["test_pr_auc"].mean(), test["test_pr_auc"].std(), "pr_auc", "PR-AUC", hline=PREVALENCE
    )
    plot_metric(axes[1], history, test["test_roc_auc"].mean(), test["test_roc_auc"].std(), "roc_auc", "ROC-AUC", hline=0.5)
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Configuración final -- train/valid por época + resultado de test (3 semillas)")
    fig.tight_layout()

    out_path = f"{OUTPUT_DIR}/final_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
