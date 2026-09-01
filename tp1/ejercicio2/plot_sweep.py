"""
Gráfico de un barrido de hiperparámetro (ej. Experimento 3: `n_layers`) --
lee el CSV que ya armó run_experiment<n>.py agrupando por semilla y valor
del dial, no recalcula ni reentrena nada (separación cómputo/gráficos, ver
CLAUDE.md). Genérico: cualquier experimento que barra un solo dial y guarde
sus resultados con las columnas estándar de train.py::run() puede
reusarlo.

Uso: python3 plot_sweep.py <csv> <columna_del_dial>
  (ej. `python3 plot_sweep.py output/experiment3_results.csv n_layers`)
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

COLOR_TRAIN = "#2a78d6"
COLOR_VALID = "#eb6834"


def plot_dial(ax, summary, dial: str, train_col, valid_col, ylabel: str):
    # Eje X categórico (posiciones 0..n-1), no numérico: los valores del dial
    # (ej. 1/2/4/8) son configuraciones discretas a comparar, no una
    # magnitud continua -- un eje lineal o log exageraría o distorsionaría
    # la distancia visual entre puntos según el dial de turno.
    positions = range(len(summary))
    labels = [str(v) for v in summary.index]

    if train_col is not None:
        ax.errorbar(
            positions, summary[(train_col, "mean")], yerr=summary[(train_col, "std")],
            color=COLOR_TRAIN, marker="o", markersize=6, linewidth=2, capsize=3, label="train",
        )
    ax.errorbar(
        positions, summary[(valid_col, "mean")], yerr=summary[(valid_col, "std")],
        color=COLOR_VALID, marker="o", markersize=6, linewidth=2, capsize=3,
        linestyle="--" if train_col is not None else "-", label="valid",
    )

    best_pos = int(summary[(valid_col, "mean")].values.argmax())
    best_y = summary.iloc[best_pos][(valid_col, "mean")]
    ax.scatter([best_pos], [best_y], s=140, facecolors="none", edgecolors="#0b0b0b", linewidths=1.5, zorder=5)

    ax.set_xlabel(dial)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.4, len(summary) - 0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV de resultados (una fila por seed x valor del dial)")
    parser.add_argument("dial", help="nombre de la columna con el valor del dial (ej. n_layers)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    summary = df.groupby(args.dial)[
        ["best_train_pr_auc", "best_valid_pr_auc", "best_valid_roc_auc"]
    ].agg(["mean", "std"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_dial(axes[0], summary, args.dial, "best_train_pr_auc", "best_valid_pr_auc", "PR-AUC")
    plot_dial(axes[1], summary, args.dial, None, "best_valid_roc_auc", "ROC-AUC")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Barrido de {args.dial} (círculo = mejor valor por PR-AUC de valid)")
    fig.tight_layout()

    base = os.path.splitext(os.path.basename(args.csv))[0]  # ej. experiment3_results
    out_path = f"output/{base.replace('_results', '')}_sweep.png"
    fig.savefig(out_path, dpi=150)
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
