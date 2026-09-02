"""
Gráfico del Experimento 8: compara el mejor PR-AUC de valid alcanzable
con presupuesto de 20 épocas vs. 40, para mostrar que la ganancia de
entrenar el doble es chica. Lee output/runs/exp8_seed*.csv (ya corridos a
40 épocas) -- no reentrena nada, solo recorta la curva ya existente hasta
la época 20 para reconstruir qué hubiera dado ese presupuesto más chico
(entrenar es determinístico dada la semilla, así que las épocas 1-20 de
la corrida de 40 son idénticas a las de una corrida de 20).

Uso: python3 plot_experiment8_epochs.py
"""
import glob

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "output"
COLOR_TRAIN = "#2a78d6"
COLOR_VALID = "#eb6834"


def load_runs() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(f"{OUTPUT_DIR}/runs/exp8_seed*.csv")):
        seed = int(path.split("seed")[1].split(".")[0])
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    history = load_runs()

    per_seed_20 = history[history["epoch"] <= 20].groupby("seed")["valid_pr_auc"].max()
    per_seed_40 = history.groupby("seed")["valid_pr_auc"].max()

    mean20, std20 = per_seed_20.mean(), per_seed_20.std()
    mean40, std40 = per_seed_40.mean(), per_seed_40.std()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    mean_curve = history.groupby("epoch")["valid_pr_auc"].mean()
    std_curve = history.groupby("epoch")["valid_pr_auc"].std()
    ax.plot(mean_curve.index, mean_curve.values, color=COLOR_VALID, linewidth=2, label="valid PR-AUC")
    ax.fill_between(mean_curve.index, mean_curve - std_curve, mean_curve + std_curve, color=COLOR_VALID, alpha=0.15)
    ax.axvline(20, color="#8a8a86", linewidth=1, linestyle=":")
    ax.scatter([20], [mean20], s=90, facecolors="white", edgecolors=COLOR_VALID, linewidths=2, zorder=5)
    ax.annotate(f"presupuesto 20 épocas\nmejor: {mean20:.3f}", xy=(20, mean20), xytext=(3, 0.50),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#555"))
    best40_epoch = mean_curve.idxmax()
    ax.scatter([best40_epoch], [mean_curve.max()], s=90, facecolors="white", edgecolors=COLOR_TRAIN, linewidths=2, zorder=5)
    ax.annotate(f"presupuesto 40 épocas\nmejor: {mean40:.3f}", xy=(best40_epoch, mean_curve.max()),
                xytext=(best40_epoch + 4, 0.42),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_ylim(0.28, 0.88)
    ax.set_xlabel("época")
    ax.set_ylabel("valid PR-AUC (media sobre 3 semillas)")
    ax.set_title("Curva completa (40 épocas) con presupuesto de 20 marcado")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    bars = ax.bar(["20 épocas", "40 épocas"], [mean20, mean40], yerr=[std20, std40],
                   color=[COLOR_VALID, COLOR_TRAIN], capsize=5, width=0.55)
    for bar, val in zip(bars, [mean20, mean40]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", fontsize=10)
    ax.annotate("", xy=(1, mean40 + 0.045), xytext=(0, mean40 + 0.045),
                arrowprops=dict(arrowstyle="-", color="#555"))
    ax.text(0.5, mean40 + 0.05, f"+{mean40 - mean20:.3f}", ha="center", fontsize=9, color="#555")
    ax.set_ylabel("mejor valid PR-AUC (media ± std, 3 semillas)")
    ax.set_title("Ganancia de duplicar el entrenamiento")
    ax.set_ylim(0.75, 0.86)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.suptitle("Experimento 8 — 20 vs. 40 épocas: la ganancia de entrenar el doble es chica")
    fig.tight_layout()
    out_path = f"{OUTPUT_DIR}/experiment8_epochs.png"
    fig.savefig(out_path, dpi=150)
    print(f"Guardado en {out_path}")
    print(f"20 épocas: {mean20:.4f} ± {std20:.4f}")
    print(f"40 épocas: {mean40:.4f} ± {std40:.4f}")
    print(f"diferencia: {mean40 - mean20:.4f}")


if __name__ == "__main__":
    main()
