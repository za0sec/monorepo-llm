"""
Gráfico del Experimento 6 -- lee output/experiment6_results.csv (no
recalcula ni reentrena, ver CLAUDE.md). Comparación puntual (no un
barrido de un solo dial, por eso no usa plot_sweep.py): para cada
d_model probado, dim_feedforward=64 (fijo, reusado del Experimento 4)
contra dim_feedforward escalado a 4x -- para decidir si conviene escalar
ambos juntos o no.
"""
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "output"
COLOR_FIXED = "#2a78d6"  # dim_feedforward=64 fijo (lo ya corrido en el Exp. 4)
COLOR_4X = "#eb6834"  # dim_feedforward escalado a 4x


def main() -> None:
    df = pd.read_csv(f"{OUTPUT_DIR}/experiment6_results.csv")
    summary = df.groupby(["d_model", "dim_feedforward"])[
        ["best_valid_pr_auc", "best_pr_auc_gap"]
    ].agg(["mean", "std"])

    d_models = sorted(df["d_model"].unique())
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, ylabel in [
        (axes[0], "best_valid_pr_auc", "PR-AUC"),
        (axes[1], "best_pr_auc_gap", "gap PR-AUC (train-valid)"),
    ]:
        for i, d_model in enumerate(d_models):
            ff_fixed = 64
            ff_4x = d_model * 4
            mean_fixed = summary.loc[(d_model, ff_fixed), (metric, "mean")]
            std_fixed = summary.loc[(d_model, ff_fixed), (metric, "std")]
            mean_4x = summary.loc[(d_model, ff_4x), (metric, "mean")]
            std_4x = summary.loc[(d_model, ff_4x), (metric, "std")]

            ax.bar(i - width / 2, mean_fixed, width, yerr=std_fixed, capsize=3, color=COLOR_FIXED,
                   label="dim_feedforward=64 (Exp. 4)" if i == 0 else None)
            ax.bar(i + width / 2, mean_4x, width, yerr=std_4x, capsize=3, color=COLOR_4X,
                   label=f"dim_feedforward=4×d_model" if i == 0 else None)

        ax.set_xticks(range(len(d_models)))
        ax.set_xticklabels([f"d_model={d}" for d in d_models])
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Experimento 6 -- ¿escalar dim_feedforward junto con d_model ayuda?")
    fig.tight_layout()

    out_path = f"{OUTPUT_DIR}/experiment6_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
