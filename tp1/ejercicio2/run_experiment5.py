"""
Corre el Experimento 5 (ver Experimentos.md): barre `dim_feedforward`
solo, manteniendo fijo el resto de la mejor config encontrada hasta ahora
(`n_heads=1`, `n_layers=2`, `d_model=16` sin cambios).

Valores probados: 16, 32, 128. El valor 64 (la base, proporción 4x sobre
d_model=16 sugerida por el paper original) se reusa de la fila
`n_layers=2` del Experimento 3 sin reentrenar.

Va de la mano con el Experimento 4 (d_model solo, dim_feedforward fijo):
la idea es correr los dos por separado y, si ambos muestran que el mejor
valor de uno depende del valor del otro, recién ahí armar un tercer
experimento moviendo ambos juntos (grid) -- ver Experimentos.md.

Solo cómputo, sin gráficos (ver plot_sweep.py). Guarda:
- output/experiment5_results.csv: una fila por (dim_feedforward, seed).
- output/runs/exp5_ff<n>_seed<semilla>.csv: historial por época.
"""
import pandas as pd

from train import run

FF_VALUES = [16, 32, 128]
SEEDS = [0, 1, 2]
EPOCHS = 20


def main() -> None:
    results = []

    baseline = pd.read_csv("output/experiment3_results.csv")
    baseline = baseline[baseline["n_layers"] == 2].drop(columns=["n_layers"]).copy()
    baseline["dim_feedforward"] = 64
    results.append(baseline)

    for dim_feedforward in FF_VALUES:
        rows = []
        for seed in SEEDS:
            r = run(seed=seed, tag=f"exp5_ff{dim_feedforward}", epochs=EPOCHS, dim_feedforward=dim_feedforward, n_heads=1, n_layers=2, d_model=16)
            r["dim_feedforward"] = dim_feedforward
            rows.append(r)
        results.append(pd.DataFrame(rows))

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv("output/experiment5_results.csv", index=False)

    summary = results_df.groupby("dim_feedforward")[["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]].agg(["mean", "std"])
    print(summary)
    best = summary["best_valid_pr_auc"]["mean"].idxmax()
    print(f"\nMejor dim_feedforward por PR-AUC de valid: {best}")


if __name__ == "__main__":
    main()
