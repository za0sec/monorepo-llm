"""
Corre el Experimento 4 (ver Experimentos.md): barre `d_model` solo,
manteniendo fijo el resto de la mejor config encontrada hasta ahora
(`n_heads=1`, `n_layers=2` -- ganador del Experimento 3 -- y
`dim_feedforward=64` sin cambios).

Valores probados: 8, 32, 64. El valor 16 (la base) se reusa de la fila
`n_layers=2` del Experimento 3 (que ya corrió con d_model=16,
dim_feedforward=64) sin reentrenar.

Va de la mano con el Experimento 5 (dim_feedforward solo, d_model fijo):
la idea es correr los dos por separado y, si ambos muestran que el mejor
valor de uno depende del valor del otro, recién ahí armar un tercer
experimento moviendo ambos juntos (grid) -- ver Experimentos.md.

Solo cómputo, sin gráficos (ver plot_sweep.py). Guarda:
- output/experiment4_results.csv: una fila por (d_model, seed).
- output/runs/exp4_dmodel<n>_seed<semilla>.csv: historial por época.
"""
import pandas as pd

from train import run

DMODEL_VALUES = [8, 32, 64]
SEEDS = [0, 1, 2]
EPOCHS = 20


def main() -> None:
    results = []

    baseline = pd.read_csv("output/experiment3_results.csv")
    baseline = baseline[baseline["n_layers"] == 2].drop(columns=["n_layers"]).copy()
    baseline["d_model"] = 16
    results.append(baseline)

    for d_model in DMODEL_VALUES:
        rows = []
        for seed in SEEDS:
            r = run(seed=seed, tag=f"exp4_dmodel{d_model}", epochs=EPOCHS, d_model=d_model, n_heads=1, n_layers=2, dim_feedforward=64)
            r["d_model"] = d_model
            rows.append(r)
        results.append(pd.DataFrame(rows))

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv("output/experiment4_results.csv", index=False)

    summary = results_df.groupby("d_model")[["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]].agg(["mean", "std"])
    print(summary)
    best = summary["best_valid_pr_auc"]["mean"].idxmax()
    print(f"\nMejor d_model por PR-AUC de valid: {best}")


if __name__ == "__main__":
    main()
