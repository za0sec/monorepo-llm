"""
Corre el Experimento 3 (ver Experimentos.md): barre varios valores de
`n_layers` (encoders apilados), manteniendo el resto de la arquitectura
del Experimento 1 fija (`n_heads=1`, `d_model=16`, `dim_feedforward=64`).

Valores probados: 2, 4, 8 -- son los que la profesora menciona
explícitamente en `transformers.VTT` ("apilan N encoders... paper
original 6; prueban 2/4/8 también"). El valor 1 no se recorre de nuevo
acá: ya está corrido como el Experimento 1 (`output/experiment1_results.csv`),
así que se reusa sin reentrenar -- solo se le agrega la columna `n_layers=1`
al mergear.

Solo cómputo, sin gráficos (ver plot_sweep.py). Guarda:
- output/experiment3_results.csv: una fila por (n_layers, seed), incluyendo
  el baseline n_layers=1 reusado del Experimento 1, para poder comparar
  los 4 valores juntos y elegir el mejor.
- output/runs/exp3_layers<n>_seed<semilla>.csv: historial por época de
  cada corrida nueva.
"""
import pandas as pd

from train import run

LAYER_VALUES = [2, 4, 8]
SEEDS = [0, 1, 2]
EPOCHS = 20


def main() -> None:
    results = []

    baseline = pd.read_csv("output/experiment1_results.csv")
    baseline["n_layers"] = 1
    results.append(baseline)

    for n_layers in LAYER_VALUES:
        rows = []
        for seed in SEEDS:
            r = run(seed=seed, tag=f"exp3_layers{n_layers}", epochs=EPOCHS, n_layers=n_layers)
            r["n_layers"] = n_layers
            rows.append(r)
        results.append(pd.DataFrame(rows))

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv("output/experiment3_results.csv", index=False)

    summary = results_df.groupby("n_layers")[["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]].agg(["mean", "std"])
    print(summary)
    best = summary["best_valid_pr_auc"]["mean"].idxmax()
    print(f"\nMejor n_layers por PR-AUC de valid: {best}")


if __name__ == "__main__":
    main()
