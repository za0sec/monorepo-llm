"""
Corre el Experimento 5 (ver Experimentos.md): barre `dim_feedforward`
solo, manteniendo fijo el resto de la mejor config encontrada hasta ahora
(`n_heads=1`, `n_layers=2`, `d_model=16` sin cambios).

Valores probados: 16, 32, 128, 256, 512. El valor 64 (la base, proporción
4x sobre d_model=16 sugerida por el paper original) se reusa de la fila
`n_layers=2` del Experimento 3 sin reentrenar.

**Extensión (mismo criterio que el Experimento 4)**: la tanda original
(16/32/64/128) ya mostraba un pico interior en 64, con un solo punto por
encima (128) confirmando la caída. Se agregan 256 y 512 para confirmar la
tendencia con más de un punto, igual que se hizo con `d_model`, en vez de
quedarse con un solo punto de caída.

Va de la mano con el Experimento 4 (d_model solo, dim_feedforward fijo):
la idea es correr los dos por separado y, si ambos muestran que el mejor
valor de uno depende del valor del otro, recién ahí armar un tercer
experimento moviendo ambos juntos (grid) -- ver Experimentos.md.

Solo cómputo, sin gráficos (ver plot_sweep.py). Guarda:
- output/experiment5_results.csv: una fila por (dim_feedforward, seed).
- output/runs/exp5_ff<n>_seed<semilla>.csv: historial por época.
"""
import os

import pandas as pd

from train import run

FF_VALUES = [16, 32, 128, 256, 512]
SEEDS = [0, 1, 2]
EPOCHS = 20
RESULTS_PATH = "output/experiment5_results.csv"


def main() -> None:
    results = []
    already_run = set()
    if os.path.exists(RESULTS_PATH):
        prev = pd.read_csv(RESULTS_PATH)
        results.append(prev)
        already_run = set(prev["dim_feedforward"].unique())
    else:
        baseline = pd.read_csv("output/experiment3_results.csv")
        baseline = baseline[baseline["n_layers"] == 2].drop(columns=["n_layers"]).copy()
        baseline["dim_feedforward"] = 64
        results.append(baseline)
        already_run = {64}

    for dim_feedforward in FF_VALUES:
        if dim_feedforward in already_run:
            continue
        rows = []
        for seed in SEEDS:
            r = run(seed=seed, tag=f"exp5_ff{dim_feedforward}", epochs=EPOCHS, dim_feedforward=dim_feedforward, n_heads=1, n_layers=2, d_model=16)
            r["dim_feedforward"] = dim_feedforward
            rows.append(r)
        results.append(pd.DataFrame(rows))

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    summary = results_df.groupby("dim_feedforward")[["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]].agg(["mean", "std"])
    print(summary)
    best = summary["best_valid_pr_auc"]["mean"].idxmax()
    print(f"\nMejor dim_feedforward por PR-AUC de valid: {best}")


if __name__ == "__main__":
    main()
