"""
Corre el Experimento 6 (ver Experimentos.md): prueba si escalar
`dim_feedforward` junto con `d_model` (manteniendo la proporción 4x del
paper original) mejora el resultado de los mejores `d_model` del
Experimento 4, en vez de dejar `dim_feedforward=64` fijo como se hizo ahí.

Combinaciones nuevas probadas:
- d_model=32, dim_feedforward=128 (4x -- en el Exp. 4 este d_model tenía
  dim_feedforward=64, o sea 2x)
- d_model=64, dim_feedforward=256 (4x -- en el Exp. 4 este d_model tenía
  dim_feedforward=64, o sea 1x, y fue el mejor resultado hasta ahora)

Se comparan contra las filas ya corridas de esos mismos d_model con
dim_feedforward=64 (Experimento 4), reusadas sin reentrenar.

Solo cómputo, sin gráficos. Guarda:
- output/experiment6_results.csv: una fila por (config, seed).
- output/runs/exp6_d<d_model>ff<ff>_seed<semilla>.csv: historial por época.
"""
import pandas as pd

from train import run

CONFIGS = [(32, 128), (64, 256)]
SEEDS = [0, 1, 2]
EPOCHS = 20


def main() -> None:
    results = []

    baseline = pd.read_csv("output/experiment4_results.csv")
    baseline = baseline[baseline["d_model"].isin([32, 64])].copy()
    baseline["dim_feedforward"] = 64
    results.append(baseline)

    for d_model, dim_feedforward in CONFIGS:
        rows = []
        for seed in SEEDS:
            r = run(
                seed=seed,
                tag=f"exp6_d{d_model}ff{dim_feedforward}",
                epochs=EPOCHS,
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                n_heads=1,
                n_layers=2,
            )
            r["d_model"] = d_model
            r["dim_feedforward"] = dim_feedforward
            rows.append(r)
        results.append(pd.DataFrame(rows))

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv("output/experiment6_results.csv", index=False)

    summary = results_df.groupby(["d_model", "dim_feedforward"])[
        ["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]
    ].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
