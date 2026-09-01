"""
Corre el Experimento 7 (ver Experimentos.md): el sistema completo --
texto (rama Transformer ganadora de los Experimentos 1 a 6: n_heads=1,
n_layers=2, d_model=64, dim_feedforward=64) combinado con las features
tabulares.

Solo cómputo, sin gráficos (ver plot_experiment.py). Guarda:
- output/experiment7_results.csv: una fila por semilla.
- output/runs/exp7_seed<semilla>.csv: historial por época.
"""
import pandas as pd

from train import run

SEEDS = [0, 1, 2]
EPOCHS = 20
MODEL_KWARGS = {"n_heads": 1, "n_layers": 2, "d_model": 64, "dim_feedforward": 64}


def main() -> None:
    results = [
        run(seed=seed, tag="exp7", model_type="combined", epochs=EPOCHS, **MODEL_KWARGS) for seed in SEEDS
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv("output/experiment7_results.csv", index=False)

    summary = results_df[["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
