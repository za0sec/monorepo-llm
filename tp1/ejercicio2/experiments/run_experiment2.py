"""
Corre el Experimento 2 (ver Experimentos.md): mismo Transformer de texto
puro del Experimento 1, cambiando solo `n_heads` de 1 a 2 -- el resto de
la arquitectura (`d_model`, `n_layers`, `dim_feedforward`) queda fijo, para
aislar el efecto de multi-head attention.

Solo cómputo, sin gráficos (ver plot_experiment2.py). Guarda:
- output/experiment2_results.csv: una fila por semilla.
- output/runs/exp2_seed<semilla>.csv: historial por época de cada corrida.
"""
import pandas as pd

from train import run

SEEDS = [0, 1, 2]
EPOCHS = 20
MODEL_KWARGS = {"n_heads": 2}


def main() -> None:
    results = [run(seed=seed, tag="exp2", epochs=EPOCHS, **MODEL_KWARGS) for seed in SEEDS]

    results_df = pd.DataFrame(results)
    results_df.to_csv("output/experiment2_results.csv", index=False)

    summary = results_df[["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
