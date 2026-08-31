"""
Corre el Experimento 1 (Transformer de texto puro, ver Notas.md y
Experimentos.md) con 3 semillas y promedia, según la aclaración de
consigna.VTT de no reportar una sola corrida.

Solo cómputo, sin gráficos (ver plot_experiment1.py). Guarda:
- output/experiment1_results.csv: una fila por semilla (métricas de valid
  en la mejor época).
- output/runs/exp1_seed<semilla>.csv: historial por época de cada corrida
  (lo genera train.py).
"""
import pandas as pd

from train import run

SEEDS = [0, 1, 2]
EPOCHS = 20


def main() -> None:
    results = [run(seed=seed, epochs=EPOCHS) for seed in SEEDS]

    results_df = pd.DataFrame(results)
    results_df.to_csv("output/experiment1_results.csv", index=False)

    summary = results_df[["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
