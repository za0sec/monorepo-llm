"""
Corre el plan de experimentos de Notas.md / Experimentos.md: baseline
solo-tabular vs. fusión tardía (Encoder-only + tabular), cada uno con
varias semillas (para promediar corridas en vez de reportar una sola,
según la aclaración de consigna.VTT).

Solo cómputo, sin gráficos (ver plot_experiments.py). Guarda:
- output/experiment_results.csv: una fila por corrida (modelo, semilla,
  métricas de valid en el mejor epoch).
- output/runs/<modelo>_seed<semilla>.csv: historial por época de cada
  corrida (lo genera train.py).
"""
import pandas as pd

from train import run

MODEL_TYPES = ["tabular", "fusion"]
SEEDS = [0, 1, 2]
EPOCHS = 20


def main() -> None:
    results = []
    for model_type in MODEL_TYPES:
        for seed in SEEDS:
            results.append(run(model_type, seed=seed, epochs=EPOCHS))

    results_df = pd.DataFrame(results)
    results_df.to_csv("output/experiment_results.csv", index=False)

    summary = results_df.groupby("model")[["best_valid_pr_auc", "best_valid_roc_auc"]].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
