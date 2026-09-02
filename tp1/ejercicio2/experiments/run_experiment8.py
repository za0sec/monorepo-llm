"""
Corre el Experimento 8 (ver Experimentos.md): mismo sistema completo del
Experimento 7 (texto + tabular), agregando una capa oculta de 64
unidades en la cabeza de salida (`Linear(139 → 64) → ReLU → Dropout →
Linear(64 → 1)` en vez de `Linear(139 → 1)` directo) -- para probar si el
problema del Experimento 7 era la falta de una no-linealidad que cruce
texto y tabular.

`hidden=64` iguala el ancho de la rama de texto (`d_model=64`), sin
agrandar el modelo mucho más de lo que ya está -- consistente con
"arrancar chico".

`EPOCHS=40` (subido de 20): con 20 épocas, valid PR-AUC todavía estaba
subiendo sin señales de meseta en las 3 semillas -- se dobla el
entrenamiento para confirmar si sigue mejorando o si recién ahí aparece
un techo.

Solo cómputo, sin gráficos (ver plot_experiment.py). Guarda:
- output/experiment8_results.csv: una fila por semilla.
- output/runs/exp8_seed<semilla>.csv: historial por época.
"""
import pandas as pd

from train import run

SEEDS = [0, 1, 2]
EPOCHS = 40
MODEL_KWARGS = {"n_heads": 1, "n_layers": 2, "d_model": 64, "dim_feedforward": 64, "hidden": 64}


def main() -> None:
    results = [
        run(seed=seed, tag="exp8", model_type="combined", epochs=EPOCHS, **MODEL_KWARGS) for seed in SEEDS
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv("output/experiment8_results.csv", index=False)

    summary = results_df[["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
