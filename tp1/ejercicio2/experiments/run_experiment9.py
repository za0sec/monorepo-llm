"""
Corre el Experimento 9 (ver Experimentos.md): ablation de `country_of_origin`
y `nutrition_score` sobre el sistema completo ya cerrado (Experimento 8:
n_heads=1, n_layers=2, d_model=64, dim_feedforward=64, hidden=64,
20 épocas). Ambas quedaron marcadas como "dudosas" en el EDA de
`ejercicio1/Notas.md` por señal univariada débil -- acá se prueba si
sacarlas cambia algo en el sistema completo.

4 variantes, 3 semillas cada una:
- full: todas las features (baseline, igual al Experimento 8).
- sin_country_of_origin: sin las 10 columnas one-hot de country_of_origin.
- sin_nutrition_score: sin nutrition_score_z.
- sin_ambas: sin las dos.

Solo cómputo, sin gráficos (ver plot_sweep.py). Guarda:
- output/experiment9_results.csv: una fila por (variante, seed).
- output/runs/exp9_<variante>_seed<semilla>.csv: historial por época.
"""
import pandas as pd

from train import run

VARIANTS = {
    "full": (),
    "sin_country_of_origin": ("country_of_origin_",),
    "sin_nutrition_score": ("nutrition_score_z",),
    "sin_ambas": ("country_of_origin_", "nutrition_score_z"),
}
SEEDS = [0, 1, 2]
EPOCHS = 20
MODEL_KWARGS = {"n_heads": 1, "n_layers": 2, "d_model": 64, "dim_feedforward": 64, "hidden": 64}


def main() -> None:
    results = []
    for variant, exclude_prefixes in VARIANTS.items():
        for seed in SEEDS:
            r = run(
                seed=seed,
                tag=f"exp9_{variant}",
                model_type="combined",
                epochs=EPOCHS,
                exclude_prefixes=exclude_prefixes,
                **MODEL_KWARGS,
            )
            r["variant"] = variant
            results.append(r)

    results_df = pd.DataFrame(results)
    results_df.to_csv("output/experiment9_results.csv", index=False)

    summary = results_df.groupby("variant")[["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]].agg(
        ["mean", "std"]
    )
    print(summary)


if __name__ == "__main__":
    main()
