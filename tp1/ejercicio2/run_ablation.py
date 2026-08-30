"""
Estudio de ablación (pedido explícitamente por la consigna: "comparación de
alternativas de los distintos módulos"). Varía un "dial" por vez desde la
configuración base (la misma de Experimentos.md: d_model=64, 4 heads, 2
encoders, MLP interno 128, todas las tabulares) y compara contra esa base.

1 sola semilla por variante (no 3): esto es una exploración de qué dial
importa, no el número principal que se reporta (ese ya usó 3 semillas en
run_experiments.py) -- así el costo computacional no se dispara, como pide
la consigna.

Guarda:
- output/ablation_results.csv: una fila por variante.
- output/runs/ablation_<variante>.csv: historial por época de cada una.
"""
import pandas as pd

from train import train_model

SEED = 0
EPOCHS = 20
BASE_CONFIG = {"d_model": 64, "n_heads": 4, "n_layers": 2, "dim_feedforward": 128}

# Variantes de arquitectura: se cambia un solo hiperparámetro desde la base.
ARCHITECTURE_VARIANTS = {
    "base": {},
    "heads_2": {"n_heads": 2},
    "heads_8": {"n_heads": 8},
    "layers_1": {"n_layers": 1},
    "layers_4": {"n_layers": 4},
    "d_model_32": {"d_model": 32},
    "d_model_96": {"d_model": 96},
    "ff_64": {"dim_feedforward": 64},
    "ff_256": {"dim_feedforward": 256},
}

# Variantes de features tabulares: se sacan columnas por prefijo, arquitectura
# en la config base.
TABULAR_VARIANTS = {
    "sin_country_of_origin": ("country_of_origin_",),
    "sin_nutrition_score": ("nutrition_score_z",),
    "sin_ambas": ("country_of_origin_", "nutrition_score_z"),
}


def run_variant(name: str, model_overrides: dict = None, exclude_prefixes: tuple = ()) -> dict:
    config = {**BASE_CONFIG, **(model_overrides or {})}
    model, history = train_model(
        "fusion", seed=SEED, epochs=EPOCHS, exclude_prefixes=exclude_prefixes, **config
    )
    history.to_csv(f"output/runs/ablation_{name}.csv", index=False)

    best = history.loc[history["valid_pr_auc"].idxmax()]
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "variant": name,
        "n_params": n_params,
        "best_epoch": int(best["epoch"]),
        "best_valid_pr_auc": best["valid_pr_auc"],
        "best_valid_roc_auc": best["valid_roc_auc"],
        "excluded_features": ",".join(exclude_prefixes) if exclude_prefixes else "-",
        **config,
    }


def main() -> None:
    rows = []
    for name, overrides in ARCHITECTURE_VARIANTS.items():
        print(f"\n=== variante: {name} ===")
        rows.append(run_variant(name, model_overrides=overrides))
    for name, prefixes in TABULAR_VARIANTS.items():
        print(f"\n=== variante: {name} ===")
        rows.append(run_variant(name, exclude_prefixes=prefixes))

    results_df = pd.DataFrame(rows)
    results_df.to_csv("output/ablation_results.csv", index=False)
    print("\n" + results_df[["variant", "n_params", "best_valid_pr_auc", "best_valid_roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
