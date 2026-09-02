"""
Corre el Experimento 11 (ver Experimentos.md): con/sin positional
encoding, sobre el Transformer de texto solo (no el sistema completo) --
para aislar el efecto sin que las features tabulares puedan compensar la
pérdida de información de orden. Arquitectura: la ganadora de texto solo
de los Experimentos 4/6 (n_heads=1, n_layers=2, d_model=64,
dim_feedforward=64).

A diferencia de otros dials, este sí está explícitamente cubierto en
clase (transformers.VTT: "lo hacés para darle un orden a tus tokens") --
ver discusión en Experimentos.md de por qué se prioriza este dial sobre
el de pooling (que no está enseñado).

Se reusa sin reentrenar el punto "con positional encoding" del
Experimento 4 (fila d_model=64, que ya usa esta arquitectura con
`use_positional_encoding=True` por default).

Solo cómputo, sin gráficos (ver plot_sweep.py). Guarda:
- output/experiment11_results.csv: una fila por (variante, seed).
- output/runs/exp11_sin_pos_seed<semilla>.csv: historial por época.
"""
import pandas as pd

from train import run

SEEDS = [0, 1, 2]
EPOCHS = 20
FIXED = {"n_heads": 1, "n_layers": 2, "d_model": 64, "dim_feedforward": 64}


def main() -> None:
    results = []

    con_pos = pd.read_csv("output/experiment4_results.csv")
    con_pos = con_pos[con_pos["d_model"] == 64].drop(columns=["d_model"]).copy()
    con_pos["variant"] = "con_positional_encoding"
    results.append(con_pos)

    rows = []
    for seed in SEEDS:
        r = run(
            seed=seed,
            tag="exp11_sin_pos",
            model_type="text",
            epochs=EPOCHS,
            use_positional_encoding=False,
            **FIXED,
        )
        r["variant"] = "sin_positional_encoding"
        rows.append(r)
    results.append(pd.DataFrame(rows))

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv("output/experiment11_results.csv", index=False)

    summary = results_df.groupby("variant")[["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]].agg(
        ["mean", "std"]
    )
    print(summary)


if __name__ == "__main__":
    main()
