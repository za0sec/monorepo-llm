"""
Corre el Experimento 10 (ver Experimentos.md): barrido de `hidden`
(ancho de la capa oculta de la cabeza de salida del sistema completo),
manteniendo el resto de la arquitectura cerrada (`n_heads=1, n_layers=2,
d_model=64, dim_feedforward=64`, 20 épocas).

Cierra con el mismo rigor que los demás dials la comparación binaria de
los Experimentos 7/8 (sin capa oculta vs. hidden=64) -- ver discusión en
Experimentos.md sobre por qué no hay una arquitectura "de cátedra" para
esta pieza.

Valores nuevos: 32, 128, 256, 512 -- el barrido se extendió a 512 en una
segunda tanda (mismo Experimento 10, ver Experimentos.md) porque 256
todavía no mostraba meseta; con 512 sí aparece (ganancia marginal chica,
+0,004 de PR-AUC, contra +0,018 del paso anterior). Se reusan sin
reentrenar:
- hidden=0 (sin capa oculta, `Linear` directo): fila `full` que no
  existe como tal en experiment7 -- ahí no hay variantes, se usa tal
  cual (Experimento 7, 20 épocas).
- hidden=64: fila `full` del Experimento 9 (20 épocas, mismas semillas y
  mismo resto de arquitectura que acá) -- no la del Experimento 8, que
  quedó reentrenado a 40 épocas y no es comparable en igualdad de
  condiciones con este barrido.

Solo cómputo, sin gráficos (ver plot_sweep.py). Guarda:
- output/experiment10_results.csv: una fila por (hidden, seed).
- output/runs/exp10_h<hidden>_seed<semilla>.csv: historial por época.
"""
import pandas as pd

from train import run

HIDDEN_VALUES = [32, 128, 256, 512]
SEEDS = [0, 1, 2]
EPOCHS = 20
FIXED = {"n_heads": 1, "n_layers": 2, "d_model": 64, "dim_feedforward": 64}


def main() -> None:
    results = []

    no_hidden = pd.read_csv("output/experiment7_results.csv").copy()
    no_hidden["hidden"] = 0
    results.append(no_hidden)

    exp9 = pd.read_csv("output/experiment9_results.csv")
    hidden64 = exp9[exp9["variant"] == "full"].drop(columns=["variant"]).copy()
    hidden64["hidden"] = 64
    results.append(hidden64)

    for hidden in HIDDEN_VALUES:
        rows = []
        for seed in SEEDS:
            r = run(seed=seed, tag=f"exp10_h{hidden}", model_type="combined", epochs=EPOCHS, hidden=hidden, **FIXED)
            r["hidden"] = hidden
            rows.append(r)
        results.append(pd.DataFrame(rows))

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv("output/experiment10_results.csv", index=False)

    summary = results_df.groupby("hidden")[["best_valid_pr_auc", "best_valid_roc_auc", "best_pr_auc_gap"]].agg(
        ["mean", "std"]
    )
    print(summary)
    best = summary["best_valid_pr_auc"]["mean"].idxmax()
    print(f"\nMejor hidden por PR-AUC de valid: {best}")


if __name__ == "__main__":
    main()
