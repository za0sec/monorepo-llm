"""
Scatter de BTR real vs. predicho por búsqueda, en test -- lee
output/btr_test.csv (generado por evaluate_test.py), no recalcula ni
reentrena nada (separación cómputo/gráficos, ver CLAUDE.md).

Cada punto es una query_id de test (una búsqueda), con su BTR real
(promedio de `bought` entre sus filas) en x y su BTR predicho (promedio
de la probabilidad predicha) en y. Las 3 semillas se grafican juntas,
con distinta transparencia, para no promediar el eje x de golpe (el BTR
real de una query no depende de la semilla, solo el predicho).
"""
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "output"
COLOR_POINTS = "#2a78d6"
COLOR_DIAG = "#8a8a86"


def main() -> None:
    btr = pd.read_csv(f"{OUTPUT_DIR}/btr_test.csv")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(btr["btr_real"], btr["btr_predicted"], s=14, color=COLOR_POINTS, alpha=0.35, linewidth=0)

    lims = [0, 1]
    ax.plot(lims, lims, color=COLOR_DIAG, linewidth=1, linestyle=":", label="predicción perfecta")

    # MAE y r por semilla, promediados -- mismo cálculo que evaluate_test.py
    # (test_results.csv), no la correlación sobre las 3 semillas mezcladas
    # en un solo pool (da un número distinto, no comparable con la tabla).
    per_seed = btr.groupby("seed")[["btr_real", "btr_predicted"]].apply(
        lambda g: pd.Series(
            {"mae": (g["btr_real"] - g["btr_predicted"]).abs().mean(), "r": g["btr_real"].corr(g["btr_predicted"])}
        )
    )
    mae_mean, mae_std = per_seed["mae"].mean(), per_seed["mae"].std()
    r_mean, r_std = per_seed["r"].mean(), per_seed["r"].std()
    ax.text(
        0.02, 0.98,
        f"MAE = {mae_mean:.3f} ± {mae_std:.3f}\nr = {r_mean:.3f} ± {r_std:.3f}\n"
        f"({btr['seed'].nunique()} semillas x {btr['query_id'].nunique()} queries)",
        transform=ax.transAxes, va="top", ha="left", fontsize=9, color="#4a4a46",
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("BTR real de la búsqueda (promedio de bought)")
    ax.set_ylabel("BTR predicho (promedio de la probabilidad predicha)")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("BTR por búsqueda en test: real vs. predicho")
    fig.tight_layout()

    out_path = f"{OUTPUT_DIR}/btr_test_scatter.png"
    fig.savefig(out_path, dpi=150)
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
