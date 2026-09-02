"""
Gráfico de la comparación de estabilidad 70/15/15 vs. 80/10/10: distribución
de la tasa de bought de valid/test a lo largo de 300 splits con semillas
distintas (generados por compare_split_stability.py). Lee
output/split_stability.csv -- no recalcula nada.
"""
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "output"

data = pd.read_csv(f"{OUTPUT_DIR}/split_stability.csv")
global_rate = 13.01  # ver Notas.md -- tasa de bought sobre el dataset completo

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for ax, split in zip(axes, ["valid", "test"]):
    subset = data[data["split"] == split]
    groups = [subset[subset["proporcion"] == p]["bought_rate"] * 100 for p in ["70/15/15", "80/10/10"]]
    ax.boxplot(groups, tick_labels=["70/15/15", "80/10/10"], widths=0.5)
    ax.axhline(global_rate, color="black", linestyle="--", linewidth=1, label="tasa global (13,01%)")
    ax.set_title(f"{split}: tasa de bought en 300 splits distintos")
    ax.set_ylabel("% bought" if split == "valid" else "")
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/split_stability.png", dpi=150)
print(f"Guardado {OUTPUT_DIR}/split_stability.png")
