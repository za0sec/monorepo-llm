"""
Genera los 3 graficos pendientes para la presentacion del Ejercicio 1,
leyendo los CSVs de experiments/presentation_extra_eda.py. No recalcula nada.

Uso (desde ejercicio1/): python3 plots/plot_presentation_extra.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# 1) campana de precio relativo
campana = pd.read_csv("output/price_position_bought.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(campana["bin_pos_relativa"], campana["pct_bought"] * 100, color="#4A6FC5")
for i, row in campana.iterrows():
    ax.text(i, row["pct_bought"] * 100 + 0.3, f"n={row['n']}", ha="center", fontsize=9)
ax.set_xlabel("posición relativa del precio dentro del rango filtrado")
ax.set_ylabel("% bought")
ax.set_title("Tasa de compra según posición del precio en el rango buscado")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("output/price_position_bought.png", dpi=150)
plt.close()

# 2) boxplots original vs z-score
norm = pd.read_csv("output/normalizacion_boxplot.csv")
numericas = ["price", "net_weight_oz", "nutrition_score"]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, etapa, titulo in zip(axes, ["original", "z-score"], ["Original", "Z-score"]):
    sub = norm[norm["etapa"] == etapa][numericas]
    ax.boxplot([sub[c].dropna() for c in numericas], labels=numericas)
    ax.set_title(titulo)
    ax.tick_params(axis="x", rotation=20)
plt.suptitle("Features numéricas antes/después de estandarizar (z-score)")
plt.tight_layout()
plt.savefig("output/normalizacion_boxplot.png", dpi=150)
plt.close()

# 3) % bought por tag de reputacion
tag = pd.read_csv("output/reputation_tag_bought.csv")
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["#4A6FC5" if v > 0 else "#B0B0B0" for v in tag["pct_bought"]]
ax.barh(tag["tag"][::-1], tag["pct_bought"][::-1] * 100, color=colors[::-1])
ax.set_xlabel("% bought")
ax.set_title("Tasa de compra según tag de reputación en title")
plt.tight_layout()
plt.savefig("output/reputation_tag_bought.png", dpi=150)
plt.close()

print("OK: 3 PNGs generados")
