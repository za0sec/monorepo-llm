"""
Gráficos para validar visualmente el split train/valid/test: que la tasa de
`bought` y la composición por franja de estratificación queden parejas entre
splits, y que los tamaños (filas/queries) por split queden a la vista para
justificar la proporción 70/15/15 elegida. Lee output/split_summary.csv y
output/query_splits.csv (generados por split_data.py) — no recalcula nada.
"""
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "output"
SPLIT_ORDER = ["train", "valid", "test"]
STRATA_ORDER = ["0%", "1-33%", "34-100%"]

summary = pd.read_csv(f"{OUTPUT_DIR}/split_summary.csv", index_col="split").loc[SPLIT_ORDER]
query_splits = pd.read_csv(f"{OUTPUT_DIR}/query_splits.csv")

overall_rate = (
    (query_splits["bought_rate"] * query_splits["n_rows"]).sum()
    / query_splits["n_rows"].sum()
)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

ax = axes[0, 0]
ax.bar(SPLIT_ORDER, summary["bought_rate"] * 100, color="#4C72B0")
ax.axhline(overall_rate * 100, color="black", linestyle="--", linewidth=1, label="tasa global")
for i, split in enumerate(SPLIT_ORDER):
    ax.text(i, summary.loc[split, "bought_rate"] * 100 + 0.15, f"{summary.loc[split, 'bought_rate']*100:.2f}%", ha="center", fontsize=9)
ax.set_ylabel("% bought (a nivel fila)")
ax.set_title("Tasa de bought por split")
ax.legend()

ax = axes[0, 1]
counts = (
    query_splits.groupby(["split", "strata"]).size().unstack("strata")[STRATA_ORDER].loc[SPLIT_ORDER]
)
proportions = counts.div(counts.sum(axis=1), axis=0) * 100
proportions.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
ax.set_ylabel("% de queries del split")
ax.set_title("Composición por franja de bought-rate (query-level)")
ax.legend(title="franja", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.tick_params(axis="x", rotation=0)

ax = axes[1, 0]
bars = ax.bar(SPLIT_ORDER, summary["n_rows"], color="#55A868")
ax.bar_label(bars, padding=3)
ax.set_ylabel("cantidad de filas")
ax.set_title("Filas por split")

ax = axes[1, 1]
bars = ax.bar(SPLIT_ORDER, summary["n_queries"], color="#C44E52")
ax.bar_label(bars, padding=3)
ax.set_ylabel("cantidad de queries")
ax.set_title("Queries por split")

fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/split_balance.png", dpi=150)
print(f"Guardado {OUTPUT_DIR}/split_balance.png")
