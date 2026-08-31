"""
Calcula los datos de los 3 graficos pendientes para la presentacion del
Ejercicio 1 (ver seccion "Para la presentacion" de Notas.md). Solo escribe
CSVs -- el ploteo va en plot_presentation_extra.py, por separado.
"""
import pandas as pd

df = pd.read_csv("../docs/supermarket_products.csv")

# 1) "campana" de bought segun posicion relativa del precio dentro del rango filtrado
pos = (df["price"] - df["filter_price_min"]) / (df["filter_price_max"] - df["filter_price_min"])
df["pos_relativa"] = pos.clip(0, 1)
bins = pd.cut(df["pos_relativa"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
campana = df.groupby(bins, observed=True)["bought"].agg(["mean", "count"]).reset_index()
campana.columns = ["bin_pos_relativa", "pct_bought", "n"]
campana["bin_pos_relativa"] = campana["bin_pos_relativa"].astype(str)
campana.to_csv("price_position_bought.csv", index=False)

# 2) boxplots antes/despues de z-score para las numericas usadas
numericas = ["price", "net_weight_oz", "nutrition_score"]
crudo = df[numericas].copy()
crudo["etapa"] = "original"
z = (df[numericas] - df[numericas].mean()) / df[numericas].std()
z["etapa"] = "z-score"
pd.concat([crudo, z], ignore_index=True).to_csv("normalizacion_boxplot.csv", index=False)

# 3) % bought por tag de reputacion en title
tag = df["title"].str.extract(r"\(([^)]+)\)\s*$")[0].fillna("(sin tag)")
tag_tabla = df.groupby(tag)["bought"].agg(["mean", "count"]).reset_index()
tag_tabla.columns = ["tag", "pct_bought", "n"]
tag_tabla = tag_tabla.sort_values("pct_bought", ascending=False)
tag_tabla.to_csv("reputation_tag_bought.csv", index=False)

print("OK:", len(campana), len(numericas) * 2 * len(df), len(tag_tabla))
