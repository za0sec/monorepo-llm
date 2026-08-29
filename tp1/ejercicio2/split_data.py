"""
Split train/valid/test (70/15/15) agrupado por query_id y estratificado por
la tasa de `bought` de cada query — ver ejercicio2/Notas.md, sección
"Split train / valid / test".

Solo cómputo, sin gráficos (ver plot_split.py). Guarda en output/:
- query_splits.csv: query_id -> split, con su tasa de bought y la franja
  usada para estratificar.
- split_summary.csv: resumen por split (cantidad de queries, filas, filas
  con bought=True y tasa de bought a nivel fila).
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "../docs/supermarket_products.csv"
OUTPUT_DIR = "output"
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALID_SIZE = 0.15
TEST_SIZE = 0.15

# Franjas para estratificar por tasa de bought a nivel de query. No se usa la
# tasa exacta como clase porque varias tasas (ej. 0.57, 1.00) tienen solo 2-10
# queries en todo el dataset -- insuficiente para partir en 3 (train/valid/
# test) de forma confiable. "0%" queda separada porque es más de la mitad de
# las queries (1058/2012) y es un caso distinto (ninguna compra en esa
# búsqueda); el resto se agrupa en una franja de tasa baja y una alta.
BIN_EDGES = [-0.01, 0.0, 0.33, 1.0]
BIN_LABELS = ["0%", "1-33%", "34-100%"]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    query_stats = df.groupby("query_id").agg(
        bought_rate=("bought", "mean"), n_rows=("bought", "size")
    )
    query_stats["strata"] = pd.cut(
        query_stats["bought_rate"], bins=BIN_EDGES, labels=BIN_LABELS
    )

    query_ids = query_stats.index.to_numpy()
    strata = query_stats["strata"].astype(str).to_numpy()

    train_ids, rest_ids, _, rest_strata = train_test_split(
        query_ids,
        strata,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=strata,
    )
    valid_ids, test_ids = train_test_split(
        rest_ids,
        train_size=VALID_SIZE / (VALID_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=rest_strata,
    )

    query_stats["split"] = "train"
    query_stats.loc[valid_ids, "split"] = "valid"
    query_stats.loc[test_ids, "split"] = "test"
    query_stats.to_csv(f"{OUTPUT_DIR}/query_splits.csv")

    row_split = df["query_id"].map(query_stats["split"])
    summary = (
        pd.DataFrame({"split": row_split, "bought": df["bought"]})
        .groupby("split")
        .agg(n_rows=("bought", "size"), n_bought=("bought", "sum"), bought_rate=("bought", "mean"))
    )
    summary.insert(0, "n_queries", query_stats.groupby("split").size())
    summary.to_csv(f"{OUTPUT_DIR}/split_summary.csv")

    print(summary.to_string())


if __name__ == "__main__":
    main()
