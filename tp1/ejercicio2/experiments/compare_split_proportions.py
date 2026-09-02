"""
Compara, con los mismos datos y la misma lógica de estratificación de
split_data.py (agrupado por query_id, estratificado por franja de bought-rate,
random_state=42), qué tamaño le tocaría a valid/test bajo dos proporciones:
70/15/15 (la elegida) vs. 80/10/10 (la alternativa considerada) -- ver
Notas.md, sección "Split train / valid / test".

Solo cómputo, sin gráficos. Guarda output/split_proportion_comparison.csv.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "../docs/supermarket_products.csv"
OUTPUT_DIR = "output"
RANDOM_STATE = 42

BIN_EDGES = [-0.01, 0.0, 0.33, 1.0]
BIN_LABELS = ["0%", "1-33%", "34-100%"]

PROPORTIONS = {
    "70/15/15 (elegida)": (0.70, 0.15, 0.15),
    "80/10/10 (alternativa)": (0.80, 0.10, 0.10),
}


def split_summary(df: pd.DataFrame, train_size: float, valid_size: float, test_size: float) -> pd.DataFrame:
    query_stats = df.groupby("query_id").agg(bought_rate=("bought", "mean"), n_rows=("bought", "size"))
    query_stats["strata"] = pd.cut(query_stats["bought_rate"], bins=BIN_EDGES, labels=BIN_LABELS)

    query_ids = query_stats.index.to_numpy()
    strata = query_stats["strata"].astype(str).to_numpy()

    train_ids, rest_ids, _, rest_strata = train_test_split(
        query_ids, strata, train_size=train_size, random_state=RANDOM_STATE, stratify=strata
    )
    valid_ids, test_ids = train_test_split(
        rest_ids, train_size=valid_size / (valid_size + test_size), random_state=RANDOM_STATE, stratify=rest_strata
    )

    query_stats["split"] = "train"
    query_stats.loc[valid_ids, "split"] = "valid"
    query_stats.loc[test_ids, "split"] = "test"

    row_split = df["query_id"].map(query_stats["split"])
    summary = (
        pd.DataFrame({"split": row_split, "bought": df["bought"]})
        .groupby("split")
        .agg(n_rows=("bought", "size"), n_bought=("bought", "sum"), bought_rate=("bought", "mean"))
    )
    summary.insert(0, "n_queries", query_stats.groupby("split").size())
    return summary.loc[["train", "valid", "test"]]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    rows = []
    for label, (train_size, valid_size, test_size) in PROPORTIONS.items():
        summary = split_summary(df, train_size, valid_size, test_size)
        for split, row in summary.iterrows():
            rows.append(
                {
                    "proporcion": label,
                    "split": split,
                    "n_queries": int(row["n_queries"]),
                    "n_rows": int(row["n_rows"]),
                    "n_bought": int(row["n_bought"]),
                    "bought_rate": row["bought_rate"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUTPUT_DIR}/split_proportion_comparison.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
