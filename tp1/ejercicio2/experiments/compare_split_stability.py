"""
Corre el split real (agrupado y estratificado por query, misma lógica de
split_data.py) muchas veces con semillas distintas, para cada proporción
(70/15/15 vs. 80/10/10), y mide qué tan lejos queda la tasa de bought de
valid/test respecto de la tasa global -- una sola semilla (random_state=42)
no alcanza para saber si una proporción es más estable que otra, hace falta
repetir. Ver Notas.md, sección "Split train / valid / test".

Solo cómputo, sin gráficos (ver plot_split_stability.py). Guarda
output/split_stability.csv: una fila por (proporción, split, semilla) con la
tasa de bought de esa corrida puntual.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "../docs/supermarket_products.csv"
OUTPUT_DIR = "output"
N_SEEDS = 300

BIN_EDGES = [-0.01, 0.0, 0.33, 1.0]
BIN_LABELS = ["0%", "1-33%", "34-100%"]

PROPORTIONS = {
    "70/15/15": (0.70, 0.15, 0.15),
    "80/10/10": (0.80, 0.10, 0.10),
}


def run_split(df, query_stats, query_ids, strata, train_size, valid_size, test_size, seed):
    train_ids, rest_ids, _, rest_strata = train_test_split(
        query_ids, strata, train_size=train_size, random_state=seed, stratify=strata
    )
    valid_ids, test_ids = train_test_split(
        rest_ids, train_size=valid_size / (valid_size + test_size), random_state=seed, stratify=rest_strata
    )
    valid_rate = df.loc[df["query_id"].isin(valid_ids), "bought"].mean()
    test_rate = df.loc[df["query_id"].isin(test_ids), "bought"].mean()
    return valid_rate, test_rate


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    query_stats = df.groupby("query_id").agg(bought_rate=("bought", "mean"), n_rows=("bought", "size"))
    query_stats["strata"] = pd.cut(query_stats["bought_rate"], bins=BIN_EDGES, labels=BIN_LABELS)
    query_ids = query_stats.index.to_numpy()
    strata = query_stats["strata"].astype(str).to_numpy()

    global_rate = df["bought"].mean()

    rows = []
    for label, (train_size, valid_size, test_size) in PROPORTIONS.items():
        for seed in range(N_SEEDS):
            valid_rate, test_rate = run_split(
                df, query_stats, query_ids, strata, train_size, valid_size, test_size, seed
            )
            rows.append({"proporcion": label, "split": "valid", "seed": seed, "bought_rate": valid_rate})
            rows.append({"proporcion": label, "split": "test", "seed": seed, "bought_rate": test_rate})

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUTPUT_DIR}/split_stability.csv", index=False)

    summary = (
        out.assign(dist=lambda d: (d["bought_rate"] - global_rate).abs() * 100)
        .groupby(["proporcion", "split"])["dist"]
        .mean()
        .unstack("split")
    )
    print(f"tasa global: {global_rate*100:.2f}%")
    print(f"distancia promedio a la tasa global, en puntos porcentuales, sobre {N_SEEDS} semillas:")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
