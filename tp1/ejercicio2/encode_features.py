"""
Encoding de features para Ejercicio 2, siguiendo las decisiones de
ejercicio1/Notas.md ("Lista de features") y el split de split_data.py
(output/query_splits.csv).

Todo lo que se "fittea" (vocabulario de texto, categorías de one-hot, media/
desvío del z-score) se ajusta solo con filas de train y se aplica igual a
valid/test, para no filtrar información de esos splits al preprocesamiento.

Solo cómputo, sin gráficos. Guarda en data/:
- train.csv / valid.csv / test.csv: una fila por producto-búsqueda, con
  todas las features ya encodeadas, `bought` (target) y `query_id` (para
  agrupar por búsqueda al calcular el BTR agregado).
- vocab.csv: vocabulario de `title`+`description` (palabra, id, frecuencia
  en train), para el embedding del Transformer.
- preprocessing_stats.csv: media/desvío usados para el z-score de cada
  numérica (por si hace falta aplicar el mismo preprocesamiento afuera).
"""
import os
import re
from collections import Counter

import pandas as pd

DATA_PATH = "../docs/supermarket_products.csv"
SPLITS_PATH = "output/query_splits.csv"
OUTPUT_DIR = "data"

TOKEN_RE = re.compile(r"\d+\.\d+|[a-záéíóúñü]+(?:'[a-z]+)?|\d+")
# Cubre el 100% de las filas de title+description sin truncar (ver
# ejercicio1/Notas.md, sección de tokenización: máximo real = 45 tokens).
MAX_LEN = 45

DIMS_RE = re.compile(r"([0-9.]+)\s*x\s*([0-9.]+)\s*x\s*([0-9.]+)")

ONE_HOT_COLUMNS = ["category", "storage_type", "brand", "allergens", "country_of_origin"]
DIRECT_NUMERIC_COLUMNS = ["net_weight_oz", "nutrition_score"]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(train_tokens: pd.Series) -> tuple[dict, Counter]:
    freqs = Counter(word for tokens in train_tokens for word in tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in freqs.most_common():
        vocab[word] = len(vocab)
    return vocab, freqs


def encode_tokens(tokens: list[str], vocab: dict) -> tuple[str, int]:
    ids = [vocab.get(w, vocab["<UNK>"]) for w in tokens[:MAX_LEN]]
    length = len(ids)
    ids = ids + [vocab["<PAD>"]] * (MAX_LEN - length)
    return " ".join(map(str, ids)), length


def parse_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    dims = df["dimensions_in"].str.extract(DIMS_RE).astype(float)
    dims.columns = ["dim_length", "dim_width", "dim_height"]
    return dims


def one_hot(df: pd.DataFrame, column: str, categories: list) -> pd.DataFrame:
    values = df[column].fillna("None")
    cat = pd.Categorical(values, categories=categories)
    return pd.get_dummies(cat, prefix=column).astype(int)


def multi_hot_ingredients(df: pd.DataFrame, ingredient_tokens: list) -> pd.DataFrame:
    lists = df["ingredients"].fillna("").apply(
        lambda s: {t.strip() for t in s.split(",") if t.strip()}
    )
    data = {f"ingredient_{tok}": lists.apply(lambda s: int(tok in s)) for tok in ingredient_tokens}
    return pd.DataFrame(data)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    splits = pd.read_csv(SPLITS_PATH)[["query_id", "split"]]
    df = df.merge(splits, on="query_id", how="left")
    df["bought"] = df["bought"].astype(int)

    df = pd.concat([df, parse_dimensions(df)], axis=1)
    df["price_relpos"] = (df["price"] - df["filter_price_min"]) / (
        df["filter_price_max"] - df["filter_price_min"]
    )

    train_mask = df["split"] == "train"

    numeric_cols = DIRECT_NUMERIC_COLUMNS + ["price_relpos", "dim_length", "dim_width", "dim_height"]
    stats = df.loc[train_mask, numeric_cols].agg(["mean", "std"])
    for col in numeric_cols:
        df[f"{col}_z"] = (df[col] - stats.loc["mean", col]) / stats.loc["std", col]
    z_cols = [f"{c}_z" for c in numeric_cols]

    onehot = pd.concat(
        [
            one_hot(df, col, sorted(df.loc[train_mask, col].fillna("None").unique()))
            for col in ONE_HOT_COLUMNS
        ],
        axis=1,
    )

    ingredient_tokens = sorted(
        {
            t.strip()
            for s in df.loc[train_mask, "ingredients"].dropna()
            for t in s.split(",")
            if t.strip()
        }
    )
    multihot = multi_hot_ingredients(df, ingredient_tokens)

    text_tokens = (df["title"].fillna("") + " " + df["description"].fillna("")).apply(tokenize)
    vocab, freqs = build_vocab(text_tokens[train_mask])
    encoded = text_tokens.apply(lambda toks: encode_tokens(toks, vocab))
    df["title_desc_tokens"] = encoded.apply(lambda x: x[0])
    df["title_desc_len"] = encoded.apply(lambda x: x[1])

    out = pd.concat(
        [
            df[["query_id", "bought"] + z_cols + ["title_desc_tokens", "title_desc_len", "split"]],
            onehot,
            multihot,
        ],
        axis=1,
    )

    for split_name in ["train", "valid", "test"]:
        subset = out[out["split"] == split_name].drop(columns=["split"])
        subset.to_csv(f"{OUTPUT_DIR}/{split_name}.csv", index=False)
        print(f"{split_name}: {len(subset)} filas, {subset.shape[1]} columnas")

    vocab_df = pd.DataFrame(
        [{"word": w, "id": i, "freq_train": freqs.get(w, 0)} for w, i in vocab.items()]
    ).sort_values("id")
    vocab_df.to_csv(f"{OUTPUT_DIR}/vocab.csv", index=False)
    print(f"vocabulario (train): {len(vocab_df) - 2} palabras + <PAD>/<UNK>")

    stats_out = stats.T.reset_index().rename(columns={"index": "feature"})
    stats_out.to_csv(f"{OUTPUT_DIR}/preprocessing_stats.csv", index=False)


if __name__ == "__main__":
    main()
