"""
Chequeo de interpretabilidad (ver Experimentos.md, sección "Pendientes"):
¿el salto de PR-AUC de la fusión tardía viene de la señal de "reputación"
escondida en el texto (el tag entre paréntesis en `title`, ver
ejercicio1/Notas.md), o de otra cosa?

Entrena el modelo de fusión una sola vez y lo evalúa sobre 3 variantes del
mismo valid, sin retocar los pesos:
1. original: texto tal cual.
2. sin_tag_title: se saca el tag entre paréntesis de `title`.
3. sin_reputacion: además de (2), se saca la frase de reputación que
   `description` agrega después de "Listed under ... storage." (se
   descubrió corriendo este mismo chequeo -- ver Experimentos.md: la
   variante 2 sola no bajaba el PR-AUC porque `description` repite la
   misma señal con otra frase).

Guarda output/reputation_tag_check.csv con el resultado de las 3 variantes.
"""
import re

import numpy as np
import pandas as pd

from encode_features import encode_tokens, tokenize
from train import evaluate, load_split, make_loader, train_model

TITLE_TAG_RE = re.compile(r"\s*\([^)]*\)\s*$")
DESC_REPUTATION_RE = re.compile(r"(Listed under .+? storage\.)\s*.+$", re.DOTALL)

SEED = 0
EPOCHS = 20


def load_vocab() -> dict:
    vocab_df = pd.read_csv("data/vocab.csv")
    return dict(zip(vocab_df["word"], vocab_df["id"]))


def load_valid_raw() -> pd.DataFrame:
    df = pd.read_csv("../docs/supermarket_products.csv")
    splits = pd.read_csv("output/query_splits.csv")[["query_id", "split"]]
    df = df.merge(splits, on="query_id", how="left")
    valid_df = df[df["split"] == "valid"].reset_index(drop=True)

    _, _, _, bought_from_csv, _ = load_split("valid")
    assert np.array_equal(valid_df["bought"].astype(int).to_numpy(), bought_from_csv.astype(int)), (
        "Las filas de valid no quedaron alineadas con data/valid.csv -- revisar el merge."
    )
    return valid_df


def build_tokens(title: pd.Series, description: pd.Series, vocab: dict):
    text = title.fillna("") + " " + description.fillna("")
    encoded = text.apply(tokenize).apply(lambda toks: encode_tokens(toks, vocab))
    tokens = np.stack(encoded.apply(lambda x: np.array(x[0].split(), dtype="int64")))
    lengths = encoded.apply(lambda x: x[1]).to_numpy(dtype="int64")
    return tokens, lengths


def main() -> None:
    print(f"Entrenando fusión (seed={SEED}, {EPOCHS} épocas) para el chequeo...")
    model, _ = train_model("fusion", seed=SEED, epochs=EPOCHS)

    _, _, tabular, bought, _ = load_split("valid")
    vocab = load_vocab()
    valid_df = load_valid_raw()

    title_no_tag = valid_df["title"].fillna("").apply(lambda s: TITLE_TAG_RE.sub("", s))
    desc_no_reputation = valid_df["description"].fillna("").apply(lambda s: DESC_REPUTATION_RE.sub(r"\1", s))

    n_title_changed = (title_no_tag != valid_df["title"].fillna("")).sum()
    n_desc_changed = (desc_no_reputation != valid_df["description"].fillna("")).sum()
    print(f"Filas con tag removido de title: {n_title_changed}/{len(bought)}")
    print(f"Filas con frase de reputación removida de description: {n_desc_changed}/{len(bought)}")

    variants = {
        "original": (valid_df["title"], valid_df["description"]),
        "sin_tag_title": (title_no_tag, valid_df["description"]),
        "sin_reputacion": (title_no_tag, desc_no_reputation),
    }

    rows = []
    for name, (title, description) in variants.items():
        tokens, lengths = build_tokens(title, description, vocab)
        loader = make_loader(tokens, lengths, tabular, bought, batch_size=128, shuffle=False)
        metrics = evaluate(model, loader, "fusion")
        print(f"{name:16s} -> PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
        rows.append({"variante": name, **metrics})

    pd.DataFrame(rows).to_csv("output/reputation_tag_check.csv", index=False)


if __name__ == "__main__":
    main()
