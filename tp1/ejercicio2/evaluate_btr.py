"""
Del `bought` por fila al BTR por búsqueda, que es lo que pide la consigna.

El modelo predice P(bought) para cada producto impreso; el BTR de una query es
el promedio de esas probabilidades sobre sus filas, y se compara contra el BTR
real (promedio de `bought`). Ver ejercicio1/Notas.md, "Unidad de análisis".

Guarda output/btr_valid.csv (una fila por query) y output/btr_summary.csv.
"""
import numpy as np
import pandas as pd
import torch

from run_final_config import CONFIG, EPOCHS
from train import load_split, make_loader, train_model

SEED = 0
SPLIT = "valid"


@torch.no_grad()
def predict(model, split: str) -> np.ndarray:
    tokens, lengths, tabular, bought, _ = load_split(split)
    loader = make_loader(tokens, lengths, tabular, bought, batch_size=128, shuffle=False)
    model.eval()
    preds = [torch.sigmoid(model(tab, tok, ln)).numpy() for tab, tok, ln, _ in loader]
    return np.concatenate(preds)


def btr_by_query(split: str, preds: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(f"data/{split}.csv")[["query_id", "bought"]]
    df["pred"] = preds
    btr = df.groupby("query_id").agg(
        n_productos=("bought", "size"), btr_real=("bought", "mean"), btr_pred=("pred", "mean")
    )
    return btr.reset_index()


def summarize(btr: pd.DataFrame) -> dict:
    error = btr["btr_pred"] - btr["btr_real"]
    return {
        "n_queries": len(btr),
        "btr_real_medio": btr["btr_real"].mean(),
        "btr_pred_medio": btr["btr_pred"].mean(),
        "mae": error.abs().mean(),
        "sesgo": error.mean(),
        "correlacion": btr["btr_real"].corr(btr["btr_pred"]),
        "correlacion_rangos": btr["btr_real"].corr(btr["btr_pred"], method="spearman"),
    }


def main() -> None:
    model, _ = train_model("fusion", seed=SEED, epochs=EPOCHS, **CONFIG)
    btr = btr_by_query(SPLIT, predict(model, SPLIT))
    btr.to_csv(f"output/btr_{SPLIT}.csv", index=False)

    summary = pd.DataFrame([{"split": SPLIT, **summarize(btr)}])
    summary.to_csv("output/btr_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
