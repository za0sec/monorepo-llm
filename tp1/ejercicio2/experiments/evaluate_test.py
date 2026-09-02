"""
Evalúa en test la configuración final del sistema completo, una sola vez
-- ver Notas.md: test no se toca hasta este punto, se usó solo train/valid
en los Experimentos 1 a 11. Es el único script del Ejercicio 2 que lee
data/test.csv.

Config final (ver Experimentos.md para el detalle de cada experimento):
- CombinedModel: n_heads=1, n_layers=2, d_model=64, dim_feedforward=64,
  hidden=256, positional encoding senoidal (confirmado, Experimento 11).
- Features tabulares: todas menos country_of_origin (Experimento 9).
- 20 épocas, seleccionando el checkpoint de la época con mejor PR-AUC de
  valid -- mismo criterio de "mejor época" que en el resto del estudio,
  pero acá sí se guarda una copia de los pesos de esa época (no solo la
  métrica) para poder evaluar ese modelo concreto en test.
- 3 semillas, promediadas (consigna.VTT: no reportar una sola corrida).

`bought` a nivel fila es la variable con la que se entrena y se mide
PR-AUC/ROC-AUC (ver Notas.md), pero lo que pide la consigna es el BTR de
una búsqueda -- el promedio de `bought` (o de la probabilidad predicha)
agrupando por query_id. Ese agregado no se calculaba en ningún lado del
estudio (Experimentos 1 a 11 solo miran la métrica a nivel fila), así que
acá, sobre las mismas 3 corridas ya entrenadas, se agrupan las
predicciones de test por query_id y se compara BTR real vs. predicho por
búsqueda -- sin reentrenar ni tocar test una segunda vez.

Guarda:
- output/test_results.csv: PR-AUC/ROC-AUC de test por semilla (fila) más
  MAE y correlación de Pearson del BTR agregado por query_id.
- output/btr_test.csv: una fila por (semilla, query_id) con el BTR real y
  el predicho de esa búsqueda, para el scatter de plots/plot_btr_test.py.
- output/runs/final_seed<semilla>.csv: historial por época (train/valid),
  mismo formato que el resto de los experimentos.
"""
import copy
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from model import CombinedModel
from train import DATA_DIR, MAX_LEN, OUTPUT_DIR, VOCAB_SIZE, evaluate, forward, load_split, make_loader

SEEDS = [0, 1, 2]
EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
EXCLUDE_PREFIXES = ("country_of_origin_",)
MODEL_KWARGS = {"n_heads": 1, "n_layers": 2, "d_model": 64, "dim_feedforward": 64, "hidden": 256}


def train_and_select_best(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_tokens, train_lengths, train_tabular, train_bought, tabular_cols = load_split(
        "train", EXCLUDE_PREFIXES
    )
    valid_tokens, valid_lengths, valid_tabular, valid_bought, _ = load_split("valid", EXCLUDE_PREFIXES)

    train_loader = make_loader(train_tokens, train_lengths, train_tabular, train_bought, BATCH_SIZE, shuffle=True)
    train_eval_loader = make_loader(
        train_tokens, train_lengths, train_tabular, train_bought, BATCH_SIZE, shuffle=False
    )
    valid_loader = make_loader(valid_tokens, valid_lengths, valid_tabular, valid_bought, BATCH_SIZE, shuffle=False)

    model = CombinedModel(VOCAB_SIZE, len(tabular_cols), max_len=MAX_LEN, **MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best_valid_pr_auc = -1.0
    best_state = None
    best_epoch = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for tokens, lengths, tabular, bought in train_loader:
            optimizer.zero_grad()
            logits = forward(model, "combined", tokens, lengths, tabular)
            loss = loss_fn(logits, bought)
            loss.backward()
            optimizer.step()

        rng_state = torch.get_rng_state()
        train_metrics = evaluate(model, train_eval_loader, "combined")
        torch.set_rng_state(rng_state)
        valid_metrics = evaluate(model, valid_loader, "combined")

        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"valid_{k}": v for k, v in valid_metrics.items()},
            }
        )
        print(
            f"[final seed={seed}] epoch {epoch:02d} "
            f"train_pr_auc={train_metrics['pr_auc']:.4f} valid_pr_auc={valid_metrics['pr_auc']:.4f}"
        )

        if valid_metrics["pr_auc"] > best_valid_pr_auc:
            best_valid_pr_auc = valid_metrics["pr_auc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch, len(tabular_cols)


@torch.no_grad()
def predict_test(model, loader) -> np.ndarray:
    """Probabilidad predicha (sigmoid del logit) fila por fila, en el mismo
    orden que test.csv (el loader no mezcla, shuffle=False)."""
    model.eval()
    preds = []
    for tokens, lengths, tabular, _bought in loader:
        logits = forward(model, "combined", tokens, lengths, tabular)
        preds.append(torch.sigmoid(logits).numpy())
    return np.concatenate(preds)


def btr_by_query(query_ids: np.ndarray, bought: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Agrupa fila->búsqueda: BTR real (promedio de bought) vs. predicho
    (promedio de la probabilidad predicha), una fila por query_id."""
    df = pd.DataFrame({"query_id": query_ids, "bought": bought, "pred": pred})
    return df.groupby("query_id").agg(
        n_rows=("bought", "size"), btr_real=("bought", "mean"), btr_predicted=("pred", "mean")
    ).reset_index()


def main() -> None:
    results = []
    btr_frames = []
    os.makedirs(f"{OUTPUT_DIR}/runs", exist_ok=True)

    test_query_ids = pd.read_csv(f"{DATA_DIR}/test.csv")["query_id"].to_numpy()

    for seed in SEEDS:
        model, history_df, best_epoch, n_tabular = train_and_select_best(seed)
        history_df.to_csv(f"{OUTPUT_DIR}/runs/final_seed{seed}.csv", index=False)

        test_tokens, test_lengths, test_tabular, test_bought, _ = load_split("test", EXCLUDE_PREFIXES)
        test_loader = make_loader(test_tokens, test_lengths, test_tabular, test_bought, BATCH_SIZE, shuffle=False)
        test_metrics = evaluate(model, test_loader, "combined")

        test_pred = predict_test(model, test_loader)
        btr = btr_by_query(test_query_ids, test_bought, test_pred)
        btr_mae = (btr["btr_real"] - btr["btr_predicted"]).abs().mean()
        btr_corr = btr["btr_real"].corr(btr["btr_predicted"])
        btr["seed"] = seed
        btr_frames.append(btr)

        n_params = sum(p.numel() for p in model.parameters())
        results.append(
            {
                "seed": seed,
                "n_params": n_params,
                "n_tabular_features": n_tabular,
                "best_epoch": best_epoch,
                "test_pr_auc": test_metrics["pr_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_btr_mae": btr_mae,
                "test_btr_pearson_r": btr_corr,
            }
        )
        print(
            f"[final seed={seed}] TEST pr_auc={test_metrics['pr_auc']:.4f} roc_auc={test_metrics['roc_auc']:.4f} "
            f"btr_mae={btr_mae:.4f} btr_r={btr_corr:.4f} (mejor época: {best_epoch})"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/test_results.csv", index=False)

    btr_df = pd.concat(btr_frames, ignore_index=True)
    btr_df.to_csv(f"{OUTPUT_DIR}/btr_test.csv", index=False)

    summary = results_df[["test_pr_auc", "test_roc_auc", "test_btr_mae", "test_btr_pearson_r"]].agg(
        ["mean", "std"]
    )
    print(summary)


if __name__ == "__main__":
    main()
