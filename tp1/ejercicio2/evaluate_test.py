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

Guarda:
- output/test_results.csv: PR-AUC/ROC-AUC de test por semilla.
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


def main() -> None:
    results = []
    os.makedirs(f"{OUTPUT_DIR}/runs", exist_ok=True)

    for seed in SEEDS:
        model, history_df, best_epoch, n_tabular = train_and_select_best(seed)
        history_df.to_csv(f"{OUTPUT_DIR}/runs/final_seed{seed}.csv", index=False)

        test_tokens, test_lengths, test_tabular, test_bought, _ = load_split("test", EXCLUDE_PREFIXES)
        test_loader = make_loader(test_tokens, test_lengths, test_tabular, test_bought, BATCH_SIZE, shuffle=False)
        test_metrics = evaluate(model, test_loader, "combined")

        n_params = sum(p.numel() for p in model.parameters())
        results.append(
            {
                "seed": seed,
                "n_params": n_params,
                "n_tabular_features": n_tabular,
                "best_epoch": best_epoch,
                "test_pr_auc": test_metrics["pr_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
        )
        print(
            f"[final seed={seed}] TEST pr_auc={test_metrics['pr_auc']:.4f} roc_auc={test_metrics['roc_auc']:.4f} "
            f"(mejor época: {best_epoch})"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/test_results.csv", index=False)

    summary = results_df[["test_pr_auc", "test_roc_auc"]].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
