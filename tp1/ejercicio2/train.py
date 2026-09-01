"""
Entrena una corrida de un experimento del Ejercicio 2 (ver Notas.md y
Experimentos.md) para una semilla dada. Soporta dos tipos de modelo
(`model_type`):

- "text": `TextTransformerClassifier`, solo texto (Experimentos 1 a 6).
- "combined": `CombinedModel`, texto + features tabulares.

Guarda output/runs/<tag>_seed<seed>.csv con el historial por época (loss,
PR-AUC, ROC-AUC medidos sobre train y sobre valid en modo eval, para poder
diagnosticar overfitting/underfitting comparando ambas curvas -- ver el
punto 3 del Ejercicio 2 de la consigna).
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import CombinedModel, TextTransformerClassifier

DATA_DIR = "data"
OUTPUT_DIR = "output"
NON_TABULAR_COLUMNS = {"query_id", "bought", "title_desc_tokens", "title_desc_len"}

MAX_LEN = 45
VOCAB_SIZE = 412  # 410 palabras + <PAD> + <UNK>, ver data/vocab.csv


def load_split(name: str):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv")
    tabular_cols = [c for c in df.columns if c not in NON_TABULAR_COLUMNS]
    tabular = df[tabular_cols].to_numpy(dtype="float32")
    tokens = np.stack(df["title_desc_tokens"].apply(lambda s: np.array(s.split(), dtype="int64")))
    lengths = df["title_desc_len"].to_numpy(dtype="int64")
    bought = df["bought"].to_numpy(dtype="float32")
    return tokens, lengths, tabular, bought, tabular_cols


def make_loader(tokens, lengths, tabular, bought, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(tokens), torch.from_numpy(lengths), torch.from_numpy(tabular), torch.from_numpy(bought)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_model(model_type: str, n_tabular_features: int, **model_kwargs) -> nn.Module:
    if model_type == "text":
        return TextTransformerClassifier(VOCAB_SIZE, max_len=MAX_LEN, **model_kwargs)
    if model_type == "combined":
        return CombinedModel(VOCAB_SIZE, n_tabular_features, max_len=MAX_LEN, **model_kwargs)
    raise ValueError(f"model_type desconocido: {model_type}")


def forward(model, model_type, tokens, lengths, tabular):
    if model_type == "text":
        return model(tokens, lengths)
    return model(tabular, tokens, lengths)


@torch.no_grad()
def evaluate(model, loader, model_type):
    model.eval()
    losses, targets, preds = [], [], []
    loss_fn = nn.BCEWithLogitsLoss()
    for tokens, lengths, tabular, bought in loader:
        logits = forward(model, model_type, tokens, lengths, tabular)
        loss = loss_fn(logits, bought)
        losses.append(loss.item() * len(bought))
        targets.append(bought.numpy())
        preds.append(torch.sigmoid(logits).numpy())
    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    return {
        "loss": sum(losses) / len(targets),
        "pr_auc": average_precision_score(targets, preds),
        "roc_auc": roc_auc_score(targets, preds),
    }


def train_model(
    model_type: str, seed: int, epochs: int = 20, batch_size: int = 128, lr: float = 1e-3, **model_kwargs
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_tokens, train_lengths, train_tabular, train_bought, tabular_cols = load_split("train")
    valid_tokens, valid_lengths, valid_tabular, valid_bought, _ = load_split("valid")

    train_loader = make_loader(train_tokens, train_lengths, train_tabular, train_bought, batch_size, shuffle=True)
    train_eval_loader = make_loader(
        train_tokens, train_lengths, train_tabular, train_bought, batch_size, shuffle=False
    )
    valid_loader = make_loader(valid_tokens, valid_lengths, valid_tabular, valid_bought, batch_size, shuffle=False)

    model = build_model(model_type, len(tabular_cols), **model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for tokens, lengths, tabular, bought in train_loader:
            optimizer.zero_grad()
            logits = forward(model, model_type, tokens, lengths, tabular)
            loss = loss_fn(logits, bought)
            loss.backward()
            optimizer.step()

        # Evaluación extra en modo eval (sin dropout) para que train y valid
        # sean comparables punto a punto.
        rng_state = torch.get_rng_state()
        train_metrics = evaluate(model, train_eval_loader, model_type)
        torch.set_rng_state(rng_state)
        valid_metrics = evaluate(model, valid_loader, model_type)

        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"valid_{k}": v for k, v in valid_metrics.items()},
            }
        )
        print(
            f"[{model_type} seed={seed}] epoch {epoch:02d} "
            f"train_pr_auc={train_metrics['pr_auc']:.4f} valid_pr_auc={valid_metrics['pr_auc']:.4f} "
            f"(gap={train_metrics['pr_auc'] - valid_metrics['pr_auc']:+.4f}) "
            f"valid_roc_auc={valid_metrics['roc_auc']:.4f}"
        )

    return model, pd.DataFrame(history)


def run(seed: int, tag: str, model_type: str = "text", epochs: int = 20, **model_kwargs) -> dict:
    model, history_df = train_model(model_type, seed, epochs, **model_kwargs)

    os.makedirs(f"{OUTPUT_DIR}/runs", exist_ok=True)
    history_df.to_csv(f"{OUTPUT_DIR}/runs/{tag}_seed{seed}.csv", index=False)

    best = history_df.loc[history_df["valid_pr_auc"].idxmax()]
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "seed": seed,
        "n_params": n_params,
        "best_epoch": int(best["epoch"]),
        "best_valid_pr_auc": best["valid_pr_auc"],
        "best_valid_roc_auc": best["valid_roc_auc"],
        "best_train_pr_auc": best["train_pr_auc"],
        "best_pr_auc_gap": best["train_pr_auc"] - best["valid_pr_auc"],
    }


if __name__ == "__main__":
    print(run(seed=0, tag="exp1", epochs=20))
