"""
Entrena una corrida del Experimento 1 (Transformer de texto puro, ver
Notas.md) para una semilla dada.

Guarda output/runs/exp1_seed<seed>.csv con el historial por época (loss,
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

from model import TextTransformerClassifier

DATA_DIR = "data"
OUTPUT_DIR = "output"
MAX_LEN = 45
VOCAB_SIZE = 412  # 410 palabras + <PAD> + <UNK>, ver data/vocab.csv


def load_split(name: str):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv")
    tokens = np.stack(df["title_desc_tokens"].apply(lambda s: np.array(s.split(), dtype="int64")))
    lengths = df["title_desc_len"].to_numpy(dtype="int64")
    bought = df["bought"].to_numpy(dtype="float32")
    return tokens, lengths, bought


def make_loader(tokens, lengths, bought, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(tokens), torch.from_numpy(lengths), torch.from_numpy(bought))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses, targets, preds = [], [], []
    loss_fn = nn.BCEWithLogitsLoss()
    for tokens, lengths, bought in loader:
        logits = model(tokens, lengths)
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


def train_model(seed: int, epochs: int = 20, batch_size: int = 128, lr: float = 1e-3, **model_kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_tokens, train_lengths, train_bought = load_split("train")
    valid_tokens, valid_lengths, valid_bought = load_split("valid")

    train_loader = make_loader(train_tokens, train_lengths, train_bought, batch_size, shuffle=True)
    train_eval_loader = make_loader(train_tokens, train_lengths, train_bought, batch_size, shuffle=False)
    valid_loader = make_loader(valid_tokens, valid_lengths, valid_bought, batch_size, shuffle=False)

    model = TextTransformerClassifier(VOCAB_SIZE, max_len=MAX_LEN, **model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for tokens, lengths, bought in train_loader:
            optimizer.zero_grad()
            logits = model(tokens, lengths)
            loss = loss_fn(logits, bought)
            loss.backward()
            optimizer.step()

        # Evaluación extra en modo eval (sin dropout) para que train y valid
        # sean comparables punto a punto -- ver train.py de referencia previa.
        rng_state = torch.get_rng_state()
        train_metrics = evaluate(model, train_eval_loader)
        torch.set_rng_state(rng_state)
        valid_metrics = evaluate(model, valid_loader)

        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"valid_{k}": v for k, v in valid_metrics.items()},
            }
        )
        print(
            f"[seed={seed}] epoch {epoch:02d} "
            f"train_pr_auc={train_metrics['pr_auc']:.4f} valid_pr_auc={valid_metrics['pr_auc']:.4f} "
            f"(gap={train_metrics['pr_auc'] - valid_metrics['pr_auc']:+.4f}) "
            f"valid_roc_auc={valid_metrics['roc_auc']:.4f}"
        )

    return model, pd.DataFrame(history)


def run(seed: int, tag: str, epochs: int = 20, **model_kwargs) -> dict:
    model, history_df = train_model(seed, epochs, **model_kwargs)

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
