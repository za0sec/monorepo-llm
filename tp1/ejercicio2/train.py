"""
Entrena una corrida (un modelo, una semilla) y guarda métricas por época.

No es para correr manualmente uno por uno: run_experiments.py llama a
`run()` para cada combinación de modelo x semilla del plan de experimentos
(ver Experimentos.md). Métricas evaluadas solo sobre valid (test se toca
una sola vez al final, ver Notas.md).

Guarda output/runs/<model>_seed<seed>.csv con el historial por época, con las
mismas 3 métricas (loss, PR-AUC, ROC-AUC) medidas sobre train y sobre valid.
Tener las dos permite diagnosticar overfitting/underfitting comparando ambas
curvas, que es lo que pide el punto 3 del Ejercicio 2 de la consigna
("métricas propias de modelos teniendo en cuenta overfitting y underfitting").

Ojo con `train_loss_running`: es el promedio de las losses de los minibatches
tal como se fueron calculando durante la época, con dropout activo y con los
pesos cambiando en cada paso. NO es comparable contra `valid_loss`, que se
mide en modo eval, sin dropout y con los pesos ya fijos al final de la época.
Por eso se agrega una pasada de evaluación extra sobre train en modo eval:
`train_loss`/`train_pr_auc`/`train_roc_auc` sí son comparables punto a punto
contra sus equivalentes de valid.
"""
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import EncoderFusionModel, TabularMLPBaseline

warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

DATA_DIR = "data"
OUTPUT_DIR = "output"
NON_TABULAR_COLUMNS = {"query_id", "bought", "title_desc_tokens", "title_desc_len"}

MAX_LEN = 45
VOCAB_SIZE = 412  # 410 palabras + <PAD> + <UNK>, ver data/vocab.csv


def load_split(name: str, exclude_prefixes: tuple = ()):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv")
    tabular_cols = [
        c
        for c in df.columns
        if c not in NON_TABULAR_COLUMNS and not any(c.startswith(p) for p in exclude_prefixes)
    ]
    tabular = df[tabular_cols].to_numpy(dtype="float32")
    tokens = np.stack(df["title_desc_tokens"].apply(lambda s: np.array(s.split(), dtype="int64")))
    lengths = df["title_desc_len"].to_numpy(dtype="int64")
    bought = df["bought"].to_numpy(dtype="float32")
    return tokens, lengths, tabular, bought, tabular_cols


def make_loader(tokens, lengths, tabular, bought, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(tabular),
        torch.from_numpy(tokens),
        torch.from_numpy(lengths),
        torch.from_numpy(bought),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_model(model_type: str, n_tabular_features: int, **model_kwargs) -> nn.Module:
    if model_type == "tabular":
        return TabularMLPBaseline(n_tabular_features)
    if model_type == "fusion":
        return EncoderFusionModel(VOCAB_SIZE, n_tabular_features, max_len=MAX_LEN, **model_kwargs)
    raise ValueError(f"model_type desconocido: {model_type}")


@torch.no_grad()
def evaluate(model, loader, model_type):
    model.eval()
    losses, targets, preds = [], [], []
    loss_fn = nn.BCEWithLogitsLoss()
    for tabular, tokens, lengths, bought in loader:
        if model_type == "tabular":
            logits = model(tabular)
        else:
            logits = model(tabular, tokens, lengths)
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
    model_type: str,
    seed: int,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    exclude_prefixes: tuple = (),
    **model_kwargs,
):
    """Entrena una corrida y devuelve (modelo entrenado, historial por época).

    `exclude_prefixes` saca columnas tabulares por prefijo (ej.
    "country_of_origin_" para el módulo de ablación de esa feature).
    `model_kwargs` se pasan al constructor del modelo (d_model, n_heads,
    n_layers, dim_feedforward -- ver model.py::EncoderFusionModel).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_tokens, train_lengths, train_tabular, train_bought, tabular_cols = load_split(
        "train", exclude_prefixes
    )
    valid_tokens, valid_lengths, valid_tabular, valid_bought, _ = load_split("valid", exclude_prefixes)

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
        train_losses = []
        for tabular, tokens, lengths, bought in train_loader:
            optimizer.zero_grad()
            if model_type == "tabular":
                logits = model(tabular)
            else:
                logits = model(tabular, tokens, lengths)
            loss = loss_fn(logits, bought)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item() * len(bought))

        train_loss_running = sum(train_losses) / len(train_bought)
        # Iterar un DataLoader consume el RNG global, así que se guarda y
        # restaura el estado alrededor de esta evaluación extra: sin esto, las
        # corridas dejarían de ser bit a bit iguales a las de antes de agregarla.
        rng_state = torch.get_rng_state()
        train_metrics = evaluate(model, train_eval_loader, model_type)
        torch.set_rng_state(rng_state)
        valid_metrics = evaluate(model, valid_loader, model_type)
        history.append(
            {
                "epoch": epoch,
                "train_loss_running": train_loss_running,
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


def run(model_type: str, seed: int, epochs: int = 30, batch_size: int = 128, lr: float = 1e-3) -> dict:
    model, history_df = train_model(model_type, seed, epochs, batch_size, lr)

    os.makedirs(f"{OUTPUT_DIR}/runs", exist_ok=True)
    history_df.to_csv(f"{OUTPUT_DIR}/runs/{model_type}_seed{seed}.csv", index=False)

    best = history_df.loc[history_df["valid_pr_auc"].idxmax()]
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": model_type,
        "seed": seed,
        "n_params": n_params,
        "best_epoch": int(best["epoch"]),
        "best_valid_pr_auc": best["valid_pr_auc"],
        "best_valid_roc_auc": best["valid_roc_auc"],
        "best_epoch_train_pr_auc": best["train_pr_auc"],
        "best_epoch_train_roc_auc": best["train_roc_auc"],
        "best_epoch_pr_auc_gap": best["train_pr_auc"] - best["valid_pr_auc"],
        "final_valid_pr_auc": history_df.iloc[-1]["valid_pr_auc"],
        "final_valid_roc_auc": history_df.iloc[-1]["valid_roc_auc"],
        "final_train_pr_auc": history_df.iloc[-1]["train_pr_auc"],
        "final_pr_auc_gap": history_df.iloc[-1]["train_pr_auc"] - history_df.iloc[-1]["valid_pr_auc"],
    }


if __name__ == "__main__":
    print(run("tabular", seed=0, epochs=5))
