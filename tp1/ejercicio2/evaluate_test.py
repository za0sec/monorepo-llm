"""
Evaluación final en test, con la configuración ya fijada (ver Experimentos.md).
Se corre una sola vez: test no participó de ninguna decisión de diseño.

Reporta las métricas por fila (PR-AUC / ROC-AUC) y el BTR por búsqueda, sobre
las 3 semillas, y en paralelo las mismas de valid para ver cuánto se degradan.

Guarda output/test_results.csv y output/btr_test.csv.
"""
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_btr import btr_by_query, predict, summarize
from run_final_config import CONFIG, EPOCHS, SEEDS
from train import load_split, train_model


def main() -> None:
    rows, btr_frames = [], []
    for seed in SEEDS:
        model, _ = train_model("fusion", seed=seed, epochs=EPOCHS, **CONFIG)
        for split in ["valid", "test"]:
            _, _, _, bought, _ = load_split(split)
            preds = predict(model, split)
            btr = btr_by_query(split, preds)
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "pr_auc": average_precision_score(bought, preds),
                    "roc_auc": roc_auc_score(bought, preds),
                    **summarize(btr),
                }
            )
            if split == "test":
                btr_frames.append(btr.assign(seed=seed))

    results = pd.DataFrame(rows)
    results.to_csv("output/test_results.csv", index=False)
    pd.concat(btr_frames).to_csv("output/btr_test.csv", index=False)
    print(results.groupby("split")[["pr_auc", "roc_auc", "mae", "correlacion"]].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
