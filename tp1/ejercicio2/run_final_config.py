"""
Confirma con 3 semillas la mejor variante de la ablación (d_model=32), que
había corrido con una sola. Ver Experimentos.md.

Guarda output/final_config_results.csv y output/runs/final_seed<n>.csv.
"""
import pandas as pd

from train import train_model

CONFIG = {"d_model": 32, "n_heads": 4, "n_layers": 2, "dim_feedforward": 128}
SEEDS = [0, 1, 2]
EPOCHS = 20


def main() -> None:
    rows = []
    for seed in SEEDS:
        _, history = train_model("fusion", seed=seed, epochs=EPOCHS, **CONFIG)
        history.to_csv(f"output/runs/final_seed{seed}.csv", index=False)
        best = history.loc[history["valid_pr_auc"].idxmax()]
        rows.append(
            {
                "seed": seed,
                "best_epoch": int(best["epoch"]),
                "best_valid_pr_auc": best["valid_pr_auc"],
                "best_valid_roc_auc": best["valid_roc_auc"],
                "best_epoch_pr_auc_gap": best["train_pr_auc"] - best["valid_pr_auc"],
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv("output/final_config_results.csv", index=False)
    print(results.to_string(index=False))
    print(results[["best_valid_pr_auc", "best_valid_roc_auc"]].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
