"""
Gráficos del experimento 1 vs. 2 (baseline tabular vs. fusión tardía con
Encoder-only). Lee output/experiment_results.csv y output/runs/*.csv
(generados por run_experiments.py) — no entrena ni recalcula nada.
"""
import glob

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "output"
MODEL_LABELS = {"tabular": "Baseline tabular", "fusion": "Fusión tardía (Encoder-only)"}
MODEL_COLORS = {"tabular": "#DD8452", "fusion": "#4C72B0"}

results = pd.read_csv(f"{OUTPUT_DIR}/experiment_results.csv")

# --- Gráfico 1: comparación de métricas por modelo (mean +/- std entre semillas) ---
summary = results.groupby("model")[["best_valid_pr_auc", "best_valid_roc_auc"]].agg(["mean", "std"])

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, metric, title in zip(axes, ["best_valid_pr_auc", "best_valid_roc_auc"], ["PR-AUC (valid)", "ROC-AUC (valid)"]):
    models = list(MODEL_LABELS.keys())
    means = [summary.loc[m, (metric, "mean")] for m in models]
    stds = [summary.loc[m, (metric, "std")] for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    ax.bar([MODEL_LABELS[m] for m in models], means, yerr=stds, capsize=4, color=colors)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)

fig.suptitle("Experimento 1 vs. 2 — mejor época de valid, promedio de 3 semillas")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/experiment_comparison.png", dpi=150)
print(f"Guardado {OUTPUT_DIR}/experiment_comparison.png")

# --- Gráfico 2: curvas de entrenamiento (valid PR-AUC por época, todas las semillas) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
for model_type in MODEL_LABELS:
    for path in sorted(glob.glob(f"{OUTPUT_DIR}/runs/{model_type}_seed*.csv")):
        history = pd.read_csv(path)
        ax.plot(history["epoch"], history["valid_pr_auc"], color=MODEL_COLORS[model_type], alpha=0.6)

handles = [plt.Line2D([0], [0], color=c, label=MODEL_LABELS[m]) for m, c in MODEL_COLORS.items()]
ax.legend(handles=handles)
ax.set_xlabel("época")
ax.set_ylabel("PR-AUC (valid)")
ax.set_title("Curvas de entrenamiento por semilla")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150)
print(f"Guardado {OUTPUT_DIR}/training_curves.png")
