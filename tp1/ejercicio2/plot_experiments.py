"""
Gráficos del experimento 1 vs. 2 (baseline tabular vs. fusión tardía con
Encoder-only), del chequeo de interpretabilidad de la señal de reputación, y
del estudio de ablación de arquitectura/features. Lee
output/experiment_results.csv, output/runs/*.csv (de run_experiments.py),
output/reputation_tag_check.csv (de check_reputation_tag.py) y
output/ablation_results.csv (de run_ablation.py) — no entrena ni recalcula
nada.
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

# --- Gráfico 2b: diagnóstico de overfitting / underfitting ---
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for col, model_type in enumerate(MODEL_LABELS):
    paths = sorted(glob.glob(f"{OUTPUT_DIR}/runs/{model_type}_seed*.csv"))
    for row, (metric, ylabel) in enumerate([("pr_auc", "PR-AUC"), ("loss", "loss (BCE)")]):
        ax = axes[row][col]
        for path in paths:
            history = pd.read_csv(path)
            ax.plot(history["epoch"], history[f"train_{metric}"], color="#DD8452", alpha=0.7)
            ax.plot(history["epoch"], history[f"valid_{metric}"], color="#4C72B0", alpha=0.7)
        ax.set_ylabel(ylabel)
        if row == 0:
            ax.set_title(MODEL_LABELS[model_type])
        else:
            ax.set_xlabel("época")

axes[0][0].legend(
    handles=[
        plt.Line2D([0], [0], color="#DD8452", label="train (modo eval)"),
        plt.Line2D([0], [0], color="#4C72B0", label="valid"),
    ],
    loc="lower right",
)
fig.suptitle("Diagnóstico de overfitting / underfitting — train vs. valid, 3 semillas")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/overfitting_diagnosis.png", dpi=150)
print(f"Guardado {OUTPUT_DIR}/overfitting_diagnosis.png")

# --- Gráfico 3: chequeo de interpretabilidad (con/sin señal de reputación en el texto) ---
tag_check = pd.read_csv(f"{OUTPUT_DIR}/reputation_tag_check.csv").set_index("variante")
variant_order = ["original", "sin_tag_title", "sin_reputacion"]
variant_labels = {
    "original": "Original",
    "sin_tag_title": "Sin tag en title",
    "sin_reputacion": "Sin tag\n+ sin frase en description",
}

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, metric, title in zip(axes, ["pr_auc", "roc_auc"], ["PR-AUC (valid)", "ROC-AUC (valid)"]):
    values = tag_check.loc[variant_order, metric]
    ax.bar([variant_labels[v] for v in variant_order], values, color="#4C72B0")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)

fig.suptitle("Interpretabilidad: ¿de dónde viene la señal del texto?")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/reputation_tag_check.png", dpi=150)
print(f"Guardado {OUTPUT_DIR}/reputation_tag_check.png")

# --- Gráfico 4: estudio de ablación (arquitectura + features tabulares) ---
ablation = pd.read_csv(f"{OUTPUT_DIR}/ablation_results.csv").set_index("variant")

ARCH_ORDER = ["heads_2", "base", "heads_8", "layers_1", "layers_4", "d_model_32", "d_model_96", "ff_64", "ff_256"]
TABULAR_ORDER = ["base", "sin_country_of_origin", "sin_nutrition_score", "sin_ambas"]
ARCH_LABELS = {
    "heads_2": "heads=2", "base": "base\n(64/4/2/128)", "heads_8": "heads=8",
    "layers_1": "layers=1", "layers_4": "layers=4",
    "d_model_32": "d_model=32", "d_model_96": "d_model=96",
    "ff_64": "ff=64", "ff_256": "ff=256",
}
TABULAR_LABELS = {
    "base": "base (todas)", "sin_country_of_origin": "sin\ncountry_of_origin",
    "sin_nutrition_score": "sin\nnutrition_score", "sin_ambas": "sin ambas",
}

fig, axes = plt.subplots(3, 1, figsize=(10, 11))

ax = axes[0]
values = ablation.loc[ARCH_ORDER, "best_valid_pr_auc"]
colors = ["#4C72B0" if v == "base" else "#8C8C8C" for v in ARCH_ORDER]
ax.bar([ARCH_LABELS[v] for v in ARCH_ORDER], values, color=colors)
ax.axhline(ablation.loc["base", "best_valid_pr_auc"], color="black", linestyle="--", linewidth=1)
ax.set_ylabel("PR-AUC (valid)")
ax.set_title("Ablación de arquitectura (1 semilla, 20 épocas c/u)")

ax = axes[1]
values = ablation.loc[ARCH_ORDER, "final_pr_auc_gap"]
ax.bar([ARCH_LABELS[v] for v in ARCH_ORDER], values, color=colors)
ax.axhline(ablation.loc["base", "final_pr_auc_gap"], color="black", linestyle="--", linewidth=1)
ax.set_ylabel("PR-AUC train − valid (época 20)")
ax.set_title("Brecha train-valid por variante (más alto = más overfitting)")

ax = axes[2]
values = ablation.loc[TABULAR_ORDER, "best_valid_pr_auc"]
colors = ["#4C72B0" if v == "base" else "#8C8C8C" for v in TABULAR_ORDER]
ax.bar([TABULAR_LABELS[v] for v in TABULAR_ORDER], values, color=colors)
ax.axhline(ablation.loc["base", "best_valid_pr_auc"], color="black", linestyle="--", linewidth=1)
ax.set_ylabel("PR-AUC (valid)")
ax.set_title("Ablación de features tabulares dudosas")

fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/ablation.png", dpi=150)
print(f"Guardado {OUTPUT_DIR}/ablation.png")

# --- Gráfico 5: BTR por query, predicho vs. real ---
btr = pd.read_csv(f"{OUTPUT_DIR}/btr_valid.csv")

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(btr["btr_real"], btr["btr_pred"], alpha=0.4, color="#4C72B0")
ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
ax.set_xlabel("BTR real de la búsqueda")
ax.set_ylabel("BTR predicho")
ax.set_title(f"BTR por búsqueda (valid, {len(btr)} queries)")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/btr_valid.png", dpi=150)
print(f"Guardado {OUTPUT_DIR}/btr_valid.png")
