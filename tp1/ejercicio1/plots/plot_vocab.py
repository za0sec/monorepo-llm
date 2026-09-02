"""
Gráfico para la decisión de tokenización: frecuencia de cada palabra del
vocabulario de title+description, ordenada de más a menos frecuente.

Lee output/vocab_freqs.csv (generado por experiments/vocab_eda.py) — no
recalcula nada, solo plotea, siguiendo la separación cómputo/gráficos del TP.

Uso (desde ejercicio1/): python3 plots/plot_vocab.py
"""
import matplotlib.pyplot as plt
import pandas as pd

freqs = pd.read_csv("output/vocab_freqs.csv").sort_values("frecuencia", ascending=False)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(range(1, len(freqs) + 1), freqs["frecuencia"], marker=".", linewidth=1)
ax.set_yscale("log")
ax.set_xlabel("rango de la palabra (de más a menos frecuente)")
ax.set_ylabel("frecuencia (escala log)")
ax.set_title(
    f"Frecuencia por palabra en title+description "
    f"(vocabulario = {len(freqs)} palabras, frecuencia mínima = {freqs['frecuencia'].min()})"
)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("output/vocab_freq_rank.png", dpi=150)
print("Guardado output/vocab_freq_rank.png")
