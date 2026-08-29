"""
EDA: tamaño del vocabulario de title+description, para decidir tokenización
por palabra vs. BPE chico.

Solo cómputo — no genera gráficos (ver plot_vocab.py para eso), siguiendo la
separación cómputo/gráficos acordada para el TP (CLAUDE.md).

Guarda:
- vocab_stats.csv: resumen (una fila) con los números clave de la decisión.
- vocab_freqs.csv: tabla completa palabra -> frecuencia, ordenada descendente.
"""
import re
from collections import Counter

import pandas as pd

DATA_PATH = "../docs/supermarket_products.csv"

# Tokenización simple por palabra: minúsculas, números (incluye decimales
# tipo "1.5") y palabras con apóstrofe (ej. contracciones) como un solo token;
# el resto de la puntuación (guiones, paréntesis, comas) se descarta como
# separador, tal como haría un tokenizador por palabra estándar.
TOKEN_RE = re.compile(r"\d+\.\d+|[a-záéíóúñü]+(?:'[a-z]+)?|\d+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    corpus = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()

    tokens: list[str] = []
    for row in corpus:
        tokens.extend(tokenize(row))

    freqs = Counter(tokens)

    n_rows = len(df)
    n_tokens = len(tokens)
    vocab_size = len(freqs)
    min_freq = min(freqs.values())
    hapax = sum(1 for c in freqs.values() if c == 1)
    freq_le_2 = sum(1 for c in freqs.values() if c <= 2)
    freq_le_5 = sum(1 for c in freqs.values() if c <= 5)

    stats = pd.DataFrame(
        [
            {
                "n_filas": n_rows,
                "n_tokens_totales": n_tokens,
                "tokens_promedio_por_fila": n_tokens / n_rows,
                "vocab_size_palabra": vocab_size,
                "freq_minima": min_freq,
                "n_hapax_freq_1": hapax,
                "n_freq_menor_igual_2": freq_le_2,
                "n_freq_menor_igual_5": freq_le_5,
            }
        ]
    )
    stats.to_csv("vocab_stats.csv", index=False)

    freq_table = (
        pd.DataFrame(freqs.items(), columns=["palabra", "frecuencia"])
        .sort_values("frecuencia", ascending=False)
        .reset_index(drop=True)
    )
    freq_table.to_csv("vocab_freqs.csv", index=False)

    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
