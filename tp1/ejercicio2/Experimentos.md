# Experimentos — Ejercicio 2

Registro de cada experimento corrido: configuración, resultados y análisis, para no tener que reconstruir esto de memoria al armar la presentación. Complementa a `Notas.md` (que tiene las decisiones de diseño); acá van los **números**.

Convención: scripts de cómputo (`train.py`, `run_experiments.py`) separados de los de gráficos (`plot_experiments.py`), igual que en el resto del TP — los gráficos de esta nota se generan leyendo `output/experiment_results.csv` y `output/runs/*.csv`, nunca reentrenando.

## Setup común

- **Datos**: `data/train.csv` / `data/valid.csv` (split de `split_data.py` + encoding de `encode_features.py`). Todas las métricas de esta nota son sobre **valid** — test no se toca hasta tener una configuración final elegida (ver `Notas.md`, sección "Split train/valid/test").
- **Arquitectura del bloque de texto**: Encoder-only (confirmado con el grupo, ver `Notas.md`) — `d_model=64`, 4 heads, 2 encoders apilados, MLP interno de 128, dropout 0.1. Implementado con `nn.TransformerEncoderLayer`/`nn.TransformerEncoder` de PyTorch (los módulos estándar, no atención hecha a mano) + positional encoding senoidal (el de la clase, `transformers.VTT`) + mean-pooling sobre los tokens no-pad para resumir la secuencia en un vector (`model.py::TextEncoder`).
- **Fusión con lo tabular**: tardía — el vector de texto (64) se concatena con el vector tabular (75 columnas: numéricas z-scoreadas + one-hot + multi-hot) y pasa por una capa densa (64) + salida (1 logit) (`model.py::EncoderFusionModel`).
- **Entrenamiento**: Adam (`lr=1e-3`), `BCEWithLogitsLoss`, batch size 128, 20 épocas. **No se agregó ponderación de clases ni ninguna técnica específica para el desbalance** (13% positivos) — no vimos nada puntual en clase para esto (ver `ejercicio1/Notas.md`), así que se dejó la loss estándar y se confía en PR-AUC/ROC-AUC como métrica (más informativa que accuracy con desbalance).
- **Métricas**: PR-AUC y ROC-AUC sobre valid, sin threshold (la consigna aclara que no hace falta). Se reporta la **mejor época según PR-AUC de valid** (no la última — ver curvas más abajo, siguen mejorando lentamente incluso en la época 20).
- **Semillas**: 3 corridas por configuración (semillas 0, 1, 2), se reporta media ± desvío — siguiendo la recomendación de `consigna.VTT` de promediar varias corridas en vez de una sola.

## Experimento 1: Baseline solo-tabular

Sin Transformer, sin texto — `model.py::TabularMLPBaseline`: capas densas (64 → 1) directo sobre las 75 columnas tabulares. Sirve como piso de comparación: si el modelo con texto no supera esto, sería evidencia de que `title`/`description` no está aportando.

| semilla | mejor época | PR-AUC (valid) | ROC-AUC (valid) |
|---|---|---|---|
| 0 | 18 | 0,178 | 0,579 |
| 1 | 20 | 0,178 | 0,583 |
| 2 | 18 | 0,178 | 0,579 |
| **media ± std** | | **0,178 ± 0,000** | **0,580 ± 0,002** |

Apenas por encima de la tasa base de `bought` (13,0%) en PR-AUC, y apenas por encima del azar (0,5) en ROC-AUC. Consistente con lo que ya habíamos visto en el EDA: las features tabulares con señal real (posición relativa del precio) son débiles comparadas con lo que se escondía en el texto, y `country_of_origin`/`nutrition_score` no aportaban nada en el cruce univariado.

## Experimento 2: Fusión tardía (Encoder-only + tabular)

`model.py::EncoderFusionModel` — el mismo tabular de arriba, pero concatenado con el vector que sale del Encoder-only sobre `title`+`description`.

| semilla | mejor época | PR-AUC (valid) | ROC-AUC (valid) |
|---|---|---|---|
| 0 | 20 | 0,743 | 0,965 |
| 1 | 17 | 0,761 | 0,965 |
| 2 | 15 | 0,719 | 0,967 |
| **media ± std** | | **0,741 ± 0,021** | **0,966 ± 0,001** |

Salto enorme sobre el baseline: **+0,56 en PR-AUC, +0,39 en ROC-AUC**. Confirma con números el hallazgo cualitativo del EDA (`ejercicio1/Notas.md`, sección del tag de reputación entre paréntesis en `title`): esa señal es tan fuerte y tan fácil de aprender vía atención que domina la predicción por completo.

![Comparación experimento 1 vs 2](output/experiment_comparison.png)

## Curvas de entrenamiento — convergencia y overfitting

![Curvas de entrenamiento](output/training_curves.png)

- **Baseline tabular**: mejora muy lenta y monótona a lo largo de las 20 épocas, sin señal de overfitting (no hay caída de valid) — probablemente porque el modelo es chico y las features tabulares tienen poca señal para explotar, así que no llega a memorizar.
- **Fusión (Encoder-only)**: converge muy rápido — la mayor parte de la mejora pasa en la primera época (de ~0,2-0,3 a ~0,65-0,68 de PR-AUC), consistente con que el patrón que aprende (el tag de reputación) es un patrón simple y muy regular de detectar. Después sigue mejorando lento hasta la época 20 sin señal clara de overfitting todavía (valid sigue subiendo, no baja) — sujeto a confirmar entrenando más épocas.

## Conclusión parcial

1. **El Transformer aporta muchísimo** sobre el baseline tabular — no es un requisito formal de la consigna sin sustancia real, hay señal genuina en el texto y el modelo la encuentra.
2. **Fusión tardía queda validada empíricamente**, no solo por el argumento teórico de `Notas.md` — con esta diferencia tan grande no hace falta correr la alternativa "todo como secuencia" para decidir cuál conviene (aunque se puede correr más adelante si se quiere comparar arquitecturas, no para decidir si vale la pena el texto).
3. El resultado es coherente con el `title` (test manual: revisar si el modelo realmente está mirando el tag entre paréntesis y no otra cosa — pendiente, ver abajo).

## Pendientes / próximos experimentos

- [ ] **Interpretabilidad**: confirmar que el modelo efectivamente está usando el tag de reputación y no otra correlación espuria — ej. mirar pesos de atención sobre una muestra de filas, o probar con el tag removido del texto y ver si el PR-AUC cae a niveles cercanos al baseline tabular.
- [ ] Entrenar más épocas (30-50) para confirmar si la fusión eventualmente empieza a overfittear (todavía no se ve en 20 épocas).
- [ ] Ablación de los "diales" restantes (ver `Notas.md`): variar heads (2/4/8), encoders apilados (1/2/4), `d_model`, y presencia/ausencia de `country_of_origin`/`nutrition_score`.
- [ ] Evaluar en test **una sola vez**, cuando se termine de elegir la configuración final (no todavía).
- [ ] Validar 70/15/15 vs. 80/10/10 corriendo con más semillas (pendiente de `Notas.md`).
- [ ] *(Opcional, baja prioridad dado el resultado)* Comparar contra la alternativa "todo como secuencia" descartada en `Notas.md`, si da el tiempo.
