# Notas — Ejercicio 2 (Desarrollo del sistema)

Notas de trabajo, en construcción. Contenido trasladado desde [`../ejercicio1/Notas.md`](../ejercicio1/Notas.md): surgió naturalmente charlando el EDA del Ejercicio 1, pero son decisiones de diseño del sistema (Ejercicio 2), no del EDA en sí — ver esa nota para el detalle completo de variable objetivo, features y su preprocesamiento.

Punto de partida que deja cerrado el Ejercicio 1:
- **Variable objetivo**: `bought` a nivel fila. El BTR de una búsqueda sale de promediar `bought` (o las probabilidades predichas) agrupando por `query_id`.
- **Lista de features cerrada**: `cart` excluida por leakage; `filter_category`/`filter_storage_type`/`package_size`/`unit_of_measure` descartadas por redundancia; `country_of_origin` y `nutrition_score` sin señal univariada clara, quedan como candidatas de ablación.
- **Preprocesamiento por feature ya decidido**: one-hot para categóricas nominales (`category`, `storage_type`, `brand`, `allergens`), z-score para numéricas (`net_weight_oz`, derivadas), posición relativa de `price` dentro del rango filtrado, `dimensions_in` parseado a numéricas, tokenización **por palabra** para `title`/`description` (vocabulario cerrado de 410 palabras, sin necesidad de BPE).

Lo que falta acá es **cómo se junta todo eso en una arquitectura concreta** — eso es el contenido de este documento.

**Estructura de la carpeta:**
- `split_data.py`, `encode_features.py`: cómputo de split y encoding (ver secciones de abajo).
- `model.py`, `train.py`, `run_experiment<n>.py`: arquitectura y entrenamiento de cada experimento (empezando por el Experimento 1, texto puro — ver `Experimentos.md`). `plot_split.py`, `plot_experiment.py`: gráficos, separados del cómputo igual que en `ejercicio1/` — leen CSVs, nunca recalculan ni reentrenan. `plot_experiment.py` es genérico para cualquier experimento (`python3 plot_experiment.py <n>`), no un script por experimento.
- `data/`: datasets ya encodeados y listos para el modelo (`train.csv`, `valid.csv`, `test.csv`, `vocab.csv`, `preprocessing_stats.csv`).
- `output/`: resultados — diagnóstico del split (`query_splits.csv`, `split_summary.csv`, `split_balance.png`) y de cada experimento (`experiment<n>_results.csv`, `runs/*.csv` con el historial por época, `experiment<n>_curves.png`).
- `Experimentos.md`: registro de cada experimento corrido — arquitectura, justificación, resultados, análisis de métricas y qué cambia en el siguiente (para armar la presentación). Este documento (`Notas.md`) se queda con las decisiones de diseño más generales (split, encoding, elección Encoder-only/fusión tardía); los números de cada corrida van en `Experimentos.md`.

**Nota (2026-08-31): se reinició el modelado desde cero** para poder iterar experimento por experimento de forma más controlada — se borraron las corridas/resultados previos (`model.py`, `train.py`, `output/` de esa etapa). El Experimento 1 arrancó como Transformer de texto puro (sin fusión tabular todavía), no como baseline tabular vs. fusión — ver `Experimentos.md` para el detalle y el porqué de ese alcance.

## Split train / valid / test

**Por qué agrupar por `query_id`:** las filas están agrupadas por búsqueda — cada `query_id` trae varios productos que comparten contexto (misma categoría, mismo rango de precio filtrado). Si se particiona fila por fila al azar, filas de una misma búsqueda podrían terminar repartidas entre train y test: el modelo ya habría visto ese contexto exacto (categoría, rango de precio) durante el entrenamiento, lo cual no refleja el caso real de uso (predecir sobre búsquedas *nuevas* que nunca se vieron). Por eso: **la partición se hace por `query_id` completo** — todas las filas de una búsqueda van juntas al mismo split, nunca mezcladas.

**Por qué 3 particiones y no 2:** el PDF de la consigna lo pide explícitamente ("recordar train/valid/test split"). Además tiene sentido con el Ejercicio 2 (experimentar con configuraciones del modelo):
- **train**: ajusta los pesos del modelo.
- **valid**: compara entre configuraciones/experimentos (arquitectura, `d_model`, etc.) durante la iteración.
- **test**: se toca una sola vez, al final, para el número que se reporta — si se usara valid para elegir la mejor configuración y también para reportar el resultado final, el número quedaría inflado (overfit a valid). **Hecho** (ver [`Experimentos.md`](Experimentos.md), sección "Evaluación final en test"): PR-AUC de test 0,809 ± 0,003, prácticamente igual a lo que daba valid con la config ganadora — sin señales de haber sobreajustado las decisiones de arquitectura a valid a lo largo del estudio.

**Precedente en TPs anteriores (SIA-TP5, autoencoders):** usaron **K-fold estratificado** como estrategia principal (entrenar con varios folds y promediar resultados, `mean`/`std` en el output), con un `val_fraction=0.2` (80/20) como caso simple para cuando no se hacían folds (`mlp/data.py::train_val_split`). Esto conecta con algo que también dice el audio de `consigna.VTT` de esta materia: la profesora recomienda "promediar varias corridas (o cross-validation) en vez de una sola ejecución" — o sea, el mismo criterio de TP5 tiene aval acá también.

**Decisión para este TP:** se eligió la opción más simple — **un solo split fijo train/valid/test**, agrupado por `query_id` — en vez de K-fold, para no complicar de entrada dado el tiempo disponible. Queda como posibilidad futura correr el split final con 2-3 semillas distintas si da el tiempo, como forma liviana de aplicar la recomendación de "promediar corridas" sin ir a K-fold completo.

**Proporciones train/valid/test: 70/15/15 (decisión inicial, a validar empíricamente).**

Trade-off considerado:
- A favor de train grande (ej. 80%): el dataset es chico en términos absolutos (2012 queries, 10.000 filas) y `bought` está desbalanceado (13%) — más train significa más ejemplos positivos reales para aprender el patrón de compra.
- A favor de valid/test más grandes (ej. 15% en vez de 10%): ya vimos con `country_of_origin` que grupos de ~245-331 filas pueden mostrar diferencias de 2-3 puntos porcentuales que son puro ruido. Con 10% del dataset (~1000 filas, ~130 positivas) el PR-AUC/ROC-AUC medido en valid podría fluctuar bastante por azar — y como valid se usa repetidamente (una vez por cada configuración del estudio de ablación), un valid ruidoso puede llevar a elegir la configuración que ganó por casualidad, no la que generaliza mejor.

Con el tamaño de este dataset y la cantidad de comparaciones que se harán contra valid, se prioriza confiabilidad de la métrica por sobre maximizar train: **70/15/15**.

**Validación empírica pendiente**: la elección de 70/15/15 por sobre 80/10/10 es por ahora un argumento teórico (igual que la fusión tardía). Se puede confirmar con datos una vez que haya un modelo baseline funcionando: entrenar con distintas semillas (splits al azar distintos, misma proporción) y comparar cuánto varía el PR-AUC/ROC-AUC de valid entre corridas para 70/15/15 vs. 80/10/10. Si 80/10/10 muestra más variabilidad entre semillas, es evidencia directa de que el valid más chico es demasiado ruidoso — si no hay diferencia notable, se podría preferir 80/10/10 por el train más grande. Nota: esto no tiene que ver con "convergencia" del entrenamiento (eso depende de arquitectura/learning rate, no de la proporción del split) — el experimento mide variabilidad de la métrica final entre semillas, no la dinámica de entrenamiento.

**Decidido: sí, estratificar por tasa de `bought` a nivel de query.** Implementado en [`split_data.py`](split_data.py):

- Se calcula la tasa de `bought` de cada query (promedio de `bought` sobre sus filas) y se agrupa en 3 franjas para estratificar: **`0%`** (ninguna compra en esa búsqueda — 1058/2012 queries, más de la mitad), **`1-33%`** y **`34-100%`**. No se usa la tasa exacta como clase de estratificación porque hay tasas (ej. 0,57 con 2 queries, 1,00 con 10) con muy pocas queries — insuficiente para partir en train/valid/test de forma confiable. Se decidió separar `34-67%` de `68-100%` en una sola franja alta porque esta última sola tenía solo 10 queries en todo el dataset.
- Split en 2 pasos con `sklearn.model_selection.train_test_split(..., stratify=...)`: primero train (70%) vs. resto (30%), después resto se parte 50/50 en valid/test (15%/15% del total). `random_state=42` fijo para reproducibilidad.
- Resultado real (`output/split_summary.csv`):

| split | n queries | n filas | tasa de bought |
|---|---|---|---|
| train | 1408 | 7012 | 12,94% |
| valid | 302 | 1498 | 12,88% |
| test | 302 | 1490 | 13,49% |

Tasa global: 13,01% — las 3 tasas quedan dentro de ±0,5 puntos porcentuales de la tasa global (mucho más parejo que lo que se vería con un split sin estratificar), la estratificación funcionó (ver [`output/split_balance.png`](output/split_balance.png), generado por [`plot_split.py`](plot_split.py): tasa de bought y composición por franja prácticamente idénticas entre splits).

Output: [`output/query_splits.csv`](output/query_splits.csv) (`query_id`, `bought_rate`, `n_rows`, `strata`, `split` — 2012 filas, una por query) y [`output/split_summary.csv`](output/split_summary.csv) (resumen agregado por split). Los scripts de preprocesamiento de features deben hacer `merge` con `query_splits.csv` por `query_id` para asignarle el split a cada fila del dataset original.

## Encoding de features (implementado)

Implementado en [`encode_features.py`](encode_features.py), aplicando las decisiones de `ejercicio1/Notas.md` ("Lista de features") sobre los splits de `split_data.py`. Regla general: todo lo que hay que "fittear" (vocabulario de texto, categorías de one-hot, media/desvío del z-score) se ajusta **solo con filas de train** y se aplica igual a valid/test — si se fitteara con todo el dataset, información de valid/test se filtraría al preprocesamiento (leakage), aunque sea indirecto (estadísticos, no el target).

Por feature:
- **Numéricas → z-score** (`net_weight_oz`, `nutrition_score`, `price_relpos`, `dim_length`, `dim_width`, `dim_height`): media/desvío calculados sobre train, aplicados a los 3 splits. Verificado: en train quedan con media≈0/desvío≈1 exacto; en valid/test quedan cerca (media entre -0,01 y 0,03, desvío entre 0,95 y 0,99) pero no exacto, como se espera al aplicar estadísticos ajenos — señal de que no hay leakage.
- **`price` + `filter_price_min` + `filter_price_max` → `price_relpos`**: `(price - filter_price_min) / (filter_price_max - filter_price_min)`, después z-scoreado igual que el resto (ver nota en "Preprocesamiento — numéricas" de `ejercicio1/Notas.md`: la decisión ahí fue normalizar *todas* las numéricas, incluida esta aunque ya estuviera en [0,1]).
- **`dimensions_in` → 3 numéricas** (`dim_length`, `dim_width`, `dim_height`): parseadas con regex desde el string `"L x W x H\""`, sin fallos de parseo (0/10000). Se optó por las 3 dimensiones por separado en vez de resumir en 1 volumen — más información disponible para el modelo, se puede revisar si conviene resumir más adelante.
- **Categóricas nominales → one-hot** (`category`, `storage_type`, `brand`, `allergens`, `country_of_origin`): categorías fijadas con los valores únicos de train (`allergens` con nulo → `"None"`). Se confirmó que **ninguna categoría de ninguna columna falta en train** (chequeado contra el dataset completo) — no hay caso de categoría "nunca vista" en valid/test a manejar. `country_of_origin` y `nutrition_score` se encodean igual que el resto aunque están marcadas como "dudosas" en `ejercicio1/Notas.md`: la decisión de incluirlas o no en el modelo es de la etapa de ablación, no del encoding.
- **`ingredients` → multi-hot** sobre los 21 ingredientes individuales (columna binaria por ingrediente, `ingredient_<nombre>`), no sobre las 12 combinaciones canónicas — se eligió multi-hot porque generaliza mejor si apareciera una combinación nueva de los mismos 21 ingredientes base (no ocurre en este dataset, pero no depende de una lista cerrada de combos).
- **`title`+`description` → tokenización por palabra**: mismo criterio que `ejercicio1/vocab_eda.py` (minúsculas, números con decimales como token propio). Vocabulario construido **solo con train**: da exactamente las mismas **410 palabras** encontradas en `ejercicio1/Notas.md` sobre el dataset completo (esperable — la frecuencia mínima por palabra en todo el dataset era 24, así que ninguna se pierde al quedarse con el 70%), más `<PAD>=0` y `<UNK>=1` reservados. Secuencias paddeadas/truncadas a `MAX_LEN=45` (el máximo real de tokens por fila en todo el dataset — cubre el 100%, no trunca ninguna fila). Se guarda además `title_desc_len` (longitud real antes del padding) para poder armar la máscara de atención del Transformer más adelante.

Output en `data/`: [`data/train.csv`](data/train.csv) / [`data/valid.csv`](data/valid.csv) / [`data/test.csv`](data/test.csv) (7012 / 1498 / 1490 filas, 79 columnas: `query_id`, `bought`, 6 numéricas z-scoreadas, `title_desc_tokens` + `title_desc_len`, 48 columnas one-hot, 21 columnas multi-hot de ingredientes), [`data/vocab.csv`](data/vocab.csv) (palabra → id → frecuencia en train) y [`data/preprocessing_stats.csv`](data/preprocessing_stats.csv) (media/desvío de cada numérica, para poder aplicar el mismo preprocesamiento afuera de este script si hiciera falta).

**Pendiente:** confirmar si conviene resumir `dimensions_in` a 1 sola feature (volumen) en vez de 3, y si `ingredients` como multi-hot es mejor que la categórica de 12 combos — quedaron ambas como decisión abierta en `ejercicio1/Notas.md` y se tomó una opción por default para poder avanzar, no está cerrado.

## Integración de `title`/`description` (Transformer) con las features tabulares

Pendiente que mandó el grupo en su momento: "terminar de definir el preprocesamiento del texto: cómo se integran `title`/`description` (tokenizados) con el resto de las features tabulares en la misma arquitectura."

Revisando la letra de `DeepLearningTP0.pdf`, el punto 4 del Ejercicio 1 pedía el preprocesamiento **de cada feature por separado** ("Qué preprocesamiento tendrá **cada feature** para ser tomada como input del modelo") — eso ya está resuelto en `ejercicio1/Notas.md`. Lo que falta —**cómo se combinan todas esas representaciones dentro de una única arquitectura**— es lo que pide el Ejercicio 2 ("Deberán diseñar e implementar un sistema... Deberán decidir en qué parte de la solución [el Transformer] es pertinente y por qué"). Es una decisión de arquitectura, y el `CLAUDE.md` del TP pide releer el material de cátedra antes de definir cualquier pieza de arquitectura — no armarla por nuestra cuenta. Se buscó en las transcripciones (`transformers.VTT`, `embeddings_1.VTT`, `embeddings_2.VTT`, `demo_transformers.VTT`, `consigna.VTT`) alguna mención de cómo combinar un Transformer de texto con features tabulares y **no aparece** — las clases cubren atención/embeddings de texto en sí, no un caso de fusión multimodal texto+tabular. No asumir una técnica no vista; esto se decide como grupo, revisando si el profesor da alguna pista adicional en clase.

Dos alternativas de sentido común consideradas:
1. **Fusión tardía**: el Transformer procesa solo la secuencia de tokens de `title`+`description` y su salida se resume en un solo vector (ej. promediando los embeddings de salida de cada token, o usando un token especial tipo `[CLS]`); ese vector se concatena con el vector de features tabulares ya encodeadas (one-hot + numéricas normalizadas) y sigue por una o más capas densas hasta la salida. Ventaja: simple, separa claramente "la parte Transformer" de "la parte tabular" para el estudio de ablación (se puede sacar una u otra).
2. **Todo como secuencia de tokens**: proyectar también cada feature tabular a la dimensión `d_model` (como si fuera "un token más") y dejar que la atención combine todo dentro del Transformer. Más ambicioso y menos evidente que esté cubierto por el material de cátedra.

**Decisión: combinar recién al final (texto y tabular por ramas separadas, concatenados antes de la salida), no meter lo tabular dentro del Transformer.** Razones teóricas, de diseño:
- Ablación limpia: "sacar el Transformer" es desconectar una rama entera y queda el modelo funcionando solo con lo tabular. Con la opción 2, sacar el texto implica rediseñar la arquitectura, no apagar un módulo.
- Arrancar chico (`d_model < 100`): así el Transformer queda acotado a donde realmente hace falta atención (el texto — recordar el hallazgo del tag de reputación en `title`, un patrón contextual/posicional dentro de la secuencia de palabras), y lo tabular se combina con algo más liviano (denso), sin forzarlo a la dimensión de los embeddings de texto.
- Justificación de "dónde y por qué" el Transformer (que pide la consigna): así es directa — el Transformer resuelve la parte de lenguaje natural, que tiene dependencias de orden/contexto; lo tabular no tiene esa estructura secuencial, no hay razón para forzarlo por atención.

**Validación empírica (Experimentos 7, 8 y 10, ver [`Experimentos.md`](Experimentos.md)): confirmada, con un matiz sobre la cabeza de salida.** Con la cabeza de salida más simple (`Linear` directo sobre el vector concatenado, Experimento 7) el sistema completo no superó al Transformer de texto solo (0,718 vs. 0,724 de PR-AUC valid, dentro del ruido) -- el problema no era combinar texto y tabular en sí, sino que una capa lineal sola no puede aprender una interacción entre ambas ramas, solo pesarlas por separado. Agregando una capa oculta a la cabeza (Experimento 8) el resultado mejora, y el Experimento 10 (barrido del ancho de esa capa oculta, hasta 512) muestra que sigue subiendo hasta encontrar una meseta real entre 256 y 512 (+0,004 de PR-AUC, contra +0,018 del paso anterior) -- **`hidden=256` queda como elección final, PR-AUC valid ≈ 0,82**, la mejor marca de todo el estudio. Conclusión: combinar texto y tabular sí suma, pero la cabeza de salida necesita una no-linealidad (y bastante ancho) para aprovecharlo -- con eso resuelto, el argumento teórico de arriba queda confirmado con datos.

## Arquitectura del bloque Transformer: Encoder-only (confirmado con el grupo)

Repasando `transformers.VTT` (Clase 1, Eugenia): un **Encoder** tiene 2 capas — Multi-Head Self-Attention (+ conexión residual/skip + Layer Norm) y MLP feed-forward (+ conexión residual + Layer Norm). Un **Decoder** agrega Cross-Attention hacia la salida del Encoder, y su self-attention lleva máscara (no puede ver tokens futuros; el Encoder sí puede mirar en ambas direcciones). Se apilan N encoders/decoders iguales (paper original: 6; prueban 2/4/8 también).

Variantes mencionadas en la clase:
- **Encoder-Decoder completo**: para tareas de secuencia-a-secuencia (ej. traducción, dos secuencias de texto distintas). No aplica a nuestro caso.
- **Decoder-only** (autoregresivo, con máscara — como GPT): es lo que se armó en la demo (`demo_transformers.VTT`), porque esa demo era generación de texto (predecir el próximo carácter). No es nuestro caso: no queremos generar texto.
- **Encoder-only** (sin máscara, da un embedding/representación — ejemplo BERT, citado en `transformers.VTT` línea 2617-2621): pensado para tareas de representación/clasificación, no generación.

**Confirmado**: usar **Encoder-only** para procesar `title`+`description` — no hace falta generar texto, hace falta resumir/entender el texto completo (que ya está disponible enteramente al momento de predecir, no hay nada "futuro" que enmascarar). La salida se resume en un vector (fusión tardía, ver arriba) y sigue por el resto de la arquitectura. Implementado en [`model.py`](model.py) con `nn.TransformerEncoderLayer` de PyTorch (Multi-Head Self-Attention + residual + LayerNorm, MLP feed-forward + residual + LayerNorm — los módulos estándar, no algo hecho a mano ni distinto de lo que vimos en clase) + positional encoding senoidal. Resultados en [`Experimentos.md`](Experimentos.md).

**Positional encoding: por qué se usa, y por qué se mantiene (Experimento 11).** A diferencia de pooling, este dial sí está explicado en clase (`transformers.VTT`, Eugenia, en respuesta a la pregunta de un compañero "¿para qué hacías el positional encoding?"): "lo hacés para darle un orden a tus tokens" — sin él, la atención no distingue el orden de las palabras. Se probó sacarlo (sobre el Transformer de texto solo, para aislar el efecto de la fusión con tabular): el PR-AUC de valid casi no cambia (0,724 → 0,720), pero el overfitting se dispara (gap 0,012 → 0,174, train llega a ~0,90 sin generalizar). Se mantiene en la arquitectura final por esto último, no por el PR-AUC pico.

## "Diales" del estudio de ablación (según lo que la profesora nombra explícitamente en clase)

`transformers.VTT` línea 2177, hablando de qué hiperparámetros se ajustan en la arquitectura: *"cuántas heads voy a usar, cuántos encoders y decoders apilados voy a tener, cuál es la dimensión de cada MLP, cuántas neuronas tiene el MLP"*. Traducido a nuestro caso (Encoder-only):

- Cantidad de **heads** de atención.
- Cantidad de **encoders apilados** (N).
- **Dimensión del MLP** interno de cada encoder (cantidad de neuronas).
- `d_model` (dimensión interna del modelo — arrancar `< 100` según la consigna).

Sumado a los módulos ya identificados antes en este documento: presencia/ausencia del bloque Transformer de texto (baseline tabular vs. sistema completo), `country_of_origin`, `nutrition_score` (con/sin, ver `ejercicio1/Notas.md`).

**`country_of_origin`/`nutrition_score` corrido (Experimento 9, ver [`Experimentos.md`](Experimentos.md)):** `nutrition_score` aporta señal real (sacarla baja PR-AUC de forma consistente en las 3 semillas, -0,031 en promedio); `country_of_origin` no muestra evidencia clara de aportar (efecto mixto por semilla, -0,005 en promedio, dentro del ruido) -- candidata a sacar del modelo final.

