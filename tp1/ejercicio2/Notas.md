# Notas — Ejercicio 2 (Desarrollo del sistema)

Notas de trabajo, en construcción. Contenido trasladado desde [`../ejercicio1/Notas.md`](../ejercicio1/Notas.md): surgió naturalmente charlando el EDA del Ejercicio 1, pero son decisiones de diseño del sistema (Ejercicio 2), no del EDA en sí — ver esa nota para el detalle completo de variable objetivo, features y su preprocesamiento.

Punto de partida que deja cerrado el Ejercicio 1:
- **Variable objetivo**: `bought` a nivel fila. El BTR de una búsqueda sale de promediar `bought` (o las probabilidades predichas) agrupando por `query_id`.
- **Lista de features cerrada**: `cart` excluida por leakage; `filter_category`/`filter_storage_type`/`package_size`/`unit_of_measure` descartadas por redundancia; `country_of_origin` y `nutrition_score` sin señal univariada clara, quedan como candidatas de ablación.
- **Preprocesamiento por feature ya decidido**: one-hot para categóricas nominales (`category`, `storage_type`, `brand`, `allergens`), z-score para numéricas (`net_weight_oz`, derivadas), posición relativa de `price` dentro del rango filtrado, `dimensions_in` parseado a numéricas, tokenización **por palabra** para `title`/`description` (vocabulario cerrado de 410 palabras, sin necesidad de BPE).

Lo que falta acá es **cómo se junta todo eso en una arquitectura concreta** — eso es el contenido de este documento.

## Split train / valid / test

**Por qué agrupar por `query_id`:** las filas están agrupadas por búsqueda — cada `query_id` trae varios productos que comparten contexto (misma categoría, mismo rango de precio filtrado). Si se particiona fila por fila al azar, filas de una misma búsqueda podrían terminar repartidas entre train y test: el modelo ya habría visto ese contexto exacto (categoría, rango de precio) durante el entrenamiento, lo cual no refleja el caso real de uso (predecir sobre búsquedas *nuevas* que nunca se vieron). Por eso: **la partición se hace por `query_id` completo** — todas las filas de una búsqueda van juntas al mismo split, nunca mezcladas.

**Por qué 3 particiones y no 2:** el PDF de la consigna lo pide explícitamente ("recordar train/valid/test split"). Además tiene sentido con el Ejercicio 2 (experimentar con configuraciones del modelo):
- **train**: ajusta los pesos del modelo.
- **valid**: compara entre configuraciones/experimentos (arquitectura, `d_model`, etc.) durante la iteración.
- **test**: se toca una sola vez, al final, para el número que se reporta — si se usara valid para elegir la mejor configuración y también para reportar el resultado final, el número quedaría inflado (overfit a valid).

**Precedente en TPs anteriores (SIA-TP5, autoencoders):** usaron **K-fold estratificado** como estrategia principal (entrenar con varios folds y promediar resultados, `mean`/`std` en el output), con un `val_fraction=0.2` (80/20) como caso simple para cuando no se hacían folds (`mlp/data.py::train_val_split`). Esto conecta con algo que también dice el audio de `consigna.VTT` de esta materia: la profesora recomienda "promediar varias corridas (o cross-validation) en vez de una sola ejecución" — o sea, el mismo criterio de TP5 tiene aval acá también.

**Decisión para este TP:** se eligió la opción más simple — **un solo split fijo train/valid/test**, agrupado por `query_id` — en vez de K-fold, para no complicar de entrada dado el tiempo disponible. Queda como posibilidad futura correr el split final con 2-3 semillas distintas si da el tiempo, como forma liviana de aplicar la recomendación de "promediar corridas" sin ir a K-fold completo.

**Proporciones train/valid/test: 70/15/15 (decisión inicial, a validar empíricamente).**

Trade-off considerado:
- A favor de train grande (ej. 80%): el dataset es chico en términos absolutos (2012 queries, 10.000 filas) y `bought` está desbalanceado (13%) — más train significa más ejemplos positivos reales para aprender el patrón de compra.
- A favor de valid/test más grandes (ej. 15% en vez de 10%): ya vimos con `country_of_origin` que grupos de ~245-331 filas pueden mostrar diferencias de 2-3 puntos porcentuales que son puro ruido. Con 10% del dataset (~1000 filas, ~130 positivas) el PR-AUC/ROC-AUC medido en valid podría fluctuar bastante por azar — y como valid se usa repetidamente (una vez por cada configuración del estudio de ablación), un valid ruidoso puede llevar a elegir la configuración que ganó por casualidad, no la que generaliza mejor.

Con el tamaño de este dataset y la cantidad de comparaciones que se harán contra valid, se prioriza confiabilidad de la métrica por sobre maximizar train: **70/15/15**.

**Validación empírica pendiente**: la elección de 70/15/15 por sobre 80/10/10 es por ahora un argumento teórico (igual que la fusión tardía). Se puede confirmar con datos una vez que haya un modelo baseline funcionando: entrenar con distintas semillas (splits al azar distintos, misma proporción) y comparar cuánto varía el PR-AUC/ROC-AUC de valid entre corridas para 70/15/15 vs. 80/10/10. Si 80/10/10 muestra más variabilidad entre semillas, es evidencia directa de que el valid más chico es demasiado ruidoso — si no hay diferencia notable, se podría preferir 80/10/10 por el train más grande. Nota: esto no tiene que ver con "convergencia" del entrenamiento (eso depende de arquitectura/learning rate, no de la proporción del split) — el experimento mide variabilidad de la métrica final entre semillas, no la dinámica de entrenamiento.

**Pendiente:** decidir si además de agrupar por `query_id`, conviene estratificar por `bought` a nivel de query (por ejemplo, usando la tasa de compra de cada búsqueda) para que el desbalance de clases quede parejo entre splits — no se decidió todavía.

## Integración de `title`/`description` (Transformer) con las features tabulares

Pendiente que mandó el grupo en su momento: "terminar de definir el preprocesamiento del texto: cómo se integran `title`/`description` (tokenizados) con el resto de las features tabulares en la misma arquitectura."

Revisando la letra de `DeepLearningTP0.pdf`, el punto 4 del Ejercicio 1 pedía el preprocesamiento **de cada feature por separado** ("Qué preprocesamiento tendrá **cada feature** para ser tomada como input del modelo") — eso ya está resuelto en `ejercicio1/Notas.md`. Lo que falta —**cómo se combinan todas esas representaciones dentro de una única arquitectura**— es lo que pide el Ejercicio 2 ("Deberán diseñar e implementar un sistema... Deberán decidir en qué parte de la solución [el Transformer] es pertinente y por qué"). Es una decisión de arquitectura, y el `CLAUDE.md` del TP pide releer el material de cátedra antes de definir cualquier pieza de arquitectura — no armarla por nuestra cuenta. Se buscó en las transcripciones (`transformers.VTT`, `embeddings_1.VTT`, `embeddings_2.VTT`, `demo_transformers.VTT`, `consigna.VTT`) alguna mención de cómo combinar un Transformer de texto con features tabulares y **no aparece** — las clases cubren atención/embeddings de texto en sí, no un caso de fusión multimodal texto+tabular. No asumir una técnica no vista; esto se decide como grupo, revisando si el profesor da alguna pista adicional en clase.

Dos alternativas de sentido común consideradas:
1. **Fusión tardía**: el Transformer procesa solo la secuencia de tokens de `title`+`description` y su salida se resume en un solo vector (ej. promediando los embeddings de salida de cada token, o usando un token especial tipo `[CLS]`); ese vector se concatena con el vector de features tabulares ya encodeadas (one-hot + numéricas normalizadas) y sigue por una o más capas densas hasta la salida. Ventaja: simple, separa claramente "la parte Transformer" de "la parte tabular" para el estudio de ablación (se puede sacar una u otra).
2. **Todo como secuencia de tokens**: proyectar también cada feature tabular a la dimensión `d_model` (como si fuera "un token más") y dejar que la atención combine todo dentro del Transformer. Más ambicioso y menos evidente que esté cubierto por el material de cátedra.

**Decisión inicial (a confirmar con resultados, ver plan de experimentos abajo): fusión tardía.** Razones (teóricas, de diseño — todavía no respaldadas con números):
- Ablación limpia: "sacar el Transformer" es desconectar una rama entera y queda el modelo funcionando solo con lo tabular. Con la opción 2, sacar el texto implica rediseñar la arquitectura, no apagar un módulo.
- Arrancar chico (`d_model < 100`): con fusión tardía el Transformer queda acotado a donde realmente hace falta atención (el texto — recordar el hallazgo del tag de reputación en `title`, un patrón contextual/posicional dentro de la secuencia de palabras), y lo tabular se combina con algo más liviano (denso), sin forzarlo a la dimensión de los embeddings de texto.
- Justificación de "dónde y por qué" el Transformer (que pide la consigna): con fusión tardía es directa — el Transformer resuelve la parte de lenguaje natural, que tiene dependencias de orden/contexto; lo tabular no tiene esa estructura secuencial, no hay razón para forzarlo por atención.

**Importante**: esto es un punto de partida razonado, no una decisión cerrada solo por argumento teórico — se valida (o se descarta) con el plan de experimentos de la siguiente sección.

## Arquitectura del bloque Transformer: Encoder-only (propuesta, a confirmar)

Repasando `transformers.VTT` (Clase 1, Eugenia): un **Encoder** tiene 2 capas — Multi-Head Self-Attention (+ conexión residual/skip + Layer Norm) y MLP feed-forward (+ conexión residual + Layer Norm). Un **Decoder** agrega Cross-Attention hacia la salida del Encoder, y su self-attention lleva máscara (no puede ver tokens futuros; el Encoder sí puede mirar en ambas direcciones). Se apilan N encoders/decoders iguales (paper original: 6; prueban 2/4/8 también).

Variantes mencionadas en la clase:
- **Encoder-Decoder completo**: para tareas de secuencia-a-secuencia (ej. traducción, dos secuencias de texto distintas). No aplica a nuestro caso.
- **Decoder-only** (autoregresivo, con máscara — como GPT): es lo que se armó en la demo (`demo_transformers.VTT`), porque esa demo era generación de texto (predecir el próximo carácter). No es nuestro caso: no queremos generar texto.
- **Encoder-only** (sin máscara, da un embedding/representación — ejemplo BERT, citado en `transformers.VTT` línea 2617-2621): pensado para tareas de representación/clasificación, no generación.

**Propuesta**: usar **Encoder-only** para procesar `title`+`description` — no hace falta generar texto, hace falta resumir/entender el texto completo (que ya está disponible enteramente al momento de predecir, no hay nada "futuro" que enmascarar). La salida se resume en un vector (fusión tardía, ver arriba) y sigue por el resto de la arquitectura. **Pendiente confirmar con el grupo.**

## "Diales" del estudio de ablación (según lo que la profesora nombra explícitamente en clase)

`transformers.VTT` línea 2177, hablando de qué hiperparámetros se ajustan en la arquitectura: *"cuántas heads voy a usar, cuántos encoders y decoders apilados voy a tener, cuál es la dimensión de cada MLP, cuántas neuronas tiene el MLP"*. Traducido a nuestro caso (Encoder-only):

- Cantidad de **heads** de atención.
- Cantidad de **encoders apilados** (N).
- **Dimensión del MLP** interno de cada encoder (cantidad de neuronas).
- `d_model` (dimensión interna del modelo — arrancar `< 100` según la consigna).

Sumado a los módulos ya identificados antes en este documento: presencia/ausencia del bloque Transformer de texto (baseline tabular vs. fusión tardía), `country_of_origin`, `nutrition_score` (con/sin, ver `ejercicio1/Notas.md`).

## Plan de experimentos / estudio de ablación (borrador)

Pregunta que motivó esto: la elección de fusión tardía por sobre "todo como secuencia" fue por argumento teórico — ¿se puede respaldar con resultados en vez de quedar solo en teoría? Sí, es exactamente lo que pide la consigna al evaluar "comparación de alternativas de los distintos módulos". Plan (de menor a mayor esfuerzo):

1. **Baseline solo tabular** — sin Transformer, sin texto: capas densas sobre las features tabulares únicamente. Sirve como piso de comparación.
2. **Fusión tardía** (la decisión actual) — Transformer sobre texto + tabular combinados al final. Comparar contra (1) con PR-AUC/ROC-AUC: si no mejora sobre el baseline tabular, es evidencia de que el texto no está aportando (raro dado el hallazgo del tag de reputación, pero hay que confirmarlo).
3. *(Opcional, si da el tiempo/cómputo)* **Todo-como-secuencia** — implementar la alternativa descartada y comparar contra (2) con las mismas métricas, para respaldar (o refutar) con números la elección de fusión tardía, no solo con la intuición de "más fácil de aislar".

Prioridad: (1) y (2) son el corazón del estudio de ablación (las más baratas, y las que más responden "¿aporta el Transformer/texto?"). (3) queda como estiramiento si sobra tiempo — la consigna aclara que no hace falta que el BTR prediga perfecto, el foco es el abordaje.

## Pendientes para retomar

- [ ] Confirmar con el grupo la propuesta de arquitectura Encoder-only para el bloque de texto (vs. Decoder-only o Encoder-Decoder completo).
- [ ] Correr el plan de experimentos de arriba (1 y 2 como mínimo) para respaldar con resultados la elección de fusión tardía.
- [ ] Validar empíricamente 70/15/15 vs. 80/10/10: correr con varias semillas y comparar variabilidad del PR-AUC/ROC-AUC de valid entre corridas (una vez que haya un baseline funcionando).
- [ ] Decidir si estratificar por tasa de `bought` a nivel de query (además de agrupar por `query_id`).
- [ ] Arrancar con arquitectura chica (`d_model < 100`) como sugiere la consigna, antes de escalar.
- [ ] Diseñar el estudio de ablación en paralelo con la arquitectura: qué módulos se van a poder prender/apagar (el bloque Transformer de texto, `country_of_origin`, `nutrition_score`, distintas cantidades de heads/capas, etc.) — no dejarlo para el final.
- [ ] Definir métricas de evaluación (PR-AUC/ROC-AUC, sin threshold) y cómo promediar varias corridas (semillas) para reportar resultados, según la aclaración de `consigna.VTT`.
