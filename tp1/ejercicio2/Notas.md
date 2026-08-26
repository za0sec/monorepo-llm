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

**Pendiente:** definir las proporciones exactas (ej. 70/15/15) y si además de agrupar por `query_id`, conviene estratificar por `bought` a nivel de query (por ejemplo, usando la tasa de compra de cada búsqueda) para que el desbalance de clases quede parejo entre splits — no se decidió todavía.

## Integración de `title`/`description` (Transformer) con las features tabulares

Pendiente que mandó el grupo en su momento: "terminar de definir el preprocesamiento del texto: cómo se integran `title`/`description` (tokenizados) con el resto de las features tabulares en la misma arquitectura."

Revisando la letra de `DeepLearningTP0.pdf`, el punto 4 del Ejercicio 1 pedía el preprocesamiento **de cada feature por separado** ("Qué preprocesamiento tendrá **cada feature** para ser tomada como input del modelo") — eso ya está resuelto en `ejercicio1/Notas.md`. Lo que falta —**cómo se combinan todas esas representaciones dentro de una única arquitectura**— es lo que pide el Ejercicio 2 ("Deberán diseñar e implementar un sistema... Deberán decidir en qué parte de la solución [el Transformer] es pertinente y por qué"). Es una decisión de arquitectura, y el `CLAUDE.md` del TP pide releer el material de cátedra antes de definir cualquier pieza de arquitectura — no armarla por nuestra cuenta. Se buscó en las transcripciones (`transformers.VTT`, `embeddings_1.VTT`, `embeddings_2.VTT`, `demo_transformers.VTT`, `consigna.VTT`) alguna mención de cómo combinar un Transformer de texto con features tabulares y **no aparece** — las clases cubren atención/embeddings de texto en sí, no un caso de fusión multimodal texto+tabular. No asumir una técnica no vista; esto se decide como grupo, revisando si el profesor da alguna pista adicional en clase.

Quedan anotadas acá **dos alternativas de sentido común** a discutir (ninguna elegida todavía):
1. **Fusión tardía (la más simple)**: el Transformer procesa solo la secuencia de tokens de `title`+`description` y su salida se resume en un solo vector (ej. promediando los embeddings de salida de cada token, o usando un token especial tipo `[CLS]`); ese vector se concatena con el vector de features tabulares ya encodeadas (one-hot + numéricas normalizadas) y sigue por una o más capas densas hasta la salida. Ventaja: simple, separa claramente "la parte Transformer" de "la parte tabular" para el estudio de ablación (se puede sacar una u otra).
2. **Todo como secuencia de tokens**: proyectar también cada feature tabular a la dimensión `d_model` (como si fuera "un token más") y dejar que la atención combine todo dentro del Transformer. Más ambicioso y menos evidente que esté cubierto por el material de cátedra — evaluar si vale la pena dado que la consigna sugiere arrancar chico (`d_model < 100`).

**Estado: pendiente de decidir** — no resuelto todavía para no adelantar una decisión de arquitectura sin haber revisado bien el material correspondiente y sin el resto del grupo.

## Pendientes para retomar

- [ ] Elegir entre las dos alternativas de integración texto+tabular (o proponer una tercera) — revisar antes si el profesor dio alguna pista adicional en clase sobre fusión de modalidades.
- [ ] Definir proporciones exactas del split train/valid/test (ej. 70/15/15) y decidir si estratificar por tasa de `bought` a nivel de query.
- [ ] Arrancar con arquitectura chica (`d_model < 100`) como sugiere la consigna, antes de escalar.
- [ ] Diseñar el estudio de ablación en paralelo con la arquitectura: qué módulos se van a poder prender/apagar (el bloque Transformer de texto, `country_of_origin`, `nutrition_score`, distintas cantidades de heads/capas, etc.) — no dejarlo para el final.
- [ ] Definir métricas de evaluación (PR-AUC/ROC-AUC, sin threshold) y cómo promediar varias corridas (semillas) para reportar resultados, según la aclaración de `consigna.VTT`.
