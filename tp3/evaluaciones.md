# Evaluaciones del RAG del TP3

**Estado al 11/11/2026.** Registro para la defensa del RAG textual que compara lo que exige un objetivo de estudio con lo que explica el audio de un video. El **objetivo** consulta el índice de **apuntes y transcripciones de clase**. El **VTT del video** entra al generador como evidencia de cobertura y queda fuera del índice académico.

Las consignas ubican la evaluación y la ablación del RAG en el ejercicio 2 y la comparación antes/después en el ejercicio 4. El ejercicio 3 pide el informe de Responsible AI, Safety y producción. La clase de métricas toma como base **Context Precision, Context Recall, Answer Relevance y Answer Faithfulness**. Permite anotación y cálculo propios, con RAGAS como opción. [Consigna, pp. 2–3][consigna] · [Otra versión, pp. 2–3][consigna-alt] · [Explicación del TP, 07:31–10:19][tp-vtt] · [Métricas, 23:26–24:09 y 33:20–33:39][metricas]

## Control común de las comparaciones

- El [caso manual de embeddings][caso] sirve para desarrollo. Su objetivo general pide definición (E1), papel del contexto (E2), comparación entre textos (E3), cálculo del coseno (E4) y limitación por magnitud (E5). En la revisión manual del VTT del video, E1 y E2 quedaron cubiertos, E3 parcial y E4 y E5 sin evidencia. La consulta «quiero estudiar embeddings» tiene otro alcance. Su revisión introductoria usa definición, uso y contexto, sin exigir la fórmula. El diagnóstico de entrenamiento se registra aparte.
- Para los próximos casos, fijar antes de correr el sistema el objetivo, el nivel, las afirmaciones académicas esperadas y sus pasajes con tiempo o página. Para generación, escuchar el video y fijar estados esperados con evidencia temporal. Reservar objetivos de distintas unidades y otros videos para comprobar los finalistas. El apunte derivado de embeddings se preparó con el caso 01 a la vista. Su mejora mide un piloto sobre ese material dirigido.
- Cambiar una variable por par. Guardar fuentes y versiones, modelo, parámetros, ranking completo, contexto que llegó al LLM, informe y tiempos. Registrar `n/N` por caso, promedio por caso y fallos. Para comparar contextos de distinta longitud, registrar también los tokens enviados y revisar los finalistas con un presupuesto común de tokens. El [plan experimental][plan] conserva el protocolo de reconstrucción. Los JSON de corridas y el material de `docs/` son locales. Los informes versionados conservan las observaciones principales.

## 1. Indexación (Indexing Pipeline)

### Qué ya se probó

| Prueba | Resultado observado | Decisión que respalda |
| --- | --- | --- |
| Transcripción de clase con splitter de 1.000 caracteres y 100 de solapamiento, seguido de 200 tokens y 20 de solapamiento. MiniLM en CPU (384 dimensiones) y Chroma persistente con coseno. | 125 chunks cotejados con sus offsets e IDs. Reindexar reutilizó 125/125. La primera ejecución tardó 14,502 s con carga o descarga del modelo y la reutilización 0,040 s. La autoconsulta devolvió el propio chunk primero. | Chroma y MiniLM son un punto de partida local reproducible. El control verifica persistencia, trazabilidad y coherencia de carga/consulta. La calidad semántica se mide con objetivos reales. [Revisión inicial][revision] |
| Agregar al VTT el texto de las páginas 3–5 del apunte derivado de embeddings, con el mismo splitter y embedder. | 136 chunks (125 + 11), con página y procedencia. Reindexar reutilizó 136/136. El apunte de Transformers se revisó y se reservó para objetivos de esa unidad. | Conservar VTT y apunte porque aportan pasajes distintos al piloto. El impacto en el ranking figura en recuperación. [Bitácora, §§12–13][bitacora] |
| ASR del video, piloto Wav2Vec2 local en `ASR01` (23,5 s). | **WER 14,55 % (8/55)** y **CER 5,21 % (16/307)** contra referencia humana. Incluye «envedings» por «embeddings» y diferencias como `72` frente a «setenta y dos». Whisper vía proxy devolvió 429 para ese recorte. | Wav2Vec2 permite continuar localmente. Quedan por medir los demás recortes y validar los tiempos CTC para citar. El VTT histórico de Whisper se puede reutilizar como entrada fija del RAG. [Piloto y reproducción][asr] |

### Ablaciones propuestas y métricas

| Variable | Comparación controlada | Qué medir | Sin clave API |
| --- | --- | --- | --- |
| Fuentes y carga | VTT solo vs. VTT + apunte. Luego, colección mixta de embeddings vs. colección multitema, conservando los chunks de embeddings y los parámetros. | Pasajes anotados presentes y rastreables, errores de extracción, número de chunks y CP/CR con las mismas consultas. Revisar si temas vecinos desplazan evidencia útil. | Sí, con VTT y apuntes locales. |
| Tamaño de chunk | `200` vs. `128` tokens de contenido. Fijar el corpus, MiniLM, el splitter por caracteres y el solapamiento de 20 tokens. | Cantidad y distribución de tamaños, explicaciones cortadas, truncamientos, CP/CR y tokens recuperados. Revisar el cálculo de coseno, el ejemplo de magnitud y el pasaje de Word2Vec. | Sí, con el tokenizer en caché y una colección por variante. |
| Superposición (overlap) | Sobre el tamaño elegido, comparar `0`, `20` y `40` tokens. Mantener fijo el solapamiento de 100 caracteres. | Explicaciones conservadas en los límites, redundancia de pasajes, tamaño del índice, CP/CR y tokens de contexto. | Sí, con la misma preparación local. |
| Embeddings | MiniLM vs. `text-embedding-3-small` de la demo. Alternativa si se acuerda ampliar los modelos de clase: un encoder local multilingüe, como el contemplado en el plan. Fijar textos, consultas, `k` y ranking por coseno exacto. Codificar documentos y consultas con el mismo modelo en cada brazo. | CP/CR, puestos de la evidencia en español, tokens truncados, tiempo de indexación y consulta, dimensiones, espacio y costo. | MiniLM y un candidato local con pesos disponibles, sí. El candidato de la demo necesita acceso al endpoint. El indexador actual está acoplado a MiniLM y el contraste requiere un adaptador o una prueba aislada. |
| Índice | Comparar el ranking de Chroma con coseno exacto sobre los mismos vectores. | Coincidencia top-`k`, evidencia perdida por búsqueda aproximada, latencia con índice abierto y huella en disco. | Sí. |
| ASR y diapositivas | Para ASR, completar las seis ventanas con Wav2Vec2 y, cuando haya acceso, repetir con Whisper sobre los mismos recortes y la misma referencia. Para diapositivas, medir después el aporte incremental de esa fuente con el RAG textual fijo. | `WER = (S+D+I)/N` por palabras y `CER = (S+D+I)/N` por caracteres, con sustituciones, omisiones e inserciones y la misma normalización. Agregar sumando errores y unidades de referencia. Auditar términos técnicos y tiempos. En diapositivas, comprobar trazabilidad por página y cambio en CP/CR. | WER/CER sobre salidas guardadas y ASR local, sí. El brazo Whisper del piloto y la extracción visual prevista mediante API siguen pendientes. La capa de texto de un PDF que la tenga se puede evaluar localmente como fuente textual. |

El tamaño y el solapamiento son parámetros experimentales. Conviene conservar la configuración que mantenga evidencia suficiente con menos cortes y redundancia. Para auditar ASR de una clase hace falta su audio original y una referencia humana. [RAG, 39:36–45:30 y 50:23–53:18][rag] · [Demo de embeddings, celda 22][pinecone] · [Imágenes y Audio, 59:16–1:05:52][audio] · [Métricas, 38:50–40:13][metricas] · [Comandos de preparación e indexación][readme]

## 2. Recuperación (Retrieval)

### Métricas de la clase

| Métrica | Cómo medirla para este TP |
| --- | --- |
| **Context Precision@k (CP)** | Etiquetar cada chunk del ranking académico con `vᵢ=1` si aporta evidencia sustantiva a una afirmación pertinente y `0` si es tangencial o solo remite a otra fuente. Calcular `Σ(P@i × vᵢ) / Σ(vᵢ)` con `P@i` igual a la proporción de relevantes hasta la posición `i`. Reportar también `P@k = relevantes/k` para mostrar el ruido al final de la lista. Si hay evidencia esperada y se recuperan cero relevantes, reportar CP `0` como convención. |
| **Context Recall@k (CR)** | `Afirmaciones académicas de referencia respaldadas / total de afirmaciones académicas de referencia`. Evaluar la unión de los chunks finales. Se puede combinar evidencia de varios fragmentos y cada afirmación cuenta una vez. Guardar también el conteo de **criterios completos**. Un criterio puede abarcar varias afirmaciones. |

Etiquetar con el objetivo literal y la referencia humana. Los fragmentos del video quedan fuera de CP y CR. En un control diseñado sin evidencia académica de referencia, registrar la abstención y dejar CP/CR como no aplicables. Estas métricas pueden calcularse con anotaciones humanas y los JSON guardados. [Métricas, 09:58–16:57][metricas] · [Clase 11, pp. 9–15][metricas-pdf]

### Qué ya se probó y qué se puede cuantificar

| Comparación histórica | Resultado | Decisión o límite |
| --- | --- | --- |
| `k=2 → 4 → 8` en VTT solo, mismo MiniLM y coseno. | Objetivo coseno y limitaciones, **0/2 → 1/2 → 1/2 criterios completos**. Objetivo general E1–E5, **0/5 → 0/5 → 1/5**. En el último corte, los puestos 5 y 8 juntos respaldaron E1. | `k=8` quedó como referencia provisional. Agregó evidencia y ruido. [Seis corridas de ablación][ablacion] |
| VTT solo vs. VTT + apunte, con `k=8`. | Para coseno, el corpus mixto devolvió el ejemplo de magnitud en el puesto 3 (p. 5 del apunte) y la fórmula en el 6 (31:44.560–32:41.570). Para el objetivo general recuperó definición y contraste contextual de la p. 3. La fórmula y el ejemplo completo quedaron fuera de esa consulta. | Agregar el apunte ayudó a recuperar evidencia, con alcance limitado al piloto. [Bitácora, §13][bitacora] |
| Consulta directa vs. expansión `inicial` de GPT-4o, corpus mixto y 8 finales. | Para «quiero estudiar embeddings», definición y uso quedaron respaldados en ambos brazos. El contraste contextual pasó de parcial a completo. La expansión reunió 32 candidatos, 15 únicos y 8 finales. La lista de fuentes quedó primera en ambos rankings. | La expansión recuperó evidencia útil con el mismo `k` final. El ruido en el primer puesto motiva medir precisión y reordenamiento. [Bitácora, §§14–15][bitacora] |
| Redacción, prompt y modelo planificador. | Agregar «entre dos embeddings» a la consulta de cálculo recuperó la fórmula en el puesto 3. La consulta dirigida a *skip-gram* y pares positivos/negativos encontró el pasaje 29:16.470–30:22.040 en puestos 1–2. Los planes iniciales GPT-4o y GPT-4.1 dejaron ese pasaje fuera de los candidatos. El prompt `concreto` nombró Word2Vec, tampoco recuperó ese pasaje y perdió el contraste contextual en la lista final. | Se conserva `inicial`. GPT-4.1 es el default histórico del proxy y adelantó el pasaje contextual del puesto 5 al 1 en este caso (15 vs. 20 candidatos únicos para GPT-4o y GPT-4.1). Sigue abierto el problema de entrenamiento. [Bitácora, §§15 y 19–22][bitacora] |

**Cálculos retrospectivos sobre rankings guardados.** Se pueden presentar con su objetivo y denominador, como ejemplos de desarrollo.

- **CP en «entender similitud coseno y sus limitaciones»**, VTT solo. Con relevancia definida por aportar la fórmula, la regla sobre magnitud o un ejemplo interpretable de esa regla, las etiquetas del top 8 son `[0, 0, 0, 1, 0, 0, 0, 0]`. El puesto 4 contiene el cálculo (ID `5acf1645d843`). El puesto 8 conserva el cierre de un ejemplo sobre frecuencia, sin el vínculo con magnitud y coseno. Los cortes `k=2` y `k=4` son prefijos verificados. CP para `k=2/4/8` da **0 / 0,25 / 0,25** y P@k da **0/2, 1/4 y 1/8**. El último corte agrega ruido aunque CP se mantenga. [Rankings y revisión][ablacion] · [Revisión de los puestos 1–4][revision]
- **CR introductorio en «quiero estudiar embeddings»**, corpus mixto y `k=8`. Se toman tres afirmaciones de la revisión retrospectiva: un embedding representa texto como vector, la cercanía sirve para comparar significado y un embedding estático conserva el vector de una palabra mientras que uno contextual se calcula para cada aparición. La directa respalda las dos primeras en el puesto 3 y aporta menciones parciales sobre el contexto. La expansión respalda las dos primeras en el puesto 2 y aporta el contraste completo en el puesto 5, con el ejemplo de «banco». Resultado **2/3 → 3/3** con esta referencia introductoria. Los conteos E1–E5 de la ablación anterior corresponden a criterios completos de otro objetivo. [Bitácora, §15][bitacora]

### Qué probar a continuación

| Prueba | Control y lectura del resultado | Sin clave API |
| --- | --- | --- |
| `k` y sensibilidad a la redacción | Repetir `2/4/8` y consultas equivalentes con el corpus final fijo. Comparar CP, CR, P@k, tokens y latencia. Elegir el menor `k` que conserve la evidencia esencial observada en desarrollo y confirmarlo en objetivos reservados. | Sí. Se pueden anotar los JSON previos y volver a ejecutar búsqueda local. |
| Directa vs. expansión | Comparar con el mismo corpus, embedder y `k` final. Guardar candidatos por subconsulta y lista final. Medir CP/CR en el ranking final y CR en la unión de candidatos para ubicar dónde se perdió evidencia. Separar tiempo y costo del planificador del tiempo de búsqueda. | Sí para reproducir planes guardados con `--plan`. Generar planes GPT nuevos sigue dependiendo del proxy. |
| Rerankers | Obtener un pool fijo de, por ejemplo, 20 candidatos por objetivo. Comparar orden base por coseno vs. `cross-encoder/ms-marco-MiniLM-L-6-v2` de la demo, con el mismo `k` final. Para probar un segundo reranker, usar el mismo diseño con un CrossEncoder multilingüe local si se acuerda incorporar un checkpoint adicional al material. Comparar CP/CR, posiciones de pasajes esenciales, tokens y latencia. CR del pool es el techo de los rerankers para ese conjunto de candidatos. | Sí con los pesos disponibles. El reranking todavía requiere un piloto fuera de la CLI actual. |

Para diagnosticar un faltante, seguir el pasaje desde la fuente al chunk, del chunk a los candidatos, de los candidatos a los `k` finales y del contexto al informe. Esa secuencia indica si conviene corregir la carga, cambiar la búsqueda, reordenar o ajustar el generador. [RAG, 1:24:59–1:25:32][rag] · [Implementación, 1:30:13–1:32:36][rag-impl] · [Notebook Advanced RAG, celdas 48–51 y 58–61][advanced]

## 3. Generación (Generation)

### Métricas de la clase y del informe

| Métrica | Cómo medirla para este TP | Sin clave API |
| --- | --- | --- |
| **Answer Relevance** | Generar tres preguntas a partir del informe visible y promediar la similitud coseno entre sus embeddings y el de la solicitud original (objetivo + tarea de evaluar el video). Usar el mismo juez, instrucciones y encoder evaluador para todos los brazos. Retirar el encabezado que repite el objetivo antes de darle el informe al juez. Medir aparte si el informe omite criterios. | Se puede intentar con un juez local y un encoder en caché. Antes de puntuar, revisar si reconstruye preguntas pertinentes en español. Una rúbrica humana de pertinencia sirve como indicador separado mientras se calibra el juez. |
| **Answer Faithfulness** | `Afirmaciones verificables respaldadas por el contexto enviado / total de afirmaciones verificables generadas`. Separar afirmaciones de las filas, recomendaciones y conclusión. Comprobar afirmaciones académicas contra los chunks y afirmaciones sobre el video contra el VTT enviado. Revisar por separado si el pasaje **citado** sostiene la afirmación. | Sí. Auditoría humana con los mensajes y salidas guardados. Un juez LLM calibrado puede automatizarla más adelante. |
| **Cobertura, abstención y citas** | Comparar `cubierto`, `parcial` y `sin_evidencia` con la referencia del video para criterios evaluables. Si falta material de clase suficiente, se espera `respaldo_insuficiente`. Contar omisiones, contradicciones, abstenciones justificadas, citas existentes y citas sustantivas, recomendaciones útiles, informes válidos y truncamientos. | Sí, para informes guardados o nuevas salidas de Qwen local. |

Guardar los denominadores. Una salida vacía o inutilizable cuenta como fallo de informe. Faithfulness queda sin puntaje si no hay afirmaciones verificables. Comparar las métricas semánticas entre pares utilizables e informar la tasa de resolución sobre todos los casos. [Métricas, 16:57–22:17][metricas] · [Clase 11, pp. 16–20][metricas-pdf] · [Demo RAGAS, celdas 4–9][ragas]

### Qué ya se probó

| Prueba histórica | Resultado medible y observación | Decisión que respalda |
| --- | --- | --- |
| Qwen2.5-1.5B-Instruct local, pasajes seleccionados vs. Chroma con `k=8`, mismos 99 segmentos de video. | Con prompt `inicial`, la selección manual llegó al límite de 1.000 tokens y no clasificó. Con Chroma, Qwen declaró cubierto el objetivo e inventó 4 IDs de clase. | Parte del problema está en generación, ya que la salida también falla cuando recibe los pasajes adecuados. [Registro de generación][generacion] |
| Qwen `inicial → tabla` y penalización `1,10 → 1,15` con `tabla`. | La tabla produjo filas, pero ambas salidas con `1,10` llegaron al límite. Con Chroma las filas duplicadas bajaron **13 → 0** y la salida **1.000 → 197 tokens** al subir la penalización. Quedaron dos filas sin cita del video. Con pasajes seleccionados, los IDs académicos inexistentes bajaron **14 → 6**. | El formato y la penalización cambiaron la salida y redujeron repetición. Persistieron omisiones y citas débiles. [Registro de generación, segunda y tercera pruebas][generacion] |
| GPT-4o con `tabla`, selección manual vs. Chroma. | Terminó ambos informes, con **5 y 4 filas**, respectivamente, y **0 IDs inexistentes**. Con selección manual contradijo el estado de E2 y omitió E5. Con Chroma faltaron E4 y E5 y se citaron pasajes insuficientes para justificar E3. | Verificar tanto el respaldo disponible como su uso en cada fila. [Registro de generación, cuarta y quinta pruebas][generacion] |
| GPT-4o vs. GPT-4.1 con `estudiante`, mismos mensajes, plan, ranking y VTT. | Ambos terminaron con cuatro filas y una recomendación. Conteo retrospectivo de ocurrencias con ID existente, incluyendo recomendación: GPT-4o **7/7** de clase y **11/11** de video, GPT-4.1 **10/10** y **26/26**. Costos históricos de generación **US$0,0138025** y **US$0,01245**. GPT-4.1 reemplazó la lista de fuentes por un pasaje que sí respalda un uso de los embeddings. Ambos clasificaron entrenamiento como `parcial` cuando correspondía respaldo insuficiente. | GPT-4.1 quedó como referencia del proxy. El conteo de IDs confirma existencia y deja abierta la fidelidad semántica. [Bitácora, §§16–18][bitacora] |
| GPT-4.1 `estudiante` con y sin pasaje de Word2Vec, cambiando solo `C5`. | Con el pasaje 29:16.470–30:22.040 mejoró la recomendación de entrenamiento. La etiqueta siguió como `parcial` en los dos brazos y la fila mantuvo las citas tangenciales `C1` y `C2`. | Es un antes/después diagnóstico. El siguiente contraste puede mantener fijo el contexto y pedir que la explicación identifique qué parte esencial del criterio sostiene el pasaje antes de clasificar el video. [Bitácora, §23][bitacora] |

**Ejemplo de auditoría de Faithfulness.** Al aislar la afirmación implícita «el material recomendado explica cómo se entrena Word2Vec» en el par con y sin el pasaje, el respaldo pasa de **0/1 a 1/1**. En el control, el material enviado deja fuera la explicación y `C5` es una lista de fuentes. En la variante, `C5` contiene el procedimiento con pares positivos y negativos. El conteo cubre esa afirmación de la recomendación. La etiqueta de cobertura siguió requiriendo corrección. [Bitácora, §23][bitacora]

Los JSON históricos permiten completar **Answer Faithfulness** sin volver a generar. Para el primer par del formato `estudiante`, usar `informe_estudiante_embeddings_20261104.4x6c1gtg/auditoria.json` e `informe_estudiante_gpt41_20261104.41f5UB/informe.json` dentro de `ejercicio2/data/resultados/`. Falta descomponer y etiquetar todas las afirmaciones para publicar el puntaje por informe. **Answer Relevance** queda pendiente de calcular con preguntas inversas y un juez común calibrado.

### Qué probar a continuación

1. **Contexto**. Con el mismo generador, prompt `tabla` y VTT, comparar pasajes de referencia, directa y expansión o reranking. Incluir un contexto que omita deliberadamente un criterio para medir abstención. Los pasajes manuales sirven para diagnosticar generación. La corrida integrada debe usar el contexto realmente recuperado.
2. **Prompt**. Con contexto y modelo fijos, comparar la tabla actual con una instrucción que exija identificar la evidencia académica antes de asignar el estado. Medir Faithfulness, abstención correcta, citas sustantivas y criterios omitidos. `inicial` y `tabla` pueden ejecutarse con Qwen y `--solo-local` si los pesos están en caché. El formato `estudiante` de la CLI exige actualmente `--backend proxy` y una recuperación expandida guardada.
3. **Modelo y validación integral**. Auditar ahora el par histórico GPT-4o/GPT-4.1 y repetirlo cuando vuelva el acceso si hace falta confirmar. Sobre objetivos y videos reservados, comparar la configuración de referencia y la elegida usando el mismo VTT por video. Publicar las cuatro métricas, cobertura y citas, fallos, tokens, latencia y costo. Para medir el impacto del ASR en el informe, hacer otro par cambiando únicamente el VTT de entrada. [Reproducción de la generación][generacion] · [Comandos y opciones][readme] · [Plan experimental][plan]

## Orden de trabajo sin clave API

1. Anotar los rankings históricos con el caso 01, medir CP/CR sobre las afirmaciones fijadas y auditar la fidelidad de los informes guardados. Completar WER/CER de Wav2Vec2 en los recortes ya referenciados.
2. Ejecutar localmente tamaño y solapamiento de chunks, Chroma frente a coseno exacto, cortes de `k` y reproducción de expansión con `--plan`. Correr el piloto de reranking con un pool fijo cuando estén los pesos. Preparar referencias de objetivos y videos reservados.
3. Comparar generación local con `tabla` y calibrar el juez local de relevancia. Conservar para cuando vuelva el proxy las llamadas nuevas de expansión GPT, los informes `estudiante`, la comparación Whisper del piloto y la extracción visual prevista. El 429 registrado en las rutas ensayadas informa falta de créditos. [Bitácora, §§35–37][bitacora]

Para la defensa, mostrar cada decisión con el par de configuraciones, el cambio observado, un pasaje que explique el resultado y el alcance de la prueba. Ya hay tres ejemplos con evidencia: subir `k` recuperó una definición a costa de más contexto, expandir agregó el contraste contextual en el objetivo introductorio y aportar el pasaje de Word2Vec mejoró la recomendación sin corregir la clasificación.

[consigna]: ../docs/RAG/tp3.pdf
[consigna-alt]: ../docs/RAG/DeepLearningTP2.pdf
[tp-vtt]: ../docs/RAG/tp3.VTT
[rag]: ../docs/RAG/RAG.VTT
[rag-impl]: ../docs/RAG/RAG%20Implementacio%CC%81n.VTT
[audio]: ../docs/RAG/Ima%CC%81genes%20y%20Audio.VTT
[metricas]: ../docs/RAG/Metricas.VTT
[metricas-pdf]: ../docs/RAG/Clase%2011%20-%20M%C3%A9tricas%20de%20la%20Industria%20%281%29.pdf
[pinecone]: ../docs/RAG/Langchain_Pinecone.ipynb
[advanced]: ../docs/RAG/UN_Report_Advanced_RAG.ipynb
[ragas]: ../docs/RAG/RAGAS.ipynb
[caso]: ejercicio2/evaluacion/caso_01_embeddings.md
[revision]: ejercicio2/evaluacion/revision_retrieval_inicial.md
[ablacion]: ejercicio2/evaluacion/ablacion_k.md
[generacion]: ejercicio2/evaluacion/generacion_caso01.md
[readme]: ejercicio2/README.md
[asr]: avances/transcripcion/README.md
[plan]: avances/plan_evaluacion_experimental.md
[bitacora]: presentación.md
