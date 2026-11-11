# TP3 · Recorrido experimental hasta la versión actual

**Corte al 11 de noviembre de 2026.** La presentación HTML llega al experimento 7, GPT‑4o con el contexto recuperado por Chroma. Los experimentos posteriores dejaron una configuración de referencia con **corpus mixto, expansión de consultas y GPT‑4.1**. El recorrido explica qué se probó, qué justificó cada decisión y con qué configuración comparar la futura iteración del ejercicio 4. [Presentación HTML][html] · [Bitácora][bitacora] · [Consigna, pp. 2–4][consigna]

## 1. Punto de partida · Caso manual, índice y búsqueda directa

Queríamos determinar si el audio de un video sirve para estudiar un objetivo. El material de clase fija qué se espera aprender. El audio transcripto permite comprobar qué explica el video. El primer caso usa el video «¿Qué son los EMBEDDINGS Grandes Modelos de Lenguaje?», transcripto con Whisper, y el objetivo «entender qué son los embeddings y cómo se compara la similitud entre textos». La revisión manual encontró **E1 definición y E2 contexto cubiertos, E3 comparación parcial y E4 fórmula del coseno y E5 limitación por magnitud sin evidencia en la transcripción**. Es la referencia de desarrollo para ese objetivo. [Caso manual y tiempos de clase y video][caso]

**Experimento 1 · Primer índice.** Partimos de las 415 intervenciones del VTT de la clase de embeddings. Aplicamos corte recursivo de 1.000 caracteres con 100 de solapamiento y luego ventanas de hasta 200 tokens de contenido con 20 de solapamiento. Obtuvimos **125 chunks**, con un máximo de **202 tokens con especiales** frente al límite de 256 del encoder. Usamos `all-MiniLM-L6-v2` en CPU, vectores de 384 dimensiones y Chroma local persistente con distancia coseno. La configuración retomó las demos de clase, cabía en el encoder y permitía conservar los tiempos y reutilizar la carga. La segunda indexación reutilizó los 125 chunks. Quedaron cortes dentro de palabras y errores del VTT. Tamaño, solapamiento, embedder y base quedaron como decisiones iniciales sin una ablación entre alternativas. [Revisión inicial][revision] · [Demo del asistente virtual, celda 9][asistente] · [Demo del reporte ONU, celdas 11–23][advanced]

**Experimento 2 · Cuántos chunks recuperar.** La búsqueda directa empezó con `k=4`, sin umbral. Probamos `k=2, 4 y 8` con el mismo VTT, índice, embedder y distancia. Los criterios completos se revisaron al leer en conjunto los fragmentos devueltos. La precisión mide el ranking de chunks para la consulta A. [Ablación y rankings][ablacion] · [Cálculo retrospectivo de precisión][evaluaciones]

| Resultado | k=2 | k=4 | k=8 |
| --- | ---: | ---: | ---: |
| A · Coseno y limitaciones, criterios completos | 0/2 | 1/2 | 1/2 |
| B · Objetivo del caso 01, criterios completos | 0/5 | 0/5 | 1/5 |
| A · Context Precision retrospectiva | 0 | 0,25 | 0,25 |
| A · Precision@k retrospectiva | 0/2 | 1/4 | 1/8 |

**Decisión.** Usamos `k=8` provisionalmente para integrar el generador. Fue el único corte que permitió respaldar E1 en B, combinando los puestos 5 y 8. En A, la fórmula siguió en el puesto 4 y los cuatro resultados adicionales no completaron el ejemplo de magnitud. Por eso se mantuvo Context Precision y cayó Precision@k. La decisión priorizó recuperar evidencia para el objetivo general, con el costo de sumar ruido. [Ablación, resultados y configuración elegida][ablacion]

## 2. Primera integración · De Qwen local a GPT‑4o

Se usaron el objetivo general del caso 01 y los mismos **99 segmentos del video**. Los seis pasajes académicos seleccionados a mano respaldaban E1–E5 y servían para diagnosticar el generador. El brazo automático usó los ocho chunks de la búsqueda directa, que juntos respaldaban por completo E1. [Registro de generación][generacion]

| Experimento del HTML | Cambio y resultado | Decisión |
| --- | --- | --- |
| **3 · Qwen2.5-1.5B-Instruct, prompt `inicial`** | Con pasajes manuales copió texto y llegó al límite de 1.000 tokens antes de clasificar. Con Chroma produjo 608 tokens, declaró cubierto el objetivo e inventó 4 IDs de clase. | El fallo con evidencia suficiente motivó probar una instrucción de salida más concreta. |
| **4 · Qwen, prompt `tabla`** | El prompt pidió una fila por criterio con estado, justificación y citas de las dos fuentes. Con pasajes manuales aparecieron 19 filas completas y 14 IDs inexistentes. Con Chroma hubo 22 filas completas, 13 duplicadas, y se mezclaron citas de las fuentes. Ambas salidas llegaron al límite sin cerrar el informe. | La tabla logró que empezara a clasificar. Mantuvimos el prompt para probar si reducir la repetición ayudaba a terminar. |
| **5 · Qwen, penalización 1,10 → 1,15** | Con Chroma, los duplicados bajaron de 13 a 0 y la salida de 1.000 a 197 tokens, con solo 2 filas y sin citas del video en ellas. Con pasajes manuales, los IDs inexistentes bajaron de 14 a 6 y la salida de 1.000 a 473 tokens. | Mejoró el problema de repetición. Las omisiones y los errores de citas motivaron comparar otro generador. |
| **6 · Qwen 1,15 vs. GPT‑4o, mismos pasajes manuales y prompt** | Los IDs inexistentes bajaron de **6 a 0**. GPT‑4o produjo 5 filas y cerró con recomendación y conclusión. Qwen produjo 8 filas y un resumen general. GPT‑4o todavía se contradijo sobre E2, dejó E4 para la recomendación y omitió E5. | GPT‑4o pasó a la prueba integrada por el cierre del informe y la validez de los IDs observados. Fue una comparación entre configuraciones local y proxy, con la evidencia y el prompt fijos. |
| **7 · GPT‑4o, pasajes manuales vs. Chroma** | Con el mismo modelo y prompt, las filas bajaron de 5 a 4 y los IDs inexistentes se mantuvieron en 0. La búsqueda dejó fuera la fórmula y la limitación por magnitud. El informe omitió ambas y citó para E3 pasajes que no explicaban la medida. | Ya había una generación automática de punta a punta desde el VTT. La falta de contexto motivó ampliar el corpus. |

Los conteos de filas describen la forma de la salida. Una fila completa puede repetir un criterio, citar un pasaje tangencial o asignar un estado incorrecto. [Salida y revisión por corrida][generacion] · [Experimento 7 en el HTML][html-fin]

## 3. Después del HTML · Corpus mixto y expansión de consultas

**Se sumó el apunte de embeddings.** Primero se revisó un apunte de Transformers y se reservó para esa unidad, porque sus referencias al coseno trataban codificación posicional. Se incorporaron en cambio las páginas 3–5 del apunte derivado de la clase de embeddings. Con el mismo splitter y MiniLM, el corpus pasó de **125 a 136 chunks**. Se conservaron las dos colecciones y la procedencia de los 11 chunks añadidos. Para la consulta de coseno, el apunte aportó el ejemplo de magnitud en el puesto 3 y la fórmula de la clase quedó en el 6. Para el objetivo general, aparecieron definición y contraste contextual del apunte. La fórmula y el ejemplo completo de magnitud siguieron fuera de esa búsqueda. El apunte se preparó con los temas del caso ya conocidos. [Bitácora, §§12–13][bitacora] · [Reconstrucción de la colección][readme]

**Cálculo retrospectivo del 11/11 · VTT solo vs. VTT con apunte.** Sobre los dos rankings guardados de «entender similitud coseno y sus limitaciones», con MiniLM, coseno y `k=8` fijos, se etiquetó cada chunk como relevante cuando aportaba la fórmula, la relación entre coseno y magnitud o un ejemplo que la explicara. Para Context Recall se tomaron dos afirmaciones de referencia. La similitud coseno se calcula mediante el producto interno normalizado. Dos vectores de la misma dirección tienen similitud coseno 1 aunque difieran en magnitud. El puesto 5 del corpus mixto explica el escalado de un vector, pero deja implícito el vínculo con el coseno, por lo que quedó fuera de la etiqueta de relevancia. [Clase 11, pp. 9–15][metricas] · [Ranking del VTT][datos-k] · [Ranking mixto][datos-mixto]

| Corpus | Puestos relevantes | Context Precision@8 | Precision@8 | Context Recall@8 |
| --- | --- | ---: | ---: | ---: |
| VTT solo | 4 · fórmula (`5acf1645d843`) | 0,25 | 1/8 | 1/2 |
| VTT + apunte | 3 · magnitud (`879e8c68419d`), 6 · fórmula (`5acf1645d843`) | 0,333 | 2/8 | 2/2 |

El cálculo respalda la incorporación del apunte para esta consulta. El ejemplo añadido completó la evidencia esperada y adelantó el primer resultado relevante, aunque la fórmula quedó dos posiciones más abajo.

**Se probó Advanced RAG con expansión por subtemas.** Para el nuevo objetivo «quiero estudiar embeddings», comparamos consulta directa con un plan de GPT‑4o, prompt de expansión `inicial`. Se fijaron el corpus mixto, MiniLM, coseno y ocho chunks finales. Las cuatro subconsultas devolvieron 32 candidatos, 15 únicos. Los ocho finales se eligieron por rondas de posición y sin repetir IDs. Con una referencia introductoria de tres afirmaciones (definición, uso en comparación semántica y contraste estático/contextual), **Context Recall pasó de 2/3 a 3/3**. La expansión agregó el ejemplo de «banco» en el puesto 5. La lista de fuentes quedó primera en ambos rankings. Mantuvimos la expansión porque agregó evidencia con la misma cantidad de fragmentos finales. La fórmula se evalúa cuando el objetivo pide calcularla. [Bitácora, §§14–15][bitacora] · [Demo de expansión, celdas 48–51][advanced] · [Directa y expansión guardadas][datos-expansion]

## 4. Del primer informe para el estudiante a GPT‑4.1

**Formato `estudiante`.** Con el plan de GPT‑4o, ocho referencias automáticas y el mismo VTT, GPT‑4o produjo cuatro evaluaciones por subtema, una recomendación y una conclusión. La salida JSON permitió presentar citas con tiempos y registrar los estados por subtema. El cambio de formato coincidió con un nuevo objetivo y contexto, así que su aporte a la calidad queda sin aislar. [Bitácora, §§16–17][bitacora]

**Comparación controlada de generadores.** En el par GPT‑4o/GPT‑4.1 se conservaron el plan de GPT‑4o, el objetivo, los mensajes, las ocho referencias, los 99 segmentos, la temperatura 0 y el tope de 1.000 tokens de salida. El conteo de IDs se hizo por ocurrencia e incluyó la cita académica de la recomendación. [Bitácora, §18][bitacora] · [Auditoría GPT‑4o][datos-4o] · [Auditoría GPT‑4.1][datos-41]

| Resultado | GPT‑4o | GPT‑4.1 |
| --- | ---: | ---: |
| Filas y recomendaciones | 4 y 1 | 4 y 1 |
| Ocurrencias con ID existente de clase | 7/7 | 10/10 |
| Ocurrencias con ID existente de video | 11/11 | 26/26 |
| Tokens de entrada / salida | 3.737 / 446 | 3.737 / 622 |
| Costo histórico de generación | US$0,0138025 | US$0,01245 |

**Decisión.** GPT‑4.1 quedó como referencia del proxy. En este par mantuvo las evaluaciones respaldadas de definición y contexto, reemplazó en aplicaciones una lista de fuentes por un pasaje que sí explica un uso, y costó alrededor de **9,8 % menos** en esa llamada, con más tokens de salida. Ambos modelos marcaron entrenamiento como `parcial` con respaldo académico insuficiente y recomendaron un pasaje centrado en comparar vectores. La existencia de los IDs quedó empatada. La mejora de respaldo se observó en un subtema y el puntaje global de Faithfulness sigue pendiente. [Bitácora, §18][bitacora] · [Evaluación por componente][evaluaciones]

**Diagnóstico del plan y decisión sobre la expansión.** La consulta manual con «Word2Vec skip-gram con pares positivos y negativos» encontró el pasaje de entrenamiento en los puestos 1 y 2, lo que confirmó que estaba indexado (29:16.470–30:22.040 de la clase). Con el mismo GPT‑4o, el prompt `concreto` nombró Word2Vec, dejó ese pasaje fuera y perdió el contraste contextual que sí recuperaba `inicial`. Se conservó `inicial`. Después comparamos los planificadores GPT‑4o y GPT‑4.1 con ese prompt y los mismos parámetros de búsqueda. **Ambos respaldaron 3/3 afirmaciones introductorias**. GPT‑4.1 adelantó el contraste contextual del puesto 5 al 1, produjo 20 candidatos únicos frente a 15 y costó US$0,001456 frente a US$0,00177. El pasaje de entrenamiento quedó fuera de los candidatos en ambos. El mejor puesto para ese pasaje contextual y el costo observado respaldan usar GPT‑4.1 también como planificador, con el mismo recall introductorio en este caso. [Bitácora, §§19–22][bitacora] · [Plan y recuperación de GPT‑4.1][datos-plan-41]

**Última comparación integrada · Con y sin evidencia de Word2Vec.** Con GPT‑4.1 como planificador y generador, `estudiante` y el mismo VTT, el control usó ocho resultados automáticos. En la variante se reemplazó solo el quinto fragmento, una lista de fuentes, por el pasaje de entrenamiento. El respaldo de la afirmación «el material recomendado explica cómo se entrena Word2Vec» pasó de **0/1 a 1/1**. Ambas salidas mantuvieron `parcial` y justificaron ese estado con los mismos dos pasajes tangenciales. Sin el reemplazo correspondía `respaldo_insuficiente`. El pasaje incorporado permitió mejorar la recomendación en la prueba diagnóstica. El reemplazo manual quedó fuera de la configuración automática. [Bitácora, §23][bitacora] · [Informes con y sin pasaje][datos-respaldo]

## 5. Configuración actual para comparar en el ejercicio 4

| Componente | Referencia del ejercicio 2 |
| --- | --- |
| Entrada del video | VTT histórico de Whisper, con 99 segmentos para el caso 01. |
| Corpus y preparación | VTT de la clase de embeddings + pp. 3–5 del apunte derivado. 136 chunks con splitting 1.000/100 caracteres y 200/20 tokens. Colección `tp3_248fe3ddb4d5588934314f79`. |
| Índice y recuperación | MiniLM local de 384 dimensiones, Chroma persistente, distancia coseno. GPT‑4.1 (`openai/gpt-4.1-2025-04-14`), prompt de expansión `inicial`, hasta cuatro subconsultas, ocho candidatos por consulta y ocho referencias finales por rondas y deduplicación. |
| Generación | GPT‑4.1 por el proxy, prompt `estudiante`, JSON, temperatura 0 y tope de 1.000 tokens de salida. |
| Corrida automática de referencia | `respaldo_entrenamiento_20261104.63FuR6/informe_sin_pasaje.json`, con el ranking de `expansion_gpt41_20261104.umOHA2/expandida.json`. |

La CLI conserva Qwen local y el prompt inicial por defecto. Para reproducir la configuración de referencia se seleccionan el corpus mixto, la expansión y el backend proxy con `--prompt estudiante` y la recuperación guardada. Los artefactos bajo `data/` son locales y están ignorados por Git. La comparación de configuración y la revisión de los JSON se hicieron sobre resultados guardados. [Comandos de reproducción][readme] · [Configuración local][config] · [Corrida automática][datos-control]

El piloto de audio posterior probó **Wav2Vec2 en un recorte** (ASR01, 00:55.650–01:19.150). Contra la referencia humana obtuvo **WER 14,55 % (8/55)** y **CER 5,21 % (16/307)**. Whisper devolvió 429 para ese recorte y quedó pendiente el par sobre el mismo audio. La prueba posterior de `gpt-4o-mini` mediante la API directa verificó una consulta breve de texto. El VTT histórico sigue siendo la entrada de referencia del RAG. El corpus de varias unidades, el reranking y la evidencia de pantalla quedaron como propuestas de evaluación. [Piloto ASR][asr] · [Bitácora, §§24–37 y 47][bitacora]

## 6. Qué métricas podemos mostrar y qué falta medir

| Métrica | Evidencia disponible | Qué permite afirmar |
| --- | --- | --- |
| **Context Precision**, clase 11, pp. 9–12 | Retrospectiva. Para coseno, `k=4 → 8` mantuvo **0,25** con el VTT solo, mientras Precision@k bajó de **1/4 a 1/8**. Con `k=8` fijo, sumar el apunte llevó CP de **0,25 a 0,333**. | El tamaño del contexto y el corpus afectan aspectos distintos del ranking. Se puede calcular CP para otros rankings guardados tras etiquetar sus chunks. |
| **Context Recall**, pp. 13–15 | Retrospectiva. Para coseno con `k=8`, sumar el apunte llevó **1/2 a 2/2**. Para el objetivo introductorio, directa vs. expansión con GPT‑4o dio **2/3 a 3/3**. | Hay respaldo cuantitativo para ampliar el corpus y usar expansión en esos objetivos. Los denominadores corresponden a referencias distintas. |
| **Faithfulness**, pp. 16–18 | Auditoría de ejemplos. Para una afirmación de la recomendación de Word2Vec, **0/1 → 1/1**. Los mensajes y las respuestas guardadas permiten descomponer y anotar las afirmaciones de los informes completos sin volver a generarlos. | Hay una mejora puntual de respaldo. Las citas existentes y los estados correctos deben medirse por separado. |
| **Answer Relevance**, pp. 19–20 | Pendiente. Los informes guardados sirven de entrada. Hace falta generar preguntas a partir de cada respuesta con un juez común y comparar sus embeddings con la solicitud original. | Queda disponible para una comparación con el mismo protocolo en ambos brazos. |
| **WER**, p. 28, y **CER** complementaria | Medición directa de Wav2Vec2 sobre ASR01. Hay seis referencias revisadas para continuar el piloto. | Hay un punto de referencia de transcripción y queda pendiente comparar ambos transcriptores en los mismos recortes. |
| **Indicadores del producto y costo**, p. 24 para costos | Filas, repetición, cierre, existencia de citas, casos de abstención incorrecta, tokens y costo por llamada. | La elección Qwen → GPT‑4o se apoya en cierre e IDs válidos. La elección GPT‑4o → GPT‑4.1 se apoya en el respaldo observado y el costo histórico en la comparación controlada. |

Los cálculos retrospectivos se basan en la lectura y el etiquetado de los artefactos guardados. Se reportan como evidencia de desarrollo, principalmente sobre un video y un tema. [Clase 11, pp. 9–20, 24 y 28][metricas] · [Explicación de clase, 23:26–24:09 y 33:20–33:39][metricas-vtt] · [Detalle por componente][evaluaciones]

## 7. Qué comparar en la próxima iteración

**Limitación observada.** El informe puede marcar un subtema como `parcial` y recomendar un pasaje tangencial cuando falta evidencia académica. El control automático de GPT‑4.1 deja un ejemplo reproducible. La inserción manual de Word2Vec también mostró que mejorar el contexto puede corregir la recomendación y dejar sin resolver la justificación de la cobertura. [Bitácora, §23][bitacora]

**Propuesta para el ejercicio 4.** Mantener el corpus, el ranking, el VTT y GPT‑4.1 fijos, y cambiar solo el prompt para que identifique qué afirmación académica respalda cada pasaje antes de clasificar el video. Pedir `respaldo_insuficiente` cuando falte esa base. Comparar la versión actual con la variante en los mismos casos, con ejemplos de respaldo suficiente e insuficiente y objetivos reservados. Medir abstenciones correctas, estados por criterio, respaldo de citas y recomendaciones, Faithfulness y Answer Relevance con la misma pauta. Con el ranking fijo, CP y CR describen la entrada compartida. Si se prueba un cambio de recuperación en otra comparación, medir CP y CR sobre sus rankings finales. La iteración queda por ejecutar y documentar con resultados antes/después. [Consigna, §1.4, p. 3][consigna] · [Plan experimental][plan]

[html]: presentacion.html
[html-fin]: presentacion.html#gpt4o-chroma
[bitacora]: presentación.md
[consigna]: ../docs/RAG/tp3.pdf
[caso]: ejercicio2/evaluacion/caso_01_embeddings.md
[revision]: ejercicio2/evaluacion/revision_retrieval_inicial.md
[ablacion]: ejercicio2/evaluacion/ablacion_k.md
[generacion]: ejercicio2/evaluacion/generacion_caso01.md
[evaluaciones]: evaluaciones.md
[readme]: ejercicio2/README.md#repetir-las-pruebas-con-el-proxy
[config]: ejercicio2/config.py
[asr]: avances/transcripcion/README.md
[plan]: avances/plan_evaluacion_experimental.md
[asistente]: ../docs/RAG/Langchain_Virtual_Assitant.ipynb
[advanced]: ../docs/RAG/UN_Report_Advanced_RAG.ipynb
[metricas]: ../docs/RAG/Clase%2011%20-%20M%C3%A9tricas%20de%20la%20Industria%20%281%29.pdf
[metricas-vtt]: ../docs/RAG/Metricas.VTT
[datos-k]: ejercicio2/data/resultados/ablacion_k_20261104.oaCc7k/coseno_k8.json
[datos-mixto]: ejercicio2/data/resultados/chroma_con_apunte_20261104.7sqyKv/coseno_k8.json
[datos-expansion]: ejercicio2/data/resultados/expansion_subtemas_20261104.zQ1Iz1/
[datos-4o]: ejercicio2/data/resultados/informe_estudiante_embeddings_20261104.4x6c1gtg/auditoria.json
[datos-41]: ejercicio2/data/resultados/informe_estudiante_gpt41_20261104.41f5UB/informe.json
[datos-plan-41]: ejercicio2/data/resultados/expansion_gpt41_20261104.umOHA2/expandida.json
[datos-respaldo]: ejercicio2/data/resultados/respaldo_entrenamiento_20261104.63FuR6/
[datos-control]: ejercicio2/data/resultados/respaldo_entrenamiento_20261104.63FuR6/informe_sin_pasaje.json
