# TP3 · Roadmap experimental del ejercicio 2

**14/11/2026 · Plan de trabajo.** La [propuesta corregida del ejercicio 1][ej1] fija el problema: determinar si lo que se escucha en un video sirve para un objetivo de estudio. La materia define qué debería aprenderse y la transcripción del video permite verificar qué explica. El informe señala cobertura, recomienda material de la cátedra y cierra con una conclusión útil para el estudiante.

Las dos consignas piden dataset, pre y postprocesamiento, métricas, ablaciones incrementales y al menos una técnica avanzada **después** de completar el RAG básico ([consigna], pp. 2–3; [otra versión][consigna-alt], pp. 2–3; [explicación oral][tp-vtt], 08:52–09:45). Los parámetros de abajo son puntos de partida del grupo; los notebooks son ejemplos. El calendario y la modalidad del ejercicio 3 siguen por confirmar ([guía], §5.3). El [archivo histórico][archivo] conserva el prototipo y sus resultados para consulta.

## 0. Fijar insumos y pauta de anotación

1. **Corpus académico.** Empezar con un VTT docente para probar el recorrido, por ejemplo `embeddings_2.VTT`. Antes de comparar, congelar un corpus chico con VTT de tres unidades, por ejemplo embeddings, Transformers y fine-tuning. Registrar por fuente el ID, unidad, ruta, licencia o procedencia, SHA-256 y versión de extracción. Conservar los originales y mantener el mismo corpus en desarrollo y prueba. Agregar PDFs solo en un contraste posterior si se detectan huecos en la referencia.
2. **Casos.** Preparar un piloto de 12 pares objetivo–video, 6 de desarrollo y 6 de prueba reservada, repartidos entre las tres unidades. Agrupar por video y por objetivo equivalente para que sus variantes queden en la misma partición. Incluir casos con cobertura completa, parcial y ausente, algún objetivo sin respaldo en el corpus y algún audio difícil de interpretar. El caso 01 previo puede servir para calibrar en desarrollo después de revisarlo; ya fue visto. Los tamaños son metas de piloto: registrar el número efectivo y limitar las conclusiones a esa muestra.
3. **Referencia humana.** Antes de mirar salidas, anotar para cada objetivo y nivel entre dos y cinco criterios verificables. Para cada criterio, fijar la afirmación académica esperada y uno o más pasajes válidos localizados por documento y página o tiempo, o marcar que el corpus carece de respaldo. Escuchar el video y revisar su VTT: anotar evidencia temporal y qué parte del criterio queda explicada. Dos personas revisan el caso de calibración y una muestra de cada partición; concilian diferencias con la pauta y congelan las etiquetas para las comparaciones. Las unidades de referencia son criterios y pasajes de la fuente, por lo que la misma anotación sirve aunque cambien los IDs de chunks.
4. **Estados.** «Cubierto»: el discurso aporta evidencia de todos los elementos requeridos. «Parcial»: cubre algunos; indicar cuáles faltan. «Sin evidencia en el audio»: hay base académica y audio utilizable, pero no aparece evidencia del criterio. «No evaluable»: falta base académica suficiente en el contexto, la transcripción no permite decidir o hay un fallo de entrada; registrar el motivo. Las abstenciones por fallos del sistema cuentan como errores si la referencia humana sí permitía evaluar.

Para aislar el RAG, usar el mismo VTT de cada video en todas las variantes. Si más adelante se compara ASR, hacerlo aparte: mismo audio y transcripción humana, WER como `(sustituciones + omisiones + inserciones) / palabras de referencia`, más revisión de términos técnicos e impacto en las etiquetas. La transcripción del video llega al generador y queda fuera del índice académico.

## 1. Indexing pipeline: preparar, partir, representar y persistir

El [registro experimental](ejercicio2/experimentos.md) documenta cómo se preparó el VTT de embeddings y reúne los resultados y decisiones de cada comparación.

**Entrada:** fuentes de la materia. **Salida:** colección local de textos, vectores y metadatos; cada chunk tiene ID estable, documento, unidad y límites en los cues originales (o página si se amplía a PDF). La [clase de RAG][rag], pp. 17–36 y 41–49, separa estas operaciones.

La carga lee los bloques del VTT y conserva su texto, fuente y tiempos. Documentar las reglas de limpieza que se apliquen, como quitar marcas de formato, y usar el mismo texto resultante en todas las variantes de splitting. Leer los bloques existentes es parte de la carga. El splitting decide cómo agrupar o cortar ese texto para indexarlo.

La baseline **I0** corta el texto en ventanas de **160 tokens de contenido, sin solapamiento**, contados con el tokenizer de MiniLM. Usa `sentence-transformers/all-MiniLM-L6-v2` en CPU y una colección Chroma local persistente con coseno. MiniLM y Chroma aparecen juntos en la [demo del asistente][asistente], celda 9. Antes de aceptar cada índice, comprobar el límite del encoder incluyendo tokens especiales.

Como controles de integridad del índice, revisar la procedencia, los chunks vacíos o duplicados, el truncamiento y la correspondencia entre chunks y vectores. Verificar que reindexar los mismos datos no duplique entradas y que reabrir la colección conserve IDs, conteo y ranking de una consulta de control. Registrar tiempo de construcción y tamaño del índice.

**El primer experimento es I1**, la comparación entre splitters. Una vez que exista el recuperador básico **R0** (§2), usarlo para medir el efecto de cambiar cómo se corta el mismo texto.

| ID | Hipótesis y cambio único | Controles y criterio |
| --- | --- | --- |
| I1 · splitter | Un corte recursivo por párrafos, líneas y espacios preserva mejor explicaciones que la ventana fija. Comparar ambos con máximo 160 tokens, overlap 0 y el mismo tokenizer. | Mismos textos limpios, consultas, embedder, coseno y `k=4`. Comparar Context Recall y Context Precision, más cortes que separen una definición de su explicación. |
| I2 · tamaño | Con el splitter elegido, 96 frente a 160 y 224 tokens permite ver el efecto de contexto versus fragmentación. | Overlap 0 y el resto como en I1; si un tamaño no cabe, descartarlo antes de correr. Registrar tokens recuperados y repetir la comparación de finalistas bajo un presupuesto común. |
| I3 · overlap | Comparar 0, 10, 20 y 40 tokens de solapamiento. El cero sirve como control y los tres niveles permiten observar si más continuidad recupera evidencia adicional. | Fijar el recursivo en 160 tokens y mantener corpus, consultas, MiniLM, Chroma, coseno y `k=4`. Medir Context Recall, Context Precision, solapamiento efectivo y texto repetido entre resultados. |
| I4 · embedder | Otro embedder puede recuperar mejor los objetivos en español. Comparar MiniLM con `text-embedding-3-small`, alternativa de la [demo de indexación][pinecone], celdas 19 y 22, si se dispone de acceso y se autoriza ese uso del corpus. | Congelar exactamente los mismos chunks y comprobar que caben en ambos modelos. Reindexar en colecciones separadas, con coseno y `k=4`; cada brazo usa su propio modelo tanto para el documento como para la consulta. Comparar precisión, recall, tiempo y costo si corresponde. |

Ejecutar un contraste por vez y usar la variante seleccionada como control del siguiente. Si un problema de recall también aparece al leer la fuente original, anotarlo como límite de corpus. Reservar un eventual contraste **mismos VTT vs. VTT + PDF docente** para ese diagnóstico, con anotaciones ampliadas y congeladas para ambos brazos; sin reescribir las fuentes en función de un video.

## 2. Retrieval: medir qué evidencia llega

**Entrada:** objetivo textual y colección académica de la materia. **Salida:** lista ordenada con IDs, distancias y referencias. La baseline **R0** envía el objetivo literal, calcula su embedding, busca por coseno y devuelve hasta **4** resultados. El VTT del video no interviene en la consulta: queremos recuperar incluso lo que el video omite. Al construir R0, comprobar que consulta la colección correcta, usa el mismo embedder que el índice y devuelve referencias verificables en orden. La [clase de RAG][rag] trata la búsqueda en pp. 37–46.

| ID | Hipótesis y cambio único | Controles y criterio |
| --- | --- | --- |
| R1 · profundidad `k` | Comparar `k=2, 4, 8`. Más candidatos pueden aportar criterios que faltaban, con más ruido o repetición. | Tras I1–I4, congelar el índice elegido, la consulta, el embedder y la distancia. Tomar los prefijos de un mismo ranking de hasta 8 para medir el efecto de `k`. Elegir el menor `k` que conserve el mejor recall con el presupuesto previsto; revisar precisión y los criterios concretos ganados o perdidos. |
| R2 · reranking, después del RAG básico | Si la evidencia aparece en posiciones bajas, comparar el orden por coseno frente al CrossEncoder de la [demo del reporte ONU][onu], celdas 58–61. | Misma lista inicial de hasta 20 candidatos (o el tamaño de la colección si es menor), `k` final elegido, generador, prompt y techo de contexto. Medir recall de candidatos, luego métricas del top final, latencia y calidad del informe. Un reranker solo puede reordenar evidencia que recibió. |
| R3 · técnica alternativa, si el fallo lo justifica | Si los candidatos excluyen sistemáticamente evidencia localizada en el corpus, evaluar expansión por subtemas de la misma [demo][onu], celdas 42–57. | Comparar consulta directa y expansión con presupuesto total de candidatos y contexto final fijos. Medir qué nuevos criterios aparecen y cuánto ruido agrega. Probarla por separado del reranker; alcanza con una técnica avanzada para el ejercicio 2. |

**Métricas** ([clase de métricas][metricas], pp. 9–15, 23–25). Un chunk relevante aporta respaldo sustantivo a un criterio anotado; una mención superficial no alcanza. Revisar el conjunto combinado de rankings sin ver el nombre de la variante, con la misma pauta por objetivo.

- **Context Recall:** criterios académicos anotados y respaldados por el conjunto recuperado / criterios que tienen evidencia en el corpus. Contar cada criterio una vez; si necesita dos pasajes, exigir ambos cuando la pauta lo indique. Marcar como `N/A` un objetivo sin criterios respaldables; evaluar aparte las abstenciones de esos casos.
- **Context Precision:** promedio de la precisión acumulada en las posiciones relevantes: `sum(Precision@i * relevancia_i) / sum(relevancia_i)`. Vale 0 en una consulta evaluable sin aciertos. Como diagnóstico, guardar también **Precision@k**: relevantes / resultados devueltos, el número efectivo de resultados y duplicación. Revisar si los chunks repetidos aportan un criterio nuevo.
- Calcular las métricas sobre el ranking y volver a calcular el recall sobre los fragmentos que entren efectivamente al modelo. Contar los tokens enviados con el tokenizer de generación; si cambia la longitud de los chunks, contrastar finalistas con el mismo techo de tokens para distinguir recuperación de longitud de contexto.

## 3. Generator: convertir las dos fuentes en un informe

**Entrada:** objetivo, referencias académicas recuperadas y VTT fijo del video, en bloques identificables. **Salida:** informe con estado y evidencia por criterio, recomendaciones con fuente y conclusión. La baseline **G0** usa una sola llamada, prompt breve directo sin ejemplos, y un modelo instruct local de la [demo de clase][asistente] (`Qwen2.5-1.5B-Instruct`, celdas 11 y 22), candidato inicial para evitar depender del proxy. Fijar revisión, decodificación sin muestreo cuando aplique, límites y plantilla. Primero usar pasajes académicos seleccionados a mano como control diagnóstico; después conectar la salida real de R0/R1. El primer control permite observar fallos del generador aun con evidencia suficiente. Sus informes son automáticos con entrada manual y se registran por separado.

| ID | Hipótesis y cambio único | Controles y criterio |
| --- | --- | --- |
| G1 · modelo con evidencia manual | Comparar Qwen con GPT-4.1 usando siete chunks completos elegidos por ID de la colección `openai_i4`. | Fijar objetivo, orden de los chunks, VTT, prompt Markdown y máximo de salida. Comparar cobertura, estados, fidelidad, respaldo de citas y completitud. Registrar las diferencias de decodificación por proveedor. [Protocolo y selección](ejercicio2/experimentos.md#g1--comparación-de-modelos-con-chunks-manuales) |
| G1.5 · prompt con evidencia manual | Con GPT‑4.1 y la entrada de G1, exigir que cada fila explicite qué establece la clase y qué muestra el video antes de asignar el estado. Se ejecutó después de G2. | Adoptar el prompt si mejora los estados y preserva el respaldo de las citas. El contraste mantuvo 2/5 estados y bajó la proporción de citas con respaldo, por lo que se conserva el prompt original. [Resultados](ejercicio2/experimentos.md#g15--prompt-con-cotejo-explícito-de-las-fuentes) |
| G2 · evidencia recuperada | Comparar el informe manual de GPT‑4.1, elegido en G1, con el que genere usando los siete chunks devueltos por búsqueda sobre el mismo objetivo literal. | Fijar la revisión del modelo, la colección, el prompt, el video y la cantidad de chunks. Registrar los tokens y revisar el respaldo académico de cada criterio en los contextos enviados. |
| G3 · Advanced RAG con expansión | Comparar G2 con una expansión de tres subconsultas, cinco resultados por consulta y fusión por turnos. | Fijar el corpus, el objetivo, el límite de siete chunks y la configuración del informe. Medir evidencia en los candidatos, en el top 7 y en el informe. Se conserva G2 como baseline. [Resultados](ejercicio2/experimentos.md#g3--advanced-rag-con-expansión-de-consulta) |
| G4 · Expansión con reranking | Reordenar los mismos candidatos de G3 con el CrossEncoder de la cátedra y pasar los siete primeros al generador. | Comparar con G3 para medir el cambio de orden y con G2 como control básico. El piloto obtuvo 3/5 de Context Recall y 3/5 estados correctos. [Resultados](ejercicio2/experimentos.md#g4--expansión-con-reranking-de-los-candidatos-de-g3) |
| Formato, opcional | Si la salida en prosa pierde campos, comparar la misma pauta en prosa y tabla por criterio. | Fijar modelo, entradas y contenido exigido. Medir informes completos y validez de citas, junto con fidelidad y relevancia. |

**Métricas** ([clase][metricas], pp. 16–24). Evaluar contra las referencias humanas, sin conocer la variante cuando sea posible:

- **Coincidencia de cobertura:** estados correctos / criterios esperados; criterio omitido cuenta como error. Mostrar matriz de confusión con los cuatro estados y una columna de omitidos, y registrar criterios inventados aparte.
- **Answer Faithfulness:** descomponer el informe en afirmaciones verificables; medir respaldadas / emitidas. Una afirmación sobre lo enseñado por la materia se coteja con el contexto académico enviado; una del video, con su VTT. Informar ambas tasas y la validez de citas (localizador existente y respaldo real / citas emitidas). Si no hay afirmaciones o citas, registrar `N/A` y medir aparte si la omisión fue correcta.
- **Relevancia del informe:** adaptación humana de Answer Relevance. Puntuar 0–2 cada eje: responde al objetivo y nivel; recomendaciones pertinentes o abstención apropiada; conclusión útil y sin digresiones. 0 = ausente o desviado, 1 = parcial, 2 = adecuado; reportar `suma/6` y desglose. Si se quiere replicar la fórmula de la clase, dejar como medida separada el coseno promedio entre el embedding del objetivo y las preguntas reconstruidas a partir del informe por un mismo juez y embedder fijos. RAGAS es opcional.
- **Integridad de salida:** informes completos / intentos, más errores de infraestructura, truncamientos y latencia. Validar que los IDs de las citas correspondan a las entradas; registrar la salida cruda y el informe final.

## 4. Integración, versión del ejercicio 2 y comparación del ejercicio 4

1. **Orden de trabajo:** cerrar insumos y pauta → construir I0 → verificar R0 → correr I1–I4 y R1 en desarrollo → verificar G0 con referencias manuales y luego recuperadas → comparar G1/G2 → congelar el RAG básico integrado → probar R2 o R3 → fijar la versión del ejercicio 2. Terminar cada etapa cuando sus controles pasan y los fallos observados quedan explicados. Guardar también resultados desfavorables.
2. **Selección:** antes de cada contraste, declarar objetivo, control, una variable, métrica primaria y guardas. Para indexación y búsqueda, priorizar Context Recall, revisar Context Precision y presupuesto. Para generación, priorizar coincidencia de cobertura y exigir que fidelidad y citas no empeoren para promover una variante. Revisar diferencias caso por caso: una ventaja de un solo caso pide replicación y, ante empate o evidencia débil, conservar la opción simple. Si la variante avanzada no mejora, reportarla y conservar el resultado básico como referencia explícita; registrar cuál versión se presenta y por qué.
3. **Prueba reservada en el ejercicio 4:** cerrar las decisiones de desarrollo primero. Evaluar las configuraciones que se quieran comparar con objetivos y videos nuevos, usando las mismas entradas, pauta y presupuesto. Publicar métricas por caso, `n/N`, promedio por caso, fallos y límites de la muestra. Un error de ejecución queda visible, con su impacto en la tasa de éxito; no se convierte en un cero semántico ni desaparece del denominador de intentos.
4. **Ejercicio 4:** preservar la versión elegida del ejercicio 2 con su código, configuración, corpus, casos y resultados. A partir de una limitación observada en desarrollo, formular una mejora de una sola variable y repetir la comparación contra esa versión. Para evaluar el cambio sin ajustar sobre resultados ya abiertos, preparar y congelar casos confirmatorios nuevos y correr ambas versiones sobre ellos. Presentar antes/después, delta, ejemplos ganados y perdidos, costo y limitaciones ([consigna], p. 3).

Por cada contraste guardar una ficha Markdown y resultados por caso en CSV o JSON en una carpeta propia. Ficha mínima: **ID/fecha | pregunta e hipótesis | control y variante | constantes y presupuesto | criterio de decisión | resultados y diferencias | errores y casos representativos | decisión y siguiente paso**. Adjuntar revisión del código (hash y cambios locales si los hay), comando, versiones de entorno/modelo/tokenizer, hashes de corpus y casos, parámetros, semillas cuando correspondan, rankings con puntajes, el contexto exacto enviado, salida cruda, etiquetas de revisión, tiempos y tokens. Conservar casos con fallo y anotar por qué faltan métricas. Llevar la decisión y su evidencia a la [bitácora][bitacora].

**Integración verificada.** El flujo de objetivo y audio a informe Markdown usa G4 como modo avanzado y conserva G2 como modo básico. Se probó desde el MP3 del Caso 01. La validación con otros objetivos queda para el ejercicio 4. [Comando y artefactos](ejercicio2/README.md)

La siguiente iteración contrasta el corpus de VTT con versiones que agregan apuntes y diapositivas. El primer piloto incorpora los tres apuntes disponibles y usa el [Caso 02 de Transformers](ejercicio4/evaluacion/caso_02_transformers.md), con audio transcripto y pauta preparada antes de la búsqueda. [Plan del ejercicio 4](roadmap_ejercicio4.md)

[ej1]: ejercicio1/Grupo03-ej1-TP3.md
[archivo]: recorrido_experimental.md
[bitacora]: ejercicio2/experimentos.md
[guía]: ../docs/RAG/GUIA_TP3.md
[consigna]: ../docs/RAG/tp3.pdf
[consigna-alt]: ../docs/RAG/DeepLearningTP2.pdf
[tp-vtt]: ../docs/RAG/tp3.VTT
[rag]: ../docs/RAG/Clase%207.1%20-%20RAG.pdf
[pinecone]: ../docs/RAG/Langchain_Pinecone.ipynb
[asistente]: ../docs/RAG/Langchain_Virtual_Assitant.ipynb
[onu]: ../docs/RAG/UN_Report_Advanced_RAG.ipynb
[metricas]: ../docs/RAG/Clase%2011%20-%20M%C3%A9tricas%20de%20la%20Industria%20(1).pdf
