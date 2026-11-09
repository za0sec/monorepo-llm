# Plan de reconstrucción experimental del TP3

**Relevamiento del 09/11/2026.** Código en `f4dad33` y corridas locales registradas. Este plan reconstruye el pipeline experimental desde los insumos y vuelve a ejecutar las comparaciones, reutilizando los scripts y los datos disponibles. El estado actual queda congelado como referencia histórica. Las pruebas y reglas de decisión se fijan antes de medir.

El producto compara **lo que el estudiante debería aprender** con **lo que explica el audio del video**. El objetivo consulta la base académica. El VTT del video aporta evidencia al generador y se mantiene fuera del índice.

```text
Indexación  Fuente académica → carga y splitting → embeddings → índice Chroma
Consulta    Objetivo → embedding de consulta → candidatos ordenados → k fragmentos finales
Informe    Fragmentos de la materia + objetivo + VTT del video → cobertura, citas y recomendaciones
```

El orden de trabajo es **transcripción → corpus e índice de texto → recuperación básica → reranking y expansión → informe integrado de texto → imágenes**. El experimento de audio se hace por separado. El RAG recibe texto fijo mientras se evalúan sus módulos, de modo que una diferencia de transcripción no altere la comparación de embeddings, splitting o ranking. La consigna pide cambios incrementales, métricas y al menos una técnica de Advanced RAG después del pipeline básico. La expansión existente es la primera técnica a evaluar. El ejercicio 4 pide documentar una limitación y comparar un antes y un después. El calendario del ejercicio 3 difiere entre las dos consignas y queda por confirmar. [Consigna, pp. 2–4][consigna] · [Otra versión, pp. 2–4][consigna-alt] · [Explicación, 07:31–10:19][tp-vtt] · [Guía, §5.3][guia]

## 0. Congelar la referencia y preparar las corridas nuevas

1. Registrar `f4dad33`, dependencias, comandos, hashes y rutas del caso 01. Conservar el MP3, el VTT docente, el apunte PDF, la transcripción de Whisper y los resultados históricos. Los 125 chunks del VTT y los 136 de VTT más apunte son controles de reproducción.
2. Guardar cada reconstrucción bajo `tp3/avances/<módulo>/data/<run_id>/`, con salidas separadas para ASR, chunks, Chroma, rankings y generación. Reconstruir chunks e índice desde las fuentes originales. Usar `--salida` y `--base` para aislar las corridas. Verificar hashes, IDs, conteos y rankings. Documentar cualquier diferencia de entorno o entrada antes de seguir. [Comandos existentes][readme] · [Revisión inicial][revision] · [Bitácora, §13][bitacora]
3. Repetir los controles históricos al llegar a cada módulo. Eso incluye volver a transcribir la muestra, reconstruir las dos colecciones, volver a ejecutar los seis rankings de `k`, comparar de nuevo directa y expansión, y generar los informes emparejados. Los archivos previos sirven para verificar la reproducción. Si se reutiliza una salida, marcarla como histórica. Registrar cualquier corrida que no se pudo repetir y el motivo.

## 1. Comparación de transcriptores

**Control:** `openai/whisper-1` vía proxy. **Candidato:** `jonatasgrosman/wav2vec2-large-xlsr-53-spanish` ejecutado localmente. La clase 10 muestra el flujo de ASR de Hugging Face y lista Wav2Vec2. Su demo ejecuta un checkpoint de Speech2Text para inglés. El checkpoint en español propuesto para nuestro audio es una selección adicional y se documenta como tal. Confirmar versión, compatibilidad y recursos con una ventana piloto antes de comprometer la corrida completa. [Clase 10, pp. 31–38][clase10] · [Ficha de Speech2Text de la demo][s2t] · [Ficha de Wav2Vec2 en español][wav2vec]

1. **Muestra y referencia.** Del MP3 del caso 01 seleccionar seis ventanas de 20–40 segundos antes de mirar las predicciones. Incluir habla clara, pasajes rápidos y términos técnicos. Escuchar y anotar una transcripción humana literal con tiempos y vocabulario crítico. Un segundo integrante revisa las ventanas ambiguas. Si hay otro audio disponible, reservarlo para comprobar si se sostiene la diferencia.
2. **Par comparable.** Exportar una vez cada clip a mono, 16 kHz, con sus límites originales registrados. Enviar el mismo recorte a ambos modelos. Fijar español donde el modelo lo admita y registrar decodificación, equipo, modelo/revisión y tiempos. Whisper reutiliza archivos cuando existen, así que usar una carpeta nueva. El VTT histórico del audio completo queda como referencia de reproducción. Para comparar calidad por ventana, usar las dos inferencias sobre los mismos recortes. [Script de transcripción][asr] · [Ficha de Wav2Vec2 en español][wav2vec]
3. **Resultados.** Calcular WER por palabras y CER por caracteres como `(sustituciones + omisiones + inserciones) / unidades de referencia`. Normalizar Unicode NFC, minúsculas, espacios y puntuación de forma idéntica, preservar tildes y dígitos y excluir espacios de CER. Guardar original, normalizado y conteos de edición por ventana. Agregar con suma de errores sobre suma de unidades. Comparar además errores en términos que cambien una conclusión, disponibilidad y calidad de los tiempos, latencia y costo. [Clase 11, p. 28][clase11]
4. **Decisión.** La transcripción debe permitir citar evidencia temporal. Si un candidato gana en texto pero carece de tiempos adecuados, registrar ese compromiso y medirlo antes de elegirlo para el pipeline. Un resultado sobre un único video queda como piloto. Las transcripciones de clase ya disponibles se conservan como insumos textuales. Solo evaluar su ASR si se consigue el audio original y se puede anotar una referencia.

**Cierre.** Referencia humana, salidas frescas de ambos brazos y tabla de WER/CER, errores críticos y tiempos. Mientras se prepara esa prueba, se puede armar el corpus académico a partir del texto disponible. Para las comparaciones del informe, congelar un mismo VTT por video y usarlo en todos los brazos.

## 2. Construir el corpus y el índice de texto

### T0. Guardar un corpus de texto con varios temas de LLMs

Usar como corpus principal una **colección común de LLMs**, con transcripciones y apuntes de varias unidades. Así, una consulta sobre embeddings compite también con material sobre atención, entrenamiento o búsqueda semántica. La colección histórica acotada a embeddings se reconstruye para reproducir las pruebas previas y medir el efecto de ampliar la base.

| Unidad | Transcripción disponible | Apuntes o material de síntesis |
| --- | --- | --- |
| Embeddings y similitud | [Clase de embeddings][vtt-embeddings] | [Apunte derivado de embeddings, pp. 3–5][apunte-embeddings] |
| Transformers y atención | [Clase de Transformers][vtt-transformers] | [LLMs - Apuntes, Unidad 1, pp. 3–14][apuntes-transformers] |
| Transfer learning y fine-tuning | [Clase de fine-tuning][vtt-finetuning] | [Diapositivas de la clase 3, texto extraíble][clase-finetuning] |
| RAG y recuperación | [Clase de RAG][vtt-rag] | [Diapositivas de la clase 7.1, texto extraíble][clase7] |

- Guardar cada fuente antes del splitting con hash, procedencia, tema, intervención o página y texto. Distinguir transcripción, apunte y diapositivas. Si se redactan resúmenes adicionales, partir de la clase completa, guardar referencias al original y congelarlos antes de redactar las consultas reservadas. El apunte de embeddings ya existente se marca como derivado y preparado con el caso 01 a la vista. Los informes y respuestas esperadas de evaluación quedan fuera del índice. [Preparación][chunks] · [Comandos existentes][readme] · [Bitácora, §13][bitacora]
- Mantener **el mismo corpus multitema para todos los experimentos principales**. Usar el tema como metadato para auditar, sin prefiltrar por el objetivo. Reportar cantidad de documentos, chunks y texto por unidad y fuente. Registrar si una unidad domina la colección o aporta pasajes casi duplicados.
- Reutilizar E1–E5 y las consultas originales de embeddings como controles de desarrollo. Proponer seis objetivos nuevos de desarrollo y cuatro reservados, al menos uno reservado por unidad. Mezclar objetivos amplios y específicos y reservar juntas las paráfrasis. Incluir confusiones plausibles, por ejemplo pedir similitud coseno entre embeddings cuando el índice también contiene coseno de codificación posicional y producto escalar de atención. Un pasaje de otra unidad cuenta como relevante si realmente respalda la afirmación solicitada. [Caso 01][caso] · [Ablación de k][k] · [Bitácora, §§12 y 19–23][bitacora]
- Anotar por objetivo las afirmaciones y los pasajes que las sostienen antes de ejecutar las variantes. Etiquetar los chunks de la unión de los rankings por respaldo, con los IDs de los brazos ocultos. Medir Context Precision y Context Recall según §5 y reportar también cuántos de los primeros `k` son irrelevantes de otra unidad. Los objetivos de recuperación se pueden preparar con las fuentes de clase, mientras el VTT de video y su referencia quedan reservados para evaluar el informe.

**Cierre T0.** Corpus multitema congelado, colecciones históricas identificadas, inventario con hashes y objetivos reservados por unidad. Durante la indexación se usa `k=8` como presupuesto fijo. Su elección se prueba en §3.

### I1. Carga, corpus y splitting

- **Carga y trazabilidad.** Verificar que los pasajes del piloto se puedan seguir desde el VTT o el PDF hasta los chunks mediante texto, offsets y tiempos o páginas. El loader conserva el texto original procesado, incluidos errores de transcripción. Los 125 chunks del VTT ya tienen controles de integridad. Revisar específicamente si las explicaciones de coseno, magnitud y entrenamiento están presentes y si quedan partidas. Registrar una pérdida de contenido, una frontera problemática y un error de ASR como causas distintas. [Preparación][chunks] · [Revisión inicial][revision]
- **Tamaño de chunk.** Contrastar el splitter actual de 1.000 caracteres con solapamiento de 100, seguido de 200 tokens de contenido con solapamiento de 20, frente a **128 tokens**. Mantener el tokenizer, los demás parámetros, la fuente, MiniLM y la búsqueda. Preparar y cargar la variante en una ruta de salida separada con los flags existentes. Comparar continuidad, cantidad y tamaño de chunks, CP/CR y tokens de contexto. Luego fijar el tamaño elegido en desarrollo y probar **0, 20 y 40 tokens de solapamiento**. Conservar el solapamiento por caracteres en 100 en los tres brazos. Medir explicaciones que se mantienen completas, duplicación, cantidad de chunks y tokens enviados. Si el tamaño elegido cambia, verificar el par tamaño/solapamiento final contra 200/20 con los mismos casos. Así se prueban cuatro configuraciones únicas como máximo, contando el control, la variante de 128 tokens y los otros dos solapamientos del tamaño elegido. [Comandos de preparación][readme] · [Clase 7.1, pp. 20–36][clase7]
- **Efecto de ampliar el corpus.** Repetir las mismas consultas de embeddings en la colección histórica mixta y en la multitema, dejando fijos los chunks de embeddings, el splitter, el embedder, coseno y `k`. Comparar CP/CR, cambios de posición y resultados irrelevantes de otras unidades. Las dos colecciones históricas, VTT solo y VTT más apunte, permiten repetir el contraste anterior de fuentes. A partir de ahí, evaluar los cambios de arquitectura sobre la colección multitema fija. [Bitácora, §§13–15][bitacora]

**Cierre I1.** Los pasajes anotados tienen localizador verificable y queda documentado qué pierde o gana cada variante. Mantener los parámetros actuales si el contraste no aporta evidencia para cambiarlos.

### I2. Modelo de embeddings

Comparar **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`, revisión fijada, CPU, 384 dimensiones) con **`text-embedding-3-small`**, visto en clase, si el proxy habilita ese endpoint para el corpus. Incluir un brazo local multilingüe con **`intfloat/multilingual-e5-small`** si se acuerda usar un modelo adicional al material de cátedra. Sirve también como alternativa local si el endpoint no está habilitado. En ese brazo, aplicar los prefijos `query:` y `passage:` según su ficha y registrarlos como parte de la configuración del modelo. [Ficha de E5 multilingüe][e5] Usar exactamente los mismos textos e IDs de chunks, consultas, ordenamiento por coseno y `k`. Codificar documentos y consultas con el mismo modelo en cada brazo. Usar coseno exacto en ambos brazos como control para aislar el efecto del modelo. Revisar los límites de cada encoder y registrar cualquier entrada que no pueda procesarse completa. El tokenizer de splitting queda fijo durante el contraste. El código actual acopla tokenizer, manifiesto y encoder a MiniLM. Una eventual prueba del candidato requiere un adaptador aislado, sin tocar la colección base. [Configuración][config] · [Indexación][indice] · [Notebook Pinecone, celdas 19 y 22][pinecone]

**Cierre I2.** Comparar CP/CR, primeros puestos, fallos y tiempos de indexación y consulta. Registrar recursos de CPU para MiniLM y costo de API del candidato. La pregunta es si el cambio rescata evidencia en español con estas consultas. Si un candidato no está disponible, registrar la comparación pendiente y completar el otro par si es viable. Confirmar al final la configuración ganadora contra el baseline en objetivos reservados. La ejecución en CPU, el cacheado local y las dimensiones verificadas siguen siendo evidencia operativa de MiniLM.

### I3. Índice vectorial y distancia

Chroma local persiste texto, IDs, vectores y referencias. Ya se verificaron persistencia entre procesos, reanudación y reutilización de 125/125 y 136/136 chunks. El tiempo histórico de indexación inicial incluyó carga o descarga del modelo. Para el contraste, usar **los mismos vectores** en Chroma y en el ranking exacto por coseno. Medir coincidencia top-`k` con una tolerancia de empate predefinida, pérdida de afirmaciones relevantes, latencia de búsqueda con índice abierto y huella en disco. Registrar por separado carga de modelo, embedding y consulta. El coseno compara direcciones sin depender de la magnitud de cada vector y se mantiene fijo para aislar el efecto del índice. La evaluación de Chroma comprueba qué tan bien reproduce ese ranking a la escala del TP. [Revisión inicial, Integridad y operación][revision] · [Implementación][indice]

**Cierre I3.** Verificar que la colección persiste, mantiene los metadatos y conserva la evidencia relevante frente a la referencia exacta, dentro del presupuesto local acordado. Con eso se puede justificar Chroma para este corpus. Un benchmark contra Pinecone solo resolvería una pregunta distinta, como un requisito concreto de escala o servicio remoto.

## 3. Recuperación, k y técnicas avanzadas

### R1. Justificar k en la búsqueda directa

Repetir `k=2, 4, 8` primero sobre la colección histórica reconstruida con las consultas literales. Luego repetir esos cortes con el corpus, splitter y embedder elegidos en desarrollo, fijos en los tres brazos. Se eligió `k=8` provisionalmente porque el objetivo general pasó de **0/5 a 1/5 criterios respaldados** al subir de 4 a 8. En coseno, 4 y 8 respaldaron 1/2. Estos conteos históricos son por criterio completo y se preservan separados de Context Recall por afirmación. [Ablación de k][k]

Para cada valor, reportar CP, CR, P@k, afirmaciones ganadas, redundancia y tokens del contexto. Revisar si los rankings cortos son prefijos antes de recortar un resultado guardado. Escoger el menor `k` que conserve las afirmaciones esenciales que recupera el brazo de mayor presupuesto si también reduce ruido o contexto. Si el valor mayor agrega evidencia esencial, cuantificar el beneficio y el costo. La búsqueda vigente carece de umbral. Probar uno solo si aparece un fallo específico donde el ruido de consultas sin evidencia esté induciendo clasificaciones incorrectas.

**Cierre R1.** Tabla de los tres valores con una explicación por consulta del efecto del corte. `k=8` sigue como baseline histórico aunque el piloto favorezca otro valor. Confirmar el finalista con los objetivos reservados de distintas unidades y publicar resultados por unidad junto con el promedio.

### R2. Ver dónde se pierde la evidencia antes de elegir técnica avanzada

| Hallazgo con los artefactos del piloto | Módulo al que volver |
| --- | --- |
| El pasaje falta o es ilegible en la fuente | Fuente, carga o transcripción. |
| El pasaje aparece en la fuente pero los chunks fragmentan la explicación | Splitting y solapamiento. |
| El pasaje está indexado y queda fuera de los candidatos | Consulta, embedding o recuperación de candidatos. |
| El pasaje aparece entre candidatos, pero queda fuera del contexto final | Corte de `k`, mezcla de resultados o reranking. |
| El pasaje llega al informe y este lo ignora o lo cita mal | Generador, prompt o validación de citas. |

### R3. Expansión existente y reranker candidato

- **Expansión.** Comparar directa contra expansión `inicial`, con el mismo índice y `k` final. La expansión usa GPT-4.1 por defecto para generar hasta cuatro consultas, alterna puestos y deduplica por ID. Usar los planes guardados con `--plan` como control de la recuperación y generar un plan nuevo para la corrida completa de la estrategia. Guardar candidatos y finalistas por separado. El código usa el mismo `k` para candidatos por subconsulta y para el máximo final. Registrar ambos al estudiar esa estrategia. El prompt `concreto` ya perdió evidencia útil en una prueba. Su revaluación requiere un fallo que justifique volver a probarlo. [Recuperación][retrieval] · [Bitácora, §§14–15 y 19–22][bitacora] · [Notebook, celdas 47–51][advanced]
- **Reranker.** El TP todavía no lo integra. Hacer un piloto local con el CrossEncoder de la cátedra, `cross-encoder/ms-marco-MiniLM-L-6-v2`. Extraer una sola lista fija, por ejemplo `N=20` candidatos de la búsqueda directa, y comparar sus primeros `k` por coseno con sus primeros `k` tras reordenarlos con el CrossEncoder. Fijar consulta, corpus, candidatos y `k`. Medir CP/CR, latencia y cambios de posición. Usar el Context Recall calculado sobre los 20 candidatos como techo para ese pool. Si el pasaje falta en el pool, este reranker no puede recuperarlo. Registrar diferencias de truncamiento y tratar la calidad en español como hipótesis. [Clase 7.2, p. 10][clase7-avanzado] · [Notebook, celdas 58–61][advanced]

**Cierre R3.** Determinar si expansión o reranking agrega evidencia al contexto final con el mismo presupuesto. Medir cada estrategia contra la búsqueda directa antes de combinar cambios. La expansión ya disponible puede cubrir el requisito de Advanced RAG si se documenta su evaluación. El experimento de reranking se hace aunque la expansión resulte suficiente. Adoptar el reranker requiere una mejora observable. Si conviene evaluar expansión más reranking, congelar la expansión, reutilizar exactamente sus candidatos deduplicados en ambos brazos y comparar el orden previo contra el reordenado con el mismo `k`.

## 4. Generación y arquitectura integrada de texto

### G1. Separar modelo, contexto y prompt

El camino local usa Qwen2.5-1.5B-Instruct. Sus corridas de desarrollo mostraron repeticiones, omisiones y problemas de citas aun con pasajes seleccionados. El camino `--backend proxy --prompt estudiante` requiere una recuperación guardada con subtemas. GPT-4.1 es el default del **proxy**. La CLI sola todavía usa el backend local y el prompt `inicial`. El control estructural de `estudiante` revisa el formato y los IDs, mientras la auditoría de contenido verifica si la cita realmente sostiene la fila. [Generación histórica][gen] · [Generador][generador] · [Proxy][proxy]

1. **Generador GPT-4.1 vs GPT-4o.** Volver a ejecutar el par del caso 01 con los mismos artefactos de entrada que en la corrida histórica. Como control de Qwen local, repetir la comparación histórica con GPT-4o usando `tabla` y los mismos pasajes y VTT. Comparar la configuración local y la remota como sistemas de inferencia y registrar sus límites, formato y tiempos. Confirmar después con exactamente el mismo plan, ranking, texto y orden de los pasajes, VTT, prompt `estudiante`, temperatura 0 y tope de 1.000 tokens. Medir Relevance, Faithfulness, cobertura, citas, validez y costo. El caso histórico completó ambos informes pero ambos clasificaron entrenamiento con respaldo insuficiente. [Bitácora, §18][bitacora]
2. **Impacto de la estrategia en el informe.** Para directa vs expansión, usar el mismo modelo y `--prompt tabla` en ambos brazos. Así puede variar el contexto recuperado sin que un brazo requiera subtemas ni cambie de formato. Evaluar el informe de uso real `estudiante` por separado. Si importa la decisión de modelo **del planificador**, comparar planes de GPT-4.1 y GPT-4o con el mismo prompt y tope de 200 tokens, manteniendo generación fuera de ese contraste. [Comandos existentes][readme] · [Bitácora, §22][bitacora]
3. **Iteración del ejercicio 4.** En el par histórico con y sin el pasaje de Word2Vec, solo cambió `C5`. La recomendación mejoró y ambos informes mantuvieron la etiqueta «Parcial» con citas tangenciales. Si se reproduce el fallo, comparar el prompt actual con una variante que exija identificar en el campo `explicacion` qué componente esencial sustenta el pasaje académico citado antes de evaluar el audio. Conservar modelo, evidencia, VTT y esquema JSON. Medir abstenciones correctas, errores críticos y criterios resueltos. La inyección manual de un pasaje es diagnóstica y se reporta aparte de las corridas reales. [Bitácora, §23][bitacora]

### G2. Confirmar con casos separados después del piloto

Para confirmar **retrieval** usar los objetivos académicos reservados de varias unidades del corpus, con sus afirmaciones de referencia. Reportar cuántos objetivos y pasajes se evaluaron en cada unidad. Para confirmar el **informe**, sí hacen falta otros videos. Propuesta inicial para generación, tres videos nuevos de desarrollo y cinco de prueba, ajustable a tiempo y acceso. El video actual y sus reformulaciones pertenecen a desarrollo. Si solo alcanza para el caso actual, presentar resultados como piloto y reportar el alcance. Priorizar los pares finalistas en vez de repetir todas las variantes con cada video.

Para cada objetivo de prueba, fijar 3–5 criterios, el nivel esperado y los pasajes académicos antes de ejecutar el sistema. Para el informe, escuchar el audio completo, anotar `cubierto`, `parcial` o `sin_evidencia` por criterio y cotejar los intervalos con el VTT. Guardar cualquier diferencia audible frente al ASR. Revisar por separado si el contexto académico del brazo permite evaluar el criterio. Cuando falte, la conducta esperada es `respaldo_insuficiente`. Alinear las filas y los `T#` por significado y alcance, y contabilizar criterios omitidos, duplicados o contradictorios. Calibrar con dos revisores en desarrollo, hacer doble revisión de una muestra reservada y adjudicar todos los errores críticos con el modelo oculto. Mantener anotaciones y respuestas de referencia fuera de las entradas del pipeline.

### G3. Congelar y repetir el sistema completo

Armar dos versiones con el mismo conjunto reservado de videos y objetivos. Ambas usan el corpus multitema congelado. El control aplica la configuración histórica de splitter 1.000/100 caracteres y 200/20 tokens, MiniLM, Chroma con coseno, expansión `inicial`, `k=8` y GPT-4.1 con prompt `estudiante`. La otra usa el splitter, embedder, índice, `k`, estrategia de recuperación y generador seleccionados durante los contrastes. La reproducción con el corpus histórico de embeddings se reporta aparte, sobre consultas de esa unidad. Fijar la misma transcripción por video para atribuir la comparación a la arquitectura RAG. Si se quiere incluir el efecto del ASR elegido, medirlo en un par aparte que solo cambie la transcripción. Publicar las cuatro métricas, estados de cobertura, citas, tiempo, costo y fallos. La comparación integral mide el efecto conjunto. Las ablaciones por módulo explican qué cambios contribuyeron.

**Cierre de texto.** Manifiesto congelado, comparación pareada y resultados de la versión textual reproducibles desde los insumos. Si un módulo queda inconcluso, registrarlo como provisional. Este cierre es la dependencia para agregar imágenes.

## 5. Métricas y criterio de cálculo

| Métrica | Qué se calcula | Referencia y evaluador |
| --- | --- | --- |
| **Context Precision@k** | Sobre el ranking final académico, `sum(P@i × v_i) / sum(v_i)`, donde `v_i=1` si el chunk respalda sustancialmente al menos una afirmación académica pertinente y `P@i=sum(v_1…v_i)/i`. Una lista bibliográfica o mención sin explicación vale 0. Para objetivos con evidencia en el corpus, recuperar cero relevantes da 0. Reportar también `P@k` para hacer visible el ruido al final del ranking. | Objetivo, afirmaciones de referencia y etiquetas humanas por chunk. El mismo etiquetado sirve para todos los brazos. Juez LLM opcional tras calibración. [Clase 11, pp. 9–12][clase11] |
| **Context Recall@k** | Afirmaciones académicas de referencia respaldadas por la unión de los chunks efectivamente entregados / total de afirmaciones académicas del caso. Dar crédito a una afirmación compuesta por evidencia en varios chunks si juntos bastan. Contar cada afirmación una vez. Guardar también si cada criterio quedó completamente respaldado. | Referencia académica humana independiente del ranking. Anotación por pasaje o juez calibrado. Se excluye el VTT del video del cálculo. [Clase 11, pp. 13–15][clase11] |
| **Answer Relevance** | Promedio del coseno entre el embedding de la solicitud original del caso y los de tres preguntas que un LLM reconstruye desde el informe. Congelar una solicitud canónica por caso con el objetivo y la tarea de evaluar el video. Dar al juez solo el informe visible, sin repetir su encabezado con el objetivo ni pasarle las fuentes o el gold. Usar el mismo juez, prompt, número de preguntas y encoder evaluador para todos los brazos. | Solicitud original e informe. Requiere juez LLM para las preguntas inversas y embeddings fijos. No requiere respuesta de referencia. Auditar si el informe responde a la tarea completa con el indicador de criterios omitidos. [Clase 11, pp. 19–20][clase11] |
| **Answer Faithfulness** | Afirmaciones generadas respaldadas por el contexto suministrado / total de afirmaciones verificables del informe. Descomponer cobertura, justificaciones, recomendaciones y conclusión en afirmaciones. Verificar cada una contra la fuente que le corresponde. Para «Sin evidencia», exigir criterio académico suficiente y ausencia de explicación en el VTT completo enviado. Para abstenciones, verificar que el respaldo académico realmente falte y que el informe no atribuya la carencia al audio. | Informe y contexto exacto de la corrida. No requiere respuesta gold. Auditoría humana o juez calibrado. Separar el puntaje por tipo de fuente y auditar por separado si **la cita específica** sostiene la afirmación, aunque otro pasaje del contexto pudiera respaldarla. [Clase 11, pp. 16–18][clase11] |

**Métricas de producto.** Guardar una matriz de confusión de cobertura sobre los criterios evaluables, con columna de omisión y de abstención, y contar criterios correctamente resueltos sobre **todos** los criterios esperados del caso. Separar la abstención justificada cuando falta respaldo de la abstención innecesaria cuando sí lo hay. Reportar `n/N` de recomendaciones con cita sustantiva sobre un faltante real, y `n/N` de informes válidos. Contar por separado los errores críticos de asignar cobertura u omisiones al video sin respaldo académico suficiente, citar una fuente inexistente o confundir las fuentes. El validador estructural cubre parte de estos controles. Las 17 pruebas registradas al fijar los defaults comprueban integración y usan respuestas simuladas del proxy, por lo que la calidad semántica se mide con las corridas reales. [Bitácora, §§17 y 21][bitacora]

**Política de denominadores.** Si el gold académico está vacío o se diseñó un caso sin evidencia en el corpus, reportar CP/CR como no aplicables en ese control y medir abstención. Ante informe vacío, inválido o truncado sin salida utilizable, registrar fallo y no computar métricas semánticas de generación sobre una salida inexistente. Un informe válido sin afirmaciones verificables deja Faithfulness como no aplicable. Un error del juez se registra por separado. Publicar siempre los denominadores y la tasa de fallos. Calcular las diferencias semánticas en pares válidos comunes y la tasa de resolución sobre todos los casos asignados, incluidos los fallos. Agregar por caso con promedio macro, mediana y diferencias pareadas, con resultados por video y conteos de mejora, empate y retroceso.

**Herramienta.** Empezar con anotación humana para recuperación y fidelidad y una llamada por informe para Answer Relevance. Usar MiniLM fijo como encoder evaluador en esta primera etapa, independientemente del embedder de recuperación. Revisar a mano en desarrollo las preguntas inversas en español, la sensibilidad a omisiones y la longitud de entrada del encoder. Congelar un juez habilitado y su configuración antes de probar. Si el proxy solo admite los mismos modelos que se comparan, usar un juez fijo común, ciego a la variante, y cotejar la muestra humana incluyendo ambas procedencias. Registrar ese límite. RAGAS queda disponible si aporta automatización después de validar fórmulas, entradas y costos. Fijar su versión y variante si se usa. La demo usa un juez LLM y dos respuestas de ejemplo. Como control inicial, reproducir los ejemplos de clase con valor 0,5 para CP, CR y Faithfulness. [Notebook RAGAS, celdas 4–7][ragas] · [Clase 11 grabada, 23:26–24:09][metricas]

## 6. Agregar imágenes como una ablación posterior

Empezar por **fotogramas del video con texto visible** que complementen una explicación hablada, por ejemplo una fórmula escrita pero no verbalizada. Si se consigue el video original, seleccionar fotogramas con timestamps reproducibles y extraer texto con OCR, una técnica de la clase 10. El OCR produce evidencia del video, separada de las fuentes de la materia. En este primer contraste el corpus académico y el índice de texto quedan fijos. El benchmark visual incluye criterios con evidencia solo en pantalla y controles donde la imagen es redundante, ilegible o irrelevante. Si no se dispone de video o referencias visuales verificables, la etapa queda pendiente. [Clase 10, pp. 23–24][clase10]

| Brazo | Entrada al informe |
| --- | --- |
| Control textual | Objetivo, pasajes académicos y VTT del audio. |
| Variante visual | Las mismas entradas, más el OCR de los fotogramas con referencias temporales propias. |

Mantener modelo y presupuesto total, eligiendo casos en los que ambos brazos entren completos. Usar el mismo prompt de evaluación de audio y pantalla en ambos, con el bloque visual vacío en el control, y conservar el baseline de texto anterior como referencia aparte. Etiquetar qué puede sustentarse con audio, pantalla o ambos. Medir cobertura correcta de criterios visibles, atribución a la fuente correcta, citas temporales, Faithfulness y costo adicional. Los resultados de recuperación académica deberían coincidir entre brazos porque su entrada queda fija. Si las imágenes relevantes son diagramas o gráficos que OCR no representa, esa pregunta justifica evaluar más adelante otro método visto en clase. [Clase 10, pp. 13–27][clase10]

## 7. Registro y decisiones

Guardar resúmenes y protocolo en la carpeta de avances del módulo. Usar su subcarpeta `data/<run_id>/` para material local y salidas completas, con copia compartida autorizada cuando corresponda. El repo ignora `docs/` y los directorios `data/` de los módulos. Entregar un inventario de fuentes, hashes y pasos de reconstrucción para que otra persona del grupo pueda reproducir la corrida con acceso al material. Mantener credenciales y datos de acceso fuera de todos los artefactos.

| Archivo o registro futuro | Campos mínimos |
| --- | --- |
| `casos.jsonl` y `referencias.jsonl` | Versión de dataset y rúbrica, IDs y partición, objetivos, corpus y hashes de originales y VTT, criterios, afirmaciones y localizadores, gold de audio, diferencias VTT/audio y revisiones. |
| `manifiesto.json` por corrida | `run_id`, `experiment_id`, brazo, fecha, commit y estado local, comandos, Python y dependencias, hardware, hashes del corpus/chunks/VTT/plan, colección y métrica, split, modelos solicitados y devueltos, revisiones, prompts y hashes, parámetros, juez y encoder, costos esperados y regla de decisión. Registrar semilla si está disponible. |
| `recuperacion.json`, `plan.json`, `informe.json` | Reusar salidas actuales por caso y brazo. Conservar candidatos y ranking final con IDs, textos y distancias, mensajes exactos, mapa `C/V/T`, respuesta bruta y renderizada, errores, tiempos por etapa, tokens y costo informado. Registrar costos ausentes como desconocidos. |
| `asr/` e `imagenes/` | En ASR, guardar ID, hash y límites de cada recorte, referencia humana, salida de cada modelo, normalización y conteos de edición. En imágenes, guardar ID de video, timestamp, hash de fotograma, método, texto visible de referencia, salida de OCR y criterio asociado. |
| `anotaciones.jsonl`, `metricas.csv` y `resumen.md` | Anotaciones por caso, criterio, chunk o afirmación, citas y motivos, resultados del juez y adjudicación. En métricas, una fila por caso/variante con numerador, denominador y estado. En resumen, comparación pareada, fallos, costos, conclusión, alcance y decisión respaldada o pendiente. |

Por comparación, declarar antes de abrir prueba qué diferencia cambiaría la decisión. En recuperación, una regla inicial a calibrar es ganar 0,10 en CR macro con caída de CP de hasta 0,05, mejora en la mayoría de los objetivos y sin perder un criterio esencial. Si se gana contexto relevante con menos chunks, registrar el ahorro y el efecto por caso. Para generación, priorizar menos errores críticos y más criterios resueltos. Usar costo y latencia como desempate con calidad comparable. Con muestra pequeña, publicar diferencias pareadas, denominadores, retrocesos y límites. Si el candidato gana, conservar el baseline y justificar el cambio. Si los resultados son mixtos o el candidato no pudo correr, dejar la decisión provisional.

La preparación y anotación iniciales son locales. Presupuestar el ASR de los clips y estimar por separado embeddings, planificación, generación, juez y OCR. Fijar un tope tras el piloto. Después de cada nueva corrida, cachear los artefactos para repetir el cálculo sin volver a pagar inferencia. Una comparación de dos informes en cinco casos, con una llamada de relevancia por informe, suma hasta 20 llamadas más los planes que no estén guardados. Los JSON congelados permiten recalcular métricas aun si una inferencia del proxy varía.

**Entregables en orden.** Comparativa Whisper vs Wav2Vec2. Corpus multitema de texto reconstruido y tabla de trazabilidad por unidad. Ablaciones nuevas de splitting, solapamiento, embeddings, índice, `k`, expansión y reranking. Comparativa de generadores y arquitectura textual integrada. Contraste textual vs imágenes si hay casos verificables. La bitácora registra en cada cierre qué se repitió, qué cambió y qué evidencia respalda la decisión.

[consigna]: ../../docs/RAG/tp3.pdf
[consigna-alt]: ../../docs/RAG/DeepLearningTP2.pdf
[tp-vtt]: ../../docs/RAG/tp3.VTT
[clase11]: <../../docs/RAG/Clase 11 - Métricas de la Industria (1).pdf>
[metricas]: ../../docs/RAG/Metricas.VTT
[clase10]: <../../docs/RAG/Clase 10 - Aplicaciones en Imágenes y Audio.pdf>
[guia]: ../../docs/RAG/GUIA_TP3.md
[ragas]: ../../docs/RAG/RAGAS.ipynb
[pinecone]: ../../docs/RAG/Langchain_Pinecone.ipynb
[advanced]: ../../docs/RAG/UN_Report_Advanced_RAG.ipynb
[chunks]: ../ejercicio2/preparar_chunks.py
[config]: ../ejercicio2/config.py
[indice]: ../ejercicio2/indexar_material.py
[retrieval]: ../ejercicio2/recuperar.py
[asr]: ../ejercicio2/transcribir_audio.py
[generador]: ../ejercicio2/generar_informe.py
[proxy]: ../ejercicio2/proxy_catedra.py
[readme]: ../ejercicio2/README.md
[bitacora]: ../presentación.md
[caso]: ../ejercicio2/evaluacion/caso_01_embeddings.md
[revision]: ../ejercicio2/evaluacion/revision_retrieval_inicial.md
[k]: ../ejercicio2/evaluacion/ablacion_k.md
[gen]: ../ejercicio2/evaluacion/generacion_caso01.md
[clase7]: <../../docs/RAG/Clase 7.1 - RAG.pdf>
[clase7-avanzado]: <../../docs/RAG/Clase 7.2 - Advanced RAG(1).pdf>
[wav2vec]: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish
[s2t]: https://huggingface.co/facebook/s2t-small-librispeech-asr
[e5]: https://huggingface.co/intfloat/multilingual-e5-small
[vtt-embeddings]: ../../docs/embeddings_2.VTT
[apunte-embeddings]: <../../output/pdf/LLMs - Apuntes - Embeddings.pdf>
[vtt-transformers]: ../../docs/transformers.VTT
[apuntes-transformers]: <../../docs/LLMs - Apuntes.pdf>
[vtt-finetuning]: ../../docs/finetuning.VTT
[clase-finetuning]: <../../docs/Clase 3 - Transfer Learning & Finetuning.pdf>
[vtt-rag]: ../../docs/RAG/RAG.VTT
