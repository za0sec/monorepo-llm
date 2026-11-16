# TP3 · Estado del RAG para el grupo

El RAG **recibe un objetivo de estudio y el MP3 de un video**. Devuelve un informe en español con una tabla de cobertura, citas por página o tiempo, material de clase para estudiar y una conclusión sobre la utilidad del video. La aplicación fija el nivel introductorio. Los pasajes de clase aportan la referencia académica y el VTT del video aporta evidencia de lo que se explicó. Las pautas manuales de los seis casos se usaron para corregir los informes y quedaron fuera de los prompts.

El flujo con el corpus completo quedó **verificado de punta a punta** con el MP3 de Tokenización. La configuración de entrega usa expansión de consulta, diez resultados por consulta, reranking y `k=7`. [Prueba de cierre](ejercicio4/evaluacion/cierre_caso_04_mp3.md)

La tabla del informe distingue «Cubierto», «Parcial», «Sin evidencia en el video» y «Respaldo de clase insuficiente». El último estado permite reconocer cuándo falta material académico para juzgar el video.

## Modelos y herramientas

| Etapa | Configuración final |
| --- | --- |
| Transcripción del MP3 | **`whisper-1`** por la API directa de OpenAI, en español y con temperatura 0. Devuelve un VTT con marcas de tiempo. |
| Lectura de fuentes | Lector de VTT propio para conservar texto, bloques y tiempos. **`pypdf`** extrae el texto seleccionable de apuntes y diapositivas. |
| Contenido visual de las diapositivas | **`gpt-4.1-2025-04-14`** describe la página completa renderizada, con temperatura 0 y detalle alto. La descripción y el texto seleccionable se indexan en fragmentos separados. Las descripciones se guardan en caché. |
| División en fragmentos | **`RecursiveCharacterTextSplitter`** de LangChain, con hasta **160 tokens** y **10 de solapamiento configurado**. El tokenizer de **`sentence-transformers/all-MiniLM-L6-v2`** mide la longitud. |
| Embeddings de documentos y consultas | **`text-embedding-3-small`** de OpenAI, con vectores de **1.536 dimensiones**. **Chroma** los almacena de forma persistente y busca por distancia **coseno**. |
| Expansión y búsqueda | **`gpt-4.1-2025-04-14`**, temperatura 0 y hasta 300 tokens de salida. Genera tres subconsultas a partir del objetivo. Chroma devuelve diez resultados para cada una de las cuatro consultas, incluida la original. Los resultados se intercalan y se deduplican por ID. |
| Reranking | **`cross-encoder/ms-marco-MiniLM-L6-v2`**, local en CPU, revisión `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Puntúa cada candidato con el objetivo original y ordena de mayor a menor. El generador recibe los primeros siete. |
| Generación del informe | **`gpt-4.1-2025-04-14`**, temperatura 0 y hasta 1.000 tokens de salida, con el prompt Markdown de G1. Recibe el objetivo, los fragmentos de clase y la transcripción completa del video. |
| Citas y auditoría | Código Python que asocia IDs `[C#]` y `[V#]` con sus páginas o intervalos. Si algún ID está ausente en la entrada, el informe muestra **«Requiere revisión»**. También se guardan el texto crudo, los mensajes exactos y los datos de la corrida. |

Las comparaciones de desarrollo usaron MiniLM como modelo de embeddings, Wav2Vec2 (`jonatasgrosman/wav2vec2-large-xlsr-53-spanish`), Qwen (`Qwen/Qwen2.5-1.5B-Instruct`) y Apple Vision OCR. El último ajuste del prompt obtuvo **4/6** estados correctos frente a **5/6** del original en Razonamiento y AI Safety. Se conservó G1 y se incorporó el aviso de citas inexistentes. [Bitácora](ejercicio2/experimentos.md) · [Prompt de G1](ejercicio2/g1/prompt.md) · [Piloto visual](ejercicio4/evaluacion/piloto_imagenes.md) · [Comparación de prompts](ejercicio4/evaluacion/ajuste_prompt_informe.md)

- **Fragmentación.** Con 160 y 224 tokens se recuperaron por completo los mismos tres criterios en I2. Con 160 se enviaron menos tokens por consulta. El solapamiento de 10 conservó la cobertura y logró la mayor Context Precision de I3. [Experimentos de indexación](ejercicio2/experimentos.md#i2--tamaño-de-chunk)
- **Embeddings.** En I4, con los mismos fragmentos, consultas y `k=4`, `text-embedding-3-small` pasó de **3/14 a 7/14** criterios completos frente a MiniLM como embedder. [Comparación de embeddings](ejercicio2/experimentos.md#i4--modelo-de-embeddings)
- **Diapositivas.** La descripción de página completa llevó el fragmento con la fórmula del coseno del puesto 11 al 8 respecto del recorte. Siguió fuera del top 7 en ese piloto. La representación final combina el texto seleccionable y la descripción de la página completa. [Comparación de formatos](ejercicio4/evaluacion/piloto_imagenes.md#comparación-de-recorte-y-diapositiva-completa)

## Corpus académico

La colección **`openai_e4_completo`** reúne **1.484 fragmentos** de las clases seleccionadas anteriores a RAG.

| Fuente | Contenido | Fragmentos |
| --- | --- | ---: |
| 6 VTT docentes | Transformers, Embeddings 1 y 2, Fine-tuning, Modelos de razonamiento y AI Safety | 1.142 |
| 5 apuntes PDF de ejemplo | Atención y codificación posicional, encoder, decoder, similitud coseno y magnitud, feature extraction y fine-tuning | 21 |
| 5 PDF de diapositivas | 105 páginas seleccionadas de Transformers (25), Embeddings (24), Fine-tuning (22), Razonamiento y producción (18) y AI Safety (16) | 321 |
| **Total** | Los fragmentos de diapositivas se reparten en 111 de texto seleccionable y 210 de descripción visual. | **1.484** |

Cada fragmento conserva su fuente y su página o intervalo. En diapositivas se cita la numeración del PDF original gracias al manifiesto. La transcripción de cada video evaluado entra por separado al generador. [Inventario y carga](roadmap_ejercicio4.md#1-seleccionar-las-fuentes) · [Manifiesto de diapositivas](ejercicio4/slides/manifiesto.json)

## Resultados de evaluación

### Corpus completo con `k=7`

**Context Recall** cuenta los criterios que el contexto de clase respalda por completo. **Context Precision** mide qué tan arriba quedaron los fragmentos relevantes, incluso los que aportan evidencia parcial. **Estados correctos** compara las etiquetas del informe sobre el video con la pauta manual.

| Caso | Context Recall | Context Precision | Estados correctos |
| --- | ---: | ---: | ---: |
| 01 · Embeddings | 3/5 | 100 % | 3/5 |
| 02 · Transformers | 0/3 | 100 % | 2/3 |
| 03 · Transfer learning | 2/3 | 81,8 % | 2/3 |
| 04 · Tokenización | 3/3 | 100 % | 3/3 |
| 05 · Modelos de razonamiento | 0/3 | 80,4 % | 3/3 |
| 06 · AI Safety | 3/3 | 94,3 % | 2/3 |
| **Seis casos** | **11/20 (55 %)** | **92,7 % promedio** | **15/20 (75 %)** |

Los cuatro casos nuevos (03–06) sumaron **8/12** de Context Recall, **89,1 %** de Context Precision y **10/12** estados correctos. En Transformers, un fragmento parcialmente relevante en el primer puesto alcanzó para que Context Precision diera 100 %, mientras los tres criterios quedaron sin respaldo completo. [Rankings y corrección por caso](ejercicio4/evaluacion/k_corpus_completo_seis_casos.md)

| Comparación en seis casos | `k=7` | `k=8` | `k=10` | `k=12` |
| --- | ---: | ---: | ---: | ---: |
| Context Recall | 11/20 | 11/20 | 12/20 | 14/20 |
| Context Precision promedio | 92,7 % | 90,4 % | 81,3 % | 74,3 % |
| Estados correctos | 15/20 | 15/20 | 14/20 | 14/20 |
| Tokens de entrada por informe, promedio | 3.391 | 3.475 | 3.693 | 3.879 |

Se eligió **`k=7`** por empatar con 8 en estados correctos y cobertura académica completa con menos tokens. El corte 12 recuperó tres criterios más y acertó un estado menos. La elección se apoya en seis objetivos con una corrida por corte.

### Generación y audio

| Métrica | Resultado | Alcance |
| --- | --- | --- |
| Answer Faithfulness | **14/15** afirmaciones descriptivas respaldadas | Piloto G4 de Embeddings del ejercicio 2, con corpus de VTT. |
| Answer Relevance | **4/6** en la rúbrica manual de foco, recomendaciones y conclusión | Mismo piloto. |
| Citas con respaldo | **29/42** apariciones de citas | Mismo piloto. Las 42 referencias eran localizables y 29 aportaron respaldo a su fila o recomendación. |
| WER y CER de Whisper | **5,87 % WER** y **3,92 % CER** | Seis recortes WAV de Embeddings cotejados con referencias humanas. |
| WER y CER de Wav2Vec2 | **14,09 % WER** y **3,07 % CER** | Mismos recortes y referencias. |

Las cifras de Faithfulness y Answer Relevance corresponden al piloto G4. La evaluación de los seis casos con el corpus completo cubre las métricas de recuperación y el acierto de estados. [Evaluación del generador en G4](ejercicio2/experimentos.md#g4--expansión-con-reranking-de-los-candidatos-de-g3) · [Evaluación de audio](ejercicio2/transcripcion/README.md#comparación-sobre-los-mismos-seis-wav)

### Prueba final desde el MP3

El comando transcribió Tokenización en **55 segmentos**, reunió **24 candidatos únicos**, seleccionó siete y produjo un informe con todas las citas localizables. Acertó **2/3 estados**. Reconoció la segmentación y detectó que el video omite la asignación de IDs. Marcó embeddings como «Parcial», aunque la pauta lo considera «Cubierto». La prueba controlada de la tabla anterior había acertado 3/3 con los mismos mensajes enviados al generador. Esa diferencia quedó registrada como variación de la generación. [Informe de la prueba](ejercicio4/data/experimentos/cierre_caso_04_mp3/gpt41/informe.md) · [Auditoría del cierre](ejercicio4/evaluacion/cierre_caso_04_mp3.md)

El informe sirve como orientación de estudio. Con el top 7 quedaron **9 de 20 criterios sin respaldo académico completo**. También se observaron etiquetas inconsistentes y citas localizables que aportaban poco respaldo. El aviso comprueba que los IDs existan. La pertinencia se revisa al contrastar la cita con su fila.

## Cómo correrlo

Para usar la interfaz local, ejecutá `.venv/bin/python tp3/ejercicio4/ui_local.py` y abrí [http://127.0.0.1:8765](http://127.0.0.1:8765). Escribí el objetivo, seleccioná un MP3 de hasta 25 MB y presioná «Evaluar video». Podés leer el informe en la página o descargarlo como Markdown. La pestaña «Archivos del corpus» lista las transcripciones, los apuntes y las diapositivas. Cada cita de clase abre su archivo en otra pestaña. Los PDF abren en la página citada y los VTT conservan sus marcas de tiempo. La interfaz usa el corpus completo y el mismo evaluador que el comando. Los archivos temporales se eliminan al terminar. Whisper, los embeddings y GPT‑4.1 siguen usando la API de OpenAI.

Desde la raíz del repo, con `.venv`, la clave `OPENAI_API_KEY` en `tp3/.env`, la colección de Chroma, el manifiesto local y el snapshot del reranker disponibles.

```bash
.venv/bin/python tp3/ejercicio2/evaluar_video.py \
  'Quiero entender cómo pasa un texto de ser una cadena de caracteres a tokens, IDs y embeddings, y qué representa cada paso' \
  tp3/ejercicio4/data/casos/tokenizacion/audio.mp3 \
  --corpus completo --salida tp3/ejercicio4/data/experimentos/demo_tokenizacion
```

El directorio de salida debe ser nuevo. El comando imprime la ruta de `gpt41/informe.md`. `--corpus completo` activa la configuración final. La opción predeterminada usa los VTT del ejercicio 2 en `openai_i4`. Los MP3, índices, cachés y artefactos de ejecución se guardan fuera de Git en `data/`. En otro checkout hay que copiar los insumos locales o reconstruir el índice. [Instrucciones y dependencias](ejercicio2/README.md) · [Reproducción de la carga](roadmap_ejercicio4.md)
