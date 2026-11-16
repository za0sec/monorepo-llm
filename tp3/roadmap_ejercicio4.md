# TP3 · Plan del ejercicio 4

## Objetivo de la iteración

Completar el corpus académico de las clases previas a RAG con **VTT, apuntes de ejemplo y diapositivas docentes**. En el Caso 01, G4 llevó al generador respaldo completo para 3 de 5 criterios. Algunos faltantes provenían de la selección de candidatos y el generador también omitió un criterio con evidencia disponible. Sobre el corpus completo se compararán configuraciones de recuperación y se medirá si la evidencia adicional llega al informe. [G4](ejercicio2/experimentos.md#g4--expansión-con-reranking-de-los-candidatos-de-g3) · [Consigna, p. 3](../docs/RAG/tp3.pdf)

## 1. Seleccionar las fuentes

| Fuente | Disponible | Uso previsto |
| --- | --- | --- |
| Apuntes de prueba | [Apuntes de Transformers](ejercicio4/apuntes/apuntes_transformers.pdf), dos páginas elaboradas a partir de la clase. | La página 1 trata atención y la página 2, codificación posicional y máscara causal. |
| Apunte sobre el encoder | [El bloque del encoder](ejercicio4/apuntes/bloque_del_encoder.pdf), una página. | Autoatención, red feed-forward y Add & Norm. Fuente docente, pp. 22 y 36–37. |
| Apunte sobre el decodificador | [Del decodificador al próximo token](ejercicio4/apuntes/decodificador_y_prediccion.pdf), una página. | Atención cruzada, logits, softmax y generación autorregresiva. Fuente docente, pp. 22 y 40–44. |
| Apunte de Embeddings | [Similitud coseno y magnitud](ejercicio4/apuntes/similitud_coseno_y_magnitud.pdf), una página. | Ejemplo numérico de producto escalar y similitud coseno al escalar un vector. Indexado en `openai_e4_texto`. |
| Apunte de Fine-tuning | [Feature extraction y fine-tuning](ejercicio4/apuntes/feature_extraction_y_fine_tuning.pdf), una página. | La misma tarea de clasificación con encoder congelado o ajustando sus últimas capas. Indexado en `openai_e4_texto`. |

### PDF originales de las diapositivas

| Selección del ejercicio 4 | PDF docente original en `docs/` | Qué se conservó |
| --- | --- | --- |
| [Clase 1 · Transformers](ejercicio4/slides/clase_01_transformers.pdf), 25 de 70 páginas. | [Clase 1 - Introducción a Transformers.pdf](<../docs/Clase 1 - Introducción a Transformers.pdf>) | Arquitectura, posición, atención, encoder, decoder y entrenamiento. |
| [Clase 2 · Embeddings](ejercicio4/slides/clase_02_embeddings.pdf), 24 de 65 páginas. | [Clase 2 - Transformación de Texto a Embeddings(1).pdf](<../docs/Clase 2 - Transformación de Texto a Embeddings(1).pdf>), edición 73.64 de 2025. | Tokenización, similitud, representaciones estáticas y contextuales, Skip-gram y BERT. |
| [Clase 3 · Fine-tuning](ejercicio4/slides/clase_03_fine_tuning.pdf), 22 de 77 páginas. | [Clase 3 - Transfer Learning & Finetuning.pdf](<../docs/Clase 3 - Transfer Learning & Finetuning.pdf>) | Transferencia, feature extraction, ajuste fino, destilación, SFT, RLHF y DPO. |
| [Clase 4 · Reasoning y producción](ejercicio4/slides/clase_04_reasoning_y_produccion.pdf), 18 de 61 páginas. | [Clase 4 - LLMs y desarrollo en producción (1).pdf](<../docs/Clase 4 - LLMs y desarrollo en producción (1).pdf>) | Escalado en inferencia, RLVR, etapas de desarrollo de un LLM y codificación posicional. |
| [Clase 6 · AI Safety](ejercicio4/slides/clase_06_ai_safety.pdf), 16 de 68 páginas. | [Clase 6 - AI Safety (1).pdf](<../docs/Clase 6 - AI Safety (1).pdf>), edición 73.64 de 2026. | Riesgos, alineamiento, robustez e interpretabilidad. |

### Transcripciones docentes

Los seis archivos de `ejercicio4/transcripciones/` son copias idénticas de los VTT de `docs/`. Se validaron con el lector del ejercicio 2.

| VTT | Tema | Segmentos | Estado en Chroma |
| --- | --- | ---: | --- |
| [transformers.VTT](ejercicio4/transcripciones/transformers.VTT) | Transformers | 754 | Indexado en I4 |
| [embeddings_2.VTT](ejercicio4/transcripciones/embeddings_2.VTT) | Embeddings | 415 | Indexado en I4 |
| [finetuning.VTT](ejercicio4/transcripciones/finetuning.VTT) | Transfer Learning y Fine-tuning | 829 | Indexado en I4 |
| [embeddings_1.VTT](ejercicio4/transcripciones/embeddings_1.VTT) | Demo de Embeddings | 398 | Indexado en `openai_e4_texto` |
| [reasoningmodels.VTT](ejercicio4/transcripciones/reasoningmodels.VTT) | Modelos de razonamiento | 543 | Indexado en `openai_e4_texto` |
| [ai_safety.VTT](ejercicio4/transcripciones/ai_safety.VTT) | AI Safety | 620 | Indexado en `openai_e4_texto` |

`docs/RAG/` es material de referencia para el desarrollo del TP. `LLMs - Apuntes.pdf` queda fuera de la selección.

Los apuntes de Embeddings y Fine-tuning contienen ejemplos didácticos elaborados a partir de los VTT docentes y, para Fine-tuning, de sus diapositivas. Cada PDF se revisó visualmente y se comprobó que `pypdf` extrae su contenido.

Se prepararon 105 diapositivas de 341 páginas disponibles. La selección conserva explicaciones, fórmulas y ejemplos. Se dejaron fuera portadas, agendas, anuncios de demos, referencias sin desarrollo y repeticiones. El [manifiesto de diapositivas](ejercicio4/slides/manifiesto.json) registra los originales, las huellas y la correspondencia entre el orden de las páginas de cada selección y su posición en el PDF original. En las páginas originales 13 y 17 de Fine-tuning, `pypdf` devuelve glifos no codificables en UTF-8 junto al texto válido. La carga debe descartar esos glifos antes de tokenizar.

## 2. Incorporar PDFs con páginas citables

1. Registrar cada PDF en un manifiesto con título, tipo de fuente, ruta y SHA‑256. Mantener los VTT e índices del ejercicio 2 como referencia.
2. Extraer solo texto por página con `pypdf`, como en el notebook de la cátedra. Para diapositivas, resolver la página del PDF original con el manifiesto. Para apuntes, guardar el número de página del archivo desde 1 antes de descartar páginas sin texto. El contenido gráfico queda fuera del índice. [Notebook del reporte ONU, celda 8](../docs/RAG/UN_Report_Advanced_RAG.ipynb)
3. Aplicar el splitter recursivo de G4 dentro de cada página, con un máximo de 160 tokens medidos por MiniLM y 10 de solapamiento. La metadata de cada chunk PDF contiene `fuente`, `tipo_fuente` y `pagina`. Cada chunk tiene su propio ID y texto. Los chunks VTT conservan sus tiempos.
4. Guardar cada variante en su propia colección Chroma, con el mismo `text-embedding-3-small`, 1.536 dimensiones y distancia coseno. Verificar que los chunks de VTT coincidan entre variantes y registrar el número de chunks por tipo de fuente.
5. Resolver las referencias del generador desde los metadatos. Una cita PDF muestra el nombre del documento y su página. Una cita VTT muestra el nombre y el intervalo. Verificar con un ejemplo de cada tipo que el localizador permite encontrar la evidencia.

### Carga del apunte de Transformers

La colección `openai_e4_apuntes` se guardó en la misma base Chroma de I4 y se reabrió para verificarla. Contiene **632 chunks**, con **623 de VTT copiados de `openai_i4` y 9 del PDF**. Chroma conserva ambas colecciones. En el apunte se guardaron **4 chunks de la página 1 y 5 de la página 2**. La fórmula de atención quedó asociada a la primera página y la regla de máscara causal a la segunda. Cada chunk del PDF tiene `fuente`, `tipo_fuente` y `pagina`.

Esta primera carga usó solo el apunte de dos páginas. Sus dos borradores de una página quedaron fuera. Al trasladar el apunte se actualizó `fuente` en los nueve chunks y en el manifiesto. Los IDs, el texto, las páginas y los vectores coincidieron antes y después.

Se reutilizaron el texto, la metadata y los embeddings de los VTT. La lectura posterior a la copia de los vectores mostró diferencias de redondeo de hasta `1,49e-8` por componente. La verificación usa una tolerancia absoluta de `1e-7`. El primer intento se revirtió al detectar la diferencia con la igualdad exacta, así que se repitió la llamada de embeddings del PDF. La carga final reportó **971 tokens de entrada** para los nueve chunks con `text-embedding-3-small`. [Script de carga](ejercicio4/indexar_apuntes.py) · [Manifiesto local](ejercicio4/data/carga_apuntes.json)

Para reproducir desde la raíz con las [dependencias instaladas](ejercicio4/requirements.txt) y la colección de I4 disponible, usar un nombre y un manifiesto nuevos si la colección ya existe.

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/indexar_apuntes.py \
  --coleccion openai_e4_apuntes_repeticion \
  --salida tp3/ejercicio4/data/carga_apuntes_repeticion.json
```

### Carga de los tres apuntes

Se incorporaron los dos PDFs adicionales a `openai_e4_apuntes_ampliado`. Cada uno aportó 3 chunks de su primera página. La colección tiene **638 chunks**: 623 de VTT y 15 de los tres PDFs. Reutilizó los vectores de los VTT y los 9 del PDF original. Se enviaron a embeddings solo los 6 chunks nuevos, con 664 tokens de entrada. Al reabrir Chroma se verificaron el conteo, los IDs y la metadata de cada uno de los 15 chunks PDF. `openai_i4` y `openai_e4_apuntes` siguen disponibles para la comparación. [Manifiesto local](ejercicio4/data/carga_apuntes_ampliado.json)

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/indexar_apuntes.py \
  --desde-manifiesto tp3/ejercicio4/data/carga_apuntes.json \
  --pdf tp3/ejercicio4/apuntes/bloque_del_encoder.pdf \
  --pdf tp3/ejercicio4/apuntes/decodificador_y_prediccion.pdf \
  --coleccion openai_e4_apuntes_ampliado \
  --salida tp3/ejercicio4/data/carga_apuntes_ampliado.json
```

### Carga de los seis VTT y cinco apuntes

`openai_e4_texto` contiene **1.163 chunks**, repartidos en 1.142 de seis VTT y 21 de cinco apuntes. La carga reutilizó 638 chunks de la colección con los tres apuntes de Transformers y calculó 525 embeddings nuevos.

| Fuente nueva | Chunks |
| --- | ---: |
| `embeddings_1.VTT` | 138 |
| `reasoningmodels.VTT` | 169 |
| `ai_safety.VTT` | 212 |
| `similitud_coseno_y_magnitud.pdf` | 3 |
| `feature_extraction_y_fine_tuning.pdf` | 3 |

El [indexador](ejercicio4/indexar_apuntes.py) admite `--vtt` y usa el lector de VTT y el splitter recursivo de I4 (160 tokens de MiniLM y 10 de solapamiento). Los VTT nuevos conservan ID de bloque e intervalo y los apuntes conservan el número de página. Se usó `text-embedding-3-small` (1.536 dimensiones, distancia coseno en Chroma). OpenAI reportó 59.797 tokens de entrada en 17 lotes. Al reabrir la colección se verificaron todos los IDs, textos y metadatos, la recuperación de un chunk mediante su vector por cada fuente nueva y la coincidencia de los vectores reutilizados con tolerancia absoluta de `1e-7`. La diferencia máxima observada al guardar fue `1,49e-8`. Las diapositivas quedan para la etapa de imágenes. [Manifiesto de carga](ejercicio4/data/carga_texto.json)

Para repetir con un nombre de colección y un manifiesto nuevos:

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/indexar_apuntes.py \
  --desde-manifiesto tp3/ejercicio4/data/carga_apuntes_ampliado.json \
  --vtt tp3/ejercicio4/transcripciones/embeddings_1.VTT \
  --vtt tp3/ejercicio4/transcripciones/reasoningmodels.VTT \
  --vtt tp3/ejercicio4/transcripciones/ai_safety.VTT \
  --pdf tp3/ejercicio4/apuntes/similitud_coseno_y_magnitud.pdf \
  --pdf tp3/ejercicio4/apuntes/feature_extraction_y_fine_tuning.pdf \
  --coleccion openai_e4_texto_repeticion \
  --salida tp3/ejercicio4/data/carga_texto_repeticion.json
```

## 3. Comparación controlada

El corpus del ejercicio 4 incluirá los seis VTT, los cinco apuntes y los cinco PDF de diapositivas seleccionadas. El primer contraste previsto sobre ese corpus compara distintos valores de `k`, con `k=7` como referencia. Por objetivo se congela el audio transcripto y se usa el mismo ranking de G4, con la consulta original, sus tres subconsultas, cinco candidatos por consulta, deduplicación y el mismo CrossEncoder. Se mantienen el prompt y el snapshot de GPT‑4.1 de G4, temperatura 0 y hasta 1.000 tokens de salida. Los valores de `k` se eligen después de revisar el ranking del corpus final. [Configuración del ejercicio 2](ejercicio2/README.md)

**Casos.** Usar el Caso 01 como control de integración y el Caso 02 como diagnóstico. Sumar otro par de objetivo y video para revisar si el resultado se repite. Para cada caso, registrar el criterio, la evidencia académica y el estado esperado del video antes de ejecutar. Usar los mismos criterios y denominadores para todas las configuraciones de ese caso.

## 4. Evaluación y registro

| Nivel | Qué medir |
| --- | --- |
| Indexación | Documentos y páginas leídos, páginas con extracción deficiente, cantidad de chunks por fuente y trazabilidad de las citas. |
| Recuperación | Context Recall y Context Precision sobre los chunks finales. Registrar qué tipos de fuente entran y si aportan evidencia nueva o repetida. |
| Informe | Acierto de estados, citas localizables y con respaldo, Answer Faithfulness, relevancia manual y completitud. |

Guardar por caso las fuentes y sus hashes, los rankings, los chunks enviados, el informe y la evaluación. Para cada criterio que cambia, mostrar el pasaje que se ganó o se perdió y si el informe lo usó. Priorizar que aumente el Context Recall y revisar si los criterios adicionales se traducen en estados y citas más sólidos. Registrar el costo de indexar y la longitud real del contexto.

**Primer piloto completado.** El [Caso 02 de Transformers](ejercicio4/evaluacion/caso_02_transformers.md) comparó `openai_i4` con `openai_e4_apuntes_ampliado`, ambos con `k=7`, la misma transcripción y búsqueda sin filtro por fuente. La propuesta inicial sumaba un PDF para medir su aporte individual. La prueba sumó los tres apuntes para medir el efecto del corpus ampliado. Context Recall@7 pasó de 0/3 a 1/3 y Context Precision@7 quedó en 14,3 % en ambas variantes. [Resultados y auditoría](ejercicio4/evaluacion/resultado_caso_02_apuntes.md)

**Piloto de imágenes completado.** Se probaron texto seleccionable, OCR y descripción sobre ocho páginas con dos recortes. El Context Recall@7 quedó en 0/2 en las tres variantes. La descripción recuperó la relación general entre encoder y decoder y transcribió la fórmula de coseno en un chunk que quedó octavo. [Configuración y resultados](ejercicio4/evaluacion/piloto_imagenes.md)

**Contraste de formato visual.** Con la misma configuración para recorte y diapositiva completa, la fórmula subió del puesto 11 al 8. Ambas variantes obtuvieron Context Recall@7 de 0/2 y Context Precision@7 de 50 %. La descripción de página completa agregó el título y contexto visual, con la fórmula en un chunk separado. [Resultados del contraste](ejercicio4/evaluacion/piloto_imagenes.md#comparación-de-recorte-y-diapositiva-completa)

**Contraste de prompts de visión.** El prompt que pidió pares de origen y destino ordenó el flujo del diagrama, pero volvió a omitir la flecha al bloque específico del decoder. La fórmula mantuvo el puesto 8. Se conserva el prompt anterior para las descripciones. [Resultados por consulta](ejercicio4/evaluacion/piloto_imagenes.md#comparación-de-prompts-de-descripción)

**Siguiente paso.** Completar el corpus con las diapositivas y volver a medir la posición de la fórmula y de los demás pasajes al probar distintos valores de `k`.
