# Diseño inicial del RAG y plan de experimentos

La base vectorial elegida es **Pinecone**, por decisión del grupo para seguir la
implementación explicada en clase y disponible en
[Langchain_Pinecone.ipynb](../../docs/RAG/Langchain_Pinecone.ipynb).
El resto del diseño es una propuesta para revisar antes de implementar. Sus
parámetros son hipótesis iniciales; todavía no se midió su calidad ni su tiempo
de ejecución.
Se mantiene el alcance gradual: un VTT de clase, un audio ya transcripto y un
objetivo de estudio. PDFs e imágenes se incorporarán en etapas posteriores.

## Qué encontramos en las clases

Se buscaron referencias a modelos pequeños, costo, tokens y crecimiento de la
aplicación en los VTT disponibles en `docs/`. No encontramos una indicación
literal de empezar con un LLM pequeño e ir aumentando su tamaño para consumir
menos tokens. Sí encontramos estas recomendaciones relacionadas:

| Fuente | Ubicación | Evidencia y alcance |
| --- | --- | --- |
| [RAG Implementación](../../docs/RAG/RAG%20Implementacio%CC%81n.VTT) | 12:15.060-12:28.590, entradas 121-122 | Eugenia propone una "demo más chiquita" e ir escalando. Se refiere al desarrollo de la aplicación. |
| Misma clase | 18:58.840-19:43.870, entradas 181-188 | Discute chunks demasiado pequeños, que pierden relaciones, y demasiado grandes, que agregan contexto irrelevante y ocupan la ventana. |
| Misma clase | 23:53.820-25:38.099, entradas 229-243 | Compara dimensiones de embeddings, capacidad de representación y costo computacional; explica que una representación pequeña puede alcanzar según el caso. |
| Misma clase | 46:34.560-47:16.730, entradas 431-435 | Limita la generación a 128 tokens para la demo y advierte que los modelos pequeños pueden requerir más restricciones. |
| Misma clase | 48:13.080-48:34.649, entradas 444-446 | Usa top-2 y aclara que `k` debe probarse experimentalmente. El valor pequeño también busca mostrar fallos. |
| Misma clase | 60:26.030-61:11.220, entradas 561-567 | Se discute que el mensaje y el contexto enviado consumen tokens. |
| Misma clase | 71:04.520-71:59.510, entradas 646-651 | Explica la división por caracteres y el posterior splitting por tokens para controlar los chunks que se pasan de longitud. |
| [Consigna explicada](../../docs/RAG/tp3.VTT) | 07:31.150-08:06.960 y 08:52.160-09:45.160, entradas 65-69 y 78-87 | Pide justificar los módulos, comparar cambios incrementales y terminar el pipeline básico antes de Advanced RAG y extras. |
| [Métricas](../../docs/RAG/Metricas.VTT) | 32:36.490-33:12.819, entradas 292-295 | Pide considerar tokens, llamadas a API e infraestructura además de la calidad. |

**Distinción para nuestro diseño:** tamaño del modelo, dimensión del embedding
y cantidad de tokens son variables diferentes. Un generador pequeño puede tener
menor costo por token, pero no garantiza consumir menos tokens para la misma
tarea. Un embedding de 384 dimensiones tampoco significa 384 tokens. Para
controlar consumo debemos medir el contexto enviado, la salida y las llamadas.

## Qué ofrecen los notebooks

Las celdas se numeran desde 1, contando todas las celdas del archivo.

| Notebook | Implementación observada | Qué tomamos |
| --- | --- | --- |
| [Langchain_Pinecone](../../docs/RAG/Langchain_Pinecone.ipynb) | Celda 15: splitter recursivo, 1000 caracteres y 200 de overlap. Celda 19: `all-MiniLM-L6-v2`. Celda 22: `text-embedding-3-small`, con `large` mencionado como alternativa. Celda 26: índice Pinecone de dimensión configurable y métrica coseno. | Referencia principal para conectar MiniLM con Pinecone, crear el índice, cargar chunks y recuperar documentos. La alternativa de embeddings por API queda para una comparación posterior. |
| [Langchain_Virtual_Assitant](../../docs/RAG/Langchain_Virtual_Assitant.ipynb) | Celda 9: MiniLM en CPU, Chroma y chunks de 2000 caracteres/200 de overlap. Celda 10: bloque API comentado con GPT-4.1 nano, salida de 512 tokens y `k=2`. Celdas 11 y 22: Qwen2.5-1.5B-Instruct local, salida de 128 tokens y `k=2` o `k=3`. | Referencia para el generador local y el armado del contexto. Adaptaremos esa parte al retriever de Pinecone, conservando Chroma solo como ejemplo del notebook. |
| [UN_Report_Advanced_RAG](../../docs/RAG/UN_Report_Advanced_RAG.ipynb) | Celdas 11 y 14: 1000 caracteres/20 de overlap, luego 256 tokens/20 de overlap. Celdas 18-20: Chroma. Celdas posteriores: expansión de consultas y reranking con CrossEncoder. | Splitting en dos pasos; una técnica avanzada para agregar después del baseline. |
| [RAGAS](../../docs/RAG/RAGAS.ipynb) | Celdas 4-7: faithfulness, answer relevancy, context precision y context recall; GPT-4o mini como juez. | Referencia para una evaluación posterior. Ese juez agrega llamadas y costo; al principio usamos revisión de casos. |

Los notebooks son demos: hay bloques comentados, valores globales, configuraciones
de Colab/CUDA y ejemplos de Chroma sin persistencia explícita. Hay que adaptar sus
componentes al proyecto y al equipo. `RetrievalQA` sirve para estudiar el armado de la cadena;
la salida de nuestro sistema debe comparar cobertura, no responder un Q&A genérico.

## Configuración inicial recomendada

| Componente | Valor inicial propuesto | Justificación |
| --- | --- | --- |
| Corpus | `docs/embeddings_2.VTT` | Permite revisar fuentes y resultados antes de ampliar el material. |
| Limpieza | La actual, conservando contenido y tiempos | Mantiene trazabilidad y evita correcciones semánticas no verificadas. |
| Primer splitting | `RecursiveCharacterTextSplitter`, 1000 caracteres, overlap 100 | Se apoya en los ejemplos de clase; buscamos fragmentos legibles antes del límite por tokens. |
| Segundo splitting | `SentenceTransformersTokenTextSplitter`, modelo MiniLM explícito, 200 tokens de contenido, overlap 20 | Sigue la demo del reporte. Se propone 200, en lugar de sus 256, dejando margen respecto del límite del encoder. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, CPU, 384 dimensiones | Está en los notebooks y en la propuesta aprobada. Ejecutado localmente no consume saldo de la API de OpenAI. |
| Base vectorial | Pinecone, índice remoto de 384 dimensiones con métrica `cosine` | Sigue la celda 26 del notebook de la cátedra y conserva los vectores entre ejecuciones en el servicio. Los embeddings se calculan con MiniLM antes de cargarlos. |
| Retrieval | Similitud coseno, `k=4`, sin umbral inicial | Valor intermedio para empezar a medir cobertura y ruido. No se supone óptimo. |
| Consulta | Objetivo de estudio escrito por el usuario, una búsqueda inicial | Permite buscar contenidos esperados aunque el video no los mencione. |
| Generador del primer pipeline completo | `Qwen/Qwen2.5-1.5B-Instruct` local | Es el generador activo del notebook, pequeño y sin gasto de saldo de OpenAI al ejecutarlo localmente. Su calidad, RAM y latencia deben probarse en el equipo. |
| Generación | `do_sample=False`, máximo 1000 tokens nuevos | El informe necesita etiquetas, evidencias, recomendaciones y conclusión. Los 128 tokens de la demo pueden cortar esa salida. Es un techo, no una longitud obligatoria ni una garantía de determinismo absoluto. |
| Límite de contexto recuperado | Hasta 1200 tokens medidos con el tokenizer del generador | Acota el material de referencia enviado y permite registrar qué candidatos quedaron fuera. Es independiente de los tokens de MiniLM. |
| Límite de entrada al generador | 6000 tokens, contando plantilla, objetivo, transcripción y referencias | Presupuesto conservador para el caso inicial; debe verificarse al implementar. Si se excede, se informa para ajustar el alcance, sin truncar silenciosamente el audio. |
| Llamadas de generación | Una por evaluación inicial | Evita introducir desde el comienzo cadenas de llamadas, expansión y jueces automáticos. |

La elección de Pinecone es una decisión del grupo; los límites y demás valores
anteriores son recomendaciones del asistente, no valores exigidos por las
profesoras.

### Configuración de Pinecone según el notebook de la cátedra

Tomamos como referencia las celdas 26 a 31 de
[Langchain_Pinecone.ipynb](../../docs/RAG/Langchain_Pinecone.ipynb):

- **Cliente y conexión:** `Pinecone` y `PineconeVectorStore`. La credencial
  `PINECONE_API_KEY` se leerá del entorno o de `tp3/.env`, sin escribirla en el
  código ni en los resultados.
- **Índice:** nombre propio del TP3, configurable; no reutilizar el índice de la
  demo. Como configuración inicial seguimos `ServerlessSpec(cloud='aws',
  region='us-east-1')` de la celda 26, verificando que esté disponible para la
  cuenta del grupo. Esa región es un valor del ejemplo, no una exigencia docente.
- **Dimensión y métrica:** 384 para MiniLM y `cosine`. Antes de reutilizar un índice
  existente, comprobar que su dimensión, métrica y modelo de embedding coincidan
  con la configuración de la corrida.
- **Contenido persistido:** ID estable, embedding, texto del chunk y metadatos
  de fuente, intervenciones y tiempos. Los originales y la transcripción del
  video se conservan separados del índice académico.
- **Recuperación:** `PineconeVectorStore` con el mismo embedder usado al indexar,
  búsqueda por similitud y `k=4`. La celda 31 muestra una búsqueda con umbral;
  nuestro baseline comienza sin umbral para medir primero cobertura y ruido.

Al adaptar la celda 26 usaremos el nombre de índice recibido por la función en
lugar de depender de la global `INDEX_NAME` que utiliza la demo. La carga deberá
evitar duplicar chunks al repetir una indexación. Estas son pautas para la futura
implementación: todavía no se creó ni se cargó un índice desde este proyecto.

La [ficha original de MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
confirma las 384 dimensiones y el truncamiento por defecto por encima de 256
word pieces. Un texto que tenga 1000 caracteres no tiene necesariamente menos
de 256 tokens. El límite es de tokens, aunque un comentario del notebook del
reporte use la palabra "chars". Se debe comprobar cada chunk con el tokenizer
real, incluidos los tokens especiales, y conservar sus posiciones en el VTT.

La ficha de MiniLM está etiquetada como inglés. Usarlo aquí es un baseline visto
en clase, no una conclusión sobre su calidad para español. Si la recuperación
falla, una comparación disponible en los notebooks es `text-embedding-3-small`.

La [ficha de Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
documenta el modelo de 1.54B parámetros y soporte de español. Eso no demuestra
que clasifique correctamente la cobertura de nuestro caso. No se han descargado
ni ejecutado estos modelos para esta recomendación.

### Qué hacemos con el GPT-4.1 nano de la propuesta

`gpt-4.1-nano-2025-04-14` aparece en el bloque API comentado y en el ejercicio 1.
Al verificar la [documentación oficial actual de OpenAI](https://developers.openai.com/api/docs/models/gpt-4.1-nano),
la ficha lo marca como "Shut down". No damos por sentado que la API pública lo
siga sirviendo. Su disponibilidad y la posible resolución de alias en el proxy
del ITBA deben confirmarse antes de usarlo. Por eso proponemos Qwen local como
primer generador y dejamos la elección de un generador por API para cuando
podamos verificar su disponibilidad. No sustituimos silenciosamente el snapshot
de la propuesta por otro modelo.

## Flujo de la aplicación

Preparación del material, una vez por configuración de índice:

```text
VTT de la clase
  -> lectura y limpieza con referencias temporales
  -> splitting por caracteres y por tokens
  -> embeddings MiniLM
  -> índice remoto persistente de Pinecone
```

Evaluación de un video:

```text
Audio -> Whisper -> transcripción guardada
Objetivo del usuario -> embedding MiniLM -> búsqueda en Pinecone
Objetivo + transcripción + referencias recuperadas
  -> una generación del informe de cobertura
  -> resultado con evidencias y recomendaciones
```

La transcripción del video se mantiene separada del corpus de la materia. Para
este audio inicial se conserva completa en la comparación: recuperar solo sus
fragmentos más parecidos podría ocultar evidencia necesaria para distinguir
parcial de sin evidencia. Su extensión real se contará antes de generar.

El prompt deberá pedir que los contenidos esperados estén respaldados por la
materia, justificar la cobertura con evidencia del video y recomendar fuentes
recuperadas identificables. Si faltan referencias de la materia, se deberá
informar esa limitación: una falla de retrieval no demuestra una omisión del video.

## Estructura propuesta de archivos

Se conserva lo existente y se agregan piezas pequeñas cuando se implemente cada
etapa. Los nombres siguientes son una propuesta, no archivos ya implementados.

```text
tp3/ejercicio2/
  config.py                   # Valores del experimento, sin credenciales.
  preparar_transcripcion.py   # Ya implementado: lector de VTT.
  transcribir_audio.py        # Ya implementado: Whisper y caché de respuesta.
  indexar_material.py         # Construcción del corpus, chunks e índice.
  recuperar.py                # Objetivo -> referencias y puntajes.
  evaluar_video.py            # Arma entradas y genera el informe.
  ejecutar_experimentos.py    # Repite casos guardados con configuraciones.
  evaluacion/
    caso_01_embeddings.md     # Ejemplo revisable existente.
  data/
    audios/
    transcripciones/
    procesados/
    resultados/
```

El índice vectorial se conserva en Pinecone; no requiere una carpeta local
`data/chroma/`. `config.py` identificará el índice y su configuración, sin incluir
la API key. Los datos procesados y los resultados de las corridas se guardan
localmente para revisión.

Cada chunk guardará `chunk_id`, fuente, IDs de intervenciones, inicio y fin en
milisegundos y versión de configuración. Cada corrida guardará la configuración
efectiva, IDs recuperados, puntajes, contexto realmente enviado, respuesta,
tiempos y conteos de tokens. El historial de chat no se agregará a cada evaluación.

Al cambiar el embedder se reconstruye el índice y se usa un índice separado,
incluso si el nuevo modelo tiene la misma dimensión. Chunks y consultas deben
usar el mismo embedder. Al cambiar el splitting también se reconstruye el índice;
al cambiar solo `k` o el generador se reutiliza.

## Orden de iteración

| Paso | Qué cambia | Qué observamos |
| --- | --- | --- |
| 0 | Implementar solo splitting, índice y retrieval | Legibilidad de chunks, límites por tokens, tiempos correctos, pertinencia y contenidos importantes recuperados. |
| 1 | Completar el pipeline con el generador propuesto | Etiquetas justificadas, referencias reales, informe completo y respuesta a la consigna. |
| 2 | `k`: comparar 2, 4 y 8 | Cobertura de las referencias frente a ruido, tokens y latencia. Registrar si el presupuesto de contexto limita el `k` efectivo. |
| 3 | Tamaño por tokens: comparar 128, 200 y 240 | Conservación de explicaciones y pertinencia; mantener el resto fijo. |
| 4 | Overlap por tokens: comparar 0, 20 y 40 | Cortes de ideas frente a redundancia; mantener el resto fijo. |
| 5 | MiniLM frente a `text-embedding-3-small` | Recuperación en español, latencia y costo de embeddings; generar índices separados. |
| 6 | Comparar generadores disponibles en el entorno elegido | Calidad y costo con el mismo contexto recuperado. Un generador más grande no repara por sí solo referencias faltantes. |
| 7 | Agregar una técnica avanzada del notebook, por ejemplo reranking | Calidad antes y después, tiempo y costo adicional. Recién con el pipeline básico funcionando. |

Se cambia una variable por vez y se conserva una referencia de comparación.
Al cambiar el máximo por tokens se debe registrar cuántos chunks efectivos
cambiaron: la primera pasada por caracteres puede dejar algunos por debajo
del límite en todas las configuraciones.

Pinecone es la base del primer pipeline completo. La elección busca seguir la
implementación de clase; no supone que por sí sola mejore la cobertura frente a
otra base. En los experimentos se mantiene esta base fija para aislar el efecto
del splitting, `k`, embeddings y generación, registrando también persistencia,
operación y latencia.

Para empezar, revisamos pertinencia de los chunks recuperados y cobertura de los
contenidos de referencia del caso 01. Tras validar etiquetas, podemos calcular
precisión de recuperación y coincidencia de clasificaciones. Ese caso servirá
para desarrollo; harán falta otros objetivos y videos separados para evaluar las
mejoras. RAGAS puede incorporarse después, adaptando sus métricas al caso de uso
y midiendo las llamadas de su juez.

## Control del presupuesto

- Reutilizar la transcripción del audio y los embeddings ya cargados en Pinecone;
  no reconstruir el índice en cada evaluación de un video.
- Revisar el splitting localmente y probar la recuperación en Pinecone antes de
  ejecutar generación.
- Medir con el tokenizer de cada modelo: los tokens de MiniLM y Qwen no son
  intercambiables. Si se usa una API, registrar también su uso reportado.
- Separar tokens de entrada y salida, tiempo de ejecución y costo monetario.
  Los embeddings y el generador locales usan recursos del equipo aunque no
  consuman saldo de OpenAI; Pinecone agrega almacenamiento y llamadas remotas.
- Verificar límites y costos del plan de Pinecone antes de cargar el corpus y
  registrar las operaciones de indexación y consulta. No asumir que el servicio
  remoto es gratuito por ejecutar MiniLM localmente.
- Registrar el motivo de finalización: si se alcanza el límite de salida y se
  corta el informe, no tratarlo como un resultado completo.
- Verificar tarifas y disponibilidad del proxy antes de incorporar modelos por
  API. No se hicieron nuevas solicitudes de inferencia para este diseño.
