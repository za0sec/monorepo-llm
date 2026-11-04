# Ejercicio 2: material de referencia y primer audio

Para ponerse al día con el proyecto, leer la [presentación y recorrido del TP3](../presentación.md):
alcance inicial, decisiones, pasos realizados y resultados de las pruebas.

El usuario final cargará el audio de un video y los temas que quiere estudiar.
El sistema deberá compararlo con los materiales de la materia e informar contenidos
cubiertos, parciales y sin evidencia, recomendar material y concluir sobre la
utilidad del video. Ya tenemos una transcripción de clase, la transcripción
automática de un audio de ejemplo, un caso de cobertura revisado, chunks del
material académico e indexación y recuperación local.
Todavía no implementamos la generación ni la comparación automática de cobertura.

La base vectorial elegida es **Chroma local con persistencia en `data/chroma/`**.
Usamos los embeddings de MiniLM para cargar los chunks y buscar referencias de
la materia. La elección reemplaza la propuesta intermedia de Pinecone y retoma
Chroma, presente en los notebooks de la cátedra. El 31/10/2026 se indexaron los
125 chunks y se probaron búsquedas reales. La recuperación inicial muestra ruido
y omisiones; no implica que el sistema ya pueda evaluar videos correctamente.

La [propuesta de diseño inicial](diseno_rag_inicial.md) reúne las recomendaciones
encontradas en las clases, los componentes de los notebooks, una configuración
de partida y el orden de experimentos. Distingue las etapas implementadas de la
generación y los experimentos todavía propuestos.

## Fuente inicial

Usamos `docs/embeddings_2.VTT`, que incluye embeddings y similitud coseno.
`embeddings_1.VTT` se concentra en la demo de implementación del Transformer.
Conservamos completa la transcripción seleccionada: todavía no filtramos temas,
preguntas de estudiantes ni intervenciones administrativas.

## Ejecutar

Desde la raíz del repositorio, con Python 3.9 o superior y sin instalar paquetes:

```bash
python3 tp3/ejercicio2/preparar_transcripcion.py
```

También se puede elegir otro VTT de texto de la cátedra:

```bash
python3 tp3/ejercicio2/preparar_transcripcion.py docs/embeddings_1.VTT
```

Los archivos se generan en `tp3/ejercicio2/data/procesados/`:

- `embeddings_2.json`: fuente del documento y segmentos con identificador,
  inicio, fin, tiempos en milisegundos y texto.
- `embeddings_2.txt`: vista para revisar manualmente, con tiempos por intervención.

Son resultados regenerables y están excluidos de Git. Los VTT originales están
en `docs/`, que ya está excluido de Git en este repositorio; deben estar presentes
localmente para ejecutar el script.

## Decisiones de limpieza

- Quitamos del cuerpo del texto la cabecera VTT, los identificadores y las líneas
  de tiempos. Los identificadores y tiempos se conservan en campos separados.
- Unimos líneas de una misma intervención y normalizamos espacios en blanco.
- Conservamos nombres de hablantes, puntuación y palabras originales. No separamos
  automáticamente por dos puntos porque también pueden ser parte del discurso.
- No corregimos errores de transcripción ni completamos explicaciones. Por ejemplo,
  la fuente contiene variantes como "Meding"; corregirlas requeriría una decisión
  posterior y comparar el efecto de ese preprocesamiento.
- Cada segmento corresponde a una entrada original del VTT. **Todavía no es un
  chunk para embeddings**: muchas entradas son frases incompletas que luego habrá
  que agrupar, conservando sus referencias.
- Se omiten comentarios `NOTE`. Este lector está acotado a los VTT de texto de la
  cátedra; no interpreta estilos, regiones ni etiquetas de subtítulos en HTML.
  Ante bloques desconocidos, texto vacío, IDs repetidos o tiempos inválidos,
  informa un error en vez de descartarlos silenciosamente. Permite intervenciones
  superpuestas, pero exige que los inicios estén ordenados.

## Revisión y siguiente paso

El caso manual inicial ya fue revisado por el usuario. Se cotejó el texto de los
125 chunks persistidos contra sus offsets en el VTT limpio y se revisaron los
fragmentos de coseno y magnitud. Se conservan texto y referencias, aunque hay
cortes dentro de palabras e intervenciones. La revisión de recuperación está en
[validación inicial](evaluacion/revision_retrieval_inicial.md).

El [caso 01: embeddings](evaluacion/caso_01_embeddings.md) contiene ese primer
ejemplo: objetivo propuesto, criterios apoyados en la clase, evidencias temporales,
clasificaciones, recomendaciones y conclusión. Fue elaborado por el asistente a
partir de las transcripciones y el usuario confirmó su revisión el 31/10/2026. Sirve
para concretar la salida esperada antes de automatizarla.

## Preparar chunks de la clase

Desde la raíz, activar el entorno e instalar las dependencias actualizadas:

```bash
source .venv/bin/activate
python3 -m pip install -r tp3/ejercicio2/requirements.txt
python3 tp3/ejercicio2/preparar_chunks.py
```

El script reutiliza el lector de VTT. Une las intervenciones con separadores de
párrafo, aplica `RecursiveCharacterTextSplitter` (1000 caracteres, overlap 100)
y divide de nuevo por tokens de MiniLM (máximo 200 de contenido, overlap 20).
Los valores son parámetros iniciales del diseño, no requisitos docentes.

La segunda pasada usa el tokenizer rápido de `all-MiniLM-L6-v2` y sus offsets
para tomar el texto original: no decodifica tokens, no elimina acentos ni cambia
mayúsculas. Se adapta así el splitting por tokens del notebook del reporte ONU
para conservar la trazabilidad. Los límites por tokens pueden cortar palabras
o intervenciones: conservar todo el texto no garantiza por sí solo que cada
chunk tenga contexto suficiente; la legibilidad debe revisarse.

Se fija la revisión del tokenizer en el código y se lee el límite de 256 tokens
de `sentence_bert_config.json`, incluyendo los especiales. Esta etapa solo carga
tokenizer y configuración, sin pesos de modelos ni GPU. En la primera ejecución
puede descargar esos archivos de Hugging Face; si ya están en caché:

```bash
python3 tp3/ejercicio2/preparar_chunks.py --solo-local
```

Genera `data/chunks/embeddings_2.chunks.json` y `.txt`, ignorados por Git. Cada
chunk incluye ID estable, texto, fuente, IDs de intervenciones, offsets de
caracteres, tiempos y conteos de tokens. Los tiempos delimitan las intervenciones
originales involucradas; no estiman el instante exacto de cada palabra cortada.
Los offsets corresponden al texto limpio unido con `\n\n`, no al archivo VTT bruto.
Se guardan hash de la fuente, parámetros y versión de configuración.

Se comprueba que no desaparezca ningún carácter no blanco y que cada chunk
respete los límites de contenido y del encoder. Los IDs distinguen apariciones
repetidas y cambian si cambia la fuente o la configuración. Si se genera otra
configuración en la misma carpeta, se reemplazan los resultados; usar `--salida`
para conservar experimentos distintos:

```bash
python3 tp3/ejercicio2/preparar_chunks.py --tokens 128 --overlap-tokens 20 \
  --salida /tmp/tp3-chunks-128
```

La ejecución inicial sobre 415 intervenciones produjo **125 chunks**, con máximo
de **202 tokens incluidos los especiales**. Estos son controles de integridad y
tamaño; todavía no son métricas de recuperación ni de cobertura del video.

Las pruebas usan el tokenizer real previamente descargado, incluida una prueba
con la clase local `docs/embeddings_2.VTT`:

```bash
python3 -m unittest discover -s tp3/ejercicio2/tests -v
```

`preparar_chunks.py` solo prepara el JSON/TXT. La indexación y las búsquedas se
ejecutan con los comandos siguientes.

## Indexar el material en Chroma

Con el entorno activo y las dependencias instaladas:

```bash
python3 tp3/ejercicio2/indexar_material.py
```

Lee `data/chunks/embeddings_2.chunks.json`, genera embeddings con
`sentence-transformers/all-MiniLM-L6-v2` en CPU y persiste texto, vectores de
384 dimensiones, IDs y metadatos en `data/chroma/`. Configura explícitamente
distancia coseno. La transcripción del video no se agrega a esta colección.

La primera ejecución descarga los pesos públicos a `data/modelos/`. No hace
llamadas de inferencia a OpenAI ni utiliza sus credenciales. Con el modelo en
caché se puede trabajar sin descargas:

```bash
python3 tp3/ejercicio2/indexar_material.py --solo-local
```

El nombre de colección se deriva del hash del JSON de chunks, modelo/revisión,
métrica y versiones de las bibliotecas. El manifiesto se guarda en sus metadatos.
Repetir una carga completa reutiliza los IDs sin volver a ejecutar el embedder.
Una carga interrumpida queda incompleta y puede reanudarse; se bloquean consultas
hasta completarla. Cambiar corpus, splitting, embedder o versiones produce otra
colección. Si se fuerza `--coleccion` con una configuración incompatible, falla
sin reemplazar la colección existente.

Opciones: `entrada` permite seleccionar otro JSON de chunks y `--base` otra
carpeta de Chroma. Conservar el JSON correspondiente para consultar ese índice.
Los límites de tokens se comprueban antes de codificar; no se permite truncamiento
silencioso de los chunks ni del objetivo.

Referencias: notebook `Langchain_Virtual_Assitant.ipynb`, celda 9, y
`UN_Report_Advanced_RAG.ipynb`, celdas 18-23. La adaptación usa las APIs oficiales
de [persistencia](https://docs.trychroma.com/reference/python/client),
[configuración de distancia](https://docs.trychroma.com/docs/collections/configure)
y [SentenceTransformer](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).

## Recuperar referencias desde un objetivo de estudio

```bash
python3 tp3/ejercicio2/recuperar.py \
  "entender similitud coseno y sus limitaciones" --solo-local
```

Usa el mismo modelo y revisión que la colección. Busca desde el objetivo del
usuario, con `k=4` y sin umbral como punto de partida. Muestra texto y tiempos,
y guarda cada corrida en un JSON nuevo en `data/resultados/`: consulta,
configuración, IDs, metadatos, textos, distancias y tiempos medidos. No invoca
al generador ni clasifica cobertura del video.

Chroma devuelve **distancia coseno: menor es mejor**. También guardamos
`similitud_coseno = 1 - distancia_coseno`; no es probabilidad ni porcentaje de
cobertura. Puede haber pequeños errores numéricos alrededor de 0 o 1.

Opciones: `--k`, `--chunks`, `--base`, `--coleccion` y `--salida ruta.json`.
Si `k` supera la cantidad de chunks, se devuelven los disponibles y se registran
el `k` solicitado y el efectivo. No se introduce un filtro por puntaje todavía:
una consulta fuera de dominio también devuelve vecinos y requiere revisión.

**Resultado inicial:** la consulta de coseno recuperó la explicación del cálculo
en el puesto 4, pero omitió el ejemplo sobre magnitud. El objetivo más amplio del
caso 01 también recuperó material poco pertinente. Ver las corridas, referencias
y límites en [la revisión inicial](evaluacion/revision_retrieval_inicial.md).
El siguiente paso es diagnosticar y mejorar esta recuperación antes de usarla
como contexto para generar el informe.

Las 13 pruebas incluyen integridad de chunks y una integración con Chroma real
y vectores controlados (persistencia en otro proceso, repetición, reanudación,
incompatibilidades y trazabilidad). La calidad semántica de MiniLM se revisó
por separado con las consultas reales; no se infiere de esas pruebas.

## Transcribir el audio de ejemplo

Este paso convierte el audio del video a texto. El material de la materia se
mantiene separado: no se envía como parte de la transcripción.
Usamos Whisper, visto en la clase de imágenes y audio, a través del proxy de la
cátedra y con las credenciales existentes en `tp3/.env`.

Instalar las dependencias en el entorno de Python que se vaya a utilizar:

```bash
python3 -m pip install -r tp3/ejercicio2/requirements.txt
```

Desde la raíz del repositorio:

```bash
python3 tp3/ejercicio2/transcribir_audio.py \
  "tp3/ejercicio2/data/audios/¿Qué son los EMBEDDINGS Grandes Modelos de Lenguaje.mp3"
```

Se generan tres archivos dentro de `tp3/ejercicio2/data/transcripciones/`:

- Un `.respuesta.json` con la respuesta original del servicio, guardada antes de
  interpretarla para poder revisarla sin repetir la llamada.
- Un `.vtt` con las palabras reconocidas y sus intervalos temporales.
- Un `.txt` con tiempos para revisar el resultado de forma cómoda.

La primera solicitud de salida VTT no produjo el formato esperado por el lector.
Por eso solicitamos `verbose_json` y generamos el VTT localmente con los tiempos de sus
segmentos. No calculamos tiempos a partir de la longitud del texto.

El script reutiliza el VTT o la respuesta JSON si ya existen para evitar repetir
la llamada. Si se quiere volver a transcribir, usar `--salida` con otra carpeta. Esta primera
versión asume audio en español y admite archivos de hasta 25 MB; no aplica
filtros de audio ni corrige términos técnicos automáticamente. Los intervalos
provienen de Whisper y son aproximados; no son chunks de recuperación.

Configuración: `OPENAI_API_KEY`, `OPENAI_BASE_URL` y, opcionalmente,
`OPENAI_TRANSCRIPTION_MODEL` (por defecto `openai/whisper-1`, identificador
publicado por el proxy). Para la API directa, configurar su URL y `whisper-1`.
La clave se lee sin imprimirla ni guardarla en los resultados.

Referencia de la llamada: [documentación oficial de transcripción de OpenAI](https://developers.openai.com/api/docs/guides/speech-to-text).

La transcripción automática está pendiente de revisión, especialmente en términos
como embeddings, nombres de modelos y fórmulas. Generar texto no demuestra aún
que el video cubra los contenidos de la materia.
