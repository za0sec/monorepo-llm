# Ejercicio 2: material de referencia y primer audio

El usuario final cargará el audio de un video y los temas que quiere estudiar.
El sistema deberá compararlo con los materiales de la materia e informar contenidos
cubiertos, parciales y sin evidencia, recomendar material y concluir sobre la
utilidad del video. Estamos preparando las entradas de esa comparación:
una transcripción de clase y la transcripción automática de un audio de ejemplo.
Todavía no implementamos recuperación, comparación ni el informe de cobertura.

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

Primero revisar algunos segmentos contra el original, incluidos los que mencionan
similitud coseno, para comprobar que se conserva el contenido y su ubicación.
Antes de agregar chunks y recuperación, revisaremos el texto del video y
definiremos un ejemplo manual de cobertura contra el material de clase.

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
