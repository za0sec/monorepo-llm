# TP3 · Evaluación de videos de estudio con RAG

El sistema recibe un objetivo de estudio y un MP3. Whisper transcribe el audio, el RAG recupera material académico de Chroma y GPT-4.1 genera un informe Markdown con cobertura por tema, citas y recomendaciones para estudiar.

## Preparación

Ejecutá los comandos desde la raíz del repositorio. Si todavía no tenés el entorno `.venv`, crealo e instalá las [dependencias del TP3](ejercicio2/requirements.txt).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r tp3/ejercicio2/requirements.txt
```

Guardá tu clave `OPENAI_API_KEY` en el archivo local `tp3/.env`. El flujo envía el audio, las consultas y el contexto de generación a la API directa de OpenAI.

El comando y la interfaz necesitan estos insumos locales:

- **Corpus final.** La colección `openai_e4_completo` vive en la base persistente de Chroma en `tp3/ejercicio2/data/experimentos/i4/chroma/`. Conservá también el manifiesto `tp3/ejercicio4/data/carga_completa.json`. En otro checkout, copiá esos insumos o seguí la [carga del corpus final](roadmap_ejercicio4.md#carga-de-las-105-diapositivas).
- **Reranker.** El modo avanzado usa el snapshot local de `cross-encoder/ms-marco-MiniLM-L6-v2`. Si falta, descargalo con el [comando y la revisión fijada](ejercicio2/README.md#insumos-locales).
- **Audio.** Podés usar tu propio MP3. El ejemplo usa el MP3 local de Tokenización, que se guarda fuera de Git.

## Ejecutar el pipeline por terminal

```bash
.venv/bin/python tp3/ejercicio2/evaluar_video.py \
  'Quiero entender cómo pasa un texto de ser una cadena de caracteres a tokens, IDs y embeddings, y qué representa cada paso' \
  tp3/ejercicio4/data/casos/tokenizacion/audio.mp3 \
  --corpus completo
```

`--corpus completo` usa las transcripciones de clase, los apuntes y las diapositivas. El modo avanzado es el predeterminado. Busca diez vecinos por consulta, combina la búsqueda del objetivo con las de tres subconsultas y reordena los candidatos con MiniLM. El generador recibe los siete primeros chunks junto con el VTT del video.

Al terminar, el comando imprime la ruta de `gpt41/informe.md`. La ejecución queda en una carpeta nueva dentro de `tp3/ejercicio2/data/ejecuciones/`, con `transcripcion.vtt`, `entrada.json` y `gpt41/corrida.json`. Si querés elegir la carpeta, agregá `--salida <carpeta-nueva>`.

## Abrir la interfaz web

```bash
.venv/bin/python tp3/ejercicio4/ui_local.py
```

1. Abrí [http://127.0.0.1:8765](http://127.0.0.1:8765).
2. Escribí el objetivo de estudio y elegí un MP3 de hasta **25 MB**.
3. Presioná **Evaluar video**. Podés leer el informe en pantalla o guardarlo con **Descargar Markdown**.

La interfaz usa el mismo pipeline con el corpus completo. En **Archivos del corpus** podés abrir el material de clase. La interfaz elimina los archivos temporales del servidor al terminar cada solicitud.

Dejá la terminal abierta mientras uses la página. Si el puerto está ocupado, arrancá con `--puerto 8766` y abrí [http://127.0.0.1:8766](http://127.0.0.1:8766). Para detener el servidor, usá `Ctrl+C` en la terminal.

La [prueba de cierre con Tokenización](ejercicio4/evaluacion/cierre_caso_04_mp3.md) documenta una ejecución completa y enlaza su informe.
