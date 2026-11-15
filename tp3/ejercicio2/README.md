# Ejercicio 2 · Evaluación de cobertura de un video

El comando recibe un objetivo de estudio y el archivo de audio del video. Whisper transcribe el audio en español, el sistema recupera material de clase desde Chroma y GPT‑4.1 genera un informe Markdown con una tabla de cobertura, material para estudiar y una conclusión. El nivel introductorio se fija dentro de la aplicación.

## Ejecutar

Desde la raíz del repositorio, con las [dependencias](requirements.txt) instaladas en `.venv`, una clave `OPENAI_API_KEY` en `tp3/.env` y el índice local preparado:

```bash
.venv/bin/python tp3/ejercicio2/evaluar_video.py \
  'entender qué son los embeddings y cómo se compara la similitud entre textos' \
  'tp3/archivo/ejercicio2_anterior/data/audios/¿Qué son los EMBEDDINGS Grandes Modelos de Lenguaje.mp3'
```

El modo predeterminado es **avanzado**. Genera tres subconsultas, busca con ellas y el objetivo original, deduplica los candidatos y reordena los chunks localmente con el CrossEncoder de G4. Envía al generador los siete primeros. Si el solapamiento deja menos de siete candidatos, completa la lista con la búsqueda del objetivo original antes de reordenar y registra esa búsqueda adicional.

**`--modo basico`** recupera directamente los siete chunks más cercanos al objetivo, como en G2. El mismo generador recibe el objetivo original, la transcripción y los pasajes seleccionados.

## Insumos locales

- **Índice académico.** Se usa la colección `openai_i4` de Chroma y su `ranking.json`. La ruta predeterminada es `tp3/ejercicio2/data/experimentos/i4/`. Para reconstruirlo a partir de los VTT docentes, seguir la [reproducción de I4](experimentos.md#i4--modelo-de-embeddings) y de su baseline de I3. Se puede indicar otra ubicación del índice de I4 con `--indice`.
- **Reranker.** El modo avanzado usa el snapshot de `cross-encoder/ms-marco-MiniLM-L6-v2` fijado en G4. Si falta, descargarlo en la caché esperada:

  ```bash
  HF_HUB_DISABLE_XET=1 .venv/bin/hf download cross-encoder/ms-marco-MiniLM-L6-v2 \
    --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a \
    --cache-dir tp3/ejercicio2/data/experimentos/g4/modelos \
    --include config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt
  ```

  Si el snapshot está en otra carpeta, pasar su ruta con `--snapshot`.

Whisper y GPT‑4.1 se ejecutan en la API directa de OpenAI. El audio se envía a Whisper. La expansión y los embeddings usan las consultas de texto. La generación recibe el contexto académico seleccionado y la transcripción.

## Revisar una ejecución

El comando imprime la ruta del informe. Cada ejecución usa una carpeta nueva en `tp3/ejercicio2/data/ejecuciones/`. Se puede fijar una carpeta nueva con `--salida`.

| Archivo | Contenido |
| --- | --- |
| `transcripcion.vtt` | Transcripción y tiempos devueltos por Whisper. |
| `entrada.json` | Hashes de las fuentes, objetivo, parámetros, consultas generadas, rankings y mensajes exactos del generador. |
| `gpt41/informe.md` | Informe para el estudiante, con los tiempos de las citas. |
| `gpt41/informe_crudo.md` y `gpt41/corrida.json` | Texto original, revisión del modelo, tokens, motivo de fin e IDs inválidos. |

Si falla una llamada, el comando devuelve un error y conserva los archivos ya guardados. Si `entrada.json` quedó grabada, se puede repetir solo la generación con `generar_g1.py gpt41 --entrada <archivo> --carpeta <directorio-nuevo>`.

El recorrido completo se verificó con el MP3 del Caso 01. Los resultados de cobertura por modelo y las limitaciones del informe se documentan en la [bitácora](experimentos.md#integración-del-ejercicio-2).
