# Ejercicio 2: primer paso con texto

Por ahora trabajamos con una sola transcripción de clase. El objetivo de este paso
es obtener texto con referencias temporales antes de dividirlo en chunks o
conectarlo a una base vectorial.

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
Después agruparemos intervenciones en chunks y revisaremos sus límites antes de
agregar embeddings y recuperación. La comparación con un video vendrá después.
