# Cierre del flujo desde MP3 con el Caso 04

Se conectó el comando de evaluación al corpus completo de 1.484 chunks con la opción `--corpus completo`. Se mantiene `openai_i4` como colección predeterminada del ejercicio 2. El ejercicio 4 usa Whisper en español, `text-embedding-3-small`, la consulta original y tres subconsultas, diez resultados de Chroma por consulta, el CrossEncoder de G4 y `k=7`. El informe usa el prompt original de G1 y GPT‑4.1 a temperatura 0. [Modo de uso](../../ejercicio2/README.md)

Desde la raíz del repositorio, con el corpus local y la clave configurados, usar una carpeta de salida nueva.

```bash
.venv/bin/python tp3/ejercicio2/evaluar_video.py \
  'Quiero entender cómo pasa un texto de ser una cadena de caracteres a tokens, IDs y embeddings, y qué representa cada paso' \
  tp3/ejercicio4/data/casos/tokenizacion/audio.mp3 \
  --corpus completo --salida tp3/ejercicio4/data/experimentos/tokenizacion_mp3
```

La ejecución guardada transcribió el MP3 en **55 segmentos**. Chroma devolvió diez resultados en cada búsqueda. Quedaron **24 candidatos únicos** y el reranker seleccionó siete, entre ellos tres fragmentos de diapositivas. El informe terminó con `stop` y todos sus IDs se ubicaron en una página o marca de tiempo. El top 7 conservó el respaldo académico de los tres criterios. [Transcripción](../data/experimentos/cierre_caso_04_mp3/transcripcion.vtt) · [Entrada, rankings y procedencia](../data/experimentos/cierre_caso_04_mp3/entrada.json) · [Informe final](../data/experimentos/cierre_caso_04_mp3/gpt41/informe.md) · [Auditoría](../data/experimentos/cierre_caso_04_mp3/gpt41/corrida.json)

| Criterio | Estado de la pauta | Estado del informe |
| --- | --- | --- |
| TK1 · Segmentación | Cubierto | Cubierto |
| TK2 · Asignación de IDs | Sin evidencia en el video | Sin evidencia en el video |
| TK3 · Embedding | Cubierto | Parcial |

El informe acertó **2/3 estados**. Reconoce que el video describe el embedding como el vector que representa al token, pero lo marca «Parcial» al exigir una explicación más profunda sobre qué captura. Ese detalle queda fuera de TK3. Además, recomienda C6 para el orden completo de los pasos, aunque ese diagrama muestra tokenización y embeddings sin el paso de IDs. Los IDs citados existen y el respaldo de cada cita requiere revisión de contenido. [Pauta del caso](casos_03_06.md#caso-04--tokenización)

Las subconsultas cambiaron respecto del experimento de `k`. Aun así, los siete chunks finales quedaron en el mismo orden, Whisper reprodujo el VTT y los mensajes enviados a GPT‑4.1 fueron idénticos. El control acertó 3/3 estados y el nuevo informe 2/3. Se registra como variación observada de la generación con la misma entrada. Esta prueba verifica la integración desde el audio y conserva `k=7` como referencia. [Control del Caso 04](k_corpus_completo_seis_casos.md#resultados-de-los-cuatro-casos-nuevos)
