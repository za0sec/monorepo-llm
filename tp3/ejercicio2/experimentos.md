# Registro experimental del ejercicio 2

Este registro documenta el corpus de partida, las comparaciones y las decisiones. El [roadmap](../roadmap_ejercicio2.md) describe el protocolo y las métricas. Al ejecutar un experimento, registrar aquí la fecha, las configuraciones comparadas, los resultados por caso, dónde quedaron las salidas y el motivo de la decisión. La [clase de RAG](../../docs/RAG/Clase%207.1%20-%20RAG.pdf), pp. 17–20, ubica la carga y el splitting dentro de la indexación.

## 1. Indexing Pipeline

### Preparación del corpus · 14/11/2026

- **Punto de partida:** [`docs/embeddings_2.VTT`](../../docs/embeddings_2.VTT), transcripción docente en español de la clase de embeddings. Su SHA-256 es `a1fa8324b6d63c7c6d90fd4dc467b65095524aba8e134d806c2118278b24cd18`.
- **Implementación:** se adaptó el [lector archivado](../archivo/ejercicio2_anterior/preparar_transcripcion.py). Une las líneas de texto de cada bloque y reduce secuencias de espacios en blanco a un espacio. Conserva las palabras, la puntuación y el hablante tal como aparecen en el VTT. Guarda la ruta y el hash de la fuente y, por bloque, el ID, el texto y el inicio y fin en formato VTT y en milisegundos.
- **Resultado:** 415 de 415 bloques coinciden con el original en ID, texto normalizado y tiempos. Abarcan de `00:00:03.410` a `01:01:48.330`. Una segunda ejecución en otro directorio produjo el mismo JSON byte a byte.
- **Decisión:** usar [`data/procesados/embeddings_2.json`](data/procesados/embeddings_2.json) como único insumo de texto del piloto. Los tiempos permiten revisar la definición de embedding (bloque 123, 18:32.360–18:50.210), el papel de la magnitud (bloques 182–188, 25:37.950–27:13.979) y el cálculo del coseno (bloque 222, 31:55.350–32:14.820). Son referencias iniciales del [caso manual](../archivo/ejercicio2_anterior/evaluacion/caso_01_embeddings.md).

Para regenerar el JSON desde la raíz del repositorio con Python 3.9 o posterior y el VTT docente disponible localmente:

```bash
python3 tp3/ejercicio2/preparar_transcripcion.py
```

El JSON es local y queda fuera de Git. Para generar una copia en otro directorio, usar `--salida <directorio>`.

### I1 · Estrategia de splitting

- **Punto de partida:** JSON de la clase preparado arriba y consultas de desarrollo sobre definición de embedding, cálculo del coseno y efecto de la magnitud.
- **Qué evaluar:** ventanas fijas frente a corte recursivo, con máximo de 160 tokens y solapamiento cero. Fijar consultas, tokenizer, MiniLM, Chroma, coseno y `k=4`. Anotar los pasajes esperados antes de ejecutar la búsqueda. Comparar Context Recall y Context Precision, y revisar cortes que separen partes de una explicación.
- **Resultados:** pendientes de implementar los dos splitters y la búsqueda básica.
- **Decisión:** pendiente. El piloto de una clase permite revisar cómo se mide y dónde falla cada variante. Ampliar a más objetivos y unidades antes de elegir un splitter.

### I2 · Tamaño de chunk

- **Punto de partida:** splitter elegido en I1.
- **Qué evaluar:** 96, 160 y 224 tokens con solapamiento cero. Medir recuperación de evidencia y cantidad de tokens recuperados.
- **Resultados:** pendientes.
- **Decisión:** pendiente.

### I3 · Solapamiento

- **Punto de partida:** splitter y tamaño elegidos. Revisar si los cortes dejaron evidencia fuera.
- **Qué evaluar:** si aparecen pérdidas en los bordes, comparar 0 y 20 tokens de solapamiento. Medir criterios recuperados y contexto repetido.
- **Resultados:** pendientes.
- **Decisión:** pendiente.

### I4 · Modelo de embeddings

- **Punto de partida:** chunks elegidos y MiniLM como baseline.
- **Qué evaluar:** comparar con `text-embedding-3-small` si está disponible para el grupo y el corpus. Usar los mismos chunks, consultas y `k`. Medir recall, precisión y recursos.
- **Resultados:** pendientes.
- **Decisión:** pendiente.

## 2. Retrieval

### R0 · Búsqueda básica para medir I1

- **Punto de partida:** objetivo literal y un índice académico por variante de splitting.
- **Qué evaluar:** verificar que una consulta use el mismo modelo de embeddings que su índice y devuelva hasta cuatro resultados ordenados por distancia coseno, con IDs y tiempos de origen.
- **Resultados:** pendientes.
- **Decisión:** usar esta configuración como control planificado de I1.

### R1 · Cantidad de resultados

- **Punto de partida:** índice elegido tras las comparaciones de indexación.
- **Qué evaluar:** comparar `k=2, 4 y 8` sobre el mismo ranking. Medir recall, precisión, redundancia y tamaño del contexto recuperado.
- **Resultados:** pendientes.
- **Decisión:** pendiente.

### R2 · Reranking

- **Punto de partida:** RAG básico integrado y evidencia relevante en posiciones bajas de los candidatos.
- **Qué evaluar:** comparar coseno y CrossEncoder con el mismo conjunto de candidatos, `k` final y presupuesto de contexto. Medir el ranking y la calidad del informe.
- **Resultados:** pendientes.
- **Decisión:** pendiente.

### R3 · Expansión de consulta

- **Punto de partida:** RAG básico integrado y diagnóstico de evidencia ausente entre los candidatos.
- **Qué evaluar:** contrastar consulta directa y expansión por subtemas con el mismo presupuesto total de candidatos y contexto. Medir criterios nuevos y ruido.
- **Resultados:** pendientes.
- **Decisión:** pendiente. El diagnóstico indicará si conviene probar R2 o R3 primero.

## 3. Generator

### G0 · Informe básico

- **Punto de partida:** objetivo, VTT fijo del video y Qwen2.5-1.5B-Instruct como candidato inicial, con un prompt directo.
- **Qué evaluar:** generar primero con referencias de clase seleccionadas a mano y luego con la salida de la búsqueda. Medir acierto de cobertura, fidelidad, validez de citas y relevancia.
- **Resultados:** pendientes.
- **Decisión:** pendiente.

### G1 · Prompt

- **Punto de partida:** informe básico con el mismo modelo, audio transcripto y referencias académicas para ambas variantes.
- **Qué evaluar:** comparar la instrucción directa con una pauta explícita de respaldo académico, cotejo con el audio y abstención. Medir cobertura, fidelidad, citas y relevancia.
- **Resultados:** pendientes.
- **Decisión:** pendiente.

### G2 · Modelo generador

- **Punto de partida:** prompt elegido en G1 con entradas y formato fijos.
- **Qué evaluar:** comparar Qwen con GPT-4.1 si está habilitado para el grupo. Comparar calidad, truncamiento y latencia.
- **Resultados:** pendientes.
- **Decisión:** pendiente.

### G3 · Formato del informe

- **Punto de partida:** modelo y prompt elegidos. Revisar si faltan campos en la salida en prosa.
- **Qué evaluar:** si se observan omisiones, comparar prosa y tabla por criterio con las mismas entradas y exigencias de contenido. Medir completitud y citas válidas.
- **Resultados:** pendientes.
- **Decisión:** pendiente.
