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

**Piloto de desarrollo · 14/11/2026.** Se usó el JSON local de la clase de embeddings. Antes de la búsqueda se registraron tres consultas específicas, una consulta que combina los tres temas y los pasajes y criterios esperados. [Consultas y pauta del piloto](consultas_i1.json)

La hipótesis era que los cortes recursivos recuperarían más evidencia al preservar explicaciones. Ambos métodos recibieron el mismo texto unido con un salto de línea entre bloques y usaron MiniLM en CPU, un máximo de 160 tokens de contenido, solapamiento cero y Chroma con distancia coseno y `k=4`. La ventana fija usa offsets del tokenizer. El recursivo de LangChain aplica la jerarquía de separadores de la demo del reporte ONU y mide la longitud con el mismo tokenizer. [Clase de RAG, pp. 24–30](../../docs/RAG/Clase%207.1%20-%20RAG.pdf) · [Demo ONU, celdas 11 y 14](../../docs/RAG/UN_Report_Advanced_RAG.ipynb)

Los 112 chunks de ventanas fijas y los 141 del recursivo reconstruyen el texto completo, conservan los IDs y tiempos de origen y tienen como máximo 160 tokens. La mediana fue de 160 y 136 tokens, respectivamente. En los cuatro resultados por consulta, las ventanas fijas recuperaron 640 tokens en promedio y el recursivo 499,25. Son tokens de MiniLM medidos antes de la generación.

Una segunda ejecución en un directorio nuevo produjo el mismo JSON byte a byte. Al abrir nuevamente las colecciones persistidas, se verificaron el conteo, la distancia coseno y la correspondencia entre IDs y texto.

El asistente anotó el texto de los candidatos agrupados por consulta, con identificadores de contenido y sin ver el método ni la posición en el ranking. Un fragmento es relevante si aporta evidencia directa a un criterio esperado, aun cuando falte otra parte de la explicación. El recall requiere que el conjunto recuperado sostenga el criterio completo. La anotación queda abierta a revisión humana.

| Consulta | Recall fijo | Recall recursivo | Context Precision fija | Context Precision recursiva |
| --- | ---: | ---: | ---: | ---: |
| Q1 · Definición | 0/1 | 0/1 | 0 | 0 |
| Q2 · Fórmula | 0/1 | 1/1 | 0 | 0,50 |
| Q3 · Magnitud | 0/1 | 1/1 | 1,00 | 0,75 |
| Q4 · Los tres temas | 0/3 | 0/3 | 1,00 | 0,33 |
| Promedio por consulta | 0 | 0,50 | 0,50 | 0,40 |

Por suma de criterios, el fijo cubrió 0/6 y el recursivo 2/6. Las posiciones relevantes en orden Q1–Q4 fueron `[]`, `[]`, `[1]`, `[1]` para fijo y `[]`, `[2]`, `[1, 4]`, `[3]` para recursivo. Precision@4 promedio fue 0,125 y 0,25. Context Precision mide la posición de los aciertos, por lo que un único fragmento relevante en el primer puesto puede dar 1 aunque el contexto deje el criterio incompleto. [Clase de métricas, pp. 9–15](../../docs/RAG/Clase%2011%20-%20M%C3%A9tricas%20de%20la%20Industria%20(1).pdf)

- **Q2.** El recursivo recuperó `c075` en el puesto 2, con la fórmula del bloque 222 (31:55.350–32:14.820). La fórmula también quedó completa en `c059` de la colección fija, fuera de los primeros cuatro resultados. El corte cambió el pasaje que acompañaba a la fórmula y su posición en la búsqueda.
- **Q3.** El recursivo recuperó `c060` y `c061`. Entre ambos aparece el efecto de la magnitud y el ejemplo de vectores con la misma dirección (bloques 182–186). En la colección fija, `c047` aporta la idea general, pero corta antes del ejemplo.
- **Q1 y Q4.** Los pasajes sobre conteo de palabras sirven como antecedente y no aportan la definición esperada de embedding. En Q4 ambos métodos recuperaron evidencia parcial sobre magnitud.

**Decisión.** El recursivo pasa como candidato a la siguiente comparación porque aumentó el recall, la métrica principal. Las ventanas fijas quedan como control. Una sola clase y cuatro consultas alcanzan para detectar estos fallos. Para elegir el splitter y después variar el tamaño, falta repetir el contraste con más objetivos y unidades.

**Reproducción.** Instalar las versiones de [requirements.txt](requirements.txt), preparar el JSON con el comando anterior y usar una copia local de `sentence-transformers/all-MiniLM-L6-v2`, revisión `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. La carpeta de salida debe ser nueva. El [ranking local](data/experimentos/i1/ranking.json) guarda texto, offsets, tiempos, distancias, versiones y hashes de los insumos. La misma carpeta conserva las dos colecciones de Chroma.

```bash
python3 -m pip install -r tp3/ejercicio2/requirements.txt
python3 tp3/ejercicio2/comparar_splitters.py \
  --modelo-local "/ruta/a/la/copia/local/de/MiniLM" \
  --salida tp3/ejercicio2/data/experimentos/i1_repeticion
```

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
- **Resultados:** se construyeron dos colecciones persistentes independientes, con una entrada por chunk. Ambas usaron los mismos embeddings para las cuatro consultas y devolvieron cuatro vecinos por distancia coseno en cada consulta.
- **Decisión:** usar esta búsqueda básica como control de I1 y de la próxima repetición con más casos.

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
