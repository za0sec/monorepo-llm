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

#### I1 · Repetición con tres unidades · 14/11/2026

Se sumaron 754 bloques de Transformers y 829 de fine-tuning a los 415 de embeddings. Cada variante corta cada documento por separado, conserva la fuente y los tiempos, y busca en una colección con las tres clases. Se mantuvieron MiniLM en CPU, Chroma, coseno, el límite de 160 tokens, solapamiento cero y `k=4`. El único cambio entre variantes fue el splitter.

La referencia se escribió antes de correr la búsqueda. Conserva las cuatro consultas del piloto y agrega seis sobre la máscara causal, query/key/value, ajuste de pesos y destilación. Hay 14 instancias de criterios, 8 en las consultas nuevas. El asistente revisó los candidatos agrupados por consulta sin etiqueta de variante ni posición. Las anotaciones quedan abiertas a revisión humana. [Consultas y criterios](consultas_i1_ampliado.json) · [Transformers, 24:52–25:31 y 41:29–42:32](../../docs/transformers.VTT) · [Fine-tuning, 43:12–43:47 y 55:06–55:29](../../docs/finetuning.VTT)

| Resultado | Ventanas fijas | Corte recursivo |
| --- | ---: | ---: |
| Chunks | 508 | 623 |
| Criterios completos, todas las consultas | 1/14 | 3/14 |
| Criterios completos, solo consultas nuevas | 1/8 | 2/8 |
| Context Recall promedio por consulta | 0,05 | 0,25 |
| Context Precision promedio por consulta | 0,358 | 0,433 |
| Precision@4 promedio por consulta | 0,175 | 0,225 |
| Tokens recuperados en promedio por consulta | 640 | 495,9 |

| Consulta | Puestos relevantes, fijo | Puestos relevantes, recursivo | Recall fijo | Recall recursivo |
| --- | --- | --- | ---: | ---: |
| Q1 · Definición de embedding | — | — | 0/1 | 0/1 |
| Q2 · Fórmula del coseno | — | 3 | 0/1 | 1/1 |
| Q3 · Magnitud | 1 | 1 | 0/1 | 0/1 |
| Q4 · Objetivo de embeddings | 3 | — | 0/3 | 0/3 |
| Q5 · Máscara causal | — | — | 0/1 | 0/1 |
| Q6 · Query, key y value | — | — | 0/1 | 0/1 |
| Q7 · Objetivo de Transformers | 3 | 4 | 1/2 | 0/2 |
| Q8 · Ajuste de pesos | 3, 4 | 1, 2, 4 | 0/1 | 1/1 |
| Q9 · Destilación | 1 | 1 | 0/1 | 0/1 |
| Q10 · Objetivo de fine-tuning | 2 | 1, 3 | 0/2 | 1/2 |

- **Coseno.** La fórmula se mantuvo en el top 4 del recursivo, ahora en el puesto 3. El ejemplo que completaba el criterio de magnitud quedó fuera del top 4 en el corpus ampliado.
- **Transformers.** Para Q7, el fijo recuperó los bloques 205–207 y explicó por qué la máscara impide mirar tokens futuros. El recursivo recuperó solo una parte de esa explicación y ninguno recuperó el papel de query, key y value.
- **Fine-tuning.** El recursivo recuperó los bloques 310–311 en Q8 y Q10, con la explicación de reutilizar y ajustar todas o parte de las capas. Los dos aciertos corresponden al mismo criterio en dos consultas. En destilación, ambos recuperaron menciones útiles a entrenar un modelo con salidas de otro. Faltó la explicación de la relación profesor–estudiante y del menor tamaño típico del estudiante.

**Verificación.** Se reabrieron las colecciones y se comprobó que los chunks reconstruyen cada VTT preparado y conservan su fuente, tiempos y límite de tokens. Al construir las dos colecciones desde cero por segunda vez, coincidieron los chunks y 19 de los 20 rankings. Para Q7, el índice fijo dejó fuera el pasaje de la máscara causal en la segunda ejecución. En esa corrida el fijo cubrió 0/14 y el recursivo mantuvo 3/14. La búsqueda aproximada mostró variación entre reconstrucciones, incluso con la misma configuración. [Ranking de la segunda corrida](data/experimentos/i1_ampliado_repeticion/ranking.json)

**Decisión.** Usar el recursivo como punto de partida para I2, la comparación de tamaño de chunk. Recuperó más criterios y usó menos tokens con el mismo `k`. El fijo logró un acierto propio en Q7 en una de las corridas y el recall absoluto sigue siendo bajo.

**Reproducción.** Usar el entorno y la revisión local de MiniLM de I1. La corrida deja el [ranking ampliado](data/experimentos/i1_ampliado/ranking.json) y sus colecciones en una carpeta nueva. Cada chunk guarda su fuente, sus offsets en ese documento, los IDs de los bloques y los tiempos. El ranking del piloto original queda disponible en su carpeta.

```bash
base=tp3/ejercicio2
python3 "$base/preparar_transcripcion.py" docs/embeddings_2.VTT
python3 "$base/preparar_transcripcion.py" docs/transformers.VTT
python3 "$base/preparar_transcripcion.py" docs/finetuning.VTT
python3 "$base/comparar_splitters.py" \
  --modelo-local "/ruta/a/la/copia/local/de/MiniLM" \
  --corpus "$base/data/procesados/embeddings_2.json" \
  --corpus "$base/data/procesados/transformers.json" \
  --corpus "$base/data/procesados/finetuning.json" \
  --consultas "$base/consultas_i1_ampliado.json" \
  --salida "$base/data/experimentos/i1_ampliado_nueva_corrida"
```

### I2 · Tamaño de chunk

**Prueba de desarrollo · 14/11/2026.** Se compararon máximos de 96, 160 y 224 tokens. El control fue el recursivo de 160 tokens de I1. Se mantuvieron las tres transcripciones, las diez consultas y sus criterios, MiniLM en CPU, Chroma, coseno, `k=4` y solapamiento cero. La hipótesis era que un tamaño mayor preservaría explicaciones partidas, a cambio de traer más texto por consulta. Se comprobó que los tres máximos más los tokens especiales caben en el encoder. [Consultas y criterios](consultas_i1_ampliado.json) · [Clase de RAG, pp. 31–33](../../docs/RAG/Clase%207.1%20-%20RAG.pdf)

| Resultado | 96 | 160 (control) | 224 |
| --- | ---: | ---: | ---: |
| Chunks | 1.162 | 623 | 417 |
| Mediana de tokens por chunk | 74 | 136 | 201 |
| Context Recall promedio | 0% | **25,0%** | 23,3% |
| Context Precision promedio | 26,4% | 43,3% | **60,8%** |
| Precision@4 promedio | 22,5% | 22,5% | 22,5% |
| Criterios completos | 0/14 | **3/14** | **3/14** |
| Tokens recuperados por consulta | 239,1 | 495,9 | 722,0 |

El asistente anotó los candidatos agrupados por consulta, sin mostrar el tamaño ni la posición. Las posiciones útiles cuentan para *Context Precision* aunque solo aporten una parte de un criterio. Para *Context Recall* se exige respaldo completo. La revisión humana de estas etiquetas está pendiente. La tabla da primero los puestos útiles y después la fracción de criterios completos.

| Consulta | 96 | 160 | 224 |
| --- | --- | --- | --- |
| Q1 · Definición de embedding | — · 0/1 | — · 0/1 | 1 · 1/1 |
| Q2 · Fórmula del coseno | 3 · 0/1 | 3 · 1/1 | 3 · 1/1 |
| Q3 · Magnitud | 3, 4 · 0/1 | 1 · 0/1 | 2 · 0/1 |
| Q4 · Objetivo de embeddings | — · 0/3 | — · 0/3 | 2 · 1/3 |
| Q5 · Máscara causal | — · 0/1 | — · 0/1 | — · 0/1 |
| Q6 · Query, key y value | — · 0/1 | — · 0/1 | — · 0/1 |
| Q7 · Objetivo de Transformers | — · 0/2 | 4 · 0/2 | 1 · 0/2 |
| Q8 · Ajuste de pesos | 1, 4 · 0/1 | 1, 2, 4 · 1/1 | 1, 4 · 0/1 |
| Q9 · Destilación | 2 · 0/1 | 1 · 0/1 | 1 · 0/1 |
| Q10 · Objetivo de fine-tuning | 2, 3, 4 · 0/2 | 1, 3 · 1/2 | 1 · 0/2 |

- **96 tokens.** En Q2 recuperó solo el final del bloque 222. La mención al producto dividido por las normas quedó separada de la explicación de que se trata del producto interno. Quedó evidencia parcial, pero la fórmula completa quedó fuera de los cuatro resultados.
- **160 y 224 tokens.** Ambos recuperaron la fórmula. Con 160 entró la explicación de ajustar todas o algunas capas en Q8 y Q10 (fine-tuning, bloques 310–311). Con 224 entró un ejemplo que explica cómo un vector numérico representa una palabra en Q1 y Q4 (embeddings, bloques 130–134). Los tres aciertos de cada variante incluyen un criterio que se repite en dos consultas.
- **Presupuesto común.** Para separar recuperación de longitud de contexto se revisaron los dos finalistas con hasta 384 tokens de MiniLM por consulta, el máximo teórico de cuatro chunks de 96. Se recorrió el ranking en orden y, cuando el siguiente chunk excedía el saldo, se conservó solo su prefijo hasta el último token que cabía, sin tokens especiales. Con 160 se conservaron Q2 y el criterio de ajuste de pesos de Q10. Con 224 se conservaron Q1 y el criterio de definición de Q4. Ambos quedaron en 2/14 criterios, con recall promedio de 15,0% y 13,3%, respectivamente. [Contextos con el tope común](data/experimentos/i2/contextos_384.json)

**Verificación.** Se construyó dos veces cada índice. Coincidieron los chunks, los rankings y sus distancias en las dos corridas de cada tamaño. Se comprobaron la trazabilidad de los chunks, la reconstrucción del texto y el conteo de registros de las colecciones persistidas.

**Decisión.** Mantener 160 tokens como control. Cubrió la misma cantidad de criterios que 224 con un 31% menos de tokens por consulta. Bajo el tope común quedaron empatados en cantidad de criterios completos. El tamaño de 224 sigue siendo una alternativa para consultas que necesitan una explicación repartida en varios bloques. El número de consultas es bajo y todos los resultados son de desarrollo. Para I3, el caso de magnitud permite comprobar si el solapamiento ayuda a recuperar el ejemplo que quedó en el chunk contiguo.

**Reproducción.** Usar las mismas fuentes y la revisión de MiniLM de I1. Los seis rankings y las colecciones quedan en los subdirectorios `tamano_<máximo>_r<repetición>`. [Salidas locales de I2](data/experimentos/i2/)

```bash
base=tp3/ejercicio2
modelo="/ruta/a/la/copia/local/de/MiniLM"
for tamano in 96 160 224; do
  for repeticion in 1 2; do
    python3 "$base/comparar_splitters.py" \
      --modelo-local "$modelo" \
      --corpus "$base/data/procesados/embeddings_2.json" \
      --corpus "$base/data/procesados/transformers.json" \
      --corpus "$base/data/procesados/finetuning.json" \
      --consultas "$base/consultas_i1_ampliado.json" \
      --solo-recursivo --tamano "$tamano" \
      --salida "$base/data/experimentos/i2_nueva/tamano_${tamano}_r${repeticion}"
  done
done
```

### I3 · Solapamiento

**Prueba de desarrollo · 14/11/2026.** Se compararon 0, 10, 20 y 40 tokens de solapamiento con 0 como control. Se mantuvieron el recursivo de 160 tokens, las tres fuentes, las diez consultas de I2, MiniLM en CPU, Chroma, coseno y `k=4`. La hipótesis era que compartir texto en los bordes ayudaría a recuperar explicaciones partidas, en especial la de magnitud. [Consultas y criterios](consultas_i1_ampliado.json) · [Clase de RAG, pp. 22–24](../../docs/RAG/Clase%207.1%20-%20RAG.pdf)

El asistente evaluó los candidatos por consulta sin mostrar su variante ni su puesto. Se usó la misma pauta de relevancia y cobertura de I2. La revisión humana queda pendiente. Las métricas de búsqueda promedian las diez consultas y las dos reconstrucciones de cada índice.

| Resultado | 0 (control) | 10 | 20 | 40 |
| --- | ---: | ---: | ---: | ---: |
| Chunks | 623 | 623 | 635 | 691 |
| Context Recall | 25,0% | 25,0% | 25,0% | 25,0% |
| Context Precision | 43,8% | **45,0%** | 36,4% | 39,6% |
| Precision@4 | 22,5% | 22,5% | 25,0% | 25,0% |
| Criterios completos en cada corrida | 3/14 | 3/14 | 3/14 | 3/14 |
| Pares de chunks contiguos con solapamiento efectivo | 0/620 | 47/620 | 162/632 | 394/688 |
| Tokens compartidos por par contiguo, promedio | 0 | 0,6 | 3,6 | 14,9 |
| Tokens recuperados por consulta, promedio | 497,8 | 496,5 | 491,6 | 538,9 |
| Tokens de intervalos repetidos por consulta, promedio | 0 | 0 | 1,6 | 6,0 |

El solapamiento efectivo se midió con las intersecciones de los offsets en cada documento. Para el top 4 se contó cada aparición adicional de un tramo de la misma fuente y se tokenizó el texto compartido. Así, el conteo registra redundancia del splitter aunque el resultado venga en otro puesto y deja fuera coincidencias literales de tramos diferentes.

| Consulta con evidencia útil | Puestos útiles con 0 | Con 10 | Con 20 | Con 40 | Criterios completos con cada valor |
| --- | --- | --- | --- | --- | --- |
| Q2 · Fórmula del coseno | 3 | 2 | 4 | 3 | 1/1 |
| Q3 · Magnitud | 1 | 1 | 2, 4 | 2, 3 | 0/1 |
| Q7 · Objetivo de Transformers | 3 en primera, 4 en segunda | 4 | 3 | 3 en primera, 4 en segunda | 0/2 |
| Q8 · Ajuste de pesos | 1, 2, 4 | 1, 2, 4 | 1, 3, 4 | 1, 2, 3 | 1/1 |
| Q9 · Destilación | 1 | 1 | 1 | 1 | 0/1 |
| Q10 · Objetivo de fine-tuning | 1, 3 | 1, 3 | 1, 4 | 1, 4 | 1/2 |

Q1, Q4, Q5 y Q6 quedaron sin evidencia útil con los cuatro valores. En Q3, 20 y 40 recuperaron una parte más del ejemplo de magnitud (bloques 186–187). El bloque 185, que compara 100 y 100 frente a 1 y 1, quedó fuera de los cuatro resultados. Con 10, el chunk de la fórmula del coseno conservó el mismo contenido del control y subió del puesto 3 al 2. [Embeddings VTT, 25:37–27:13 y 31:55–32:14](../../docs/embeddings_2.VTT)

**Verificación.** Se construyó dos veces cada índice y se verificó la cobertura del texto usando los offsets, el tamaño de los chunks y el conteo de las colecciones reabiertas. Coincidieron 38 de los 40 rankings. El índice de 0 y el de 40 cambiaron en Q7 entre las dos corridas, sin cambiar los criterios completos recuperados. La segunda corrida de 0 reprodujo los rankings de I2.

**Decisión revisada.** En la primera lectura elegimos 0 por ser el menor solapamiento. Para comparar embeddings vamos a usar el recursivo de **160 tokens con 10 de solapamiento**. Ambas opciones cubrieron los mismos 3 de 14 criterios, generaron 623 chunks y devolvieron cero tokens repetidos en el top 4. Con ese empate, usamos Context Precision para decidir. Subió de 43,8% a 45,0% porque la fórmula del coseno pasó del tercer al segundo puesto. La ventaja viene de una consulta y la decisión se aplica a este conjunto de desarrollo.

**Reproducción.** Usar las fuentes y la revisión local de MiniLM de I2. Los ocho rankings guardan los offsets y los conteos de texto compartido. [Salidas locales de I3](data/experimentos/i3/)

```bash
base=tp3/ejercicio2
modelo="/ruta/a/la/copia/local/de/MiniLM"
for solape in 0 10 20 40; do
  for repeticion in 1 2; do
    python3 "$base/comparar_splitters.py" \
      --modelo-local "$modelo" \
      --corpus "$base/data/procesados/embeddings_2.json" \
      --corpus "$base/data/procesados/transformers.json" \
      --corpus "$base/data/procesados/finetuning.json" \
      --consultas "$base/consultas_i1_ampliado.json" \
      --solo-recursivo --tamano 160 --solapamiento "$solape" \
      --salida "$base/data/experimentos/i3_nueva/solape_${solape}_r${repeticion}"
  done
done
```

### I4 · Modelo de embeddings

- **Punto de partida:** mismos chunks del recursivo de 160 tokens con 10 de solapamiento en ambos brazos, con MiniLM como baseline.
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
