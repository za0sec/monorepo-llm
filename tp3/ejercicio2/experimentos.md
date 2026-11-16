# Registro experimental del ejercicio 2

Este registro documenta el corpus de partida, las comparaciones y las decisiones. El [roadmap](../roadmap_ejercicio2.md) describe el protocolo y las métricas. Al ejecutar un experimento, registrar aquí la fecha, las configuraciones comparadas, los resultados por caso, dónde quedaron las salidas y el motivo de la decisión. La [clase de RAG](../../docs/RAG/Clase%207.1%20-%20RAG.pdf), pp. 17–20, ubica la carga y el splitting dentro de la indexación.

## 1. Indexing Pipeline

### Preparación del corpus · 14/11/2026

- **Punto de partida:** [`docs/embeddings_2.VTT`](../../docs/embeddings_2.VTT), transcripción docente en español de la clase de embeddings. Su SHA-256 es `a1fa8324b6d63c7c6d90fd4dc467b65095524aba8e134d806c2118278b24cd18`.
- **Implementación:** se adaptó el lector del prototipo que dio origen al [lector activo](preparar_transcripcion.py). Une las líneas de texto de cada bloque y reduce secuencias de espacios en blanco a un espacio. Conserva las palabras, la puntuación y el hablante tal como aparecen en el VTT. Guarda la ruta y el hash de la fuente y, por bloque, el ID, el texto y el inicio y fin en formato VTT y en milisegundos.
- **Resultado:** 415 de 415 bloques coinciden con el original en ID, texto normalizado y tiempos. Abarcan de `00:00:03.410` a `01:01:48.330`. Una segunda ejecución en otro directorio produjo el mismo JSON byte a byte.
- **Decisión:** usar [`data/procesados/embeddings_2.json`](data/procesados/embeddings_2.json) como único insumo de texto del piloto. Los tiempos permiten revisar la definición de embedding (bloque 123, 18:32.360–18:50.210), el papel de la magnitud (bloques 182–188, 25:37.950–27:13.979) y el cálculo del coseno (bloque 222, 31:55.350–32:14.820). Son referencias iniciales del [caso manual](evaluacion/caso_01_embeddings.md).

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
for tamano in 96 160 224; do
  for repeticion in 1 2; do
    python3 "$base/comparar_splitters.py" \
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
for solape in 0 10 20 40; do
  for repeticion in 1 2; do
    python3 "$base/comparar_splitters.py" \
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

**Configuración · 14/11/2026.** Se congelaron el ranking de MiniLM de I3, los 623 chunks del recursivo de 160 tokens con 10 de solapamiento y las diez consultas. Cada modelo representó los mismos textos y consultas con una colección propia de Chroma, distancia coseno y `k=4`. Se evaluó si el cambio de modelo recuperaba más evidencia completa y mejoraba su posición. [Baseline local](data/experimentos/i3/solape_10_r1/ranking.json) · [Demo de Pinecone, celdas 19 y 22–23](../../docs/RAG/Langchain_Pinecone.ipynb)

El proxy de la cátedra rechazó la credencial con HTTP 401 en el catálogo y en las llamadas de embeddings con ambos identificadores de modelo, con y sin el prefijo `openai/`. Para esas comprobaciones se usaron frases de prueba. El grupo confirmó que la misma clave responde por la API directa de OpenAI en `prueba.py` con un modelo de chat.

El [script de comparación](comparar_embeddings.py) reutiliza los chunks de I3, registra el hash de la baseline y comprueba que el archivo de consultas coincida con el usado en esa corrida. Guarda los dos rankings, la colección nueva y los tokens y costo de OpenAI cuando el proxy los informa. La ejecución local con respuestas simuladas verificó el orden de las 633 entradas, los 20 lotes, los diez rankings y la persistencia de la colección.

La API directa de OpenAI se probó inicialmente con transporte simulado. Tras la autorización explícita para enviar los fragmentos y consultas a esa API, se completó la corrida real con `text-embedding-3-small`. La API informó **73.705 tokens de entrada** en 20 lotes y devolvió vectores de 1.536 dimensiones. A la tarifa estándar de US$0,02 por millón, el costo calculado es **US$0,0014741**. Las llamadas sumaron 27,4 segundos. El tiempo corresponde a OpenAI, incluida la red. [Tarifa oficial](https://developers.openai.com/api/docs/models/text-embedding-3-small)

| Resultado | MiniLM | `text-embedding-3-small` |
| --- | ---: | ---: |
| Context Recall promedio | 25,0% | **53,3%** |
| Context Precision promedio | 45,0% | **62,2%** |
| Precision@4 promedio | 22,5% | **37,5%** |
| Criterios completos | 3/14 | **7/14** |
| Tokens recuperados por consulta, medidos con el tokenizer de MiniLM | 496,5 | 549,8 |

El asistente revisó los candidatos agrupados por consulta sin etiqueta de modelo ni puesto. La evaluación está pendiente de revisión humana. El cálculo usa los puestos útiles para precisión y el respaldo completo entre los cuatro chunks para recall.

| Consulta | Puestos útiles de MiniLM | Puestos útiles de OpenAI | Recall MiniLM | Recall OpenAI |
| --- | --- | --- | ---: | ---: |
| Q1 · Definición de embedding | — | 3 | 0/1 | 1/1 |
| Q2 · Fórmula del coseno | 2 | 1 | 1/1 | 1/1 |
| Q3 · Magnitud | 1 | 1 | 0/1 | 0/1 |
| Q4 · Objetivo de embeddings | — | 2, 3 | 0/3 | 1/3 |
| Q5 · Máscara causal | — | — | 0/1 | 0/1 |
| Q6 · Query, key y value | — | — | 0/1 | 0/1 |
| Q7 · Objetivo de Transformers | 4 | 1, 3, 4 | 0/2 | 1/2 |
| Q8 · Ajuste de pesos | 1, 2, 4 | 1, 2, 3 | 1/1 | 1/1 |
| Q9 · Destilación | 1 | 2 | 0/1 | 1/1 |
| Q10 · Objetivo de fine-tuning | 1, 3 | 1, 2, 3 | 1/2 | 1/2 |

- OpenAI recuperó la definición de embedding en un pasaje de Transformers (Q1, `c219`, 35:03–35:22) y la fórmula del coseno también en la consulta combinada (Q4, `c075`, 31:55–32:22).
- En Q7 recuperó por qué el decoder enmascara los tokens futuros (`c198`, 25:08–25:37). Quedó incompleta la función de los values. Para la pregunta específica sobre la máscara, Q5, los dos modelos siguieron sin encontrar ese pasaje.
- En Q9 recuperó la explicación profesor–estudiante (`c464`, 55:05–55:37). Los dos modelos siguieron sin completar la explicación de magnitud de Q3.

**Verificación.** Coincidieron los 623 textos y sus referencias con la baseline congelada. Se comprobó el hash de las consultas, el conteo de 633 entradas a la API y que la colección de OpenAI se puede reabrir con los 623 documentos y distancia coseno. [Resultados locales de I4](data/experimentos/i4/ranking.json)

**Decisión.** Usar `text-embedding-3-small` como referencia de desarrollo para comparar `k`. Mejoró cobertura y posición con el mismo número de resultados. MiniLM queda como comparación local y la decisión se limita a las diez consultas de desarrollo.

**Reproducción.** Usar las dependencias de [requirements.txt](requirements.txt), una clave válida para la API directa en `tp3/.env` y un directorio nuevo.

```bash
python3 tp3/ejercicio2/comparar_embeddings.py --openai-directo \
  --salida tp3/ejercicio2/data/experimentos/i4_nueva_corrida
```

## 2. Retrieval

### R0 · Búsqueda básica para medir I1

- **Punto de partida:** objetivo literal y un índice académico por variante de splitting.
- **Qué evaluar:** verificar que una consulta use el mismo modelo de embeddings que su índice y devuelva hasta cuatro resultados ordenados por distancia coseno, con IDs y tiempos de origen.
- **Resultados:** se construyeron dos colecciones persistentes independientes, con una entrada por chunk. Ambas usaron los mismos embeddings para las cuatro consultas y devolvieron cuatro vecinos por distancia coseno en cada consulta.
- **Decisión:** usar esta búsqueda básica como control de I1 y de la próxima repetición con más casos.

### R1 · Cantidad de resultados

**Prueba de desarrollo · 14/11/2026.** Se reutilizó la colección `openai_i4` de Chroma con los 623 chunks del recursivo de 160 tokens, 10 de solapamiento y embeddings de `text-embedding-3-small` de 1.536 dimensiones. Se generaron únicamente los vectores de las diez consultas. Para cada consulta se guardó un ranking por coseno de ocho resultados y se evaluaron sus primeros 2, 4 y 8. La hipótesis era que con más resultados podía entrar evidencia completa para las consultas que fallaban en I4.

| Resultado | `k=2` | `k=4` (baseline) | `k=8` |
| --- | ---: | ---: | ---: |
| Context Recall promedio | 43,3% | 53,3% | **66,7%** |
| Context Precision promedio | 60,0% | **62,2%** | 60,6% |
| Precision@k promedio | **45,0%** | 37,5% | 31,3% |
| Criterios completos | 6/14 | 7/14 | **9/14** |
| Tokens recuperados por consulta, promedio | 266,6 | 549,8 | 1.067,7 |

La Context Precision pondera los puestos relevantes. Precision@k mide la proporción de resultados relevantes entre los devueltos. El asistente mantuvo la pauta de I4 y revisó el texto de los candidatos adicionales por consulta, sin mostrar su posición. La anotación sigue pendiente de revisión humana. Los tokens recuperados se midieron con el tokenizer de MiniLM, igual que en I4. [Clase de métricas, pp. 9–15](../../docs/RAG/Clase%2011%20-%20M%C3%A9tricas%20de%20la%20Industria%20(1).pdf)

| Consulta | Puestos útiles en el ranking de ocho | Recall `k=2` | Recall `k=4` | Recall `k=8` |
| --- | --- | ---: | ---: | ---: |
| Q1 · Definición de embedding | 3 | 0/1 | 1/1 | 1/1 |
| Q2 · Fórmula del coseno | 1 | 1/1 | 1/1 | 1/1 |
| Q3 · Magnitud | 1, 7 | 0/1 | 0/1 | 0/1 |
| Q4 · Objetivo de embeddings | 2, 3, 7 | 1/3 | 1/3 | 2/3 |
| Q5 · Máscara causal | 5 | 0/1 | 0/1 | 1/1 |
| Q6 · Query, key y value | 6, 7 | 0/1 | 0/1 | 0/1 |
| Q7 · Objetivo de Transformers | 1, 3, 4, 5, 6, 7 | 1/2 | 1/2 | 1/2 |
| Q8 · Ajuste de pesos | 1, 2, 3, 7 | 1/1 | 1/1 | 1/1 |
| Q9 · Destilación | 2 | 1/1 | 1/1 | 1/1 |
| Q10 · Objetivo de fine-tuning | 1, 2, 3, 6 | 1/2 | 1/2 | 1/2 |

En Q1 el puesto 3 contiene la definición que se pierde con `k=2`. En Q5 el puesto 5 aporta la explicación de la máscara causal (`c198`, Transformers, 25:08–25:37). En Q4 el puesto 7 agrega la definición de embedding (`c219`, Transformers, 35:03–35:22). Q6 incorpora evidencia para query y key, pero sigue sin explicar para qué se usan los values al construir el contexto. [Transcripción de Transformers](../../docs/transformers.VTT)

**Decisión.** De los tres valores probados, `k=8` cubre más criterios. Como los dos aciertos nuevos aparecen en los puestos 5 y 7, se inspeccionaron esos prefijos del ranking guardado. Con `k=5` y `k=6` quedaron 8/14 criterios. Con **`k=7`** quedan los mismos 9/14 criterios, 66,7% de recall y 60,6% de Context Precision que con 8, con **943,3 tokens promedio**. Se usa 7 como punto de partida para la generación en desarrollo. Esta elección se revisará en el conjunto reservado y parte de una inspección posterior del ranking.

**Verificación.** Los diez primeros cuatro resultados coincidieron con los de I4 en el mismo orden. Al reabrir Chroma y reutilizar los vectores de consulta guardados se reprodujeron los diez rankings de ocho resultados, incluidas sus distancias. La API informó 202 tokens de entrada en un solo lote de consultas. [Ranking de R1](data/experimentos/r1/ranking.json)

**Reproducción.** El archivo de salida guarda los vectores de consulta para repetir las búsquedas localmente. El comando genera nuevamente los vectores de las diez consultas con la clave de la API directa y usa la colección existente de I4.

```bash
python3 tp3/ejercicio2/comparar_k.py \
  --salida tp3/ejercicio2/data/experimentos/r1_nueva_corrida/ranking.json
```

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

### Control previo de la transcripción · 14/11/2026

Se revisó el VTT del Caso 01 contra seis recortes con referencia humana verificada. La estimación del VTT fue de **15,07% de WER y 12,80% de CER** al seleccionar segmentos por punto medio. Como el VTT tiene tiempos por segmento, el resultado depende de qué texto se incluye en los bordes. Las dos corridas de G0 usarán el mismo VTT. La comparación de transcriptores sobre WAV idénticos sigue aparte. [Registro de audio](transcripcion/README.md#control-del-vtt-y-seis-recortes--14112026)

En la comparación sobre los seis WAV idénticos, Whisper obtuvo **5,87% de WER y 3,92% de CER** frente a **14,09% y 3,07%** de Wav2Vec2. Este contraste evalúa la transcripción sobre los recortes. La evaluación de G0 medirá el informe con el VTT completo como entrada. [Comparación sobre los mismos seis WAV](transcripcion/README.md#comparación-sobre-los-mismos-seis-wav)

### G0 · Informe básico

- **Punto de partida:** objetivo, VTT fijo del video y Qwen2.5-1.5B-Instruct como candidato inicial, con un prompt directo.
- **Qué evaluar:** generar primero con referencias de clase seleccionadas a mano y luego con la salida de la búsqueda. Medir acierto de cobertura, fidelidad, validez de citas y relevancia.
- **Resultados:** pendientes.
- **Decisión:** pendiente.

### G1 · Comparación de modelos con chunks manuales

**Preparación · 15/11/2026.** El objetivo es «entender qué son los embeddings y cómo se compara la similitud entre textos», a nivel introductorio. Se reutiliza el VTT completo de Whisper de *¿Qué son los EMBEDDINGS Grandes Modelos de Lenguaje?* (99 segmentos). El [prompt](g1/prompt.md) pide una tabla Markdown, recomendaciones y una conclusión. El [caso](g1/caso.json) fija el objetivo, los modelos, el presupuesto y el orden de los chunks. Los resultados de G0 quedan como antecedente, porque G1 usa una selección de clase diferente.

**Control manual.** Se eligieron por ID siete documentos completos de la colección `openai_i4`. La preparación obtuvo sus textos y metadatos directamente de Chroma y verificó que coinciden con la copia de I4. Los dos modelos recibirán los mismos siete chunks y los mismos 99 segmentos en el mismo orden. En esta etapa la selección humana reemplaza la búsqueda por similitud.

| En el prompt | ID en Chroma | Fuente y bloques VTT | Intervalo | Evidencia para la pauta |
| --- | --- | --- | --- | --- |
| C1 | `c219` | Transformers, 296–297 | 35:03.390–35:22.090 | E1 · Representación semántica como vector numérico |
| C2 | `c066` | Embeddings, 192–194 | 27:57.050–28:17.709 | E2 · Embedding estático, ejemplo de «pizza» |
| C3 | `c067` | Embeddings, 195–199 | 28:17.970–28:50.049 | E2 · Cómo cambia el embedding de «pizza» con el contexto |
| C4 | `c355` | Fine-tuning, 56–58 | 07:40.540–08:12.510 | E3 · Ángulo como criterio al comparar representaciones |
| C5 | `c075` | Embeddings, 222–223 | 31:55.350–32:22.889 | E4 · Producto interno y normalización por las normas |
| C6 | `c060` | Embeddings, 182–184 | 25:37.950–26:10.769 | E5 · Cuándo importa la magnitud |
| C7 | `c061` | Embeddings, 185–186 | 26:10.830–26:47.999 | E5 · Ejemplo de la misma dirección con distinta magnitud |

Los chunks suman 934 tokens medidos por MiniLM durante I4. En inferencia se registrará el conteo de cada generador. La tabla anterior y la [pauta E1–E5](evaluacion/caso_01_embeddings.md#criterios-de-cobertura) se reservan para corregir. El [JSON congelado](data/experimentos/g1/entrada_manual.json) contiene los textos exactos, los dos mensajes y las huellas del VTT y del registro I4. Es un artefacto local ignorado por Git.

**Qué se va a evaluar.** Comparar Qwen2.5‑1.5B‑Instruct local en CPU/float32, sin muestreo y con penalización 1,1, contra GPT‑4.1 (snapshot configurado `gpt-4.1-2025-04-14`, temperatura 0). Ambos tienen un máximo de 1.000 tokens de salida. Registrar revisión efectiva del modelo, tokens de entrada y salida, latencia y motivo de fin. Los parámetros de decodificación dependen de cada backend. Las dos corridas deben registrar el mismo SHA-256 de `entrada_manual.json`.

| Métrica | Regla para G1 |
| --- | --- |
| Criterios presentes | Criterios comparables que aparecieron / 5. Registrar filas fuera de alcance y duplicados. |
| Acierto de estados | Criterios con estado correcto / 5. Omitidos o contradictorios cuentan como error. |
| Citas localizables y citas con respaldo | IDs existentes / citas emitidas, y citas que sostienen la fila / citas emitidas. Distinguir clase de video. Sin citas, registrar N/A y la omisión. |
| Answer Faithfulness | Afirmaciones descriptivas distintas respaldadas / afirmaciones descriptivas evaluables. Revisar cada afirmación contra la fuente correspondiente. |
| Relevancia manual | Sumar foco, pertinencia de las recomendaciones y utilidad de la conclusión según la rúbrica del roadmap. Cada eje vale 0–2. |
| Completitud | Tabla, «Para estudiar» y conclusión presentes y sin truncamiento. |

**Ejecución prevista.** El [script](generar_g1.py) tiene una acción local `preparar` para regenerar la entrada en otra carpeta y dos acciones de inferencia, `qwen` y `gpt41`. Estas últimas guardarán `informe_crudo.md`, `informe.md` con los tiempos de las citas y `corrida.json` en la carpeta de cada modelo. La configuración de GPT‑4.1 apunta a la API directa de OpenAI, con la clave local de `tp3/.env`.

**Resultados · 15/11/2026.** Se ejecutó una corrida por modelo con el mismo SHA-256 de la entrada congelada. La API directa recibió solo los mensajes preparados y confirmó el snapshot `gpt-4.1-2025-04-14`. Los informes automáticos se auditaron manualmente con los siete chunks y el VTT.

| Métrica | Qwen2.5‑1.5B‑Instruct | GPT‑4.1 |
| --- | ---: | ---: |
| Criterios presentes | 2/5 | 3/5 |
| Estados correctos | 2/5 | 2/5 |
| Citas de clase localizables | 7/20 | 16/16 |
| Citas de clase con respaldo | 0/20 | 14/16 |
| Citas del video localizables | N/A (sin citas) | 14/14 |
| Citas del video con respaldo | N/A (sin citas) | 11/14 |
| Answer Faithfulness descriptiva | 11/11 | 11/14 |
| Relevancia manual del informe | 1/6 | 3/6 |
| Informe completo | No | Sí |
| Tokens de entrada / salida | 4.052 / 1.000 | 3.499 / 518 |
| Motivo de fin | Límite (`length`) | Cierre normal (`stop`) |
| Tiempo de generación | 99,0 s (CPU) | 5,6 s (API, incluido traslado) |

| Criterio | Pauta manual | Qwen | GPT‑4.1 |
| --- | --- | --- | --- |
| E1 · Definición | Cubierto | Cubierto | Cubierto |
| E2 · Papel del contexto | Cubierto | Cubierto | Parcial |
| E3 · Comparación vectorial | Parcial | Omitido | Parcial |
| E4 · Cálculo del coseno | Sin evidencia en el video | Omitido | Mencionado dentro de E3 y en la recomendación, sin estado propio |
| E5 · Efecto de la magnitud | Sin evidencia en el video | Omitido | Omitido |

Se considera presente un criterio cuando tiene una fila con una evaluación identificable. Los estados se corrigen por separado de la calidad de sus citas. Qwen escribió 20 filas con seis contenidos distintos y omitió las citas del video y la justificación en todas. Inventó los IDs `[C8]` a `[C20]`. GPT‑4.1 escribió cinco filas, con dos sobre la definición y propiedades de los embeddings y una sobre aplicaciones que se aparta del objetivo. Marcó E2 como parcial pese a que el video usa el ejemplo de «banco» y explica el contraste en `[V31]` a `[V37]`. Presentó dos recomendaciones con seis pasajes en total, aunque el prompt pedía hasta dos pasajes.

Para auditar citas se cuenta cada ID emitido y se comprueba si su segmento aporta evidencia a la fila o recomendación. En los dos rangos de GPT‑4.1 se contaron solo los extremos escritos. Los rangos incumplen el formato pedido. Los tres IDs de video sin respaldo específico fueron `[V86]` en la comparación semántica y `[V63]` y `[V80]` en la lista de aplicaciones. Los dos IDs de clase sin respaldo fueron `[C4]` y `[C5]` en esa misma lista.

En Answer Faithfulness se deduplicaron las afirmaciones descriptivas y se evaluó si las fuentes recibidas las respaldan. Las once afirmaciones de Qwen tenían respaldo en la transcripción. Eso incluye descripciones de aplicaciones que repitió y dejó sin citar. En GPT‑4.1 quedaron tres afirmaciones sin respaldo: que el video carece de un ejemplo del papel del contexto, que no explica cómo cambia la representación según el contexto y que la lista de aplicaciones está respaldada por los pasajes de clase. La revisión contabiliza las omisiones, los estados y las citas en sus propias métricas.

**Relevancia manual.** Se aplicó a los dos informes guardados la adaptación de Answer Relevance del [roadmap](../roadmap_ejercicio2.md#3-generator-convertir-las-dos-fuentes-en-un-informe). Cada eje vale 0 si está ausente o desviado, 1 si es parcial y 2 si es adecuado.

| Eje | Qwen | GPT‑4.1 |
| --- | --- | --- |
| Foco en el objetivo y el nivel | 1/2. Presenta la definición, pero se desvía hacia imágenes y aplicaciones y omite la comparación de textos. | 1/2. Trata la definición y la comparación, pero dedica una fila a aplicaciones y deja sin evaluar el cálculo y la magnitud. |
| Recomendaciones pertinentes | 0/2. Sección ausente. | 1/2. Recomienda pasajes pertinentes para estudiar el coseno, pero también propone cubrir una supuesta falta de ejemplos de contexto. |
| Conclusión útil | 0/2. Sección ausente. | 1/2. Señala que el video sirve de introducción y que faltan métricas, pero señala el ejemplo de contexto como faltante aunque el video lo incluye. |

**Decisión.** Elegir GPT‑4.1 para G2. Empató en acierto de estados, pero completó el informe y 25 de sus 30 citas aportaron respaldo a la fila o recomendación. E2, E4 y E5 quedan como fallos a seguir en G2. La selección es para este piloto de un solo caso.

[Informe de Qwen](data/experimentos/g1/qwen/informe.md) · [Informe de GPT‑4.1](data/experimentos/g1/gpt41/informe.md)

La [pauta manual de corrección](evaluacion/caso_01_embeddings.md) se incorporó al ejercicio 2 para revisar los criterios junto con G1 y reutilizarlos en G2. Se conservó la tabla E1–E5 del caso original.

### G2 · Modelo elegido con chunks recuperados

**Configuración · 15/11/2026.** Se comparó el control manual de G1 contra una corrida con los siete chunks que devolvió Chroma para el objetivo literal. Se mantuvieron el prompt, el objetivo, el nivel, el VTT de 99 segmentos y el snapshot `gpt-4.1-2025-04-14`, con temperatura 0 y máximo de 1.000 tokens de salida. Se verificó que en el mensaje al generador solo cambió el bloque de clase. La búsqueda usó `openai_i4`, distancia coseno y una llamada a `text-embedding-3-small` para vectorizar solo el objetivo (18 tokens de entrada). No se volvió a indexar el corpus.

| Puesto | Chunk | Fuente | Distancia coseno |
| ---: | --- | --- | ---: |
| 1 | `c357` | Fine-tuning | 0,4083 |
| 2 | `c434` | Fine-tuning | 0,4320 |
| 3 | `c219` | Transformers | 0,5005 |
| 4 | `c360` | Fine-tuning | 0,5109 |
| 5 | `c437` | Fine-tuning | 0,5143 |
| 6 | `c049` | Embeddings | 0,5177 |
| 7 | `c048` | Embeddings | 0,5204 |

**Respaldo académico revisado antes de generar.** La pauta manual cubre los cinco criterios con los pasajes de G1. En G2, el ranking respalda por completo **2/5**. E1 aparece en `c219`. E3 queda respaldado al combinar la comparación de consulta y elementos de `c357` con el ángulo como criterio en `c048` y `c049`. El apoyo a E2 es parcial porque hay pistas sobre cómo se obtiene un embedding con transformers, pero falta el contraste entre embeddings fijos y contextuales. El apoyo a E5 es parcial porque `c049` muestra un ángulo pequeño a pesar de distinta magnitud y carece del ejemplo de igual similitud ante un cambio de escala. E4 carece de respaldo en los siete resultados porque ninguno contiene la fórmula del coseno. Este 2/5 mide **Context Recall de este objetivo**; se contabiliza aparte del acierto de estados del informe.

| Métrica | G1 · Manual | G2 · Recuperado |
| --- | ---: | ---: |
| Criterios con respaldo académico completo | 5/5 | 2/5 |
| Criterios presentes en el informe | 3/5 | 3/5 |
| Estados que coinciden con la pauta del video | 2/5 | 2/5 |
| Citas localizables | 30/30 | 39/39 |
| Citas con respaldo | 25/30 | 27/39 |
| Answer Faithfulness descriptiva | 11/14 | 11/12 |
| Relevancia manual | 3/6 | 4/6 |
| Informe completo | Sí | Sí |
| Tokens de clase con MiniLM | 934 | 904 |
| Tokens de entrada / salida de GPT‑4.1 | 3.499 / 518 | 3.472 / 550 |
| Tiempo de generación vía API | 5,6 s | 5,4 s |

| Criterio | Resultado con los chunks recuperados |
| --- | --- |
| E1 · Definición | Cubierto, como en la pauta. De los tres pasajes citados, `c219` aporta la definición. |
| E2 · Papel del contexto | Cubierto, coincide con la revisión del video. Los dos pasajes citados no establecen el contraste y el informe no avisó que la clase recuperada aportaba respaldo parcial. |
| E3 · Comparación vectorial | Marcó «Cubierto» para comparar textos y «Parcial» para las medidas. La pauta reúne el procedimiento y la medida en el mismo criterio. Se registró la evaluación como ambigua y se contó como error de estado. |
| E4 · Cálculo del coseno | Omitido. Los siete chunks carecen de la fórmula y el generador tampoco lo evaluó por separado con el control manual, que sí la incluía. |
| E5 · Efecto de la magnitud | Omitido en ambos informes. La variante recuperada tenía un ejemplo parcial del vínculo entre ángulo y magnitud. |

Se contaron los extremos escritos en los cinco rangos de video, igual que en G1. El formato pedía IDs individuales. **14/21** citas de clase y **13/18** del video respaldaron al menos una parte de su fila o recomendación. En la fila sobre definiciones, `C4` y `C5` mencionan cómo se obtienen o reutilizan embeddings, pero la definición viene de `C3`. En la fila de búsqueda semántica, solo `C1` describe la comparación entre consulta y elementos. El `[V19]` de la fila de similitud entre textos describe rostros. Ningún ID fue inventado.

En Answer Faithfulness se contaron doce afirmaciones descriptivas distintas. Se respaldaron la definición vectorial, la relación entre similitud numérica y semántica, el contraste entre tipos de embeddings, el papel del contexto, la búsqueda semántica, la comparación consulta–documentos, la obtención de embeddings con redes neuronales y con transformers, la remisión a otro video, la ausencia de una medida concreta en el video y el uso del ángulo en la clase. La afirmación sin respaldo fue que el material recuperado mencionaba el producto escalar. Los pasajes `C6` y `C7` solo aportan evidencia sobre el ángulo.

La relevancia manual dio **1/2 en foco** porque abordó la definición y la similitud junto con temas laterales, **1/2 en recomendaciones** porque los pasajes sobre ángulo sirven, pero la recomendación atribuye a esos pasajes el producto escalar y agrega el proceso de obtención de embeddings, y **2/2 en conclusión** porque resume para qué sirve el video y señala correctamente que faltan detalles sobre las medidas.

**Decisión.** Conservar G2 como baseline con recuperación real y registrar sus límites. Mejorar la recuperación podría aportar la fórmula y el contraste contextual. La omisión del cálculo y de la magnitud en el control de G1 también justifica probar un ajuste de instrucciones sin cambiar las fuentes.

**Reproducción.** Recuperar una vez con `.venv/bin/python tp3/ejercicio2/recuperar_g2.py`. Para generar, usar `.venv/bin/python tp3/ejercicio2/generar_g1.py gpt41 --carpeta tp3/ejercicio2/data/experimentos/g2 --entrada tp3/ejercicio2/data/experimentos/g2/entrada_recuperada.json`. Los artefactos locales guardan el ranking con distancias, el contexto exacto y los hashes. [Entrada recuperada](data/experimentos/g2/entrada_recuperada.json) · [Informe G2](data/experimentos/g2/gpt41/informe.md) · [Registro de la corrida](data/experimentos/g2/gpt41/corrida.json)

### G1.5 · Prompt con cotejo explícito de las fuentes

**Configuración · 15/11/2026.** Se corrió después de G2 para comprobar el efecto del [nuevo prompt](g1_5/prompt.md) con el contexto manual de G1. El prompt pide evaluar contenidos por separado, establecer qué enseña la clase, cotejar el video y asignar un estado. Se verificó que solo cambió el contenido del mensaje de sistema. Se mantuvieron fijos los siete chunks, el objetivo, el nivel, el VTT, el formato Markdown, `gpt-4.1-2025-04-14`, temperatura 0 y máximo de 1.000 tokens de salida. Los criterios de la pauta se usaron solo al corregir.

| Métrica | G1 · Prompt original | G1.5 · Nuevo prompt |
| --- | ---: | ---: |
| Criterios presentes en el informe | 3/5 | 4/5 |
| Estados correctos | 2/5 | 2/5 |
| Citas localizables | 30/30 | 36/36 |
| Citas de clase con respaldo | 14/16 | 13/16 |
| Citas de video con respaldo | 11/14 | 13/20 |
| Citas con respaldo, total | 25/30 | 26/36 |
| Answer Faithfulness descriptiva | 11/14 | 18/20 |
| Relevancia manual | 3/6 | 4/6 |
| Informe completo | Sí | Sí |
| Tokens de entrada / salida | 3.499 / 518 | 3.560 / 587 |
| Tiempo de generación vía API | 5,6 s | 4,3 s |

| Criterio | Pauta | G1 | G1.5 |
| --- | --- | --- | --- |
| E1 · Definición | Cubierto | Cubierto | Cubierto |
| E2 · Papel del contexto | Cubierto | Parcial | Cubierto, en dos filas complementarias |
| E3 · Cómo comparar textos mediante vectores | Parcial | Parcial | Cubierto en la fila de similitud general, aunque no indica una medida explicada en el video |
| E4 · Cálculo del coseno | Sin evidencia en el video | Mencionado sin estado propio | Parcial en la fila de coseno, aunque la explicación reconoce que el video no explica esa medida |
| E5 · Efecto de la magnitud | Sin evidencia en el video | Omitido | Omitido, con una recomendación genérica sobre ventajas y limitaciones |

**Citas y fidelidad.** Se contaron las apariciones de IDs y solo los extremos escritos de los seis rangos de video, como en G1 y G2. La fila de aplicaciones explica usos que sí aparecen en el video, pero sus tres citas de clase (C4, C5 y C6) carecen de respaldo para esa fila. Entre los IDs de video, V19 describe rostros en vez de textos. Al deduplicar afirmaciones descriptivas, las dos sin respaldo fueron atribuir a la clase la búsqueda semántica y el agrupamiento. Se contabilizó la descripción del coseno en términos del ángulo como fiel a la formulación oral de C4. La fórmula que precisa la medida está en C5. El error de estado de la fila del coseno se contabiliza aparte.

**Relevancia manual.** Foco 1/2 porque aborda el objetivo pero agrega aplicaciones y omite magnitud. Recomendaciones 1/2 porque selecciona material pertinente sobre el coseno, pero remite a cuatro pasajes y deja genérico el estudio de sus limitaciones. Conclusión 2/2 porque explica la utilidad introductoria y señala qué falta complementar.

**Decisión.** Conservar el prompt de G1 y G2 como baseline. G1.5 recuperó la evaluación contextual y dio una fila propia al coseno, pero mantuvo el acierto en 2/5 y redujo la proporción de citas con respaldo de 25/30 (83,3 %) a 26/36 (72,2 %). G2b queda condicionado a un prompt que supere el control acordado.

**Reproducción.** [Preparador](preparar_variante_prompt.py) · [Prompt](g1_5/prompt.md) · [Entrada](data/experimentos/g1_5/entrada_manual.json) · [Informe](data/experimentos/g1_5/gpt41/informe.md)

```bash
.venv/bin/python tp3/ejercicio2/preparar_variante_prompt.py tp3/ejercicio2/data/experimentos/g1/entrada_manual.json tp3/ejercicio2/g1_5/prompt.md tp3/ejercicio2/data/experimentos/g1_5/entrada_manual.json
.venv/bin/python tp3/ejercicio2/generar_g1.py gpt41 --carpeta tp3/ejercicio2/data/experimentos/g1_5
```

### Diagnóstico del top 20 de G2

Se consultó el objetivo literal de G2 en `openai_i4` con `text-embedding-3-small`. Los siete primeros chunks coincidieron con G2 en el mismo orden. La llamada de embeddings reportó 18 tokens de entrada.

- **E4 · Fórmula del coseno.** Aparece completa en el puesto **10**, `c075` de Embeddings (31:55–32:23), que distingue el producto interno de su normalización por las normas.
- **E5 · Efecto de la magnitud.** El puesto **18**, `c060` de Embeddings (25:37–26:10), explica por qué el coseno reduce la importancia de la magnitud y empieza el ejemplo. El **6**, `c049`, aporta un caso de ángulo pequeño con magnitudes distintas. El cierre del ejemplo de igual similitud con vectores escalados (`c061`) queda fuera, así que el respaldo sigue siendo parcial según la pauta.
- **E2 · Papel del contexto.** El puesto **13**, `c197` de Transformers (24:45–25:08), menciona que el encoder mira tokens a ambos lados. Los chunks que contrastan el embedding fijo con el contextual (`c066` y `c067`) quedan fuera. El respaldo sigue siendo parcial.

El conjunto de 20 respalda por completo **3/5 criterios** (E1, E3 y E4), frente a **2/5** en el top 7. Se revisó solo el ranking por coseno y queda pendiente elegir una técnica para la siguiente prueba. [Ranking y textos](data/experimentos/g2/diagnostico_top20.json)

### G3 · Advanced RAG con expansión de consulta

**Configuración.** GPT‑4.1 a temperatura 0 generó tres subconsultas a partir del objetivo literal de G2. El expander recibió solo ese objetivo y una instrucción general para dividirlo en preguntas de búsqueda. Se buscaron la original y las tres nuevas con `text-embedding-3-small` y la misma colección `openai_i4`, cinco resultados por consulta. Se intercalaron los cuatro rankings por posición, empezando por la consulta original en cada turno, y se deduplicó por ID. Se pasaron los primeros siete chunks distintos al mismo GPT‑4.1 con el prompt, objetivo y VTT de G2. Se verificaron esos controles y que el ranking de la consulta original reproduce los cinco primeros puestos de G2.

| Consulta generada | Primeros cinco IDs |
| --- | --- |
| ¿Qué son los embeddings en el contexto del procesamiento de lenguaje natural? | `c434`, `c357`, `c219`, `c436`, `c437` |
| ¿Cómo se generan y representan los embeddings de textos? | `c434`, `c219`, `c357`, `c437`, `c418` |
| ¿Cuáles son los métodos para comparar la similitud entre textos utilizando embeddings? | `c357`, `c434`, `c060`, `c075`, `c223` |

**Resultados de recuperación.** Los 20 resultados se redujeron a diez candidatos distintos. El conjunto completo respaldó **3/5** criterios (E1, E3 y E4), los mismos que el diagnóstico de los 20 candidatos por coseno. La tercera subconsulta encontró la fórmula (`c075`) en su cuarto puesto. Quedó **octava** al intercalar, fuera del contexto del generador. También encontró `c060`, que pasó al **cuarto** puesto final y explica el efecto de la magnitud, pero deja incompleto el ejemplo de E5. La primera subconsulta usa «contexto del procesamiento de lenguaje natural» en un sentido general. Ninguna de las tres buscó explícitamente el contraste entre embeddings fijos y contextuales.

El top 7 fue `c357`, `c434`, `c219`, `c060`, `c360`, `c436`, `c437`. Solo quedó plenamente respaldada la definición (E1). Para E3, `c357` explica qué se compara y `c060` nombra el coseno, pero se perdieron `c048` y `c049`, que en G2 explicaban el ángulo como criterio. Para Context Precision se marcó evidencia directa aunque fuera parcial. Los puestos relevantes fueron 1, 3, 6 y 7 en G2 y 1, 3 y 4 en G3. El aumento de Context Precision refleja la posición de los aciertos. La proporción simple de chunks pertinentes bajó de 4/7 a 3/7.

| Métrica del contexto final | G2 · Coseno | G3 · Expansión |
| --- | ---: | ---: |
| Context Recall | 2/5 (40 %) | 1/5 (20 %) |
| Context Precision | 68,5 % | 80,6 % |
| Precision@7 | 4/7 | 3/7 |
| Tokens de clase con MiniLM | 904 | 880 |

**Resultados del informe.**

| Métrica | G2 · Coseno | G3 · Expansión |
| --- | ---: | ---: |
| Criterios presentes | 3/5 | 3/5 |
| Estados correctos | 2/5 | 2/5 |
| Citas localizables | 39/39 | 34/34 |
| Citas con respaldo | 27/39 | 18/34 |
| Answer Faithfulness descriptiva | 11/12 | 12/15 |
| Relevancia manual | 4/6 | 4/6 |
| Informe completo | Sí | Sí |
| Tokens de entrada / salida del generador | 3.472 / 550 | 3.427 / 496 |

Se mantuvo la corrección de G2 por criterio. E1 y E2 coincidieron con el estado del video. E2 citó pasajes que no establecen el contraste que juzga. Se contó E3 como ambiguo porque el informe marcó como cubierta la comparación semántica y como parcial la comparación de similitud entre textos en otra fila. E4 y E5 quedaron sin estado propio.

Para las citas se contaron cada ID escrito y los extremos de los cinco rangos del video, como en G2. Respaldaron al menos parte de su fila **5/13** citas de clase y **13/21** del video. La recomendación apunta a `c060` para entender cómo se mide el coseno, aunque el texto comenta su efecto sobre la magnitud y no enseña el cálculo. Las tres afirmaciones descriptivas sin respaldo atribuyeron a los chunks de clase la equivalencia entre similitud numérica y semántica, el contraste fijo/contextual y la disponibilidad de todos los pasajes pertinentes. La relevancia manual dio 1/2 en foco por temas laterales, 1/2 en recomendaciones por identificar el tema pendiente con un pasaje incompleto y 2/2 en conclusión por precisar la utilidad del video para el objetivo.

**Decisión.** Conservar G2 como baseline. La expansión encontró evidencia útil, pero la fusión por turnos dejó afuera la fórmula y redujo la cobertura del top 7. La concurrencia de consultas parecidas redujo a la mitad la cantidad de candidatos distintos. Las llamadas adicionales reportaron 129 tokens de entrada y 58 de salida para expandir y 70 tokens para vectorizar las cuatro consultas. Este contraste corresponde a un piloto de un solo objetivo.

**Reproducción.** [Expansión y prompt](data/experimentos/g3/expansion.json) · [Rankings y contexto final](data/experimentos/g3/entrada_expandida.json) · [Informe](data/experimentos/g3/gpt41/informe.md)

```bash
.venv/bin/python tp3/ejercicio2/recuperar_g3.py
.venv/bin/python tp3/ejercicio2/generar_g1.py gpt41 --carpeta tp3/ejercicio2/data/experimentos/g3 --entrada tp3/ejercicio2/data/experimentos/g3/entrada_expandida.json
```

### G4 · Expansión con reranking de los candidatos de G3

**Configuración.** Se congelaron los diez candidatos distintos y su orden de G3. El CrossEncoder `cross-encoder/ms-marco-MiniLM-L6-v2` de la demo de cátedra (celdas 58–61) puntuó cada par del objetivo literal con el texto completo de un chunk. Se ordenaron los puntajes de mayor a menor, con el orden de G3 como desempate. Se ejecutó localmente en CPU, revisión `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Los pares midieron entre 124 y 179 tokens y entraron completos en el límite de 512. Los 0,06 s de puntuación excluyen la descarga y carga del modelo. El modelo fue entrenado para ranking en inglés, una limitación al interpretar este piloto en español. [Notebook de cátedra](../../docs/RAG/UN_Report_Advanced_RAG.ipynb) · [Ficha del modelo](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)

| Puesto G4 | Chunk | Puesto G3 | Puntaje |
| ---: | --- | ---: | ---: |
| 1 | `c357` | 1 | 4,2170 |
| 2 | `c434` | 2 | 2,6228 |
| 3 | `c075` | 8 | 1,4065 |
| 4 | `c219` | 3 | 0,8030 |
| 5 | `c360` | 5 | −0,1619 |
| 6 | `c060` | 4 | −0,3330 |
| 7 | `c418` | 9 | −2,1051 |
| 8 | `c437` | 7 | −3,7129 |
| 9 | `c223` | 10 | −7,7017 |
| 10 | `c436` | 6 | −9,0469 |

**Recuperación.** El top 7 conservó la definición (E1), recuperó la fórmula del coseno en `c075` (E4) y completó la evidencia del procedimiento de comparación (E3) al combinar ese pasaje con `c357`. El ejemplo de la magnitud (E5) quedó incompleto en `c060` y el contraste fijo/contextual (E2) sigue sin respaldo completo en el conjunto. Los puestos con evidencia directa fueron 1, 3, 4 y 6. La incorporación de un cuarto chunk pertinente en el puesto 6 eleva Precision@7, pero baja el promedio ponderado por posición de Context Precision respecto de G3.

| Métrica | G2 · Coseno | G3 · Expansión | G4 · Expansión + reranking |
| --- | ---: | ---: | ---: |
| Context Recall del top 7 | 2/5 (40 %) | 1/5 (20 %) | 3/5 (60 %) |
| Context Precision | 68,5 % | 80,6 % | 77,1 % |
| Precision@7 | 4/7 | 3/7 | 4/7 |
| Tokens de clase con MiniLM | 904 | 880 | 940 |
| Criterios presentes en el informe | 3/5 | 3/5 | 3/5 |
| Estados correctos | 2/5 | 2/5 | 3/5 |
| Citas localizables | 39/39 | 34/34 | 42/42 |
| Citas con respaldo | 27/39 | 18/34 | 29/42 |
| Answer Faithfulness descriptiva | 11/12 | 12/15 | 14/15 |
| Relevancia manual | 4/6 | 4/6 | 4/6 |
| Informe completo | Sí | Sí | Sí |
| Tokens de entrada / salida de GPT‑4.1 | 3.472 / 550 | 3.427 / 496 | 3.476 / 570 |

**Informe.** Se mantuvieron el GPT‑4.1, el prompt, el objetivo, el nivel y el VTT de G3. E1 y E2 coincidieron con el estado esperado del video. Los dos chunks de clase que cita la fila de E2 no establecen el contraste. E3 pasó a una sola fila con estado «Parcial» y el informe recomienda el pasaje que contiene la fórmula. E4 y E5 quedaron sin evaluación propia.

Se contaron las apariciones de IDs y los extremos escritos de los cinco rangos de video. Respaldaron al menos parte de su fila o recomendación **10/15** citas de clase y **19/27** de video. Por ejemplo, el pasaje sobre predicción de emojis `c418` no respalda la definición vectorial que le atribuye el informe. En Answer Faithfulness se deduplicaron afirmaciones descriptivas sobre definición, representación semántica y computacional, contexto, comparación y medidas, más los cuatro usos que aparecen en el video. La afirmación sin respaldo fue que el contexto académico ya tenía todos los pasajes pertinentes. La relevancia manual dio 1/2 en foco por las filas solapadas y laterales, 1/2 en recomendaciones porque los pasajes sobre similitud son útiles pero se afirma que la búsqueda no dejó faltantes, y 2/2 en conclusión porque identifica el uso introductorio y lo que hay que complementar.

**Decisión.** El reranking mejoró el contexto final y el acierto de estados frente a G3 y G2 en el Caso 01. G2 se conserva como control de RAG básico y G4 queda como configuración avanzada para integrar en el ejercicio 2. La validación con otros objetivos se reserva para el ejercicio 4. Los diez candidatos de G3 fijaron el techo en 3/5. El generador usó el pasaje del coseno en la recomendación y todavía lo omitió como criterio de cobertura.

**Reproducción.** Ejecutar [rerankear_g4.py](rerankear_g4.py) con `--snapshot` apuntando al snapshot local indicado, y después `generar_g1.py gpt41 --carpeta tp3/ejercicio2/data/experimentos/g4 --entrada tp3/ejercicio2/data/experimentos/g4/entrada_rerankeada.json`. [Ranking, puntajes y contexto final](data/experimentos/g4/entrada_rerankeada.json) · [Informe](data/experimentos/g4/gpt41/informe.md)

### Integración del ejercicio 2

Se agregó el [comando de evaluación](evaluar_video.py) para recibir solo el objetivo y el audio. Reutiliza Whisper, la recuperación básica de G2, la expansión y el reranking de G4 y el generador de los experimentos. El modo avanzado es el predeterminado. La ruta básica queda disponible como control. [Instrucciones de uso](README.md)

Se ejecutó el modo avanzado con el MP3 original del Caso 01. Whisper generó un VTT nuevo con **99 segmentos**. Las cuatro consultas dieron **10 candidatos distintos**, se seleccionaron **7** y entró el chunk `c075` con la fórmula. GPT‑4.1 terminó normalmente con **615 tokens de salida** y completó la tabla, las recomendaciones y la conclusión. Se verificaron las huellas del audio, el VTT y la entrada y se resolvieron las citas existentes a sus tiempos. La generación volvió a omitir un estado propio para el cálculo del coseno. Esta ejecución verifica la integración con audio real, con una expansión y una transcripción nuevas. Las cifras de calidad de G2 y G4 corresponden a sus corridas controladas.

Se comprobó con respuestas simuladas que el modo básico usa una sola consulta, que el modo avanzado completa los siete chunks si las búsquedas devuelven los mismos cinco y que los IDs de clase y video se resuelven a los tiempos correctos. La validación con objetivos y videos nuevos se reserva para el ejercicio 4.

[Transcripción](data/ejecuciones/caso01-integracion/transcripcion.vtt) · [Entrada y ranking](data/ejecuciones/caso01-integracion/entrada.json) · [Informe](data/ejecuciones/caso01-integracion/gpt41/informe.md)

**Insumos del ejercicio 2 · 15/11/2026.** El [audio y el VTT de referencia](evaluacion/casos/caso_01/) se copiaron desde el prototipo y se verificó que son idénticos. G1, la evaluación de transcripción y el comando de ejemplo usan ahora esas rutas. Los snapshots fijados de Qwen y MiniLM se copiaron a la caché local del ejercicio 2. Se verificaron las cargas de ambos modelos y que G1 reconstruye los mismos mensajes con acceso bloqueado a la carpeta histórica. Al recalcular WER y CER con el VTT trasladado coincidieron todos los valores. La ruta del VTT fue el único cambio en ese reporte. Las corridas congeladas conservan sus rutas originales como datos de procedencia.

### Comparación opcional del formato del informe

- **Punto de partida:** modelo y prompt elegidos. Revisar si faltan campos en la salida en prosa.
- **Qué evaluar:** si se observan omisiones, comparar prosa y tabla por criterio con las mismas entradas y exigencias de contenido. Medir completitud y citas válidas.
- **Resultados:** pendientes.
- **Decisión:** pendiente.

### Continuación en el ejercicio 4

El corpus de VTT de I4 se conserva como referencia. La colección ampliada suma tres apuntes de Transformers y tiene 638 chunks: 623 VTT y 15 PDF. Se verificaron los IDs y los localizadores del PDF al reabrir Chroma. La comparación de recuperación queda pendiente. [Configuración y carga](../roadmap_ejercicio4.md#carga-de-los-tres-apuntes)

El primer piloto del ejercicio 4 comparó el corpus base contra el corpus con los tres apuntes usando el [Caso 02 de Transformers](../ejercicio4/evaluacion/caso_02_transformers.md). Se reutilizó el VTT de Whisper de 219 segmentos y se fijó `k=7`. Con apuntes, Context Recall@7 pasó de 0/3 a 1/3 y Context Precision@7 quedó en 14,3 %. Se reutilizó la consulta a Chroma con soporte de metadatos PDF y se agregó el localizador por página al informe. El siguiente contraste previsto es `k=7` frente a `k=10` en la colección mixta. [Resultados y auditoría](../ejercicio4/evaluacion/resultado_caso_02_apuntes.md)

Para el ejercicio 4 se prepararon dos apuntes de ejemplo de una página, de Embeddings y Fine-tuning. Se verificó la extracción de texto con `pypdf`. [Inventario de apuntes](../roadmap_ejercicio4.md#1-seleccionar-las-fuentes)

Se seleccionaron 105 diapositivas de cinco clases previas a RAG. El corpus final del ejercicio 4 reunirá los VTT acordados, los cinco apuntes y los cinco PDF de diapositivas, con procedencia por página. Se compararán valores de `k` después de cargar ese corpus. [Plan actualizado](../roadmap_ejercicio4.md) · [Manifiesto de diapositivas](../ejercicio4/slides/manifiesto.json)

Se reunieron los seis VTT docentes en `ejercicio4/transcripciones/`. Cada copia coincide byte a byte con su original y pasa el lector VTT. La tabla distingue los tres ya indexados en I4 de los tres pendientes. [Inventario de transcripciones](../roadmap_ejercicio4.md#transcripciones-docentes)

Se completó la carga de texto de ejercicio 4 en `openai_e4_texto`. Se reutilizaron 638 chunks y se añadieron 519 de los tres VTT pendientes y 6 de los dos apuntes nuevos. El total es 1.163 chunks. Se verificaron los metadatos y la persistencia al reabrir Chroma. [Configuración y validación](../roadmap_ejercicio4.md#carga-de-los-seis-vtt-y-cinco-apuntes)

El piloto de imágenes del ejercicio 4 comparó texto seleccionable, Apple Vision OCR y descripciones de GPT‑4.1 sobre ocho páginas. El Context Recall@7 quedó en 0/2 en los tres brazos. La descripción conservó la fórmula pero el fragmento quedó octavo. La colección principal conserva la carga textual de 1.163 chunks. [Auditoría del piloto](../ejercicio4/evaluacion/piloto_imagenes.md)

Se contrastó el recorte con la diapositiva completa como entrada a GPT‑4.1. Ambas variantes usaron el mismo prompt y mantuvieron el texto seleccionable. La fórmula subió del puesto 11 al 8 con la página completa, con Context Recall@7 de 0/2 y Context Precision@7 de 50 % en ambos casos. [Resultados](../ejercicio4/evaluacion/piloto_imagenes.md#comparación-de-recorte-y-diapositiva-completa)

Se compararon los prompts de descripción de la diapositiva completa. Pedir pares con origen y destino dejó la relación general entre encoder y decoder al comienzo del ranking y siguió omitiendo el bloque de destino. La fórmula permaneció octava. Se conserva el prompt de control y queda medir `k` con el corpus completo. [Resultados por consulta](../ejercicio4/evaluacion/piloto_imagenes.md#comparación-de-prompts-de-descripción)
