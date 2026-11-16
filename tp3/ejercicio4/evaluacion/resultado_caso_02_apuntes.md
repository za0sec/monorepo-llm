# Caso 02 · Comparación de corpus con apuntes

Piloto con revisión manual del asistente. La [pauta de tres criterios](caso_02_transformers.md) se fijó antes de la búsqueda. El resultado corresponde a un solo objetivo y queda abierto a revisión del grupo.

## Configuración

Se comparó `openai_i4` (623 VTT) con `openai_e4_apuntes_ampliado` (los mismos 623 VTT más 15 chunks de tres PDFs). Se verificó la coincidencia de texto, metadatos y vectores de los VTT, con tolerancia absoluta de `1e-7` para los vectores. Ambas colecciones usaron los mismos parámetros HNSW y distancia coseno. El [script](../comparar_apuntes.py) reutiliza la expansión, los embeddings, la fusión, el CrossEncoder y la generación del ejercicio 2. En las citas, los PDF se localizan por fuente y página, y los VTT por intervalo.

GPT‑4.1 generó una sola expansión. Se usaron el objetivo original y tres preguntas sobre la decisión de a qué tokens atender, el procedimiento matemático y el efecto de las múltiples cabezas. Las cuatro consultas se embebieron en un único lote de `text-embedding-3-small` (1.536 dimensiones) y sus vectores se compartieron entre las dos búsquedas. Se recuperaron cinco resultados por consulta, con fusión por turnos y deduplicación por ID. El CrossEncoder `cross-encoder/ms-marco-MiniLM-L6-v2` puntuó los candidatos usando el objetivo original. Cada informe recibió los siete mejores, el mismo VTT de 219 segmentos y el mismo prompt. Se usó `gpt-4.1-2025-04-14`, temperatura 0 y máximo de 1.000 tokens de salida. La pauta quedó fuera de los mensajes.

## Resultados

| Medida | Solo VTT | VTT y apuntes |
| --- | ---: | ---: |
| Context Recall@7 | 0/3 | 1/3 |
| Context Precision@7 | 14,3 % | 14,3 % |
| Estados del video que coinciden con la pauta, incluyendo aciertos sin respaldo académico completo | 1/3 | 2/3 |
| Estados correctos con respaldo académico completo | 0/3 | 1/3 |
| Citas de clase que aportan respaldo en su fila o recomendación | 3/14 | 10/13 |
| Rangos de video que aportan respaldo en su fila | 6/7 | 7/7 |
| Answer Faithfulness descriptiva | 9/9 | 12/13 |
| Tokens MiniLM del contexto académico | 1.008 | 1.010 |
| Tokens de entrada / salida del informe | 6.279 / 443 | 6.255 / 566 |

Los dos informes terminaron con `stop` y todos sus IDs de cita existen. La expansión común consumió 141 tokens de entrada y 70 de salida. El lote de embeddings consumió 93 tokens de entrada. Se reutilizó la transcripción del caso.

### Evidencia recuperada

El sufijo `@7` indica que ambas métricas se calculan sobre los siete fragmentos finales, ya reordenados. Para Context Recall, esos fragmentos tienen que respaldar por completo un criterio. Para Context Precision, un fragmento cuenta como relevante si aporta alguna evidencia directa de A1, A2 o A3, incluso parcial. Se promedia la precisión acumulada en las posiciones relevantes. En ambas variantes la única posición relevante del top 7 fue la séptima, así que el valor fue `1/7`, aunque cada fragmento respaldó contenidos distintos.

| Criterio | Solo VTT | VTT y apuntes |
| --- | --- | --- |
| A1 · Roles de Q, K y V | El top 7 habla del propósito general de la atención, sin explicar los tres roles. | `pdf-034c4c852d0e4ab4-p1-1` define los tres roles y quedó séptimo. |
| A2 · Cálculo escalado | Sin evidencia en el top 7. | Sin evidencia en el top 7. |
| A3 · Varias cabezas | `c277`, séptimo, menciona varias cabezas, sus matrices y el costo computacional. Respaldo parcial. | Sin evidencia en el top 7. |

Entraron cuatro chunks PDF al top 7. El de decodificador quedó primero con atención cruzada, el de posición quedó quinto y el de encoder quedó sexto con una introducción general a la autoatención. Esos tres pasajes quedan fuera de la evidencia directa de la pauta.

### Auditoría del informe

| Criterio | Solo VTT | VTT y apuntes |
| --- | --- | --- |
| A1 · Cubierto | Las filas sobre decisión y cálculo, juntas, describen Q, K y V y dan un estado compatible con la pauta. Las citas académicas no explican esos roles. | Las filas sobre intuición, producto escalar y valores respaldan el estado con el audio y con la definición recuperada de la p. 1 del apunte. |
| A2 · Parcial | El informe marca «Cubierto» para la mecánica del cálculo y omite el escalado y softmax. | Describe los pasos que sí se oyen y los marca «Cubierto» por separado. Omite evaluar el cálculo escalado como criterio y omite recomendar cómo completarlo. |
| A3 · Sin evidencia en el video | Marca «Parcial» aunque la explicación reconoce que el video no lo cubre. La cita docente solo tiene respaldo parcial. | El estado coincide con lo que se escucha. La fila y la recomendación citan `[C7]`, que explica Q, K y V y termina antes de la sección de varias cabezas. Con este contexto correspondía señalar respaldo de clase insuficiente. |

El informe con apuntes agregó codificación posicional y atención cruzada, que quedan fuera de la pauta. Además, marcó «Parcial» para codificación posicional aunque el tramo citado plantea la pregunta sin explicar el mecanismo. La coincidencia de estados de la tabla se calcula solo sobre A1–A3. Un criterio omitido no cuenta como acierto.

**Citas.** Se contaron las apariciones de `[C#]` y cada rango de video como una unidad. Una cita aporta respaldo si toca directamente al menos parte de lo que se afirma en su fila o recomendación. En la base aportaron respaldo `[C1]` en la fila de la intuición y las dos apariciones de `[C7]` para múltiples cabezas. El rango `[V63]-[V66]` solo presenta el nombre del mecanismo, por lo que no respalda la explicación de la primera fila. En la variante con apuntes quedaron sin respaldo `[C4]` en la primera fila y las dos apariciones de `[C7]` sobre múltiples cabezas. Los rangos amplios cuentan si contienen evidencia directa y se conservan para revisar su precisión temporal.

**Answer Faithfulness descriptiva.** Se deduplicaron afirmaciones comprobables y se evaluaron con el contexto entregado al modelo. Los estados se midieron por separado en la tabla anterior. Los dos informes describen correctamente seis puntos del video: priorización de otros tokens, Q y K aprendidas para estimar compatibilidad, producto escalar, rol de V, suma ponderada y ausencia de explicación de múltiples cabezas. La base también describe correctamente la visualización matricial del video, la existencia de varias cabezas en la clase y su costo adicional. Con apuntes se respaldan cuatro descripciones más del video (problema de memoria de las RNN, pregunta sobre la posición, falta de explicación posicional y falta de explicación de atención cruzada) y dos de clase (incorporación del orden y origen de Q, K y V en atención cruzada). La afirmación sin respaldo es que `[C7]` explica el mecanismo y propósito de múltiples cabezas. Este conteo compara afirmaciones respaldadas dentro de cada informe.

## Decisión y reproducción

Conservar los tres apuntes para la siguiente comparación. En este caso mejoran la cobertura académica del top 7 de 0 a 1 criterio. El siguiente contraste compara `k=7` con `k=10` sobre el mismo ranking de la colección mixta. Eso incorporaría la fórmula, que quedó octava, y el fragmento sobre combinación de salidas, que quedó noveno.

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/comparar_apuntes.py \
  --salida tp3/ejercicio4/data/experimentos/caso_02_apuntes_repeticion
```

[Expansión](../data/experimentos/caso_02_apuntes/expansion.json) · [Consultas y vectores](../data/experimentos/caso_02_apuntes/consultas.json) · [Ranking base](../data/experimentos/caso_02_apuntes/openai_i4/entrada.json) · [Ranking con apuntes](../data/experimentos/caso_02_apuntes/openai_e4_apuntes_ampliado/entrada.json) · [Informe base](../data/experimentos/caso_02_apuntes/openai_i4/gpt41/informe.md) · [Informe con apuntes](../data/experimentos/caso_02_apuntes/openai_e4_apuntes_ampliado/gpt41/informe.md)
