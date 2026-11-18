# Caso 01 con el corpus completo

Se usa la [pauta revisada del Caso 01](../../ejercicio2/evaluacion/caso_01_embeddings.md) y su VTT de Whisper de 99 segmentos. El objetivo literal es «entender qué son los embeddings y cómo se compara la similitud entre textos». Se mantienen el nivel introductorio, el [prompt de G1](../../ejercicio2/g1/prompt.md), GPT‑4.1 a temperatura 0 y el límite de 1.000 tokens de salida. La pauta queda fuera de los mensajes del modelo.

La colección `openai_e4_completo` aporta 1.484 chunks. La expansión genera tres subconsultas y se recuperan cinco candidatos de Chroma por consulta, incluida la original. Se fusionan los resultados por turnos, se deduplican por ID y se aplica una vez el CrossEncoder de G4. Las configuraciones `k=7`, `k=8`, `k=10` y `k=12` usan prefijos de ese mismo ranking. Si hay menos candidatos que un corte, se usa el máximo disponible una sola vez.

Se auditan el respaldo completo de los cinco criterios en cada prefijo, la coincidencia de los estados con el VTT, el respaldo de las citas y los tokens de entrada. Un estado acertado con evidencia académica completa se registra por separado. Una mención en recomendaciones no sustituye la evaluación de cobertura del criterio.

La fusión de los 20 resultados devolvió **9 candidatos únicos**, por lo que se probaron `k=7`, `k=8` y `k=9`. Se verificó que cada entrada usó el prefijo acordado, el mismo VTT y el mismo prompt. Los tres informes terminaron con `stop` e IDs de citas válidos.

| Medida | k=7 | k=8 | k=9 |
| --- | ---: | ---: | ---: |
| Context Recall | 2/5 | 2/5 | 2/5 |
| Context Precision | 89,3 % | 83,9 % | 81,1 % |
| Estados que coinciden con la pauta | 3/5 | 3/5 | 3/5 |
| Estados acertados con evidencia académica completa recuperada | 2/5 | 1/5 | 2/5 |
| Tokens de entrada de GPT‑4.1 | 3.264 | 3.376 | 3.499 |
| Tokens de salida de GPT‑4.1 | 528 | 519 | 553 |

Para Context Recall se anotaron E1 y E3 como completos. C3 explica que el embedding representa texto mediante vectores densos. C2 muestra la comparación del embedding de la consulta con los embeddings de los elementos y C1 aporta el coseno y el ángulo como criterio. E2 queda parcial. C7 y C8 definen embeddings estáticos y dinámicos y dan ejemplos de modelos, pero el contexto recuperado carece de un ejemplo de cómo cambia una misma palabra con el contexto. E4 y E5 carecen de evidencia completa. Los puestos relevantes para Context Precision fueron 1, 2, 3, 7, 8 y 9. Se promedió la precisión acumulada de los puestos relevantes dentro de cada prefijo.

Los dos fragmentos que se sumaron repiten contenido. El puesto 8 describe el contraste estático/dinámico de la misma página que el puesto 7. El puesto 9 recupera el texto seleccionable de la página cuya descripción estaba en el puesto 3. El candidato de la página 32 que quedó primero presenta la fórmula, y el del puesto 5 es el título seleccionable. El fragmento con la ecuación, `embeddings-32-descripcion_pagina-2`, y su equivalente de la transcripción, `c075`, quedaron fuera de los resultados iniciales de las cuatro consultas.

| Criterio | Estado de la pauta | k=7 | k=8 | k=9 |
| --- | --- | --- | --- | --- |
| E1 · Qué es un embedding | Cubierto | Cubierto | Cubierto | Cubierto |
| E2 · Papel del contexto | Cubierto | Parcial | Cubierto | Cubierto |
| E3 · Procedimiento general | Parcial | Parcial | Sin evaluación separada de E4 | Parcial |
| E4 · Cálculo del coseno | Sin evidencia | Sin evidencia | Sin evidencia | Sin evaluación separada de E3 |
| E5 · Magnitud | Sin evidencia | Omitido | Omitido | Omitido |

En `k=7`, el informe usa C3 para la definición y C1 junto con C2 para el procedimiento general. En `k=8` y `k=9`, el estado «Cubierto» de E2 concuerda con el ejemplo de «banco» en el video. Las citas C7 y C8 respaldan el contraste entre los tipos de embeddings y falta un ejemplo académico equivalente. El informe de `k=9` deja C2 sin citar en la fila de comparación, aunque estaba en su contexto. Los informes de `k=7` y `k=9` recomiendan C1 y C5 como si mostraran la fórmula. Ambos fragmentos apuntan a la página correcta, pero el texto enviado carecía de la ecuación.

Se mantiene `k=7` como referencia para este caso. Los cortes mayores agregaron tokens y repeticiones con el mismo Context Recall y la misma cantidad de estados acertados. Sigue probar cuántos resultados se traen por consulta antes del reranking, con `k=7` fijo. Las métricas de este caso sirven para desarrollo y todavía requieren validación con otro objetivo y video.

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/comparar_k_completo.py \
  --salida tp3/ejercicio4/data/experimentos/caso_01_k_completo
```

[Ranking común](../data/experimentos/caso_01_k_completo/ranking.json) · [Informe k=7](../data/experimentos/caso_01_k_completo/k7/gpt41/informe.md) · [Informe k=8](../data/experimentos/caso_01_k_completo/k8/gpt41/informe.md) · [Informe k=9](../data/experimentos/caso_01_k_completo/k9/gpt41/informe.md)

## Diez candidatos por consulta

Se conserva la expansión del control, incluido el texto exacto de las cuatro consultas. Se contrasta el top 7 con cinco y diez resultados por consulta antes del reranking. Después se comparan los cortes 7, 8, 10 y 12 del ranking con diez resultados por consulta, hasta el máximo disponible. El corpus, el reranker, el VTT, el prompt y la pauta de evaluación siguen siendo los del Caso 01.

### Cinco contra diez resultados por consulta

Se volvieron a calcular los embeddings de las cuatro consultas guardadas. Los IDs y el orden de los primeros cinco resultados coincidieron con el control en las cuatro búsquedas. Las distancias coseno difirieron como máximo en `6,7e-5`. Se verificaron el mismo corpus, la misma revisión del CrossEncoder y los mismos mensajes, salvo los fragmentos de clase seleccionados.

| Resultados por consulta | Candidatos únicos | Puesto de la fórmula | Context Recall@7 | Estados correctos con k=7 |
| --- | ---: | ---: | ---: | ---: |
| 5 | 9 | Fuera de los candidatos | 2/5 | 3/5 |
| 10 | 16 | 1 | 3/5 | 3/5 |

El apunte de similitud coseno apareció sexto para la consulta original y para la subconsulta sobre métodos de comparación. El reranker lo colocó primero. Su primera página aporta en el mismo chunk la definición, la dirección como criterio y la ecuación completa. El informe de `k=7` citó ese fragmento y acertó que el video carece de la explicación del cálculo. Los dos fragmentos con la ecuación de la diapositiva y del VTT docente siguieron fuera de los candidatos. La evidencia nueva provino del apunte.

### Cortes del ranking ampliado

| Resultado | k=7 | k=8 | k=10 | k=12 |
| --- | ---: | ---: | ---: | ---: |
| Context Recall | 3/5 | 3/5 | 3/5 | 3/5 |
| Context Precision | 100 % | 100 % | 90 % | 80,4 % |
| Estados que coinciden con la pauta | 3/5 | 4/5 | 3/5 | 4/5 |
| Estados acertados con evidencia académica completa recuperada | 3/5 | 3/5 | 3/5 | 3/5 |
| Tokens de entrada de GPT‑4.1 | 3.340 | 3.358 | 3.588 | 3.785 |
| Tokens de salida de GPT‑4.1 | 548 | 543 | 609 | 571 |

Los cinco criterios se corrigieron con la misma pauta. E1, E3 y E4 quedaron completos en el contexto desde el primer corte. Para Context Precision se anotaron los puestos 1, 2, 3, 4, 10, 11 y 12 como relevantes. Se promedió la precisión acumulada en las posiciones relevantes de cada prefijo. Los puestos 10 y 11 definen embeddings estáticos y dinámicos con ejemplos de modelos. Falta una comparación de la representación de una misma palabra en dos contextos para completar E2. El fragmento con el ejemplo de magnitud no apareció y E5 quedó incompleto.

| Criterio | Estado esperado | k=7 | k=8 | k=10 | k=12 |
| --- | --- | --- | --- | --- | --- |
| E1 · Qué es un embedding | Cubierto | Cubierto | Cubierto | Cubierto | Cubierto |
| E2 · Papel del contexto | Cubierto | Mención general del contexto | Cubierto | Parcial | Cubierto |
| E3 · Procedimiento general | Parcial | Parcial | Parcial | Parcial | Parcial |
| E4 · Cálculo del coseno | Sin evidencia | Sin evidencia | Sin evidencia | Sin evidencia | Sin evidencia |
| E5 · Magnitud | Sin evidencia | Omitido | Omitido | Omitido | Omitido |

Los cuatro informes citaron el primer fragmento del apunte para la fórmula. Los informes de `k=8` y `k=10` también lo recomiendan por su ejemplo numérico, aunque ese chunk termina en el encabezado del ejemplo. En `k=8` y `k=12`, el informe separa el papel del contexto y cita el tramo del video con el ejemplo de «banco». Con `k=7` solo describe el contexto en general. Con `k=10` marca el contraste como parcial por la terminología del video. Los cuatro informes terminaron con `stop` e IDs de citas válidos.

Se eligen diez candidatos por consulta y `k=8` para validar con otro objetivo. En este caso, `k=8` igualó los cuatro aciertos de `k=12` con 427 tokens de entrada menos. El único fragmento añadido al pasar de 7 a 8 es el título de la diapositiva de coseno. La evidencia académica completa se mantuvo igual, así que la mejora en el informe se observó en la salida de generación de este caso.

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/comparar_k_completo.py \
  --vecinos 10 --control tp3/ejercicio4/data/experimentos/caso_01_k_completo/ranking.json \
  --salida tp3/ejercicio4/data/experimentos/caso_01_vecinos10
```

[Ranking ampliado](../data/experimentos/caso_01_vecinos10/ranking.json) · [Informe k=7](../data/experimentos/caso_01_vecinos10/k7/gpt41/informe.md) · [Informe k=8](../data/experimentos/caso_01_vecinos10/k8/gpt41/informe.md) · [Informe k=10](../data/experimentos/caso_01_vecinos10/k10/gpt41/informe.md) · [Informe k=12](../data/experimentos/caso_01_vecinos10/k12/gpt41/informe.md)

### Embeddings con la configuración final

Esta ficha resume el corte `k=7` ya ejecutado con el corpus completo. Usa el mismo video y objetivo del Caso 01 de la integración del ejercicio 2 («entender qué son los embeddings y cómo se compara la similitud entre textos»). El ejercicio 4 reutilizó el VTT de referencia de 99 segmentos. Chroma devolvió diez vecinos por cada una de las cuatro consultas, quedaron 16 candidatos únicos y el reranker seleccionó los siete que recibió GPT-4.1.

| Indicador | Resultado | Cómo se verificó |
| --- | ---: | --- |
| Context Recall@7 | **3/5 (60 %)** | El contexto respalda por completo E1 (definición), E3 (comparación vectorial) y E4 (cálculo del coseno). |
| Context Precision@7 | **100 %** | Los cuatro chunks relevantes quedaron en los puestos 1–4. La precisión acumulada en cada uno de esos puestos es 1. |
| Estados correctos | **3/5** | Coinciden con la pauta E1 («Cubierto»), E3 («Parcial») y E4 («Sin evidencia en el video»). |
| IDs de cita inexistentes | **0** | La corrida registra `citas_invalidas: []`. |

Con `k=7`, el informe menciona el contexto y deja E2 sin una evaluación propia del contraste entre embeddings fijos y contextuales. E5, el efecto de la magnitud, queda omitido. La corrida terminó con `stop`, 3.340 tokens de entrada y 548 de salida. El apunte sobre similitud coseno queda como `C1` y se recomienda para estudiar el cálculo que el video omite.

En la integración del ejercicio 2, el mismo caso había dado 3/5 de recall, 78,6 % de precisión, 2/5 estados correctos y ningún ID inexistente. La comparación muestra un mejor orden y un estado correcto adicional. También cambiaron el corpus, el VTT y la cantidad de vecinos por consulta, de cinco a diez. La corrida desde MP3 de la diapositiva 37 corresponde a Tokenización y tiene una pauta de tres criterios.

[Pauta E1–E5](../../ejercicio2/evaluacion/caso_01_embeddings.md#criterios-de-cobertura) · [Ranking de 16 candidatos](../data/experimentos/caso_01_vecinos10/ranking.json) · [Entrada del corte de siete](../data/experimentos/caso_01_vecinos10/k7/entrada.json) · [Informe](../data/experimentos/caso_01_vecinos10/k7/gpt41/informe.md) · [Registro de la corrida](../data/experimentos/caso_01_vecinos10/k7/gpt41/corrida.json) · [Entrada del ejercicio 2](../../ejercicio2/data/ejecuciones/caso01-integracion/entrada.json)
