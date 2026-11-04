# Primera prueba de generación del informe

Fecha: 04/11/2026. Queremos separar las omisiones que provienen de la búsqueda
de los errores del generador al interpretar evidencia disponible.

## Comparación

Las dos corridas usan Qwen2.5-1.5B-Instruct local (revisión
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`), en CPU y float32, con
`do_sample=False` y hasta 1.000 tokens nuevos. Se toma el modelo del
notebook `Langchain_Virtual_Assitant.ipynb` (celdas 11 y 22). Se adapta la
instrucción de usar solo el contexto al análisis con dos fuentes. El
generador recibe el mismo objetivo y los 99 segmentos de la misma
transcripción de Whisper en las dos corridas.

| Corrida | Material académico | Tokens de entrada |
| --- | --- | ---: |
| Diagnóstico | Seis pasajes del VTT que respaldan los criterios del caso manual | 4.097 |
| Chroma | Ocho fragmentos recuperados desde el objetivo general | 4.151 |

Se suministra el texto original de los pasajes. Las etiquetas esperadas y el
informe manual quedan reservados para revisar las salidas. Se citan IDs de la
clase `[C#]` y del video `[V#]`. Los tiempos se resuelven a partir de las
fuentes después de generar. Ambas entradas están por debajo del límite de
6.000 tokens, con todos los fragmentos y segmentos de video incluidos.

Para revisar, se cotejan los temas identificados, las clasificaciones, los
pasajes citados y si se distingue la falta de respaldo académico de la falta
de evidencia en el video. Esta es una prueba diagnóstica sobre el
[caso manual](caso_01_embeddings.md), con la [ablación de k](ablacion_k.md)
como antecedente de la búsqueda.

## Validación inicial

Se verificó con el tokenizer de Qwen que ambas entradas están completas. Una
salida controlada permitió comprobar el mapeo de citas a sus tiempos y la
detección de un ID inexistente. Pasaron las 13 pruebas de chunks e índice del
ejercicio 2. Se verificó el SHA-256 de los pesos antes de la inferencia. Las
dos corridas reales se ejecutaron localmente. Se comprobó la igualdad de las
instrucciones, la transcripción y los parámetros de generación. Los ocho
fragmentos de Chroma coinciden con los de la ablación anterior.

## Resultados de la inferencia

| Corrida | Salida | Tiempo de generación | Resultado |
| --- | ---: | ---: | --- |
| Diagnóstico | 1.000 tokens | 175,5 s | Copió los pasajes de clase y llegó al límite de salida antes de clasificar o recomendar. |
| Chroma | 608 tokens | 109,9 s | Marcó el objetivo global como cubierto, inventó cuatro referencias de clase y atribuyó afirmaciones a segmentos que tratan otros temas. |

En el diagnóstico, los seis pasajes aportaban respaldo académico para los
cinco criterios E1–E5. La salida llegó a copiar el inicio del último pasaje
(`C6`, entrada 222, 31:55.350–32:14.820), sin evaluar la transcripción.
Esto muestra un fallo de la generación con evidencia académica disponible.

Con Chroma solo estaban disponibles las referencias suficientes para E1,
según la revisión de la ablación. La salida omitió el aviso de respaldo
académico insuficiente y concluyó que el video cubría todos los puntos del
objetivo. Citó `C9` a `C12`, aunque el contexto solo incluía `C1` a `C8`.
También usó rangos de IDs del video en lugar de citas individuales. Por
ejemplo, atribuyó una explicación de *clustering* a `V30–V35`
(04:02.660–04:40.580), donde el video explica el significado de «banco»
según el contexto. Atribuyó sistemas de recomendación a `V91–V97`
(09:25.580–09:49.580), que contienen el cierre y la despedida. La revisión
semántica de las citas se hizo manualmente. El script verifica existencia
de citas individuales con el formato previsto.

Ninguna salida permite comparar las etiquetas de los cinco criterios con el
caso manual. Los JSON completos de la prueba quedaron en
`data/resultados/generacion_caso01_20261104.03XLjo/` como `manual.json` y
`chroma.json`. Los tiempos corresponden a la generación, incluida la
evaluación de la entrada, y excluyen la carga del modelo y la búsqueda.

## Siguiente prueba propuesta

Variar solo las instrucciones de salida. Pedir una fila por criterio con
etiqueta, cita académica y cita del video cuando corresponda. Mostrar un
ejemplo breve ajeno al caso y pedir que la salida empiece por la primera fila.
Repetir las dos corridas con el mismo Qwen, la misma transcripción y `k=8`.
Comparar si produce clasificaciones utilizables, si reconoce el respaldo
académico insuficiente y si las citas existen y sostienen la afirmación.

## Reproducción

Desde la raíz, con las dependencias y los datos locales listos:

```bash
source .venv/bin/activate
objetivo="entender qué son los embeddings y cómo se compara la similitud entre textos"
video="tp3/ejercicio2/data/transcripciones/¿Qué son los EMBEDDINGS Grandes Modelos de Lenguaje.vtt"
salida=$(mktemp -d tp3/ejercicio2/data/resultados/generacion_caso01_20261104.XXXXXX)
python3 tp3/ejercicio2/generar_informe.py "$objetivo" "$video" \
  --clase-vtt docs/embeddings_2.VTT --pasaje 123 123 --pasaje 137 139 \
  --pasaje 181 181 --pasaje 182 188 --pasaje 191 197 --pasaje 222 222 \
  --prompt inicial --salida "$salida/manual.json"
python3 tp3/ejercicio2/generar_informe.py "$objetivo" "$video" \
  --k 8 --prompt inicial --salida "$salida/chroma.json"
```

Con Qwen y MiniLM en caché, agregar `--solo-local` a ambos comandos.

## Segunda prueba con una tabla por criterio — 04/11

Se cambió solo el mensaje de sistema. El prompt `tabla` pide que la salida
empiece por una tabla con criterio, estado, cita de clase, cita de video y
justificación. Distingue explícitamente entre ausencia de evidencia en el
video y falta de respaldo en el material recuperado. Incluye una fila
ficticia sobre el ciclo del agua para mostrar el formato. Se conservaron
el mensaje con el objetivo y las fuentes, el modelo y los parámetros de
generación. Para reproducir, usar los comandos anteriores con
`--prompt tabla` y una nueva carpeta de salida.

La revisión compara si la salida respeta el formato, qué criterios del caso
identifica y clasifica, si los IDs existen y si los pasajes citados respaldan
las afirmaciones. La coincidencia de una etiqueta y el respaldo de su
justificación se revisan por separado.

| Material académico | Prompt | Entrada | Salida | Filas completas | IDs de cita individuales inexistentes |
| --- | --- | ---: | ---: | ---: | ---: |
| Pasajes de referencia | Inicial | 4.097 | 1.000 (límite) | 0 | 0 |
| Pasajes de referencia | Tabla | 4.194 | 1.000 (límite) | 19 | 14 |
| Chroma con `k=8` | Inicial | 4.151 | 608 | 0 | 4 |
| Chroma con `k=8` | Tabla | 4.248 | 1.000 (límite) | 23 | 0 |

Las cifras de entrada y salida están en tokens. El conteo de IDs inexistentes
abarca citas individuales con el formato `[C1]` o `[V1]`. Se revisaron
aparte los rangos y las citas en la columna equivocada.

- **Pasajes de referencia.** Qwen empezó por la tabla. Marcó como cubierto
  el tema de la definición (E1), pero la justificación también atribuyó una
  explicación de similitud al tramo de introducción `V1–V3` (00:00.000–00:22.220).
  Marcó la comparación (E3) como cubierta, aunque `V4–V6`
  (00:22.220–00:50.300) anuncia el objetivo del video y contiene publicidad.
  Clasificó el papel del contexto (E2) como parcial con un ID académico
  inexistente. Omitió el cálculo del coseno (E4) y la limitación de magnitud
  (E5), aunque los pasajes de ambos estaban en la entrada. Inventó `C7` a
  `C20`, usó rangos de video y se cortó antes de la recomendación y la
  conclusión.
- **Chroma.** La fila sobre comparación de textos coincidió con la etiqueta
  parcial de E3, pero puso un rango del video en la columna de clase
  (`[V10-V12]`) y justificó la fila con el ejemplo de imágenes. Usó
  «Sin respaldo académico» en varias filas y lo justificó con supuestas
  ausencias en el video. Repitió las mismas seis filas hasta agotar la
  salida. Tampoco llegó a la recomendación ni a la conclusión.

Se verificó que el mensaje con los datos, los IDs y textos de clase y video,
la revisión del modelo y los parámetros de generación coincidían con la
primera prueba. Pasaron las 13 pruebas del ejercicio. Las salidas de la segunda
prueba están en `data/resultados/generacion_tabla_caso01_20261104.dlLtMO/`
como `manual.json` y `chroma.json`.

**Decisión:** conservar la tabla como variante experimental. Con estas dos
corridas no hay una configuración de generación apta para entregar informes.
Como siguiente prueba, usar el prompt de tabla y variar únicamente
`repetition_penalty` de `1.1` (valor de la configuración de Qwen) a
`1.15` (notebook `Langchain_Virtual_Assitant.ipynb`, celda 22). Medir si reduce las
filas duplicadas y si logra cerrar el informe, y volver a auditar las citas
y los criterios.
