# Revisión inicial de indexación y recuperación

Fecha: 31/10/2026. Ejecuciones locales reales de MiniLM y Chroma, revisadas por
el asistente. Esta revisión evalúa referencias académicas recuperadas; no genera
un informe sobre el video ni sustituye el [caso manual](caso_01_embeddings.md).

## Configuración y fundamento

- Corpus: `docs/embeddings_2.VTT`, 415 intervenciones, 125 chunks.
- SHA-256 de la fuente: `a1fa8324b6d63c7c6d90fd4dc467b65095524aba8e134d806c2118278b24cd18`.
- SHA-256 del documento de chunks según `hash_json`: `00b58f1116024bb51f74aca18ed5c73b491a60d4396d2cda6b7170daa5132a21`.
- Splitting: 1000 caracteres/100 de overlap, luego 200 tokens/20 de overlap.
- Embedder: `sentence-transformers/all-MiniLM-L6-v2`, revisión
  `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, CPU, 384 dimensiones,
  `normalize_embeddings=False`. Límite observado: 256 tokens.
- Chroma 1.5.9, Sentence Transformers 5.1.2, PyTorch 2.8.0,
  Transformers 4.57.1, tokenizers 0.22.2.
- Colección: `tp3_c18deda7c17b0e0f81ea5513`, persistida en `data/chroma/`.
- Recuperación inicial: distancia coseno explícita, `k=4`, sin umbral.

MiniLM + Chroma se apoya en `Langchain_Virtual_Assitant.ipynb`, celda 9;
la indexación y búsqueda, en `UN_Report_Advanced_RAG.ipynb`, celdas 18-23.
La persistencia y el aislamiento por configuración adaptan esas demos. `k=4`
es una hipótesis del diseño local, no un requisito ni un valor optimizado.

## Integridad y operación

Se cotejaron los 125 textos persistidos contra sus offsets en el VTT limpio:
coinciden con el original procesado y conservan IDs de intervenciones. Cada
vector tiene 384 dimensiones. Se revisaron especialmente los chunks 50-53
(25:22.170-27:47.049) y 63 (31:44.560-32:41.570), que contienen coseno,
magnitud y su cálculo. Se mantienen los errores originales de ASR y algunos
cortes dentro de palabras o intervenciones; esto limita la legibilidad.

La primera indexación agregó 125 chunks; una segunda ejecución en otro proceso
reutilizó los 125 y agregó cero, sin ejecutar el embedder. Tiempos dentro de la
función de indexación: 14,502 s iniciales (incluye carga/descarga del modelo) y
0,040 s al reutilizar. No incluyen arranque e importaciones de Python ni son
un benchmark de latencia. Tamaños observados aproximados: Chroma 2,3 MB y caché
del modelo 87 MB.

Una autoconsulta con el texto exacto del chunk 63 devolvió su ID en primer lugar,
con distancia aproximada de -1,19e-7 por redondeo. Esto verifica coherencia entre
embeddings de carga y consulta; no mide recuperación semántica de objetivos.

Pasaron 13 pruebas: siete de chunks y seis de integración con Chroma real y
vectores controlados. Incluyen persistencia en otro proceso, trazabilidad,
distancia/orden, repetición sin recálculo, reanudación tras fallo y rechazo de
configuraciones, métricas o entradas incompatibles. `pip check` no encontró
dependencias incompatibles. No se llamó a servicios de inferencia por API.

## Consulta A: coseno y limitaciones

Consulta exacta: `entender similitud coseno y sus limitaciones`.
Resultado local completo: `data/resultados/coseno_k4.json`.
Los IDs se abrevian a sus primeros 12 caracteres en las tablas.

| Puesto | Prefijo del chunk ID | Tiempo de clase | Distancia coseno | Revisión del contenido |
| --- | --- | --- | --- | --- |
| 1 | `31e13bc27aae` | 49:37.400-50:04.219 | 0,447976 | Atención, parámetros y límites del Transformer; no desarrolla coseno. |
| 2 | `753c837681d6` | 55:47.820-56:17.560 | 0,518801 | Posiciones/frecuencias y distancia entre tokens; no desarrolla coseno. |
| 3 | `6592f82fe3b3` | 56:35.360-57:01.270 | 0,527981 | Tamaño de vectores y restricciones del Transformer; no desarrolla coseno. |
| 4 | `5acf1645d843` | 31:44.560-32:41.570 | 0,545727 | Incluye la entrada 222: cálculo mediante producto interno y normas (E4). |

La referencia E4 se recupera, pero las entradas 182-188 de E5 (magnitud y
ejemplo de series, 25:37.950-27:13.979) quedan fuera de los cuatro resultados.
La mayor similitud numérica no coincide con la referencia más pertinente.
El contexto recuperado sería insuficiente para explicar ambas partes del objetivo.

Tiempo observado dentro de la recuperación: 0,154 s para carga del modelo y
embedding del objetivo; 0,00164 s para búsqueda. Las importaciones quedan fuera.

## Consulta B: objetivo del caso 01

Consulta exacta: `entender qué son los embeddings y cómo se compara la similitud entre textos`.
Resultado local completo: `data/resultados/caso01_k4.json`.

| Puesto | Prefijo del chunk ID | Tiempo de clase | Distancia coseno | Revisión del contenido |
| --- | --- | --- | --- | --- |
| 1 | `edab6e7f243e` | 19:52.630-20:15.640 | 0,440729 | Representación de documentos mediante conteos; relacionado con representaciones, pero no explica el objetivo completo. |
| 2 | `f868030552ed` | 01:02.930-01:47.279 | 0,457591 | Tipos de tokens y tokens especiales. |
| 3 | `544ad9b64437` | 48:22.470-49:04.670 | 0,462638 | Atención y positional encoding. |
| 4 | `97d111e4cbd1` | 15:14.610-15:52.110 | 0,475494 | Frecuencia de tokens y comportamiento del modelo. |

No se recuperan las intervenciones citadas como referencia para E1-E5 en el
caso manual. Esto compara localización de evidencias; no demuestra que todo
fragmento alternativo carezca de utilidad ni constituye una métrica de RAGAS.
Tiempo observado: 0,152 s de carga/embedding y 0,00199 s de búsqueda.

## Reproducción y siguiente decisión

Desde la raíz, con el entorno activo y los chunks preparados:

```bash
python3 tp3/ejercicio2/indexar_material.py
python3 tp3/ejercicio2/indexar_material.py --solo-local
python3 tp3/ejercicio2/recuperar.py \
  "entender similitud coseno y sus limitaciones" --solo-local \
  --salida tp3/ejercicio2/data/resultados/coseno_k4.json
python3 tp3/ejercicio2/recuperar.py \
  "entender qué son los embeddings y cómo se compara la similitud entre textos" \
  --solo-local --salida tp3/ejercicio2/data/resultados/caso01_k4.json
python3 -m unittest discover -s tp3/ejercicio2/tests -v
```

Los JSON completos, Chroma y pesos son regenerables y están ignorados por Git;
este informe conserva la configuración y la revisión para compartirlas.
Las latencias y mínimos errores numéricos pueden variar entre equipos.

La operación e integridad están verificadas, pero la recuperación del baseline
todavía no aporta contexto suficiente para estos objetivos. El paso siguiente es
diagnosticarla y comparar una variable por vez, por ejemplo `k=2,4,8`, y después
una alternativa de embeddings vista en clase si el ruido persiste. Español,
errores de ASR y cortes de chunks son hipótesis a estudiar, no causas demostradas
por estas dos consultas. Conservar estas corridas como baseline; no atribuir
faltantes del retrieval a omisiones del video. La generación queda pendiente,
al igual que Advanced RAG y la evaluación en casos independientes.
