# Ablación de recuperación por cantidad de fragmentos

Fecha de ejecución: 04/11/2026. Comparación exploratoria del recuperador del
ejercicio 2. La consigna pide comparar cambios incrementales (p. 2) y la
explicación oral menciona variar parámetros de la recuperación
(`docs/RAG/tp3.VTT`, 08:52–09:19).

## Método

Se ejecutaron las dos consultas de la [revisión inicial](revision_retrieval_inicial.md)
con `k=2, 4 y 8`, en seis corridas locales. `k` es la cantidad de fragmentos
devueltos por búsqueda. Se mantuvieron los 125 chunks de `docs/embeddings_2.VTT`,
el modelo MiniLM en CPU, la colección Chroma `tp3_c18deda7c17b0e0f81ea5513`,
la distancia coseno y la búsqueda sin umbral. El manifiesto coincide en las
seis corridas y con la configuración de la revisión inicial.

- **A:** `entender similitud coseno y sus limitaciones`. Se revisaron E4 y E5.
- **B:** `entender qué son los embeddings y cómo se compara la similitud entre textos`.
  Se revisaron E1–E5.

Se usaron los criterios y las entradas VTT del [caso manual](caso_01_embeddings.md#criterios-de-cobertura)
como referencia. Se cruzaron los IDs para localizar pasajes y se leyó el texto
efectivamente recuperado. Un criterio cuenta como respaldado si el conjunto de
fragmentos aporta la explicación esperada. Una coincidencia de ID o un pasaje
que solo aporta parte de la explicación se registra por separado. Estos conteos
describen las referencias académicas disponibles para cada consulta y usan
criterios adaptados al caso de desarrollo.

## Resultados

| Consulta | `k=2` | `k=4` | `k=8` |
| --- | --- | --- | --- |
| A · Coseno y limitaciones | 0/2 | 1/2: E4 en el puesto 4 | 1/2: E4. Aparece parte del pasaje de E5 en el puesto 8. |
| B · Objetivo del caso 01 | 0/5 | 0/5 | 1/5: E1, respaldado al combinar los puestos 5 y 8. |

Los resultados de `k=4` coincidieron con los registros anteriores. Para ambas
consultas, `k=2` y `k=4` devolvieron los primeros dos y cuatro puestos de
`k=8` en el mismo orden. Los puestos 1–4 están detallados en la revisión
inicial. Estos son los fragmentos que se agregaron al pasar a `k=8`:

| Consulta | Puesto | Prefijo del ID | Tiempo en la clase | Distancia coseno | Qué aporta |
| --- | ---: | --- | --- | ---: | --- |
| A | 5 | `2c4a6c25c926` | 32:14.890–32:41.570 | 0,552748 | Continúa con producto interno y entrenamiento, con solapamiento con el puesto 4. |
| A | 6 | `9cc37a19e304` | 17:53.110–18:22.379 | 0,555102 | Tamaño de vocabulario y tokenización. |
| A | 7 | `4afe1972fcf1` | 24:43.310–25:02.360 | 0,559794 | Esparsidad de matrices. |
| A | 8 | `fb4d20804426` | 26:58.290–27:47.049 | 0,562335 | Incluye la entrada 188, el cierre del ejemplo sobre frecuencia de palabras. Las entradas 182–187 con la explicación sobre magnitud y el ejemplo de series quedaron fuera. |
| B | 5 | `567321612e88` | 18:22.460–19:08.759 | 0,477211 | La entrada 123 explica la motivación de representar el significado de palabras y documentos. |
| B | 6 | `84cfee9a27cc` | 52:27.320–52:44.660 | 0,479961 | Menciona un embedding en el flujo del Transformer, con una referencia breve al contexto. |
| B | 7 | `448e01a14529` | 30:06.130–30:46.010 | 0,481373 | Fragmento de entrenamiento de Word2Vec. |
| B | 8 | `bccd4afb2b60` | 24:56.890–25:37.590 | 0,481439 | La entrada 181 describe los embeddings como vectores densos. Con el puesto 5 aporta los dos componentes de E1. |

En B, el puesto 1 ya describe una representación por conteo y el 6 alude a la
incorporación de contexto. Se registraron como aportes parciales porque quedan
fuera la explicación del criterio de comparación entre vectores y el contraste
entre representaciones fijas y contextuales. En A, el puesto 8 aporta un
pasaje relacionado con E5, pero queda fuera la explicación que conecta magnitud
y coseno. Los conteos dependen de la lectura de estos dos objetivos de una
misma clase. La evaluación del video y la generalización requieren pruebas
separadas.

## Reproducción

Desde la raíz, con el entorno preparado y el modelo y el índice en caché:

```bash
source .venv/bin/activate
salida=$(mktemp -d tp3/ejercicio2/data/resultados/ablacion_k_20261104.XXXXXX)
for k in 2 4 8; do
  python3 tp3/ejercicio2/recuperar.py \
    "entender similitud coseno y sus limitaciones" \
    --solo-local --k "$k" --salida "$salida/coseno_k$k.json"
  python3 tp3/ejercicio2/recuperar.py \
    "entender qué son los embeddings y cómo se compara la similitud entre textos" \
    --solo-local --k "$k" --salida "$salida/caso01_k$k.json"
done
```

Los seis JSON de esta ejecución quedaron en
`data/resultados/ablacion_k_20261104.oaCc7k/`. Guardan manifiesto, referencias
completas y tiempos de búsqueda.

## Configuración elegida

Se adopta `k=8` por defecto para la siguiente etapa. Entre los tres valores
probados, fue el único que aportó respaldo para E1 en la consulta general y
conservó el respaldo de E4 en la consulta de coseno. Se priorizó la cantidad de
criterios respaldados en estos dos objetivos, con el costo de agregar también
fragmentos poco útiles. La calidad del informe todavía queda por medir.
Las corridas de `k=4` quedan como referencia histórica y se pueden reproducir
con `--k 4`.

## Próxima decisión

El paso propuesto es implementar el informe básico con dos fuentes separadas.
Para depurar el generador se puede hacer una corrida de diagnóstico con pasajes
académicos del caso manual. Luego se puede repetir con los fragmentos realmente
recuperados y registrar cuándo falta respaldo académico para evaluar un
criterio. La primera corrida probaría el generador con referencias conocidas.
La segunda mostraría el comportamiento del pipeline completo.

Como camino alternativo si se prioriza estudiar la recuperación, se puede
comparar otro embedder visto en clase manteniendo fijo el corpus y el splitting.
El aumento de `k` por sí solo dejó fuera referencias importantes en ambas
consultas.
