# Notas — Ejercicio 1 (Formulación del problema y EDA)

Notas de trabajo, en construcción. Objetivo: dejar registrado el *por qué* de cada decisión para el informe final, no solo el resultado.

## Unidad de análisis y variable objetivo

Cada fila del CSV **no es una búsqueda (`query_id`) completa** — es **un producto puntual mostrado como resultado dentro de una búsqueda**. Cada búsqueda (`query_id`) tiene entre 1 y 8 productos impresos (~5 en promedio, 2012 queries en total).

El BTR que define la consigna (comprados / impresos) es una métrica agregada a nivel búsqueda. Pero como el dataset ya viene desagregado a nivel fila-producto-búsqueda, con una columna `bought` (True/False) por fila, tiene más sentido:

- **Variable objetivo: `bought`, a nivel fila.** Predecir la probabilidad de que ESE producto puntual, mostrado en ESA búsqueda puntual, sea comprado.
- El BTR de una búsqueda completa sale "gratis" después: es el promedio de esas probabilidades (o de los `bought` reales) sobre los productos que se mostraron en esa query.
- Esto es consistente con que la consigna pida evaluar con **PR-AUC / ROC-AUC** (métricas de clasificación binaria, no de error de una tasa) y aclare que no hace falta definir un threshold — no nos interesa una decisión sí/no con un corte fijo, sino qué tan bien el modelo *ordena* los productos por probabilidad de compra.

## Relación `cart` y `bought`

No son lo mismo, y la relación es asimétrica:

- `bought = True` **siempre** implica `cart = True` (0% de casos de compra sin haber pasado por el carrito). Tiene sentido de negocio: no se compra sin agregar al carrito antes.
- Lo inverso **no** se cumple: de los productos que llegaron al carrito (`cart = True`, 30,07% del total), solo el **43,3%** terminó comprado (`bought = True`). El resto (17,06 puntos porcentuales del total de filas) quedó en el carrito sin concretar la compra.

Distribución completa (sobre el total de filas):

| cart | bought | % filas |
|---|---|---|
| True | True | 13,01% |
| True | False | 17,06% |
| False | True | 0,00% |
| False | False | 69,93% |

Implicancia: `cart` es una señal fuertemente correlacionada con `bought` pero **no puede usarse como feature de entrada** si el objetivo es predecir `bought` de forma realista — en el momento en que se hace la predicción (cuando se muestran los resultados de búsqueda), todavía no sabemos si el usuario va a agregar al carrito. Usar `cart` como feature sería *data leakage* (usar información que en la práctica no está disponible al momento de predecir). Falta decidir/confirmar esto formalmente al armar la lista de features.

## Balance de clases

- `bought = True`: **13,01%** de las filas.
- `cart = True`: 30,07% de las filas.

Desbalance moderado (no extremo). Implicancias a tener en cuenta más adelante:
- El PR-AUC es más informativo que el ROC-AUC en este contexto, porque el ROC-AUC puede verse artificialmente alto por la cantidad de negativos (no-compras) fáciles de acertar.
- Pendiente: pensar cómo particionar train/valid/test dado este desbalance (no vimos en clase ninguna técnica específica para esto — no asumir nada, decidirlo entre nosotras cuando lleguemos a ese punto).

## Estructura de las queries

- Las 2012 `query_id` tienen, cada una, un único valor de `filter_category` — es decir, cada búsqueda está acotada a una sola categoría (el usuario buscó "dentro de" una categoría con el filtro activado). Confirmado con `df.groupby('query_id')['filter_category'].nunique()` → siempre 1.
- Lo mismo pasa con **los otros 3 filtros**: `filter_storage_type`, `filter_price_min`, `filter_price_max` también son constantes dentro de cada `query_id` (2012 de 2012 queries con 1 solo valor). Confirmado por columna con `df.groupby('query_id')[c].nunique()`.
- Conclusión: los 4 `filter_*` describen **la búsqueda** (lo que pidió el usuario: categoría, tipo de almacenamiento, rango de precio), no el producto — son un conjunto fijo de "condiciones" aplicado a todos los productos devueltos por esa query. Son distintos conceptualmente de los atributos propios del producto (`category`, `storage_type`, `price`).
- **`category == filter_category` y `storage_type == filter_storage_type` en el 100% de las filas.** Es decir, esos dos filtros son copias exactas de columnas que ya tenemos del producto — **no aportan información nueva, se pueden descartar como features** (`filter_category`, `filter_storage_type`). Distinto es el caso de `filter_price_min`/`filter_price_max`: no hay una columna "price del filtro" única sino un rango, y ahí sí hay señal nueva (la posición relativa de `price` dentro de ese rango, ver sección de `price` más abajo).

## Nulos en `allergens`

`allergens` es la única columna con nulos (44,55%), pero **no es un dato faltante al azar**: depende 100% de la `category` del producto.

- Nulo siempre (100%) en: `Household`, `Meat`, `Personal Care`, `Produce` — categorías donde no aplica declarar alérgenos alimentarios (detergente, fruta suelta, etc.).
- Nunca nulo (0%) en: `Bakery`, `Dairy`, `Seafood` — panificados, lácteos y pescado, que casi siempre contienen alguno de los alérgenos que aparecen en los datos.
- Parcial en el resto: `Beverages` (50,8%), `Baby` (34,0%), `Frozen` (25,2%), `Pantry` (25,1%), `Snacks` (18,6%) — categorías con productos mixtos (algunos declaran, otros no).

Valores no nulos (7 categorías de alérgeno, son los "big allergens" típicos de USA): `Wheat` (1649), `Milk` (1469), `Soy` (815), `Tree nuts` (529), `Peanuts` (521), `Shellfish` (291), `Fish` (271). Es un solo alérgeno por fila, no una lista.

Conclusión: el nulo acá **codifica información real** ("este producto no declara alérgenos"), no un dato perdido. No corresponde imputar ni descartar la columna — conviene tratar el nulo como una categoría más (ej. `"None"`) al momento de encodear.

**`allergens` no es consistente con `ingredients`.** Se probó si el alérgeno declarado aparece como substring dentro de la lista de ingredientes de la misma fila — y no lo es: la misma combinación exacta de ingredientes (ej. "Prepared ingredients, Spices, Salt" en `Frozen`) aparece unas veces con `allergens = Wheat` y otras con `Soy`. El % de coincidencia por alérgeno es 0% para `Fish`, `Peanuts`, `Shellfish`, `Soy`, `Tree nuts`, y solo parcial (55-68%) para `Wheat`/`Milk` (por casualidad de palabras como "Wheat flour"). Conclusión: `allergens` parece generado de forma independiente de `ingredients` (probablemente condicionado solo por `category`, igual que los nulos) — no se puede usar una columna para derivar o validar la otra, son señales separadas.

## Panorama general de columnas (22 en total)

Tipo, % de nulos y cardinalidad (valores únicos) de cada columna, para agrupar por tipo de tratamiento:

| columna | tipo | nulos | nunique |
|---|---|---|---|
| title | str | 0% | 9910 |
| description | str | 0% | 9112 |
| price | float | 0% | 2135 |
| category | str | 0% | 12 |
| timestamp | str | 0% | 9999 |
| query_id | str | 0% | 2012 |
| filter_category | str | 0% | 12 |
| filter_price_min | float | 0% | 393 |
| filter_price_max | float | 0% | 1167 |
| filter_storage_type | str | 0% | 3 |
| cart | bool | 0% | 2 |
| bought | bool | 0% | 2 |
| brand | str | 0% | 15 |
| package_size | str | 0% | 27 |
| unit_of_measure | str | 0% | 5 |
| net_weight_oz | float | 0% | 3753 |
| dimensions_in | str | 0% | 9864 |
| storage_type | str | 0% | 3 |
| ingredients | str | 0% | 190 |
| allergens | str | 44,55% | 7 |
| nutrition_score | int | 0% | 83 |
| country_of_origin | str | 0% | 10 |

Agrupando por tipo de tratamiento que probablemente necesiten:

- **Texto libre / casi-únicos**: `title` (9910), `description` (9112), `dimensions_in` (9864 — llama la atención que una medida física tenga casi tantos valores únicos como filas; a revisar por qué).
- **Categóricas de pocas opciones** (candidatas a one-hot u otro encoding simple): `category`/`filter_category` (12), `storage_type`/`filter_storage_type` (3), `unit_of_measure` (5), `brand` (15), `country_of_origin` (10), `allergens` (7).
- **Numéricas continuas**: `price`, `filter_price_min`, `filter_price_max`, `net_weight_oz`, `nutrition_score`.
- **Identificadores / contexto de búsqueda**: `query_id`, `timestamp`.
- **Booleanas**: `cart` (leakage, no usar como feature), `bought` (target).
- **A revisar más**: `package_size` (27 únicos — sospecho que es texto tipo "10 oz", redundante con `net_weight_oz` + `unit_of_measure`), `ingredients` (190 combinaciones únicas, no texto libre por producto — parece ser un conjunto acotado de listas de ingredientes reutilizadas).

## `dimensions_in`

Formato: texto tipo `"3.3 x 4.0 x 4.1\""` (largo x ancho x alto, en pulgadas).

Hipótesis inicial: no hay un estándar de envase, cada producto tiene su propia medida. Confirmado:
- Volumen (largo×ancho×alto) vs `net_weight_oz`: correlación **0,38** — relación moderada-baja, no es que el peso determine la dimensión de forma exacta (hay variación/ruido por producto).
- Solo **136 de 10.000** filas tienen `dimensions_in` exactamente repetido — casi no hay envases "estándar" reutilizados entre productos distintos.

Decisión para preprocesamiento: no tratar como categórica (quedaría con ~9864 categorías, inmanejable). Conviene parsear el string y descomponerlo en 3 features numéricas (largo, ancho, alto) o resumir en 1 (volumen).

## `package_size` / `unit_of_measure` vs `net_weight_oz`

`unit_of_measure` tiene 5 valores: `oz`, `ct`, `lb`, `gal`, `fl oz`. `package_size` es el texto tipo "10 oz", "12 ct" (número + unidad).

Hipótesis: `package_size`/`unit_of_measure` son la etiqueta nominal del envase, y `net_weight_oz` ya es el peso real medido y normalizado a onzas para todos los productos (resolviendo el problema de unidades mixtas). Verificado:
- Aún para `unit_of_measure = 'oz'`, el número de `package_size` casi nunca coincide exacto con `net_weight_oz` (0,4% de match exacto) — están cerca pero no son iguales (ej. "10 oz" nominal con net_weight_oz = 10.14). Es la diferencia típica entre lo que dice la etiqueta y lo que realmente pesa el producto.
- Para `ct`, `lb`, `gal`, `fl oz` no hay una fórmula de conversión limpia (mismo `package_size` aparece con distintos `net_weight_oz`, ej. "12 ct" con 141.34 en una fila y 155.14 en otra) — son productos distintos con la misma cantidad de unidades pero distinto peso por unidad.

Conclusión: `package_size` y `unit_of_measure` son redundantes para el modelo si usamos `net_weight_oz`, que ya viene en una unidad común. No hace falta parsear/convertir `package_size` — se puede descartar como feature (o dejarlo solo como referencia textual, no como input numérico).

## `ingredients`

Hipótesis: son combinaciones reutilizadas entre productos, donde varía el orden en que aparecen listadas dentro del string. Confirmado:
- `nunique` crudo: 190. Pero al ordenar alfabéticamente los ingredientes dentro de cada fila (para ignorar el orden), baja a solo **12 combinaciones reales** — el resto eran las mismas recetas con el orden mezclado (ej. "Prepared ingredients, Spices, Salt" == "Salt, Prepared ingredients, Spices").
- Vocabulario total: solo **21 ingredientes individuales** distintos (Salt, Natural flavors, Wheat flour, Cream, Milk, Household materials, Baby-safe ingredients, etc. — varios ligados claramente a `category`).

Conclusión para preprocesamiento: no tratar como texto libre ni como 190 categorías. Dos caminos razonables: (a) **multi-hot** sobre los 21 ingredientes individuales (columna binaria por ingrediente), o (b) usar las 12 combinaciones canónicas (ordenadas) como una categórica de 12 valores. A decidir cuál conviene más cuando definamos la arquitectura.

## `price` vs `filter_price_min` / `filter_price_max`

- `price` siempre cae dentro de `[filter_price_min, filter_price_max]` — **100% de las filas**. Los resultados de búsqueda respetan estrictamente el filtro de precio del usuario, nunca se muestra algo fuera de rango.
- Se calculó la **posición relativa del precio dentro del rango filtrado**: `pos = (price - filter_price_min) / (filter_price_max - filter_price_min)`, entre 0 (en el mínimo) y 1 (en el máximo). Tasa de `bought` por bins de esa posición:

| posición relativa | % bought | n |
|---|---|---|
| 0.0 - 0.2 | 7,49% | 1935 |
| 0.2 - 0.4 | 13,35% | 2554 |
| 0.4 - 0.6 | 17,56% | 2569 |
| 0.6 - 0.8 | 13,38% | 2265 |
| 0.8 - 1.0 | 9,01% | 677 |

Forma de campana: se compra menos lo más barato del rango buscado (¿desconfianza de calidad?) y también lo más caro (cerca del tope de lo que el usuario dijo que pagaría), y más lo que queda en el medio. No es simplemente "a menor precio relativo, mayor conversión". **Es una feature con señal real** — conviene calcular esta posición relativa como feature derivada, no solo usar `price` en términos absolutos.

## Encoding de categóricas nominales

Categóricas candidatas a feature, todas **nominales** (sin orden natural entre valores): `category` (12), `storage_type` (3), `brand` (15), `country_of_origin` (10), `allergens` (7, incluyendo el nulo como categoría propia — ver sección de nulos más arriba).

**One-hot encoding** (sugerido por la consigna): convierte una columna categórica de N valores en N columnas binarias (0/1), una por categoría, con un 1 solo en la que corresponde a esa fila. Evita usar un único número (0,1,2,...) para representar las categorías, porque eso le impondría al modelo una relación de orden/magnitud inexistente (ej. "Frozen"=2 no es "el doble" de "Ambient"=1).

Distribución de cada candidata:
- `category`: 12 valores, entre 209 (`Baby`) y 1412 (`Pantry`) filas — desbalance moderado.
- `storage_type`: 3 valores (`Ambient` 5524, `Refrigerated` 2911, `Frozen` 1565) — sin mayor problema.
- `brand`: 15 valores, bastante parejo (597-726 filas cada uno).
- `country_of_origin`: 10 valores, **muy desbalanceado** — 75% (7500/10000) es "United States", el resto se reparte entre 9 países con 245-331 filas cada uno.

**Riesgo del desbalance en one-hot**: no es el mismo problema que el desbalance del target (`bought` 13/87%, que afecta qué tan bien el modelo detecta la clase minoritaria). Acá el riesgo es que una categoría con pocos ejemplos (ej. `country_of_origin = Peru`, 245 filas) le da al modelo poca evidencia para aprender un peso confiable para esa columna one-hot — puede terminar ajustando ese peso a ruido específico de esos casos en vez de señal real y generalizable. Con `country_of_origin`, además, la columna de "United States" (75%) domina el gradiente frente a las otras 9, mucho más débiles. En este dataset no es un caso extremo (la categoría más chica tiene ~210-245 filas, no un puñado), pero vale la pena, más adelante, chequear si `country_of_origin` realmente aporta señal prediciendo `bought` o si conviene simplificarla (ej. "United States" vs "Resto").

## Lista de features (borrador, a confirmar)

Repasando todo lo visto hasta ahora, columna por columna:

**Descartadas — leakage:**
- `cart`: se sabe después del momento de predicción, no puede ser input. (ver sección "Relación cart y bought")

**Descartadas — redundantes con otra columna que ya se usa:**
- `filter_category`: igual a `category` en el 100% de las filas.
- `filter_storage_type`: igual a `storage_type` en el 100% de las filas.
- `package_size` / `unit_of_measure`: redundantes con `net_weight_oz`, que ya normaliza todo a onzas.

**A usar directo (numéricas):**
- `net_weight_oz`
- `nutrition_score` (todavía no lo miramos en detalle — pendiente)

**A usar como derivadas (no la columna cruda):**
- `price` + `filter_price_min` + `filter_price_max` → **posición relativa del precio dentro del rango filtrado** (la "campana" que vimos). Evaluar si además conviene dejar `price` absoluto.
- `dimensions_in` → parsear a 3 numéricas (largo, ancho, alto) o resumir en volumen.

**A usar con one-hot (categóricas nominales, pocas opciones):**
- `category` (12)
- `storage_type` (3)
- `brand` (15)
- `allergens` (7, nulo = categoría propia "sin declarar")

**A usar con encoding especial (multi-valor):**
- `ingredients` → multi-hot sobre 21 ingredientes individuales, o categórica de 12 combinaciones canónicas (a decidir).

**Dudosa / a definir en la experimentación:**
- `country_of_origin`: no mostró señal clara sobre `bought` (rango angosto, diferencias compatibles con ruido). Probar con y sin ella (o simplificada a "US" vs "Resto") en el estudio de ablación.

**Para el modelo Transformer (texto):**
- `title`, `description`: texto libre, van a necesitar tokenización/embeddings (Clase 2) — es la parte más relacionada con la arquitectura Transformer en sí, hay que pensar bien cómo se integra con el resto de las features tabulares.

**Descartada (por ahora) — sin señal encontrada:**
- `timestamp`: se probó la hipótesis de que la hora del día prediga `bought` (ej. alcohol/`Beverages` a la noche, desayuno/`Bakery` a la mañana). Resultado: tasa de `bought` por hora, tanto en general como dentro de `Beverages` y `Bakery`, salta de forma errática entre horas consecutivas (ej. `Beverages` 21h=21,3% pero 20h=10,5%; `Bakery` 6h=25% pero 8h=7,7%), sin un patrón consistente. Con 30-58 filas por hora dentro de cada categoría, es ruido estadístico, no señal real. No se probaron otras granularidades (día de la semana, fin de semana vs. semana) — se podría revisar si hace falta, pero por ahora no hay evidencia de que `timestamp` aporte al target.

**Descartada — sin señal, pero útil como clave de agrupación:**
- `query_id`: el ID en sí no debe usarse como feature (es un identificador arbitrario, no generaliza — el modelo podría "memorizar" IDs de train en vez de aprender algo real). Se probó la feature derivada "cantidad de productos mostrados en esa búsqueda" (`n_resultados`, entre 2 y 8): tasa de `bought` se mantiene entre 12-15% sin importar el tamaño de la búsqueda, sin señal clara. Conclusión: no se encontró ninguna feature útil derivada de `query_id` — pero sigue siendo necesario como **clave para agrupar filas y calcular el BTR agregado por búsqueda** (no como input del modelo).

## Split train / valid / test

> **Nota: esto pertenece al Ejercicio 2** ("Desarrollo del sistema"), no al Ejercicio 1 — el PDF lo pide explícitamente ahí ("¿Cómo particiono mi conjunto de datos? Sugerencia: recordar train/valid/test split", primer aspecto de diseño del Ejercicio 2). Se dejó discutido y anotado acá porque surgió naturalmente charlando el EDA, pero cuando se arme la carpeta `ejercicio2/` esta sección se debería mover/reubicar ahí, y no debería figurar en el informe final del Ejercicio 1.

**Por qué agrupar por `query_id`:** las filas están agrupadas por búsqueda — cada `query_id` trae varios productos que comparten contexto (misma categoría, mismo rango de precio filtrado). Si se particiona fila por fila al azar, filas de una misma búsqueda podrían terminar repartidas entre train y test: el modelo ya habría visto ese contexto exacto (categoría, rango de precio) durante el entrenamiento, lo cual no refleja el caso real de uso (predecir sobre búsquedas *nuevas* que nunca se vieron). Por eso: **la partición se hace por `query_id` completo** — todas las filas de una búsqueda van juntas al mismo split, nunca mezcladas.

**Por qué 3 particiones y no 2:** el PDF de la consigna lo pide explícitamente ("recordar train/valid/test split"). Además tiene sentido con el Ejercicio 2 (experimentar con configuraciones del modelo):
- **train**: ajusta los pesos del modelo.
- **valid**: compara entre configuraciones/experimentos (arquitectura, `d_model`, etc.) durante la iteración.
- **test**: se toca una sola vez, al final, para el número que se reporta — si se usara valid para elegir la mejor configuración y también para reportar el resultado final, el número quedaría inflado (overfit a valid).

**Precedente en TPs anteriores (SIA-TP5, autoencoders):** usaron **K-fold estratificado** como estrategia principal (entrenar con varios folds y promediar resultados, `mean`/`std` en el output), con un `val_fraction=0.2` (80/20) como caso simple para cuando no se hacían folds (`mlp/data.py::train_val_split`). Esto conecta con algo que también dice el audio de `consigna.VTT` de esta materia: la profesora recomienda "promediar varias corridas (o cross-validation) en vez de una sola ejecución" — o sea, el mismo criterio de TP5 tiene aval acá también.

**Decisión para este TP:** se eligió la opción más simple — **un solo split fijo train/valid/test**, agrupado por `query_id` — en vez de K-fold, para no complicar de entrada dado el tiempo disponible. Queda como posibilidad futura correr el split final con 2-3 semillas distintas si da el tiempo, como forma liviana de aplicar la recomendación de "promediar corridas" sin ir a K-fold completo.

**Pendiente:** definir las proporciones exactas (ej. 70/15/15) y si además de agrupar por `query_id`, conviene estratificar por `bought` a nivel de query (por ejemplo, usando la tasa de compra de cada búsqueda) para que el desbalance de clases quede parejo entre splits — no se decidió todavía.

## Preprocesamiento — numéricas: normalización

Las features numéricas están en escalas muy distintas entre sí: `price` (1.2 - 35), `net_weight_oz` (2.8 - 155), `nutrition_score` (0 - 99), y las derivadas (posición relativa del precio ya está en [0,1], dimensiones de `dimensions_in` en pulgadas). Sin normalizar, una columna con magnitud más grande (ej. `net_weight_oz`) pesaría más que otra solo por su escala, no porque sea más relevante para predecir `bought`.

Decisión: normalizar todas las features numéricas antes de pasarlas al modelo, con **z-score** (`(x - media) / desvío`).

Por qué z-score y no min-max:
- Min-max (`(x - min) / (max - min)`) comprime todo a `[0,1]` pero es muy sensible a outliers — un solo valor extremo aplasta al resto de los datos en un rango chico cerca de 0.
- `net_weight_oz` en particular tiene una distribución asimétrica con cola larga (rango 2.8-155, media 28.7, desvío 29 — hay productos livianos mayoría y pocos bien pesados). Con min-max, la mayoría quedaría apretada cerca de 0 mientras los pocos productos pesados estiran la escala. Con z-score ese efecto pesa menos.
- Consistencia con el criterio ya usado en TP4 (SIA) para PCA/Kohonen/Oja, donde también se estandarizó con z-score.

## `title` — señal escondida: tag de reputación entre paréntesis

Pregunta que motivó esto: ¿para qué sirve el texto (`title`/`description`) si el objetivo es predecir `bought`? No dar por sentado que aporta solo porque la consigna pide un Transformer — se buscó evidencia.

Se notó que muchos `title` terminan con un tag entre paréntesis (ej. "... (Well Reviewed)", "... (Limited Feedback)"). Se extrajo con regex (`\(([^)]+)\)\s*$`) y se cruzó con `bought`:

| tag en `title` | % bought | n |
|---|---|---|
| Customer Favorite | 67,75% | 493 |
| Best Seller | 65,74% | 470 |
| Top Rated | 62,71% | 472 |
| #1 Pick | 62,50% | 496 |
| Well Reviewed | 3,77% | 477 |
| Shopper Favorite | 2,76% | 507 |
| Highly Rated | 2,12% | 520 |
| Popular Choice | 1,92% | 469 |
| (sin tag) | 0,00% | 511 |
| Discontinuing Soon | 0,00% | 550 |
| Limited Feedback | 0,00% | 524 |
| Low Feedback | 0,00% | 466 |
| Current Stock | 0,00% | 515 |
| Rarely Reordered | 0,00% | 500 |
| Recently Added | 0,00% | 481 |
| Regular Listing | 0,00% | 494 |
| Standard Listing | 0,00% | 506 |
| Clearance Listing | 0,00% | 510 |
| Unrated Listing | 0,00% | 517 |
| New Listing | 0,00% | 522 |

**Es la señal más fuerte encontrada en todo el EDA**, muy por encima del promedio general (13,01%). 4 tags concentran casi toda la probabilidad de compra (62-68%), 4 tags dan algo de señal baja (2-4%), y las 11 restantes (incluyendo "sin tag") están en 0%.

**Decisión de diseño**: aunque el patrón es muy regular (siempre entre paréntesis al final del título, fácil de parsear con regex y convertir en categórica), **se decidió NO parsearlo aparte** — dejar `title` como texto crudo para que el modelo lo aprenda vía tokenización + embeddings + atención. Parsearlo manualmente le sacaría al Transformer justo la parte de "entender el texto" que la consigna pide mostrar que se comprende (foco del TP: comprensión de la arquitectura, no solo el mejor resultado numérico). Esto también da una justificación concreta y verificable de *por qué* tiene sentido meter un Transformer sobre el texto en este problema — no es solo un requisito formal de la consigna, hay señal real ahí escondida.

## Para la presentación

- [ ] Gráfico de la "campana" de `bought` según posición relativa del precio dentro del rango filtrado (ver tabla en sección `price` vs `filter_price_min/max`) — mostrar visualmente el efecto de no comprar ni lo más barato ni lo más caro del rango buscado.
- [ ] Gráfico de boxplots (o rangos) de las features numéricas antes/después de normalizar, para mostrar visualmente por qué hacía falta (escalas muy distintas: `price` vs `net_weight_oz` vs `nutrition_score`).
- [ ] **Gráfico de barras de % bought por tag de reputación en `title`** — el hallazgo más fuerte de todo el EDA, buena pieza para justificar por qué usar el Transformer sobre texto.

## `country_of_origin` vs `bought`

Tasa de `bought` por país (promedio general: 13,01%):

| país | % bought | n |
|---|---|---|
| Vietnam | 16,61% | 271 |
| Italy | 14,55% | 275 |
| Peru | 13,88% | 245 |
| Mexico | 13,77% | 305 |
| United States | 12,99% | 7500 |
| Canada | 12,73% | 267 |
| New Zealand | 12,32% | 284 |
| Spain | 12,08% | 331 |
| Chile | 11,11% | 261 |
| Thailand | 10,73% | 261 |

Rango angosto (10,7% - 16,6%), sin ningún país que se dispare claramente. Para los 9 países que no son EEUU, cada uno tiene solo 245-331 filas — con muestras tan chicas, una diferencia de 2-3 puntos porcentuales es compatible con ruido estadístico (con n≈245 y tasa base 13%, el margen esperado "por azar" ronda ±4 puntos).

Conclusión: `country_of_origin` **no muestra señal clara** sobre `bought`. Candidata a simplificar (ej. "United States" vs "Resto") o directamente evaluar en el estudio de ablación si aporta algo al predecir.
- [ ] Definir lista final de features candidatas y descartar las que generan leakage (`cart` entre ellas).
- [ ] Pensar cómo particionar train/valid/test dado el desbalance de clases (no vimos en clase ninguna técnica específica para esto — no asumir nada, decidirlo entre nosotras cuando lleguemos a ese punto).
