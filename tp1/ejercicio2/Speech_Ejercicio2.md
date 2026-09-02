# Speech — Ejercicio 2 (Desarrollo del sistema)

Guión para presentar `presentacion_tp1.pdf` completo, slide por slide (30 páginas: portada, split, métricas, arquitectura general, setup, Experimentos 1 a 11, modelo final y evaluación en test).

**Cómo usar esto:** no es para leer literal arriba del escenario — es para que cada uno lo lea, lo entienda, y lo diga con sus propias palabras. Donde dice "nosotros decidimos X porque Y", el Y tiene que ser algo que puedan explicar si alguien del jurado pregunta "¿y por qué eso, específicamente?" sin volver a leer la slide. Marco en **[si preguntan más]** algunas líneas de repregunta típicas con la respuesta ya pensada.

---

## Slide 1 — Portada "Ejercicio 2: Desarrollo del sistema"

Arranque corto, no hay contenido técnico todavía:

> "Esta parte es el desarrollo del sistema: cómo armamos el split, cómo definimos la arquitectura, y el estudio de ablación — ir cambiando un componente del Transformer a la vez, con evidencia de por qué cada cambio ayuda o no. No arrancamos tirando una arquitectura grande a ver qué pasa: cada decisión que van a ver a partir de acá tiene un experimento propio atrás, con datos, no con intuición."

Esto planta la idea central de toda la sección: **todo lo que viene está justificado con una corrida real**, no elegido a ojo. Vale la pena decirlo acá arriba porque es el hilo que conecta las 11 slides siguientes.

---

## Slide 2 — "Split train / valid / test"

**Qué se ve en pantalla:** por qué agrupar por `query_id`, y para qué sirve cada uno de los tres splits.

**Speech:**

> "Antes de tocar el modelo, hay una decisión de split que tiene que ver directamente con cómo funciona la evaluación de cualquier modelo: necesitamos que valid y test midan generalización real, no memoria.
>
> El dataset no es un conjunto de filas independientes entre sí. Cada `query_id` agrupa varios productos que se mostraron en la misma búsqueda, compartiendo los mismos cuatro filtros exactos — categoría, tipo de almacenamiento, y el rango de precio. Si particionáramos fila por fila al azar, es perfectamente posible que dos productos de la misma búsqueda —con los mismos filtros, compitiendo por el mismo click— terminen uno en train y el otro en test. El modelo no estaría generalizando a una búsqueda nueva, estaría viendo una versión levemente distinta de una búsqueda que ya vio en parte durante el entrenamiento. Esa correlación infla la métrica sin que el modelo haya aprendido nada más.
>
> Por eso la partición mueve la búsqueda completa —todas sus filas— al mismo split. Es la misma lógica de fondo por la que en el split de train/test en general uno evita que información quede filtrada de un lado al otro: acá el vector de filtración no es el target, es el contexto compartido entre filas hermanas.
>
> Y por qué tres particiones y no dos: train ajusta los pesos. Valid es el que consultamos una y otra vez a lo largo de los 11 experimentos para decidir qué cambio de arquitectura conservar. Test se toca una sola vez, al final — si lo usáramos también para elegir la config ganadora, el número final quedaría inflado por el mismo motivo que cualquier selección de modelo sobre el propio set de evaluación: terminaríamos eligiendo la configuración que más le acertó a las particularidades de ese split puntual, no la que generaliza mejor."

**[si preguntan más] "¿Por qué no usar directamente time-based split?"** → "Se evaluó esa opción. En este dataset el histórico no es muy profundo (~2 años) y no es un requisito para este TP — lo dejamos como simplificación consciente, no como algo que no se nos ocurrió. En un caso de producción real sí correspondería, porque mezclar fechas puede inflar la métrica por estacionalidad."

---

## Slide 3 — "Split: 70/15/15 estratificado por tasa de bought"

**Qué se ve en pantalla:** los 4 paneles de balance del split, y la tabla comparando 70/15/15 contra 80/10/10.

**Speech:**

> "Antes de entrar en la proporción, vale la pena explicar el panel de la derecha —composición por franja de bought-rate—, porque es la prueba de que la estratificación funcionó, no solo el argumento de por qué la hicimos. Agrupar por `query_id` evita que una búsqueda quede partida entre splits, pero no garantiza que los tres splits tengan una mezcla parecida de *tipos* de búsqueda: por puro azar, un split al azar podría terminar con una proporción distinta de búsquedas sin ninguna compra frente a búsquedas con mucha conversión, y eso rompería la comparabilidad entre train, valid y test aunque cada búsqueda individual esté bien asignada.
>
> Por eso estratificamos: agrupamos las 2.012 queries en tres franjas según su propia tasa de bought — 0% (sin ninguna compra, más de la mitad del dataset, 1.058 de 2.012 queries, un caso cualitativamente distinto, no es 'poca conversión', es 'cero'), 1-33% y 34-100% — y usamos esa franja como la clase a estratificar en el split, en vez de la tasa exacta, porque muchos valores puntuales (0,57, 1,00) tienen apenas un puñado de queries en todo el dataset, insuficiente para repartir en tres splits de forma confiable.
>
> El panel de composición es lo que confirma que esa estratificación agarró: para cada split, qué porcentaje de sus queries cae en cada franja. Y da prácticamente idéntico en los tres — 52,6% / 31,2% / 16,2% en train, 52,6% / 31,1% / 16,2% en valid, la misma composición en test. Esto mide algo distinto de lo que muestra el panel de la izquierda: ese otro panel promedia —la tasa de bought a nivel fila, que también da parecida entre splits—, pero un promedio parecido se puede armar con mezclas de queries completamente distintas: un split con muchas queries en 0% compensadas por unas pocas con conversión altísima, y otro split con puras queries de conversión media, podrían dar el mismo promedio y esconder que en realidad no vieron el mismo tipo de búsquedas. El panel de composición muestra que acá no pasa eso: los tres splits ven, en la misma proporción, búsquedas sin compra, con compra baja y con compra alta — el modelo entrena y se evalúa sobre una mezcla comparable de estos tres comportamientos de búsqueda, no solo sobre un promedio que coincide por casualidad.
>
> Con el criterio de agrupar por `query_id` ya resuelto, y la estratificación confirmada, quedaba la proporción. Achicar el train tiene un costo real: `bought` está desbalanceado, 13% de positivos, así que un train más chico da menos ejemplos positivos para aprender el patrón de compra. Pero agrandar train achicando valid y test tiene otro costo, menos obvio: con muy pocas filas, el PR-AUC que medimos deja de ser una medición confiable y empieza a ser, en parte, ruido de muestreo.
>
> Esto lo llevamos a un número, no lo dejamos en argumento teórico: con 80/10/10, test queda con 956 filas y sólo 123 positivos — el mismo orden de magnitud que los grupos de `country_of_origin` que en el EDA de Ejercicio 1 ya habíamos visto que mostraban diferencias de 2-3 puntos por pura casualidad de muestra chica. Con 70/15/15, test tiene 201 positivos, un 56% más para estimar la misma curva Precisión-Recall.
>
> Y acá está el punto que más vale la pena remarcar: con una sola semilla fija, mirando nada más el resultado puntual, casi elegimos mal — con `random_state=42` la tasa de bought de test daba más parecida a la tasa global usando el split más chico (80/10/10), lo cual parecía sugerir que 80/10/10 era mejor. Con una sola corrida no hay forma de distinguir si eso es patrón real o casualidad de esa semilla en particular. Por eso repetimos el split 300 veces con semillas distintas y medimos, en promedio, qué tan lejos queda la tasa de bought de valid/test respecto de la tasa real del dataset completo. Ahí sí, en las 300 corridas, 70/15/15 gana en promedio tanto en valid como en test — la corrida de la semilla 42 donde parecía ganar 80/10/10 era la excepción, no la regla. Es la misma idea de fondo del promediar-varias-semillas que van a ver en cada experimento de acá en adelante: una sola corrida puede engañar, por eso no confiamos en un solo número."

**[si preguntan más] "¿Y esto no debería validarse con el modelo entrenado, no solo con la tasa de bought?"** → "Correcto, y es justo la limitación que dejamos anotada: lo que falta es repetir esta comparación pero midiendo cuánto varía el PR-AUC del modelo real entre las 300 corridas de cada proporción, no solo la tasa de bought. Es evidencia real a favor de 70/15/15, pero no reemplaza esa validación más completa."

---

## Slide 4 — "Cómo evaluamos el modelo"

**Qué se ve en pantalla:** definiciones de PR-AUC y ROC-AUC, overfitting como gap, promedio sobre semillas, selección de mejor época.

**Speech:**

> "La consigna pide evaluar sin threshold — no queremos una decisión sí/no con un corte fijo de probabilidad, queremos saber si el modelo ordena bien: si a un producto que efectivamente se compró le asigna más probabilidad que a uno que no. Eso es exactamente lo que miden PR-AUC y ROC-AUC, cada una barriendo todos los cortes posibles en vez de fijar uno.
>
> La razón por la que miramos PR-AUC primero, y no ROC-AUC, tiene que ver directamente con el desbalance de clases: 87% de las filas son negativas. ROC-AUC mira la tasa de falsos positivos, que se calcula sobre el total de negativos — con tantos negativos 'fáciles' de acertar, la tasa de falsos positivos se mantiene baja casi sin esfuerzo, y el número puede verse artificialmente alto (0,94-0,97 en casi todos nuestros experimentos) sin que eso signifique que el modelo distingue bien la clase que realmente nos importa. PR-AUC en cambio no le da ningún crédito por acertar negativos — compara contra la prevalencia real (13%, no contra 0), así que un modelo sin señal alguna da 0,13, no un valor engañosamente alto. Por eso PR-AUC es la métrica que usamos para decidir entre configuraciones a lo largo de todo el estudio, y ROC-AUC queda como chequeo cruzado.
>
> El gap de PR-AUC entre train y valid es nuestro termómetro de sobreajuste: si el modelo memoriza patrones específicos de las 7.012 filas de train que no se sostienen en valid, ese gap crece con las épocas. Lo vamos a ver repetirse en cada experimento — no es solo mirar qué configuración da el número más alto en valid, es mirar también si ese número vino acompañado de un gap chico o de uno que se disparó.
>
> Y por último, cada configuración corre con 3 semillas — 0, 1 y 2 — y reportamos media y desvío, no un solo número. Esto no es un detalle prolijo: el punto de partida del entrenamiento (cómo se inicializan al azar los pesos) puede aterrizar en soluciones distintas dentro de una superficie de pérdida que no es convexa, así que una sola corrida puede ganar o perder por esa inicialización, no por la arquitectura en sí. Van a ver en el Experimento 1 mismo un caso muy concreto de esto: una semilla con gap de overfitting de 0,114 contra 0,02-0,04 de las otras dos — si hubiéramos corrido una sola vez, y nos tocaba esa semilla, la conclusión sobre esa arquitectura hubiera sido otra."

---

## Slide 5 — "Arquitectura general del sistema"

**Qué se ve en pantalla:** por qué Encoder-only y no Decoder-only ni Encoder-Decoder, y por qué texto y tabular se concatenan recién antes de la salida.

**Speech — esta es la slide más importante para demostrar que entendemos la arquitectura, tomarse el tiempo acá:**

> "Un bloque Encoder tiene dos partes, cada una envuelta en una conexión residual y una normalización: primero self-attention multi-cabeza, después una red feed-forward que se aplica posición a posición. La conexión residual suma la entrada de la capa a su salida antes de normalizar — eso le da al gradiente, durante el backward, un camino directo hacia atrás que no depende de atravesar toda la transformación no lineal, mitigando el problema de que el gradiente se desvanezca a medida que se apilan capas. La normalización después de esa suma reescala todo a la misma escala antes de pasar a la siguiente capa. Esto es simplemente cómo está construido un Encoder — no lo armamos nosotros a mano, usamos `nn.TransformerEncoderLayer` de PyTorch, que ya implementa exactamente esto.
>
> La diferencia entre Encoder y Decoder no es esa estructura — es una máscara. El Decoder agrega una máscara que le impide a cada posición atender a posiciones futuras en la secuencia, porque genera texto token por token y en el momento de predecir el token N no puede haber visto el token N+1 todavía. El Encoder no tiene esa restricción: cada token puede atender libremente a cualquier otro token de la secuencia, tanto para adelante como para atrás.
>
> Nuestro caso es exactamente el escenario donde esa máscara no hace falta: no generamos texto, `title` y `description` están completos y disponibles enteros en el momento de predecir `bought`. No hay nada "futuro" que esconder. Necesitamos comprimir una secuencia entera en una representación que capture su contenido — eso es tarea de Encoder, no de Decoder. Y Encoder-Decoder completo tampoco aplica: esa arquitectura está pensada para mapear una secuencia a otra secuencia distinta, como traducir de un idioma a otro — nosotros no transformamos una secuencia en otra secuencia, la resumimos en un vector para clasificar.
>
> La segunda decisión de esta slide es por qué texto y tabular se concatenan recién en la última capa, en vez de meter lo tabular dentro del Transformer como si fuera un token más. La razón de fondo está en qué hace mecánicamente la atención: self-attention calcula, para cada posición, una combinación ponderada de los vectores 'value' del resto de la secuencia, con pesos que salen de comparar 'query' contra 'key' — es un mecanismo pensado para relaciones entre elementos de una secuencia ordenada, donde importa qué tan cerca o lejos está un token de otro y en qué orden aparecen. Las 75 features tabulares no tienen esa estructura: no hay una noción de 'la columna anterior' o 'la columna siguiente', es un vector plano de atributos del mismo producto. Forzarlas a pasar por atención sería aplicarle a datos sin orden un mecanismo construido específicamente para aprovechar el orden — no hay nada que ganar ahí, y si lo intentáramos ni siquiera podríamos justificar por qué elegimos un orden arbitrario de columnas como si fuera una secuencia.
>
> Además, combinar recién al final nos da una ablación limpia: si queremos sacar el Transformer del sistema y quedarnos solo con lo tabular, es desconectar una rama entera, no rediseñar la arquitectura. Esa separación de ramas es la que después, en los Experimentos 7 y 8, nos permite aislar exactamente qué está aportando el texto y qué está aportando lo tabular."

**[si preguntan más] "¿Por qué no usar `[CLS]` como BERT en vez de mean-pooling?"** → (esto no está en esta slide del PDF pero puede salir en preguntas) "Evaluamos esa opción y la descartamos: el material de la materia no explica el uso de `[CLS]` como técnica de pooling para clasificación, solo aparece atado a cómo BERT arma su entrada de pre-entrenamiento. Usarlo hubiera sido importar una convención que no vimos desarrollada. Mean-pooling en cambio se apoya en algo que sí vimos: la atención en sí misma se explica como un promedio ponderado de vectores value — mean-pooling es el caso particular de pesos uniformes de esa misma idea."

---

## Slide 6 — "Setup de entrenamiento"

**Qué se ve en pantalla:** tabla con Adam, batch size 128, semillas 0/1/2, 20 épocas.

**Speech:**

> "Antes de entrar experimento por experimento, esta tabla son los valores que quedaron fijos en las 11 corridas — no son diales que hayamos ido variando, son la base común sobre la que se compara cada cambio de arquitectura. La decisión de no convertir a Adam, el learning rate o el batch size en diales de ablación fue una decisión de alcance: con el tiempo disponible, decidimos concentrar el estudio en los parámetros que efectivamente definen la arquitectura del Transformer y su capacidad — heads, layers, `d_model`, el ancho del feed-forward — en vez de repartir ese tiempo también en hiperparámetros de entrenamiento estándar para los que existen defaults ampliamente usados en la práctica.
>
> Lo que sí tiene justificación puntual son las semillas: usamos siempre las mismas tres — 0, 1 y 2 — en todos los experimentos, nunca semillas distintas de una corrida a otra. Si cada experimento usara semillas diferentes, una diferencia de resultado entre dos configuraciones podría deberse a qué semillas le tocaron a cada una, no al cambio de arquitectura que estamos evaluando. Fijarlas aísla esa fuente de variación del efecto que realmente queremos medir."

---

## Slide 7 — "Experimento n°1"

**Qué se ve en pantalla:** objetivo (aislar el comportamiento del Transformer) y arquitectura (`d_model=16`, `n_heads=1`, `n_layers=1`, `dim_feedforward=64`, positional encoding senoidal, cabeza lineal).

**Speech:**

> "Este es el punto de partida de todo el estudio: solo `title`+`description` a través del Transformer, sin ninguna feature tabular todavía. La razón de arrancar así es aislar una pregunta a la vez — antes de complicar la arquitectura combinando texto con tabular, queríamos saber si hay señal real aprovechable en el texto solo. En el EDA del Ejercicio 1 habíamos encontrado un tag de reputación entre paréntesis al final de `title` con una relación durísima con `bought` — 67% de compra en los tags positivos contra 0% en la mayoría del resto. La pregunta que responde este experimento es si la atención puede efectivamente capturar eso.
>
> Cada valor de esta arquitectura tiene una razón de ser chica a propósito, no aleatoria. `d_model=16` es la dimensión que atraviesa todo el bloque: es el ancho del embedding de cada token, y también el ancho de las proyecciones Q, K y V, y el ancho del vector que sale del Encoder. Elegir 16 fuerza al modelo a comprimir agresivamente cada uno de los 412 tokens del vocabulario en apenas 16 números — es la prueba más exigente posible de si la atención sola, con muy poco lugar para guardar información, ya alcanza para encontrar ese patrón.
>
> `n_heads=1` es el mínimo que sigue siendo, técnicamente, atención multi-cabeza — se deja en 1 a propósito para poder aislar el efecto de subir a 2 heads en el próximo experimento sin mezclarlo con otro cambio. `n_layers=1` es la unidad más chica que la teoría define como 'un Encoder': una capa de self-attention más una feed-forward, cada una con su residual y su normalización — no hay una versión más chica que siga siendo, por definición, un Encoder.
>
> El positional encoding es senoidal, una función fija de la posición, no una tabla de embeddings que se entrena — eso significa que no agrega ni un parámetro nuevo al modelo, coherente con el criterio de arrancar chico en todo sentido, no solo en `d_model`. Y la cabeza de salida es un `Linear` directo del vector del Encoder a un solo logit, sin capa intermedia — la versión más mínima posible de clasificador, para no esconder si la señal viene del Transformer o de una cabeza con capacidad propia."

---

## Slide 8 — Resultados Experimento 1

**Qué se ve en pantalla:** curvas train/valid de PR-AUC y ROC-AUC, tabla con PR-AUC 0,688 ± 0,024, ROC-AUC 0,954 ± 0,012, gap 0,058 ± 0,049.

**Speech:**

> "El número clave para leer este resultado es la prevalencia: 0,130. Es lo que daría un modelo que ignora completamente el input y ordena al azar los productos por probabilidad de compra. Nuestro PR-AUC de valid da 0,688 — más de cinco veces ese piso. Con apenas 16 dimensiones de compresión y un solo head, el Transformer ya está encontrando señal real en el texto, no ruido.
>
> El gap de overfitting confirma por qué insistimos tanto con promediar semillas: en promedio da 0,058, pero el desvío estándar es casi tan grande como la media — 0,049. Eso significa que las tres semillas no se comportan igual: una tiene un gap de 0,114 mientras las otras dos están en 0,02-0,04. Con un modelo de menos de diez mil parámetros y un dataset de siete mil filas, veinte épocas ya alcanzan para que al menos una inicialización empiece a sobreajustar, aunque las otras dos todavía no. Si hubiéramos corrido una sola semilla, la conclusión sobre 'cuánto sobreajusta esta arquitectura' hubiera dependido pura y exclusivamente de cuál nos tocó.
>
> En el gráfico se ve además algo consistente con la explicación de la slide anterior sobre por qué usamos PR-AUC como métrica principal: las curvas de train y valid en PR-AUC se separan de forma sostenida a partir de la época 5, mientras que en ROC-AUC casi no hay separación visible. Es exactamente el efecto del desbalance de clases — ROC-AUC, al no penalizar tanto los negativos fáciles, es menos sensible a mostrar el sobreajuste que PR-AUC sí deja ver con claridad.
>
> La conclusión de este experimento no es solo 'el modelo funciona' — es que confirma con evidencia empírica algo que en el Ejercicio 1 era solo una hipótesis del EDA: el tag de reputación, y el contenido de `title`/`description` en general, es predictivo, y el Transformer lo está aprovechando incluso en la configuración más chica posible. Con esa base confirmada, tiene sentido seguir ablacionando desde acá hacia arriba en vez de agrandar todo de una."

---

## Slide 9 — "Experimento n°2 - aumento de heads"

**Qué se ve en pantalla:** objetivo (si separar tipos de relación entre palabras ayuda) y arquitectura (`n_heads=2`, resto igual).

**Speech:**

> "Acá subimos `n_heads` de 1 a 2, dejando todo lo demás exactamente igual que el Experimento 1, para aislar solamente este cambio.
>
> Vale la pena explicar qué hace mecánicamente separar en más de un head, porque es la parte que más se presta a confusión. Con un solo head, el modelo calcula una única combinación ponderada de todo el resto de la secuencia para cada token, basada en un único patrón de similitud entre queries y keys. Con dos heads, en cambio, el mismo `d_model=16` se reparte en dos subespacios de 8 dimensiones cada uno, y cada subespacio calcula su propio patrón de atención en paralelo, con su propia proyección Q/K/V independiente — un head podría terminar especializándose en captar, por ejemplo, la relación entre el tag de reputación y el resto de la oración, y el otro en alguna otra regularidad del texto. Al final, las salidas de los dos heads se concatenan de vuelta a los mismos 16 valores totales.
>
> Esa mecánica es justamente la razón por la que el conteo de parámetros no cambia entre este experimento y el anterior: 9.889 en los dos casos. Las proyecciones Q, K, V y de salida siguen siendo matrices de `d_model × d_model` en total, sin importar en cuántos heads se reparta esa dimensión internamente — heads no agrega parámetros nuevos, reparte los que ya existían en subespacios paralelos más chicos. Es una distinción importante: agregar heads no es lo mismo que agrandar el modelo."

---

## Slide 10 — Resultados Experimento 2

**Qué se ve en pantalla:** curvas train/valid, tabla comparando Exp. 1 vs Exp. 2, conclusión sobre estabilidad.

**Speech:**

> "El resultado es exactamente coherente con la mecánica que acabamos de explicar: si heads no agrega parámetros, no debería moverse la performance pico — y no se mueve, 0,688 en los dos experimentos, prácticamente idéntico.
>
> Lo que sí cambia, y bastante, es la estabilidad entre semillas: el desvío estándar de PR-AUC baja de 0,024 a 0,007 — más de tres veces menos variable. Nuestra hipótesis para explicar esto, que dejamos anotada como hipótesis y no como conclusión cerrada porque son solo 3 semillas: con un solo head, toda la capacidad de encontrar un patrón de atención útil depende de una única inicialización aleatoria de una matriz de 16×16 — si esa inicialización arranca en una zona poco favorable de la superficie de pérdida, el entrenamiento entero puede quedar condicionado por eso. Con dos heads, en cambio, hay dos inicializaciones independientes de 8×8 corriendo en paralelo, cada una explorando su propio subespacio — es menos probable que las dos arranquen mal al mismo tiempo, y el resultado final termina siendo un promedio más consistente entre semillas.
>
> La conclusión que sacamos de este experimento es específica al rango en el que estamos trabajando: con `d_model=16` y un texto de a lo sumo 45 tokens, `n_heads` no es el cuello de botella de performance de esta arquitectura — es un dial que afecta más la estabilidad del entrenamiento que la capacidad del modelo. Eso nos llevó a priorizar los diales que sí controlan capacidad real —la cantidad de encoders apilados, primero— en vez de seguir barriendo heads."

---

## Slide 11 — "Experimento n°3 - layers"

**Qué se ve en pantalla:** objetivo, arquitectura con `n_layers = 1 / 2 / 4 / 8`.

**Speech:**

> "Este experimento cambia algo estructuralmente distinto a heads, y es importante remarcar la diferencia porque es fácil confundirlas. Apilar encoders no reparte una capacidad existente — cada encoder nuevo trae su propio set completo de proyecciones Q, K, V y de salida, más su propia red feed-forward, todo con parámetros propios que se suman a los que ya había. Además, la segunda capa no recibe el embedding original: recibe la salida de la primera capa, que ya es una versión de cada token contextualizada con el resto de la secuencia a través de esa primera atención. Entonces la segunda capa puede volver a aplicar atención, pero ahora sobre relaciones de segundo orden — relaciones entre tokens que ya están mezclados entre sí, no entre los embeddings crudos.
>
> Por eso acá sí esperábamos que se moviera tanto la capacidad del modelo como el riesgo de sobreajuste, a diferencia de lo que vimos con heads. Probamos una progresión de duplicado — 1, 2, 4 y 8 capas — para cubrir un orden de magnitud completo de profundidad sin tener que correr un barrido exhaustivo de todos los valores intermedios."

---

## Slide 12 — Resultados Experimento 3

**Qué se ve en pantalla:** barrido de `n_layers`, tabla con PR-AUC/ROC-AUC/gap por valor, conclusión.

**Speech:**

> "`n_layers=2` gana en PR-AUC de valid —0,696 contra 0,688 de una sola capa— y además tiene el gap de overfitting más chico de los cuatro valores probados, 0,028, la mitad que con una sola capa. No es solo el número más alto por casualidad: mejora en dos frentes al mismo tiempo, performance y generalización, lo cual lo hace una elección sólida y no un empate estadístico que ganó por ruido.
>
> Lo interesante, y coherente con la mecánica que explicamos en la slide anterior, es lo que pasa después: con 4 y con 8 capas, tanto PR-AUC como ROC-AUC bajan de forma monótona, y la variabilidad entre semillas en 8 capas es más del doble que en 2. Esto no es que 'más profundidad sea mala' en abstracto — es que cada capa nueva suma un set completo de matrices Q/K/V/feed-forward propias, y con solo 7.012 filas de train, en 8 capas ya estamos en 32.849 parámetros, más de tres veces los de 2 capas, sin que haya más señal real en el texto para que esos parámetros extra aprendan algo útil. El resultado no es que el modelo 'decide' sobreajustar — es que optimizar una red bastante más grande, en pocas épocas y con pocos datos, se vuelve genuinamente más difícil de estabilizar, y esa dificultad de optimización es la que vemos reflejada en la caída de PR-AUC y en la mayor varianza entre semillas.
>
> Con esto, `n_layers=2` queda fijo, y la arquitectura base para seguir ablacionando pasa a ser `n_heads=1, n_layers=2, d_model=16, dim_feedforward=64`. Los próximos diales pendientes en el estudio —que todavía no están armados como slide— son `d_model` y el ancho del feed-forward interno, los dos diales de capacidad que faltan antes de pasar al sistema completo con las features tabulares."

---

## Slide 13 — "Experimento n°4 - d_model"

**Qué se ve en pantalla:** objetivo (ver si el modelo venía corto de capacidad) y arquitectura (`d_model = 8/16/32/64/128/256` — la slide dice "254", es una errata, en los resultados y en `Experimentos.md` el valor real es 256).

**Speech:**

> "Con `n_layers=2` ya cerrado, quedan dos diales de capacidad por explorar, y los separamos a propósito en dos experimentos distintos para no mezclar sus efectos: `d_model` acá, el ancho del feed-forward interno en el próximo.
>
> Vale la pena ser precisos con qué es `d_model`, porque no es 'un parámetro más' — es la dimensión que atraviesa literalmente todo el bloque. Es el ancho de cada vector de embedding, y por lo tanto también el ancho de las proyecciones Q, K, V y de salida de la atención (cada una es una matriz de `d_model × d_model`), y el ancho del vector que circula por las conexiones residuales entre sub-capas. Subir `d_model` no agranda una sola pieza, agranda el modelo entero, en dos frentes a la vez: más capacidad para representar cada token (16 dimensiones fuerzan una compresión mucho más agresiva de los 412 tokens del vocabulario que 64), y más parámetros en cada una de las cuatro proyecciones de atención, que crecen cuadráticamente con `d_model` porque son matrices `d_model × d_model`.
>
> Corrimos primero hasta 64 (dentro del rango sugerido para arrancar) y como seguía subiendo sin meseta, extendimos a 128 y 256 para encontrar el techo real, no quedarnos en un límite que era solo un punto de partida sugerido."

---

## Slide 14 — Resultados Experimento 4

**Qué se ve en pantalla:** barrido de PR-AUC/ROC-AUC por `d_model`, tabla completa, `d_model=64` marcado como mejor.

**Speech:**

> "`d_model=64` es el techo real de este dial: 0,724 de PR-AUC, el mejor de todo el barrido, con el gap de overfitting más chico de los seis valores (0,012). Antes de los 64 el modelo efectivamente venía corto — la performance sube de forma sostenida de 8 a 64, coherente con que cada escalón de `d_model` le da al Encoder más lugar para separar tokens y construir combinaciones de atención más ricas.
>
> Después de 64 aparece algo interesante: no es que 'deje de mejorar', empeora. En `d_model=128` el gap se dispara a 0,081, el más alto de todo el barrido — con 219.137 parámetros contra apenas 7.012 filas de train, cada una de esas cuatro matrices de atención de 128×128 tiene mucho más lugar para memorizar particularidades del train set que señal real que aprender. Y en `d_model=256` pasa algo todavía más revelador: el gap da negativo. Eso no significa que el modelo generalice mejor que en train — significa que ni siquiera está terminando de ajustar bien train en 20 épocas. Con 700.289 parámetros, moverlos con Adam en un puñado de épocas y unos pocos miles de ejemplos no alcanza para converger; el modelo necesitaría muchísimas más épocas para ni siquiera terminar de aprender train, no hablar ya de generalizar.
>
> `d_model=64` queda fijo, confirmado con evidencia como el mejor valor de todo el rango explorado, no solo el mejor dentro de un límite autoimpuesto de arranque."

---

## Slide 15 — "Experimento n°5 - dim_feedforward"

**Qué se ve en pantalla:** objetivo (probar el ancho del MLP interno), arquitectura (`dim_feedforward = 16/32/64/128/256/512`, `d_model=16` fijo).

**Speech:**

> "Este dial es distinto a `d_model` aunque los dos suenan a 'ancho'. `dim_feedforward` es el tamaño intermedio de la red feed-forward que cada encoder aplica posición por posición, después de la atención — expande el vector de `d_model` dimensiones a `dim_feedforward`, aplica una no linealidad, y lo vuelve a proyectar a `d_model` para poder seguir a la próxima capa o a la salida.
>
> Por qué importa esta pieza específicamente: la atención en sí es una combinación ponderada de vectores — una suma pesada, que sigue siendo, matemáticamente, una operación lineal sobre los value. Toda la capacidad de aplicarle una transformación no lineal a lo que la atención ya mezcló para cada token vive en este feed-forward. Si es muy angosto, se vuelve un cuello de botella: por más rica que haya sido la mezcla de atención, el modelo tiene poco margen para transformarla de forma no trivial antes de devolverla a `d_model`. Corrimos acá con `d_model=16` fijo (la base del momento) para aislar el efecto sin mezclarlo con el cambio de `d_model` que ya habíamos decidido explorar aparte."

---

## Slide 16 — Resultados Experimento 5

**Qué se ve en pantalla:** barrido de PR-AUC/ROC-AUC por `dim_feedforward`, `dim_feedforward=64` (4x sobre `d_model=16`) como mejor.

**Speech:**

> "Acá aparece un patrón distinto al de `d_model`: no es 'más es mejor hasta un techo', es un pico interior — `dim_feedforward=64` gana con 0,696, y **todos** los valores por encima (128, 256, 512) quedan peor, estabilizándose alrededor de 0,675-0,688 sin volver a subir. Esto es coherente con la idea de cuello de botella de la slide anterior, pero mirada desde el otro extremo: con `d_model=16` fijo, el 'contenido' que cada token trae para transformar ya está limitado por esas 16 dimensiones — agrandar el feed-forward más allá de cierto punto no le da al modelo más información real para procesar, solo más parámetros con los que ajustar ruido específico de train. `dim_feedforward=64` resulta ser, en este punto, exactamente la proporción 4x sobre `d_model` que también aparece en el paper original del Transformer — pero la razón de que funcione acá no es 'porque el paper usa esa proporción', es que a `d_model=16` esa amplitud alcanza para des-cuellobotellar el feed-forward sin sobrepasar la información real disponible.
>
> Esto deja una tensión abierta con el Experimento 4: ahí `dim_feedforward=64` quedó fijo mientras `d_model` subía a 64, así que en el punto ganador la proporción real terminó siendo 1x, no 4x — y aun así dio el mejor resultado de todo el estudio hasta ese momento. No podíamos saber, sin correrlo, si la proporción 4x en sí es lo que importa (y `d_model=64` con `dim_feedforward=256` daría más todavía) o si lo que pesa es la capacidad absoluta de `d_model` y 64 ya alcanza como ancho de feed-forward en cualquier caso. Por eso hace falta el próximo experimento."

---

## Slide 17 — "Experimento n°6 - d_model y dim_feedforward en conjunto"

**Qué se ve en pantalla:** objetivo (si la proporción 4x importa de verdad o alcanza con la capacidad absoluta de 64), arquitectura resaltando ambos parámetros.

**Speech:**

> "No corrimos acá un grid completo de todas las combinaciones — hubiera sido 30 configuraciones por 3 semillas para responder una pregunta puntual. Corrimos dos combinaciones nuevas manteniendo la proporción 4x en los dos puntos que ya nos interesaban: `d_model=32` con `dim_feedforward=128`, y `d_model=64` con `dim_feedforward=256`, comparadas contra las mismas filas de `d_model=32`/`64` del Experimento 4, donde `dim_feedforward` había quedado fijo en 64 (proporciones 2x y 1x respectivamente).
>
> La pregunta de fondo, conectando con la slide anterior: el feed-forward existe para darle a cada token una transformación no lineal después de la atención — la pregunta es si esa transformación necesita escalar en proporción fija con `d_model`, o si alcanza con un ancho absoluto una vez que `d_model` ya es lo bastante generoso."

---

## Slide 18 — Resultados Experimento 6

**Qué se ve en pantalla:** tabla con las 4 combinaciones y sus proporciones, gráficos de barras de PR-AUC y del gap.

**Speech:**

> "PR-AUC prácticamente no se mueve al escalar `dim_feedforward` en ninguno de los dos `d_model` — las diferencias están completamente dentro del desvío estándar de cada punto. Donde sí hay un efecto grande, y en la dirección contraria a lo esperado, es en el overfitting: en `d_model=64`, pasar de `dim_feedforward=64` a 256 casi octuplica el gap, de 0,012 a 0,099, con mucha más varianza entre semillas.
>
> Esto confirma la segunda hipótesis que habíamos dejado planteada: la proporción 4x que funcionaba a `d_model=16` no se sostiene cuando `d_model` ya es más ancho. La explicación mecánica es la misma que veníamos armando: a `d_model=64`, el vector que le llega al feed-forward ya trae mucho más contenido por token que a `d_model=16` — el feed-forward no necesita expandirse en la misma proporción para tener margen de transformarlo, porque el 'material bruto' ya es rico. Agrandarlo igual solo agrega parámetros —126.401 contra 7.012 filas de train— que el modelo termina usando para memorizar en vez de generalizar, sin que la performance de valid lo compense.
>
> Con esto cerramos los cuatro diales del Transformer de texto puro: `n_heads=1, n_layers=2, d_model=64, dim_feedforward=64` — la mejor marca de solo-texto de todo el estudio, PR-AUC valid 0,724 ± 0,021. De acá en más pasamos al sistema completo: sumar la rama tabular."

---

## Slide 19 — "Experimento n°7 - sistema completo: texto + tabular"

**Qué se ve en pantalla:** objetivo (si sumar tabular mejora sobre el texto solo), arquitectura con la rama de texto ganadora + cabeza `Linear` directa.

**Speech:**

> "Acá conectamos por primera vez las dos ramas que veníamos manteniendo separadas: el vector de 64 dimensiones que resume el texto se concatena con las 75 features tabulares ya codificadas, dando un vector de 139 dimensiones, que entra directo a una salida `Linear(139→1)` — sin capa oculta todavía, mismo criterio minimalista que veníamos aplicando desde el Experimento 1.
>
> La pregunta que responde este experimento es la validación empírica de la decisión de arquitectura que mostramos en la slide 5: concatenar tarde, cada rama por su cuenta, ¿alcanza para que el modelo aproveche lo tabular, o hace falta algo más?"

---

## Slide 20 — Resultados Experimento 7

**Qué se ve en pantalla:** curvas train/valid, tabla comparando solo-texto vs. texto+tabular, hipótesis sobre por qué no mejoró.

**Speech:**

> "El resultado es contraintuitivo si uno esperaba que 'más información siempre ayuda': PR-AUC de valid **no mejoró** al sumar lo tabular —0,718 contra 0,724 del texto solo, diferencia dentro del ruido—. ROC-AUC sí subió (0,962→0,967), así que las tabulares no son inútiles, ayudan a ordenar mejor los casos en general, pero no mueven la aguja en la métrica que más nos importa bajo este desbalance de clases.
>
> El overfitting sí empeoró y bastante, casi se cuadruplica el gap (0,012→0,045) y aparece mucho antes en el entrenamiento — con 139 dimensiones de entrada contra 64 del texto solo, y el mismo dataset chico, hay más con qué memorizar sin que haya necesariamente más señal real proporcional.
>
> Nuestra hipótesis, y es la que motiva el siguiente experimento: la salida es un único `Linear`, que matemáticamente es una suma ponderada de cada dimensión de entrada por separado —`output = Σ wᵢ·xᵢ`—. Puede aprender qué tan importante es cada feature tabular y qué tan importante es cada dimensión del texto, cada una por su cuenta, pero estructuralmente no puede construir un término que dependa de la combinación de ambas —algo como 'el tag de reputación pesa más en esta categoría de producto'—, porque eso requeriría multiplicar información de una rama por información de la otra, y una suma ponderada no puede hacer eso. No es que las tabulares no aporten: es que la cabeza, tal como está, no tiene forma de dejarlas interactuar con el texto."

---

## Slide 21 — "Experimento n°8 - capa oculta en la cabeza de salida"

**Qué se ve en pantalla:** objetivo (probar si el estancamiento se debía a la falta de no-linealidad), arquitectura con cabeza `Linear(139→64) → ReLU → Dropout(0,1) → Linear(64→1)`.

**Speech:**

> "Directamente probamos la hipótesis del experimento anterior: insertamos una capa oculta de 64 unidades con ReLU en el medio, entre la concatenación y la salida. Este cambio, chico en apariencia, es el que le da al modelo la capacidad que la suma ponderada no tenía: cada una de las 64 neuronas ocultas es, antes de la no linealidad, una combinación pesada de las 139 dimensiones de entrada —texto y tabular mezclados en la misma combinación—, y después del ReLU esa neurona se activa o no según ese patrón combinado. Recién ahí el modelo puede construir features que dependen genuinamente de ambas ramas al mismo tiempo, no solo de cada una por separado.
>
> El ancho, 64, se eligió por paridad con `d_model` como punto de partida razonable —ni tan angosto que pierda capacidad, ni tan ancho que arriesgue overfitting sin necesidad—, con la idea explícita de barrerlo más adelante en vez de darlo por cerrado con un solo valor probado."

---

## Slide 22 — Resultados Experimento 8

**Qué se ve en pantalla:** tabla comparando las tres configuraciones, curvas a 40 épocas, conclusión sobre el techo real.

**Speech:**

> "Esto confirma la hipótesis de forma contundente: 0,798 de PR-AUC, la mejor marca de todo el estudio hasta este punto, muy por encima tanto del texto solo (0,724) como del sistema sin capa oculta (0,718). El problema del experimento anterior no era que las tabulares no aportaran señal — era que la cabeza lineal no tenía forma de dejarlas interactuar con el texto.
>
> Corrimos esta configuración primero a 20 épocas y seguía subiendo sin señales de meseta, así que la repetimos a 40 para ver el techo real. Ahí se ve algo importante: valid sube fuerte hasta la época 20-25 y ahí se aplana —incluso baja un poco hacia la 40—, mientras train sigue subiendo derecho. La ganancia de duplicar el entrenamiento es chica, 0,791 a 20 épocas contra 0,798 a 40, y viene de picos puntuales de alguna semilla, no de una mejora sostenida en las tres. Con esto confirmamos que 20 épocas ya daban un resultado representativo — no hacía falta entrenar más para llegar a la conclusión correcta sobre esta arquitectura, solo para confirmar que no nos estábamos perdiendo de nada relevante."

---

## Slide 23 — "Experimento n°9 - country_of_origin y nutrition_score"

**Qué se ve en pantalla:** objetivo (testear dos features sin señal univariada clara en el EDA), arquitectura igual al Experimento 8.

**Speech:**

> "En el EDA del Ejercicio 1 estas dos features habían quedado marcadas como dudosas: `country_of_origin` mostraba tasas de compra en un rango angosto (10,7%-16,6%) compatible con ruido de muestra chica por país, y `nutrition_score` tenía correlación prácticamente nula con `bought` (-0,019) mirada sola.
>
> Pero 'sin señal univariada' no es lo mismo que 'sin señal útil' — y esto conecta directo con lo que acabamos de confirmar en el experimento anterior: ahora que la cabeza tiene una capa no lineal capaz de combinar features entre sí, una columna que no dice nada por sí sola podría aportar en combinación con otras (por ejemplo, interactuando con `category`). Por eso no la descartamos a partir del chequeo simple del EDA, la tratamos como un dial más de este estudio de ablación y la corrimos con la arquitectura ya cerrada."

---

## Slide 24 — Resultados Experimento 9

**Qué se ve en pantalla:** barrido de las 4 variantes, tabla con PR-AUC y gap, decisión de quedarse con ambas.

**Speech:**

> "Los resultados separan claramente a las dos features, que habían quedado agrupadas como 'dudosas' en el EDA pero no se comportan igual acá. Sacar `nutrition_score` baja PR-AUC de forma consistente —0,791 a 0,760—, una caída bastante mayor que el desvío estándar de cada punto, así que no es ruido: sí aporta señal real, aunque esa señal no se viera en el cruce univariado simple del Ejercicio 1. Sacar `country_of_origin`, en cambio, mueve PR-AUC apenas —0,791 a 0,786—, dentro de un margen compatible con el ruido entre semillas.
>
> La decisión que tomamos fue quedarnos con ambas de todos modos: no encontramos evidencia de que `country_of_origin` sume, pero tampoco de que perjudique, y sacarla no traía ningún beneficio claro que justificara complicar la comparación con una variante menos del modelo completo."

---

## Slide 25 — "Experimento n°10 - ancho de la capa oculta"

**Qué se ve en pantalla:** objetivo, arquitectura con `hidden` como parámetro a barrer.

**Speech:**

> "Antes de dar por cerrada la arquitectura, volvimos sobre un cabo suelto: `hidden=64` en el Experimento 8 se había elegido por paridad con `d_model`, probando un solo valor — a diferencia de los demás diales de capacidad (`n_layers`, `d_model`, `dim_feedforward`), que sí se barrieron. Era inconsistente con el resto del estudio dejarlo sin barrer, así que lo tratamos igual que a los otros: como un dial más a explorar, no como un valor ya cerrado."

---

## Slide 26 — Resultados Experimento 10

**Qué se ve en pantalla:** barrido de `hidden` de 0 a 512, tabla completa, meseta entre 256 y 512.

**Speech:**

> "El barrido confirma la misma lógica de capacidad de la cabeza que venimos viendo desde el Experimento 8: a más ancho, más lugar para que esa capa oculta construya combinaciones entre texto y tabular, y PR-AUC sube de forma sostenida de 0,718 en `hidden=0` a 0,820 en `hidden=512` — la mejor marca de todo el estudio.
>
> Pero la meseta aparece recién entre 256 y 512: el salto de 128 a 256 fue +0,018, y el de 256 a 512 es +0,004, un orden de magnitud más chico y dentro del solapamiento de los desvíos estándar. `hidden=512` gana por muy poco margen a cambio de 36.096 parámetros más.
>
> Nos quedamos con `hidden=256`, no con el número más alto probado. La razón es la misma que guio todo el estudio desde el Experimento 1: cuando la diferencia de performance es chica, preferimos el modelo más chico — `hidden=512` queda documentado como el punto que confirma dónde está la meseta real, no como la elección final."

---

## Slide 27 — "Experimento n°11 - positional encoding"

**Qué se ve en pantalla:** objetivo (ver la importancia del orden del texto), arquitectura — sin bullet de "cabeza de salida", a diferencia de las slides de los Experimentos 8-10: este experimento corre sobre el Transformer de texto solo (`TextTransformerClassifier`), no sobre el sistema completo, así que no hay cabeza combinada que mostrar acá.

**Speech:**

> "Este experimento corre sobre el Transformer de texto solo, no sobre el sistema completo — a propósito, para aislar el efecto del positional encoding sin que las features tabulares puedan compensar una eventual pérdida de información de orden.
>
> Por qué hace falta esto en primer lugar, mecánicamente: la atención calcula, para cada token, una similitud entre su vector query y los vectores key del resto de la secuencia, y esa similitud sale de un producto interno sobre el contenido de los embeddings. Nada en esa cuenta depende de en qué posición de la oración está cada token — la atención, tal como está definida, es indiferente al orden: si permutáramos las palabras de una oración, cada token seguiría viendo exactamente los mismos vecinos con los mismos pesos de atención, solo que reordenados. El modelo no tendría forma de distinguir 'bien calificado, poco confiable' de 'poco calificado, bien confiable'. El positional encoding rompe esa simetría: suma a cada embedding un vector determinado únicamente por su posición en la secuencia (la función senoidal, fija, no aprendida), así que el producto interno Q·K ya no depende solo del contenido, también arrastra información de dónde está cada token respecto de los demás."

---

## Slide 28 — Resultados Experimento 11

**Qué se ve en pantalla:** tabla con/sin positional encoding, decisión de mantenerlo.

**Speech:**

> "El resultado tiene dos lecturas que van en direcciones distintas, y las dos son informativas. En PR-AUC de valid casi no hay diferencia —0,724 con positional encoding, 0,720 sin él—: sacar la información de orden no le hace perder al modelo casi nada de su capacidad de *alcanzar* un buen resultado. Esto en principio sorprende, porque en el Ejercicio 1 el hallazgo más fuerte del EDA —el tag de reputación en `title`— parecía un patrón posicional (siempre al final, entre paréntesis). Pero puede explicarse así: la atención, incluso sin orden, sigue pudiendo agrupar tokens por *contenido* — qué palabras aparecen juntas en la misma oración — y eso puede alcanzar para captar el tag igual, sin necesitar saber en qué posición exacta cae.
>
> Donde sí hay un efecto enorme es en el overfitting: el gap se dispara de 0,012 a 0,174, train llega a 0,89-0,92 sin positional encoding contra 0,70-0,76 con él, mientras valid se queda prácticamente plano. Nuestra hipótesis, dejada como hipótesis y no como conclusión cerrada: sin la restricción estructural que impone la posición, la atención tiene más libertad para ajustarse a combinaciones específicas de palabras de cada fila de train en vez de aprender un patrón que generalice — el positional encoding, más que aportar señal para el pico de performance, actuaría acá como una forma de regularización implícita, dándole a la atención menos grados de libertad para memorizar.
>
> Se mantiene en la arquitectura final por esto último: no por el número pico, que casi no cambia, sino porque sin él el modelo sobreajusta mucho más rápido y más fuerte — algo que sería un problema real en un dataset todavía más chico, o entrenando más épocas de las que usamos acá."

---

## Slide 29 — "Modelo final elegido"

**Qué se ve en pantalla:** arquitectura completa (`d_model=64, n_heads=1, n_layers=2, dim_feedforward=64`, positional encoding senoidal, cabeza `Linear(129→256)→ReLU→Dropout→Linear(256→1)`), resultados de valid: PR-AUC 0,816 ± 0,013.

**Speech:**

> "Este es el cierre de los once experimentos: cada uno de estos números tiene un experimento propio detrás que lo respalda, no un default. `n_heads=1` porque en este rango de `d_model` demostramos que heads reparte capacidad, no la agranda — no era el cuello de botella. `n_layers=2` porque duplicar la profundidad agregó una capa de composición de segundo orden que ayudó, pero seguir apilando agregaba parámetros sin señal nueva para aprovecharlos. `d_model=64` y `dim_feedforward=64` porque encontramos el techo real de cada uno con barridos, no nos quedamos en el límite sugerido de arranque. Positional encoding porque, aunque no mueve el pico, evita un sobreajuste mucho más agresivo. Fusión tardía con una cabeza de `hidden=256` porque confirmamos que la interacción entre texto y tabular necesitaba una no linealidad para aparecer, y barrimos ese ancho hasta encontrar dónde la ganancia deja de justificar el costo.
>
> El número que queda —PR-AUC de valid 0,816 ± 0,013— no es el techo absoluto que vimos en todo el estudio (`hidden=512` daba 0,820), es el resultado de priorizar consistentemente, en cada empate chico, el modelo más simple entre dos opciones casi equivalentes — el mismo criterio de 'arrancar chico y crecer solo si el dato lo pide' aplicado de punta a punta, no solo al principio."

**[si preguntan más] "¿Por qué acá dice `Linear(129→...)` y en las slides de los Experimentos 7 a 10 decía `Linear(139→...)`?"** → "Porque el vector tabular que entra a la cabeza cambió de tamaño en el medio del estudio, y esta slide ya refleja esa decisión. Los Experimentos 7 a 10 corrieron con las 75 columnas tabulares completas —64 del texto más 75 de lo tabular, 139—, porque en ese momento todavía no habíamos ablacionado `country_of_origin`. Recién en el Experimento 9 encontramos que sacarla no perjudicaba el PR-AUC de valid —0,791 con ella, 0,786 sin ella, una diferencia menor que el propio desvío entre semillas—, así que la arquitectura final la descarta: quedan 65 columnas tabulares, 64+65=129. Es la misma metodología greedy de toda la sección 4: cada experimento corre sobre la mejor configuración encontrada hasta ese punto, así que un dial decidido en el Experimento 9 —sacar esa columna— cambia el tamaño de entrada de todo lo que viene después, incluida esta slide final."

---

## Slide 30 — "Evaluación en test"

**Qué se ve en pantalla:** curvas de train/valid con la línea de test superpuesta, tabla por semilla, PR-AUC test 0,809 ± 0,003 — y un tercer panel nuevo, el scatter de BTR real vs. predicho por búsqueda (MAE 0,067 ± 0,002, r 0,762 ± 0,014).

**Speech:**

> "Este es el único momento de los once experimentos donde tocamos `data/test.csv` — hasta acá, toda decisión de arquitectura se tomó mirando exclusivamente valid, exactamente como planteamos en la slide del split: test se reserva para un solo número al final, no para elegir entre configuraciones.
>
> El resultado es la mejor confirmación posible de que el proceso funcionó: 0,809 de PR-AUC en test es prácticamente el mismo número que veníamos viendo en valid en las épocas donde se seleccionó el mejor checkpoint de cada semilla —0,80-0,82 en el Experimento 10—, y con la varianza más baja de todo el estudio, 0,003 de desvío entre semillas. Si las once decisiones de arquitectura que tomamos mirando valid se hubieran sobreajustado a ese split en particular —eligiendo la configuración que más le acertaba a las particularidades de esas 1.498 filas de valid, no la que generaliza mejor—, test tendría que haber dado un número notablemente más bajo. No pasó eso.
>
> En números finales: PR-AUC de test 0,809, más de seis veces la prevalencia de 0,130 que tendría un modelo sin ninguna señal; ROC-AUC 0,965, contra 0,5 de azar. Y comparado contra el punto de partida —el Experimento 1, texto solo, la configuración más chica posible, PR-AUC 0,688—, la mejora es de +0,121 en PR-AUC de test, +18% relativo, construida no de un solo cambio grande sino de once decisiones encadenadas, cada una con su propio experimento, su propia hipótesis y su propia evidencia detrás.
>
> Pero PR-AUC y ROC-AUC miden algo puntual: qué tan bien el modelo ordena productos individuales por probabilidad de compra. La consigna pide predecir el BTR de una búsqueda, que es otra cosa — el promedio de esas compras agrupado por `query_id`, no la clasificación fila por fila. Nunca lo habíamos calculado en los once experimentos, así que lo agregamos acá, sobre las mismas 3 corridas ya entrenadas —mismos pesos de la mejor época por semilla, sin reentrenar ni tocar test una segunda vez—: agrupamos las predicciones de test por búsqueda y comparamos el BTR real de cada una contra el promedio de las probabilidades que el modelo le asignó a sus filas.
>
> Los dos números que reportamos miden cosas distintas y se complementan. El MAE, 0,067, es la diferencia absoluta promedio entre el BTR real y el predicho de cada búsqueda, en las mismas unidades que el dato — en promedio nos equivocamos por 6,7 puntos porcentuales. La correlación de Pearson, 0,762, mide si el BTR real y el predicho suben y bajan juntos entre las 302 búsquedas de test, sin importar la magnitud — y da bastante alta: cuando una búsqueda tiene más compras reales, el modelo en general también le pone más probabilidad. En el scatter se ve exactamente eso, la nube de puntos sube de izquierda a derecha, pero también se ve la falla más clara: la columna de puntos en `BTR real = 0` con predicciones dispersas hasta 0,3-0,45 — el modelo tiende a sobreestimar el BTR de las búsquedas donde ningún producto se compró. Tiene sentido con cómo entrena: cada fila es un caso independiente, el modelo nunca ve ni optimiza directamente el promedio de una búsqueda completa, así que en las búsquedas con pocas filas ese promedio agregado es más sensible a un error puntual del modelo en una sola fila.
>
> Eso es lo que queremos que quede claro con esta sección: no llegamos a esta arquitectura por prueba y error a ciegas, llegamos siguiendo, en cada paso, lo que el mecanismo del Transformer predice que debería pasar — confirmándolo, o corrigiéndolo, con el dato real — y cerramos mirando la métrica que la consigna realmente pide, no solo la que usamos para comparar configuraciones durante el estudio."

**[si preguntan más] "¿Por qué no entrenar directamente contra el BTR agregado, en vez de contra `bought` fila por fila?"** → "Lo consideramos, pero entrenar a nivel fila nos da muchos más ejemplos con más resolución: 7.012 filas de train contra apenas 1.408 queries si agregáramos antes de entrenar — perderíamos justo la granularidad de cuál producto puntual se compró, que es la que le permite al Transformer aprender del texto de cada producto por separado. El BTR sigue siendo recuperable después, agregando las predicciones ya hechas, sin necesidad de sacrificar esa resolución para entrenar."

**[si preguntan más] "¿Por qué el modelo sobreestima tanto las búsquedas con BTR real 0?"** → "Es la contracara de entrenar a nivel fila: el modelo optimiza para separar bien productos comprados de no comprados en general, no para acertar que una búsqueda puntual tuvo cero compras. Una búsqueda con BTR real 0 puede tener productos con texto y features parecidos a los de otras búsquedas donde sí hubo compra — el modelo les asigna una probabilidad razonable en base a eso, aunque en esa búsqueda en particular no se haya concretado ninguna. No es un error de la arquitectura, es una limitación esperable de optimizar a nivel fila cuando lo que finalmente se quiere leer es un agregado por grupo."
