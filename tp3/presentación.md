# Presentación: recorrido y decisiones del TP3

Última actualización: **04/11/2026**.

Este documento cuenta cómo fuimos construyendo el proyecto: con qué alcance
empezamos, qué decisiones tomamos, qué implementamos, qué comprobamos y qué
aprendimos. Sirve para que alguien que no siguió el desarrollo pueda ponerse
al día y como base para preparar la presentación final.

El recorrido hasta esta fecha se reconstruyó con el historial de Git, los
documentos del proyecto y las ejecuciones registradas. En adelante agregaremos
entradas a medida que avancemos. Explicamos las decisiones con sus motivos y
evidencias, distinguiendo lo implementado de las propuestas pendientes.

## 1. Qué queremos resolver

Queremos ayudar a un estudiante a decidir si un video le sirve para estudiar
un objetivo concreto de una materia. El sistema deberá comparar el contenido
hablado del video con los materiales de la cátedra y entregar:

- Contenidos cubiertos, tratados parcialmente y sin evidencia en el audio.
- Material de la materia recomendado para completar lo que falte.
- Una conclusión sobre la utilidad del video para ese objetivo.

Los materiales de la materia establecen qué se espera saber; el video aporta
evidencia de qué explica. Esta separación determina la arquitectura: **el corpus
académico y la transcripción del video cumplen funciones diferentes**.

La definición original está en la [propuesta del ejercicio 1](ejercicio1/Grupo03-ej1-TP3.md),
apartados 1-5. El objetivo es evaluar cobertura, no responder preguntas generales
sobre una carpeta de documentos.

## 2. Con qué empezamos y por qué

Elegimos comenzar con **una clase, un video y un objetivo de estudio**, trabajando
sobre texto. El video se incorpora mediante su audio transcripto; todavía no
analizamos fotogramas, imágenes ni información que aparezca solo en pantalla.
La idea fue poder revisar las fuentes y cada etapa antes de ampliar el corpus
o agregar más componentes.

| Elemento | Punto de partida | Motivo |
| --- | --- | --- |
| Material académico | Una transcripción: `docs/embeddings_2.VTT` | Contiene embeddings y similitud coseno, adecuados para revisar el primer caso. `embeddings_1.VTT` está más centrado en la demo del Transformer. |
| Video de ejemplo | Audio de «¿Qué son los EMBEDDINGS Grandes Modelos de Lenguaje», de unos 10 minutos y 9 segundos | Permite contrastar una explicación introductoria con la clase seleccionada. |
| Objetivo del caso manual | Entender qué son los embeddings y cómo se compara la similitud entre textos | Acota qué contenidos se esperan y con qué profundidad; es el objetivo de trabajo del caso inicial. |
| Modalidad del análisis | Texto de la clase y texto obtenido del audio | Permite auditar el contenido y citar tiempos antes de incorporar otras modalidades. |
| Forma de avance | Primero un pipeline básico, luego comparaciones y técnicas avanzadas | Permite revisar por separado preparación, recuperación y generación. |

Conservamos la clase completa, incluidas preguntas e intervenciones
administrativas. No corregimos automáticamente los errores de transcripción.
Esto preserva las fuentes, pero también deja ruido que habrá que estudiar.

### Estructura inicial y evolución de las tecnologías

La propuesta del ejercicio 1 planteaba **Chroma + MiniLM + GPT-4.1 nano**.
Durante el diseño se consideró Pinecone; el 31/10 se eligió Chroma local
persistente para simplificar la configuración y conservar la colección entre
ejecuciones, apoyándonos en los ejemplos de la cátedra.

MiniLM se mantuvo como primer embedder para probar una configuración vista en
clase y ejecutable en CPU. Es un punto de partida: su elección no demuestra
que recupere bien material en español.

Para la generación, el diseño posterior propuso **Qwen2.5-1.5B-Instruct local**,
presente en el notebook de la cátedra, dejando por verificar la disponibilidad
del modelo por API de la propuesta original. La implementación se registra
en el apartado 7. La justificación y las referencias están en el
[diseño inicial](ejercicio2/diseno_rag_inicial.md).

### Flujo elegido

Preparación del material académico, implementada:

```text
VTT de una clase
  -> segmentos limpios con IDs y tiempos
  -> chunks con texto y referencias
  -> embeddings MiniLM
  -> colección persistente en Chroma
```

Flujo previsto para evaluar un video:

```text
Objetivo de estudio -> búsqueda en Chroma -> referencias académicas
Audio del video -> Whisper -> transcripción guardada
Objetivo + referencias + transcripción -> generador -> informe de cobertura
```

La búsqueda parte del objetivo de estudio para poder encontrar contenidos que
el video podría omitir. La transcripción del video se mantiene separada de la
colección académica: no debe convertirse en respaldo de su propia cobertura.

## 3. En qué orden avanzamos

### Paso 1. Preparar la transcripción de la clase — 28/10

Implementamos [preparar_transcripcion.py](ejercicio2/preparar_transcripcion.py)
para leer el VTT y producir JSON y una vista TXT. Quitamos cabeceras y marcas
del cuerpo del texto, normalizamos espacios y conservamos IDs, hablantes y
tiempos en campos identificables.

**Motivo:** separar el contenido de su formato sin perder la posibilidad de
volver al original. Una entrada del VTT puede ser una frase incompleta;
estos segmentos todavía no eran los chunks para embeddings.

Referencia del historial: `96b7096` — «Limpieza de VTT».

### Paso 2. Obtener el texto del video — 28/10

Implementamos [transcribir_audio.py](ejercicio2/transcribir_audio.py) con Whisper
a través del proxy de la cátedra. Guardamos la respuesta original, un VTT y
una vista TXT. Reutilizamos los archivos existentes para evitar repetir llamadas.

La primera salida VTT no tuvo el formato esperado por el lector. Se pasó a
solicitar `verbose_json` y construir el VTT localmente con los tiempos devueltos
por Whisper.

**Resultado y límite:** obtuvimos texto para trabajar, con tiempos aproximados.
La revisión completa contra el audio y de los términos técnicos sigue pendiente;
tener una transcripción no demuestra que el video cubra los contenidos.

Referencia del historial: `50c3b73` — «transcripcion de audios».

### Paso 3. Definir una salida esperada con un caso manual — 28/10; revisión el 31/10

El asistente elaboró el [caso 01](ejercicio2/evaluacion/caso_01_embeddings.md)
comparando las transcripciones, y el usuario confirmó su revisión el 31/10.
Se definieron cinco criterios: qué es un embedding, papel del contexto,
comparación de vectores, cálculo del coseno y limitación relacionada con magnitud.

**Motivo:** concretar qué debería contener un informe y qué evidencia justificaría
cada clasificación antes de automatizarlo.

**Resultado manual:** E1 y E2 quedaron cubiertos; E3, parcial; E4 y E5, sin
evidencia en la transcripción disponible. Por ejemplo, la fórmula del coseno
está en la clase, entrada 222, 31:55.350-32:14.820; el ejemplo sobre magnitud,
en las entradas 182-188, 25:37.950-27:13.979.

Este caso sirve como referencia de desarrollo. No es una salida automática,
una auditoría del audio ni una evaluación independiente de generalización.

Referencia del historial: `1a2c2a8` — «ejemplo de analisis de cobertura».

### Paso 4. Documentar el diseño antes de completar el RAG — 31/10

Se incorporó el [diseño inicial y plan de experimentos](ejercicio2/diseno_rag_inicial.md),
con los componentes de los notebooks, una configuración inicial y el orden
propuesto para comparar alternativas. Se fijó Chroma local persistente como base.

**Motivo:** dejar explícito qué tomamos de clase, qué adaptamos y qué parámetros
eran hipótesis nuestras. Para MiniLM + Chroma, la referencia principal es
`Langchain_Virtual_Assitant.ipynb`, celda 9; para splitting e indexación,
`UN_Report_Advanced_RAG.ipynb`, celdas 11-23, ambos bajo `docs/RAG/`.

Referencia del historial: `8767f2e` — «guia tp3 y diseño inicial».

### Paso 5. Construir chunks para el encoder — 31/10

Implementamos [preparar_chunks.py](ejercicio2/preparar_chunks.py) con dos pasadas:
splitter recursivo de 1000 caracteres y 100 de overlap, seguido por ventanas
de hasta 200 tokens de contenido y 20 de overlap, con el tokenizer de MiniLM.

**Motivo:** agrupar intervenciones y respetar el límite del encoder. Usamos
offsets para conservar el texto original y asociar cada chunk a fuente, IDs de
intervenciones y tiempos. Estos parámetros son iniciales, no valores óptimos.

**Resultado:** de 415 intervenciones se obtuvieron **125 chunks**, con máximo
de **202 tokens incluidos los especiales**, por debajo del límite de 256.
Se comprobó que no desaparecieran caracteres no blancos. La integridad no
resuelve por sí sola la legibilidad: hay cortes dentro de palabras e intervenciones.

Referencia del historial: `10a2cf8` — «chunks».

### Paso 6. Indexar y buscar referencias — 31/10

Implementamos [indexar_material.py](ejercicio2/indexar_material.py) y
[recuperar.py](ejercicio2/recuperar.py), con valores compartidos en
[config.py](ejercicio2/config.py). Se cargaron los 125 chunks con embeddings
MiniLM de 384 dimensiones en Chroma persistente, con distancia coseno explícita.
La búsqueda inicial usa el objetivo de estudio, `k=4` y ningún umbral.

**Motivo:** comprobar que podemos obtener referencias académicas antes de
usarlas como contexto para un generador. Se conservaron los metadatos y se
separaron colecciones por corpus/configuración para evitar mezclar experimentos.
Las cargas completas se reutilizan y las interrumpidas pueden reanudarse.

**Resultado técnico:** la base persiste entre ejecuciones y repetir la carga
reutiliza los 125 chunks sin duplicarlos ni recalcular sus embeddings. Cada
búsqueda guarda consulta, configuración, referencias, distancias y tiempos.

Referencia del historial: `bdddab6` — «buscar material de la clase a partir de un objetivo de estudio».

## 4. Qué pruebas hicimos y qué mostraron

Hasta este punto hicimos controles de integridad y operación y dos consultas
reales. **Todavía no hicimos la comparación de configuraciones `k=2,4,8` ni
una evaluación automática del video.**

| Prueba realizada | Qué queríamos comprobar | Resultado observado |
| --- | --- | --- |
| Integridad del splitting | Conservar texto, ubicaciones y límites de tokens | Siete pruebas de chunks aprobadas; 125 chunks dentro del límite. |
| Integración con Chroma real y vectores controlados | Persistencia en otro proceso, repetición, reanudación, trazabilidad y rechazo de configuraciones inválidas | Seis pruebas aprobadas; junto con las anteriores, 13 en total. No miden calidad semántica de MiniLM. |
| Cotejo de los datos persistidos con el VTT limpio | Conservar texto y referencias tras indexar | Coinciden los 125 textos con sus offsets; vectores de 384 dimensiones. |
| Autoconsulta con el texto exacto del chunk de la fórmula | Coherencia entre embeddings de carga y consulta | Recuperó su propio ID primero, con distancia prácticamente cero. |
| Objetivo «entender similitud coseno y sus limitaciones», `k=4` | Encontrar cálculo y limitación sobre magnitud | La fórmula apareció en el puesto 4; los otros tres resultados no desarrollaban coseno y faltó el ejemplo de magnitud. |
| Objetivo general del caso 01, `k=4` | Recuperar las referencias necesarias para el caso manual | Hubo ruido y no aparecieron las intervenciones citadas como referencia para E1-E5. |

Las consultas completas, los tiempos y la revisión por fragmento están en
[revisión inicial de recuperación](ejercicio2/evaluacion/revision_retrieval_inicial.md).
Por ejemplo, la fórmula recuperada corresponde al intervalo de clase
31:44.560-32:41.570, mientras que el ejemplo de magnitud esperado en
25:37.950-27:13.979 quedó fuera de los cuatro resultados.

**Qué aprendimos:** que el índice funcione y conserve los datos no significa que
recupere el contexto necesario. Tampoco un puntaje mayor garantiza pertinencia.
Una referencia que no recuperamos no demuestra una omisión del video.
Español, errores de ASR y cortes de chunks son posibles factores a investigar;
estas dos consultas no permiten atribuirles la causa del problema.

## 5. Dónde estamos y qué sigue

| Parte del proyecto | Estado al 31/10/2026 |
| --- | --- |
| Corpus mínimo y transcripción del video | Disponibles localmente; revisión contra el audio pendiente. |
| Caso de cobertura manual | Elaborado y revisado como referencia de desarrollo. |
| Preparación de chunks | Implementada y validada en integridad/tamaño. |
| Embeddings y Chroma persistente | Implementados y ejecutados sobre 125 chunks. |
| Recuperación por objetivo | Implementada; calidad inicial insuficiente en las consultas revisadas. |
| Generación del informe y comparación automática | Pendientes. |
| Comparación de configuraciones y métricas de cobertura | Pendientes. |
| Más clases/videos, PDFs e imágenes | Posibles ampliaciones; todavía no incorporadas al pipeline. |
| Advanced RAG | Pendiente, después de completar el pipeline básico. |

En ese momento se propuso comparar `k=2,4,8` manteniendo fijo el resto y
registrar qué referencias útiles y qué ruido se agregaban. La comparación
quedó registrada en el apartado siguiente. El plan contemplaba estudiar
otras variables de a una según el resultado y continuar con generación y
evaluación del informe. Ampliar el corpus o las modalidades deberá quedar
registrado como un cambio de alcance con sus propios resultados.

## 6. Primera ablación de recuperación — 04/11

Comparamos `k=2,4,8` para los dos objetivos de la revisión inicial. Se
ejecutaron seis búsquedas locales con los mismos 125 chunks, MiniLM, Chroma
persistente, distancia coseno y sin umbral. Queríamos comprobar si aumentar
la cantidad de fragmentos aportaba referencias suficientes para los criterios
del caso manual. Las corridas de `k=4` reprodujeron las anteriores y se cotejaron
los IDs y el texto de los fragmentos recuperados.

**Resultado:** para coseno y sus limitaciones, `k=2` respaldó 0 de 2 criterios
y `k=4` y `k=8` respaldaron 1 de 2. Con ocho fragmentos apareció el cierre
de un pasaje relacionado con magnitud, pero quedaron fuera la explicación y
el ejemplo de series. Para el objetivo general, `k=2` y `k=4` respaldaron
0 de 5 criterios y `k=8` respaldó 1 de 5 al combinar dos fragmentos sobre
qué es un embedding. Al registrar la comparación se conservó inicialmente `k=4`.
El [informe de ablación](ejercicio2/evaluacion/ablacion_k.md) detalla el método,
los fragmentos nuevos y los comandos para reproducirlos.

**Decisión posterior:** adoptamos `k=8` como valor por defecto. Entre los
valores probados, respaldó más criterios del caso de desarrollo y conservó
el respaldo para el cálculo del coseno, aunque agregó fragmentos poco útiles.
Se verificó que la CLI sin `--k` reproduce las ocho referencias de la ablación
para el objetivo general y que pasan las 13 pruebas del ejercicio 2. Las
corridas originales se pueden reproducir con `--k 4`.

**Siguiente paso propuesto:** construir el generador del informe básico y
probarlo con pasajes académicos de referencia y con los resultados reales del
buscador. Así se puede observar por separado si el informe usa bien las
fuentes y qué limitaciones arrastra cuando faltan referencias de la clase.
Para la corrida con búsqueda se fija `k=8`.

Para ejecutar el estado actual, consultar el [README del ejercicio 2](ejercicio2/README.md).
Los VTT originales bajo `docs/`, los audios, los pesos, Chroma y los resultados
bajo `data/` están ignorados por Git: un clon no incluye todos esos archivos.
Este documento y los informes de revisión conservan el recorrido compartible.

## 7. Implementación del generador y prueba del caso 01 — 04/11

Implementamos [generar_informe.py](ejercicio2/generar_informe.py) con el
Qwen2.5-1.5B-Instruct local del notebook de la cátedra. El comando recibe un
objetivo y un VTT de video. Por defecto busca material académico con
`k=8` y admite pasajes conocidos para el diagnóstico. El informe usa IDs
distintos para la clase y el video. El script resuelve sus tiempos a partir
de los originales y registra los IDs citados que no existen.

Se prepararon dos entradas con el mismo objetivo, las mismas instrucciones
y los 99 segmentos del video. El diagnóstico contiene seis pasajes del caso
manual y usa 4.097 tokens. La corrida con Chroma contiene ocho fragmentos y
usa 4.151 tokens. Ambos conteos se verificaron con el tokenizer real sin
recortar las fuentes. La comparación y los comandos de reproducción están
en el [registro de generación](ejercicio2/evaluacion/generacion_caso01.md).

Pasaron las 13 pruebas del ejercicio 2. Con una salida controlada se
comprobaron el mapeo de tiempos y la detección de citas inexistentes.
Se completó la inferencia real de ambas entradas con Qwen local.

**Resultado:** con los pasajes de referencia, Qwen copió material de la clase
hasta el límite de 1.000 tokens y no clasificó los criterios. Con Chroma
terminó en 608 tokens, declaró el objetivo cubierto y citó cuatro referencias
académicas inexistentes. Algunas citas del video también respaldaban un tema
distinto del que afirmó. La primera corrida muestra un problema de generación
aun con respaldo académico disponible. En la segunda se suma el faltante de
referencias ya medido en la recuperación.

**Siguiente paso propuesto:** probar una instrucción con salida más
estructurada, manteniendo fijos Qwen, las fuentes y `k=8`, y revisar
clasificaciones y citas con el mismo caso de desarrollo.

## 8. Comparación de prompts para el informe — 04/11

Probamos una instrucción que pide empezar por una tabla con una fila por
criterio y columnas separadas para citas de clase y video. Se conservaron
Qwen, el objetivo, el texto y los IDs de ambas fuentes, y `k=8` en Chroma.
Se verificaron los datos enviados contra las corridas anteriores. Pasaron
las 13 pruebas del ejercicio 2.

**Resultado:** las dos salidas empezaron con la tabla y se cortaron en
1.000 tokens. Con los pasajes de referencia aparecieron 14 IDs académicos
inexistentes. Con Chroma se repitieron filas y se usó una cita de video como
referencia académica. En ninguno de los dos informes las clasificaciones y
citas alcanzaron para una evaluación utilizable. La tabla queda disponible
con `--prompt tabla` para reproducir la prueba. El
[registro de generación](ejercicio2/evaluacion/generacion_caso01.md#segunda-prueba-con-una-tabla-por-criterio--0411)
contiene los resultados y ejemplos.

**Siguiente paso propuesto:** comparar el efecto de la penalización por
repetición que usa el notebook de la cátedra, con el prompt de tabla, para
medir las filas duplicadas y el cierre del informe. Volver a comprobar qué
criterios aparecen y si las citas sostienen las clasificaciones.

## 9. Penalización por repetición — 04/11

Probamos `repetition_penalty=1.15`, valor usado en la celda 22 del notebook
de la cátedra, frente al `1.1` de Qwen. Las dos corridas conservaron el
prompt de tabla, el contenido de las fuentes, la revisión del modelo, el
límite de 1.000 tokens y `k=8` en Chroma. Se comprobó la equivalencia de
entradas y pasaron las 13 pruebas del ejercicio.

**Resultado:** las corridas terminaron en 473 tokens con los pasajes de
referencia y 197 con Chroma. En Chroma las filas duplicadas bajaron de
13 a cero, pero quedaron dos filas sin citas del video. Con los pasajes de
referencia persistieron seis IDs académicos inexistentes y se omitieron
coseno y magnitud. Se conserva `1.15` como variante experimental. El
[registro de generación](ejercicio2/evaluacion/generacion_caso01.md#tercera-prueba-con-penalización-por-repetición--0411)
incluye el detalle. Se propone comparar otro generador primero con los
pasajes de referencia para evaluar el uso de evidencia disponible.

## 10. Comparación con GPT-4o — 04/11

Probamos GPT-4o por el proxy de la cátedra con los mismos mensajes, el
mismo video y los seis pasajes académicos de la corrida manual de Qwen.
El proxy usó `gpt-4o-2024-08-06`. La generación terminó en 346 tokens,
con cinco filas, una recomendación, una conclusión y citas con IDs
existentes. Se revisaron las citas contra ambas fuentes. Coincidieron las
clasificaciones de E1 y E3, mientras que la respuesta dio dos estados
incompatibles al tema del contexto y las representaciones estáticas frente
a las dinámicas. El coseno quedó en la recomendación y faltó evaluar la
limitación de magnitud.

El proxy reportó US$0,0124925 por la corrida. La próxima comparación
propuesta es pasarle a GPT-4o los fragmentos recuperados por Chroma con
`k=8` para ver cómo responde cuando faltan referencias académicas.
El [registro de generación](ejercicio2/evaluacion/generacion_caso01.md#cuarta-prueba-con-gpt-4o--0411)
incluye la revisión y los archivos de la corrida.

## 11. GPT-4o con referencias recuperadas por Chroma — 04/11

Repetimos la prueba con el mismo objetivo, prompt, transcripción y versión
de GPT-4o. Los seis pasajes de referencia se reemplazaron por los ocho
resultados guardados de Chroma. Se cotejó cada cita con la fuente a la
que apuntaba.

**Resultado:** la respuesta terminó en 314 tokens y produjo cuatro filas.
La definición quedó respaldada. La comparación de similitud se marcó
como parcial con evidencia del video, pero los pasajes de clase citados
no explicaban la medida. La respuesta omitió coseno y magnitud y no
señaló que faltaba respaldo académico. El proxy reportó US$0,0123625.
El [registro de generación](ejercicio2/evaluacion/generacion_caso01.md#quinta-prueba-con-gpt-4o-y-chroma--0411)
conserva la comparación. La siguiente revisión puede comprobar si el
generador reconoce referencias insuficientes antes de asignar etiquetas.

## 12. Revisión de los apuntes como fuente — 04/11

Se revisó el texto de las 14 páginas de [LLMs - Apuntes.pdf](<../docs/LLMs - Apuntes.pdf>)
para evaluar si aportaba referencias al caso 01. El PDF tiene una capa de
texto y abarca la Unidad 1, Transformers. La página 5 introduce el embedding
como vector de entrada. La página 6 usa el coseno para la codificación de
posición, como se confirmó al cotejar la página con el texto extraído.
Las páginas 7–8 describen el producto escalar de atención. El apunte aporta
material para objetivos de Transformers, pero deja sin cubrir el cálculo
de similitud coseno entre embeddings y el ejemplo sobre magnitud que
buscamos para el caso 01. Se reservó como fuente para un caso de
Transformers. Para ampliar el caso 01, queda identificar un apunte de
la unidad de embeddings con esos temas.

## 13. Colección con la clase y el apunte derivado — 04/11

Por pedido del usuario se sumó el apunte de estudio sobre embeddings y
similitud coseno, elaborado a partir de la clase 2. Se indexó el texto de
las páginas 3–5 con MiniLM y los parámetros de splitting de la clase.
Se conservan los números de página y la marca de procedencia derivada.
La colección mixta `tp3_248fe3ddb4d5588934314f79` contiene 136 chunks
(125 de la transcripción y 11 del apunte). La colección base se conservó.
Al repetir la indexación se reutilizaron los 136 chunks. Pasaron las
13 pruebas existentes del ejercicio 2.

Se repitieron las dos consultas de la ablación con `k=8`. En «entender
similitud coseno y sus limitaciones», el ejemplo de magnitud del apunte
apareció en el puesto 3 (página 5) y la fórmula de la transcripción en el
puesto 6 (31:44.560–32:41.570). En el objetivo general del caso 01,
el PDF aportó la definición y el contraste entre embeddings estáticos y
contextuales (página 3, puestos 3 y 5). La fórmula y el ejemplo completo
de magnitud quedaron fuera, y uno de los resultados fue la lista de
fuentes del apunte. Los resultados locales quedaron en
`data/resultados/chroma_con_apunte_20261104.7sqyKv/` del ejercicio 2.

Como el apunte se preparó con los temas del caso ya conocidos, la comparación
mide el efecto de agregar una síntesis dirigida al caso. Para las próximas
pruebas queda disponible el corpus mixto mediante `--chunks`. El
[README](ejercicio2/README.md#usar-la-colección-con-la-transcripción-y-el-apunte)
registra cómo reconstruirlo y consultarlo.

## 14. Búsqueda por subtemas del objetivo — 04/11

Comparamos la consulta directa «quiero estudiar embeddings» con una expansión
por subtemas, inspirada en `UN_Report_Advanced_RAG.ipynb`, celdas 48–51.
Se mantuvieron MiniLM, distancia coseno y la colección mixta de 136 chunks.
Ambas variantes devolvieron ocho referencias finales. Para la expansión se
hizo una llamada a GPT-4o con el objetivo y la instrucción de proponer hasta
cuatro subtemas introductorios, cada uno con una pregunta en español.
El caso manual se usó para revisar el resultado después de buscar.

GPT-4o propuso concepto, tipos, entrenamiento y aplicaciones. Cada
pregunta recuperó ocho candidatos. Los 32 resultados contenían 15 IDs
distintos. Para elegir los ocho finales, se recorrieron los resultados
por puesto dentro de cada subconsulta y en orden de subtemas, omitiendo
los IDs ya incorporados.

| Referencia esperada | Consulta directa | Expansión |
| --- | --- | --- |
| Definición de embeddings (E1) | Puesto 3, página 3 | Puesto 2, página 3 |
| Contraste entre estáticos y contextuales (E2) | Menciones al contexto, sin el ejemplo completo | Puesto 5, página 3, con el ejemplo de «banco» |
| Fórmula del coseno (E4) | Una lista de fuentes remite al cálculo | Puesto 3, entrada 222 del VTT, 31:55.350–32:14.820 |
| Limitación de magnitud con ejemplo (E5) | Apareció la regla sin el ejemplo completo | Apareció la regla sin el ejemplo completo |

La expansión recuperó la fórmula mediante la pregunta de aplicaciones.
El plan dejó fuera la comparación de vectores como subtema explícito.
Quedó pendiente recuperar un pasaje que explique de forma directa el
procedimiento general de comparación de textos (E3). La lista de fuentes
de la página 5 quedó primera en ambas variantes. Estos resultados
corresponden a un solo objetivo sobre un corpus que incluye el apunte
preparado para este caso.

El proxy reportó 108 tokens de entrada, 150 de salida y US$0,00177.
La búsqueda se ejecutó localmente. El prompt, las consultas, los candidatos
y la selección quedaron en `data/resultados/expansion_subtemas_20261104.zQ1Iz1/`
del ejercicio 2, en `plan.json`, `directa.json` y `expandida.json`.
La próxima variante propuesta es pedir que el plan considere cómo se aplica
o se mide el concepto y qué limitaciones conviene entender, cuando sean
pertinentes al objetivo. Se puede comparar con la misma colección y el
mismo presupuesto final de ocho fragmentos.

## 15. Revisión de la recuperación según el objetivo — 04/11

Se revisaron las dos búsquedas guardadas para «quiero estudiar embeddings»
con criterios introductorios. Se consideró que un criterio tenía respaldo
cuando los fragmentos finales aportaban una definición vectorial del
texto, un uso explicado con un ejemplo, o un contraste entre embeddings
estáticos y contextuales. Fue una revisión retrospectiva de la misma
prueba con el objetivo literal como referencia.

| Criterio introductorio | Consulta directa | Expansión |
| --- | --- | --- |
| Qué es un embedding de texto | Respaldado, puesto 3 | Respaldado, puesto 2 |
| Para qué sirve | Respaldado por el ejemplo de textos con significados parecidos, puesto 3 | Mismo pasaje, puesto 2 |
| Cómo cambia con el contexto | Parcial: menciona el contexto sin desarrollar el contraste | Respaldado por el contraste y el ejemplo de «banco», puesto 5 |

Ambas variantes recuperaron la definición y un uso concreto. La expansión
agregó respaldo completo para el papel del contexto. La fórmula se
reservó como criterio para objetivos que pidan el cálculo. Con la misma
colección, MiniLM y `k=8`, se probaron estas consultas locales:

| Consulta sobre el cálculo | Evidencia en los primeros ocho |
| --- | --- |
| «cómo se calcula la similitud coseno» | Fragmentos relacionados, sin el cálculo completo |
| «cómo calcular la similitud coseno entre dos embeddings» | Fórmula con producto escalar y normas en el puesto 3, página 4 del apunte |

Las dos redacciones apuntan al cálculo y dieron resultados distintos.
El segundo caso recuperó la fórmula, pero el ejemplo numérico completo
quedó fuera. Los resultados se guardaron en
`data/resultados/expansion_subtemas_20261104.zQ1Iz1/` como
`calculo_coseno_k8.json` y `calculo_coseno_embeddings_k8.json`.
La revisión y las dos búsquedas fueron locales. Para una próxima prueba
del informe introductorio se pueden usar los ocho pasajes de la expansión
y revisar definición, uso y contexto.

## 16. Primer informe en el formato para el estudiante — 04/11

Se generó un informe para «quiero estudiar embeddings» con
`gpt-4o-2024-08-06` a través del proxy de la cátedra. Se reutilizaron los
cuatro subtemas del plan, los ocho resultados finales de la búsqueda
expandida y los 99 segmentos del video. El modelo devolvió una evaluación
por subtema, recomendaciones y conclusión. La vista legible muestra las
referencias de la materia en las recomendaciones y guarda los tiempos
del video bajo «Ver evidencia». La llamada usó 3.737 tokens de entrada y
446 de salida. El proxy informó US$0,0138025.

Se verificaron las cuatro filas, los estados y la existencia de los IDs
citados. El cotejo manual confirmó que el apunte y el video respaldan la
definición y el papel del contexto. La transcripción del video respalda
las aplicaciones mencionadas, pero el modelo citó como respaldo académico
`C1`, que es una lista de referencias del apunte. Para entrenamiento
asignó «Parcial» y recomendó `C3` (31:44.560–32:41.570 de la clase).
Ese tramo contiene una mención al ajuste de pesos y se centra en comparar
vectores. El audio citado solo menciona que hubo entrenamiento. El
criterio quedó sin respaldo suficiente de la materia y la recomendación
requiere corrección.

La vista de la respuesta del modelo y su auditoría quedaron en
`data/resultados/informe_estudiante_embeddings_20261104.4x6c1gtg/`
del ejercicio 2, como `informe.md` y `auditoria.json`. La prueba
muestra el formato acordado y deja pendiente mejorar la decisión sobre
cuándo un pasaje permite evaluar el subtema y recomendar material.

## 17. Incorporación de las llamadas de prueba al código — 04/11

Las llamadas al proxy de las primeras corridas se hicieron desde comandos
exploratorios. Se incorporaron al repo para que las pruebas puedan repetirse.
`recuperar.py` contiene el prompt de expansión, la selección de referencias
y la opción de reutilizar un plan guardado. `generar_informe.py` contiene
los prompts del informe, incluido `estudiante`, y acepta una recuperación
guardada. Ambos registran las llamadas a través de `proxy_catedra.py`.
El README y el registro del caso 01 conservan los comandos.

Se cotejaron los cuatro mensajes completos de las llamadas históricas
(expansión, dos informes de tabla y un informe del estudiante). Se
reprodujeron los ocho fragmentos de Chroma reutilizando el plan guardado.
Pasaron 16 pruebas automatizadas, con respuestas simuladas para el proxy.
La validación de esta integración se hizo con las respuestas ya guardadas
y sin consumo adicional de API. Queda repetir la evaluación específica
desde la CLI cuando corresponda la siguiente corrida.

## 18. Comparación de GPT-4.1 y GPT-4o en el informe del estudiante — 04/11

Se agregó `--modelo-proxy` a `generar_informe.py` y se ejecutó una
corrida con `openai/gpt-4.1-2025-04-14`. Se comprobó que el objetivo,
los mensajes, los cuatro subtemas, los ocho fragmentos y los 99 segmentos
del video coincidieran con la corrida de GPT-4o. Ambas usaron temperatura
cero, salida JSON y un límite de 1.000 tokens de salida. Pasaron las 16
pruebas automatizadas.

| Modelo | Tokens de entrada | Tokens de salida | Costo informado por el proxy |
| --- | ---: | ---: | ---: |
| GPT-4o | 3.737 | 446 | US$0,0138025 |
| GPT-4.1 | 3.737 | 622 | US$0,01245 |

GPT-4.1 conservó las evaluaciones respaldadas de definición y contexto y
citó segmentos sobre el ejemplo de «banco». En aplicaciones dejó de citar
`C1`, la lista de referencias. `C2` sí explica un uso, la comparación
semántica de textos. Los ejemplos de agrupamiento, síntesis y recomendación
se respaldan en el video, mientras que el material de clase citado solo
respalda parte de esa fila.

Ambos modelos marcaron entrenamiento como «Parcial» y recomendaron el mismo
tramo de la clase (31:44.560–32:41.570). El pasaje menciona ajuste de pesos,
pero se centra en comparar vectores y deja sin desarrollar los métodos que
recomienda el informe. Para ese criterio correspondía «No se pudo evaluar
con el material disponible». La prueba deja disponible GPT-4.1 como variante
y mantiene abierta la revisión de cómo exigir respaldo antes de clasificar.

La salida y la auditoría están en
`data/resultados/informe_estudiante_gpt41_20261104.41f5UB/`
del ejercicio 2. El [README](ejercicio2/README.md#repetir-las-pruebas-con-el-proxy)
registra cómo repetir la comparación.

## 19. Diagnóstico de la subconsulta de entrenamiento — 04/11

Se revisaron las cuatro búsquedas del plan de GPT-4o y se confirmó que los
informes de GPT-4o y GPT-4.1 recibieron la misma selección guardada de ocho
fragmentos. La pregunta original sobre entrenamiento ya se había buscado
en Chroma. Sus ocho candidatos incluían definiciones, menciones generales
al entrenamiento y un ejemplo de conteo de palabras. El pasaje que explica
el método de entrenamiento de Word2Vec quedó fuera de esos candidatos.

Se hizo una consulta local de diagnóstico con la misma colección, MiniLM
y `k=8`. La redacción «cómo se entrena Word2Vec skip-gram con pares
positivos y negativos» recuperó en los puestos 1 y 2 la explicación del
método en la transcripción (29:16.470–30:22.040). Se eligieron esos términos
después de revisar la clase. El resultado comprueba que la evidencia está
indexada y se puede encontrar con una consulta más específica. Falta
comprobar si el plan puede generar esa consulta a partir del objetivo.
El resultado quedó en
`data/resultados/diagnostico_entrenamiento_20261104.js0weJ/skipgram.json`
del ejercicio 2.

## 20. Expansión con preguntas más concretas — 04/11

Se probó `--expandir --prompt-expansion concreto` con el objetivo «quiero
estudiar embeddings». La variante pide nombrar un método cuando el
subtema describe un proceso y elegir un caso concreto cuando describe
un uso. Se conservaron GPT-4o, la colección mixta, MiniLM, ocho candidatos
por consulta y ocho fragmentos finales seleccionados por rondas. El prompt
no contenía el nombre de Word2Vec ni los términos de la consulta de diagnóstico.

GPT-4o propuso concepto, representación vectorial, Word2Vec y
clasificación de texto. La pregunta «¿Cómo funciona el método Word2Vec
para generar embeddings con un ejemplo?» no recuperó entre sus ocho
candidatos el tramo que explica los pares positivos y negativos
(29:16.470–30:22.040). La transcripción escribe variantes como «world to Beck»
y «Skidcrum». La consulta manual anterior, que también describía los pares
positivos y negativos, sí había recuperado el pasaje. Esta comparación
muestra que nombrar el método por sí solo fue insuficiente en esta corrida.

La búsqueda produjo 32 candidatos, 17 únicos y ocho finales. Entre los
finales se conservaron definición y un uso introductorio, pero se perdió
el pasaje sobre embeddings estáticos y contextuales. La variante inicial
respaldaba los tres criterios introductorios, por lo que se conserva
como configuración de referencia. La llamada consumió 165 tokens de
entrada y 142 de salida. El proxy informó US$0,0018325. Pasaron las
16 pruebas automatizadas. Los resultados quedaron en
`data/resultados/expansion_concreta_20261104.1098T9/` del ejercicio 2.

## 21. Configuración para las próximas pruebas — 04/11

Se eligió el prompt inicial de expansión y GPT-4.1
(`openai/gpt-4.1-2025-04-14`) como modelo predeterminado del proxy para
planificar subtemas y generar informes. Los comandos de reproducción
identifican explícitamente GPT-4o cuando corresponde a una corrida previa.
Pasaron las 17 pruebas automatizadas. La validación de la configuración
usó respuestas simuladas del proxy.

## 22. Cómo seguiremos actualizando este documento

En cada avance significativo agregaremos una entrada al recorrido, con fecha
y en el orden en que ocurrió. Actualizaremos también el estado actual. Al
cambiar una decisión conservaremos la anterior y el motivo del cambio, para
que se entienda cómo evolucionó el proyecto.

Cada entrada incluirá:

- **Situación de partida:** qué problema o límite intentamos resolver.
- **Cambio y motivo:** qué elegimos, qué alternativa reemplaza y por qué.
- **Configuración y alcance:** clases, videos, modalidades, modelos y parámetros que cambiaron o se mantuvieron.
- **Prueba o revisión realizada:** qué comparamos y con qué evidencia.
- **Resultado y límite:** qué observamos, incluidos resultados desfavorables, y qué sigue sin comprobarse.
- **Consecuencia:** siguiente paso y enlaces a código, informe, tiempos/páginas o commit, cuando corresponda.

Las pruebas propuestas se anotarán como pendientes hasta ejecutarse. La
configuración detallada y las salidas completas permanecerán en los informes
de cada experimento; aquí registraremos lo necesario para entender la evolución.
