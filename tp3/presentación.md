# Presentación: recorrido y decisiones del TP3

Última actualización: **31/10/2026**.

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
del modelo por API de la propuesta original. **Todavía no implementamos ni
probamos un generador.** La justificación y las referencias están en el
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

Flujo previsto para evaluar un video, todavía incompleto:

```text
Objetivo de estudio -> búsqueda en Chroma -> referencias académicas
Audio del video -> Whisper -> transcripción guardada
Objetivo + referencias + transcripción -> generador -> informe de cobertura
                                        [pendiente]
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

El próximo avance propuesto es comparar `k=2,4,8` manteniendo fijo el resto y
registrar si mejora la recuperación de las referencias o solo aumenta el ruido.
Si hace falta, se estudiarán otras variables de a una. Después continuaremos
con generación y evaluación del informe. Ampliar el corpus o las modalidades
deberá quedar registrado como un cambio de alcance con sus propios resultados.

Para ejecutar el estado actual, consultar el [README del ejercicio 2](ejercicio2/README.md).
Los VTT originales bajo `docs/`, los audios, los pesos, Chroma y los resultados
bajo `data/` están ignorados por Git: un clon no incluye todos esos archivos.
Este documento y los informes de revisión conservan el recorrido compartible.

## 6. Cómo seguiremos actualizando este documento

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
