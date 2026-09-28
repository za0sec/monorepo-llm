# Guía de estudio para la defensa oral del TP2

## *Query-Based Adversarial Prompt Generation*

**Paper:** Jonathan Hayase, Ema Borevkovic, Nicholas Carlini, Florian Tramèr y Milad Nasr. NeurIPS 2024.  
**Fuentes usadas:** [paper](paper.pdf), [consigna](tp2.pdf), diapositivas y transcripciones de [`docs/`](docs/).

Esta guía está organizada según los ejes de la consigna. No está pensada para memorizarla palabra por palabra, sino para poder responder con esta estructura:

> **Afirmación breve → evidencia del paper → interpretación → límite o matiz.**

Las referencias como `[§4.3, p. 6]` apuntan a la sección y página impresa del paper. Las conexiones marcadas como **Lectura crítica** son interpretaciones razonables, pero no afirmaciones explícitas de los autores.

---

## 0. Qué exige realmente la defensa

La consigna y la explicación de las profesoras dejan estas prioridades:

- dura aproximadamente 30 minutos y tiene formato de preguntas y debate, no de exposición;
- todos los integrantes deben poder responder sobre todo el paper;
- no hace falta reproducir demostraciones matemáticas, pero sí explicar la lógica del método;
- hay que identificar la novedad y compararla con lo que existía antes;
- cada afirmación de los autores debe contrastarse con sus resultados concretos;
- esperan opiniones propias: límites, objeciones, mejoras y experimentos adicionales;
- reconocer una incertidumbre de manera fundada es mejor que inventar una certeza.

Por eso la preparación no debería dividir el paper en “una sección por persona”. Pueden repartirse quién inicia cada tema, pero todos tienen que dominar el argumento completo.

---

## 1. El paper en un minuto

Antes de este trabajo, los ataques automáticos contra LLMs alineados seguían principalmente dos caminos:

1. **White-box:** se accede a los pesos y gradientes del modelo. GCG es el ejemplo central.
2. **Transferencia:** se optimiza un prompt adversarial en un modelo abierto y luego se lo prueba en el modelo cerrado.

El problema es que la transferencia necesita un buen modelo sustituto y funciona mal para ataques **dirigidos**, donde no alcanza con lograr que el modelo deje de rechazar: se busca que emita exactamente un string objetivo.

La idea del paper es usar consultas al propio modelo objetivo durante la optimización. Su método, **GCQ (Greedy Coordinate Query)**, genera muchos vecinos de un prompt, usa un modelo local barato para preseleccionarlos y consulta al modelo remoto para decidir cuáles realmente reducen la pérdida. También propone una variante sin modelo local.

La evidencia principal es que GCQ:

- fuerza strings exactos en GPT-3.5 Turbo Instruct con **79,6 % de éxito gastando como máximo USD 0,10 por objetivo**;
- llega a **97,9 %** para objetivos de hasta 20 tokens;
- evade un moderador de OpenAI con un sufijo universal en **99,2 %** del conjunto de validación;
- también evade Llama Guard;
- supera ampliamente a ataques basados solo en transferencia en la tarea dirigida.

La conclusión de seguridad es que **cerrar los pesos o impedir la transferencia no alcanza** si la API todavía expone una señal que permite optimizar contra el modelo real.

---

## 2. La tesis lógica del trabajo

Conviene poder reconstruir este argumento sin mirar notas:

```text
GCG funciona, pero necesita gradientes del modelo objetivo
                         ↓
cada iteración de GCG tiene una etapa de filtrado y otra de evaluación exacta
                         ↓
la evaluación exacta puede hacerse consultando al modelo objetivo
                         ↓
el filtrado puede delegarse a un proxy local, aunque no sea un buen sustituto
                         ↓
GCQ optimiza directamente contra un modelo remoto
                         ↓
puede hacer ataques dirigidos que la transferencia pura no consigue
```

La novedad conceptual no es simplemente “hacer muchas queries”. Es **separar la propuesta barata de candidatos de la decisión basada en la pérdida real del objetivo**.

---

## 3. Glosario imprescindible

- **Adversarial example:** entrada construida deliberadamente para inducir un fallo del modelo.
- **Modelo alineado:** modelo ajustado para seguir instrucciones y evitar conductas dañinas. Estar alineado en entradas normales no implica robustez ante entradas adversariales.
- **White-box:** el atacante conoce pesos y gradientes.
- **Black-box:** el atacante no conoce los pesos; solo observa alguna interfaz de entrada/salida.
- **Ataque por transferencia:** se optimiza en un modelo local y se envía el resultado terminado al modelo objetivo.
- **Ataque basado en queries:** las respuestas del modelo objetivo guían la búsqueda del ataque.
- **Ataque dirigido:** busca una salida exacta y predeterminada.
- **Ataque no dirigido / jailbreak:** busca alguna conducta indebida, sin exigir una salida exacta.
- **Harmful string:** string objetivo cuya coincidencia debe ser exacta. Es una condición más estricta que “el modelo no rechazó”.
- **Surrogate / modelo sustituto:** modelo local sobre el que se construye un ataque que después se transfiere.
- **Proxy:** modelo local usado solo para ordenar candidatos; el modelo objetivo toma la decisión final mediante su pérdida real.
- **ASR:** *attack success rate*, proporción de ataques exitosos.
- **Loss del ataque:** log-probabilidad acumulada negativa del string objetivo condicionado por el prompt.
- **Sufijo universal:** un mismo sufijo sirve para muchas entradas.
- **Sufijo no universal:** se optimiza un sufijo distinto para cada entrada.

**Matiz:** el paper no siempre usa “proxy” y “surrogate” con una separación terminológica rígida. La distinción anterior sirve para entender la función que cumple el modelo local en cada ataque.

---

# Ejes pedidos por la consigna

## 4. Problema de investigación, relevancia y contexto

### Problema

Los LLMs alineados suelen rechazar pedidos dañinos, pero esa conducta puede fallar ante prompts construidos adversarialmente. Los ataques white-box pueden ser fuertes, aunque no son realistas contra modelos cerrados. Los ataques por transferencia son más realistas, pero tienen dos limitaciones centrales [§2, pp. 2-3]:

- necesitan un sustituto suficientemente parecido al objetivo;
- no consiguen con fiabilidad que el modelo genere un string dañino exacto.

### Relevancia

Forzar un string exacto es especialmente importante cuando la salida del LLM no es solo texto para leer, sino que puede activar una herramienta, un plugin o una acción de un agente. El paper menciona pagos, lectura o envío de correos y llamadas maliciosas a plugins [§1, p. 1].

### Contexto histórico

El paper presenta un paralelismo con visión:

1. ataques white-box;
2. transferencia entre modelos;
3. optimización black-box mediante queries.

En texto aparece una dificultad adicional: los tokens son discretos. No se puede aplicar un pequeño cambio continuo a una palabra como se modifica un píxel. Métodos anteriores optimizaban en el espacio continuo de embeddings y luego proyectaban a tokens, pero no resultaban suficientemente fuertes contra transformers grandes [§2, p. 2].

### Respuesta oral corta

> El problema es cómo construir ataques dirigidos contra modelos cerrados cuando no tenemos sus pesos. La transferencia pura no alcanza para strings exactos y depende demasiado de tener un sustituto parecido. El paper propone usar la API del objetivo como señal de optimización y demuestra que eso vuelve prácticos ataques dirigidos contra generación y moderación.

---

## 5. Hipótesis, preguntas y objetivos

El paper no formula hipótesis estadísticas explícitas. Es mejor decirlo así y luego reconstruir las preguntas de investigación.

### Objetivo explícito

Diseñar un ataque que construya ejemplos adversariales **directamente sobre un modelo remoto**, sin depender exclusivamente de la transferencia [§1, pp. 1-2] .

### Preguntas reconstruidas

| Pregunta | Evidencia que la responde |
|---|---|
| ¿Las queries permiten forzar strings exactos donde la transferencia falla? | Tabla 1 y experimento con GPT-3.5, §§4.2-4.3 |
| ¿Hace falta que el proxy sea un sustituto fiel? | Mistral 7B base funciona como proxy para GPT-3.5, §4.3 |
| ¿Se puede atacar sin proxy ni gradientes? | Variante query-only, §§3.3 y 4.4 |
| ¿El enfoque también sirve contra clasificadores? | OpenAI Moderation y Llama Guard, §§4.5-4.6 |
| ¿Qué factores condicionan el éxito? | Longitud del objetivo, longitud del prompt e inicialización, §4.3 |

### Hipótesis implícitas

- La etapa de selección exacta de GCG puede reemplazarse por consultas al objetivo.
- Para filtrar candidatos, un proxy aproximado puede ser suficiente.
- Como el gradiente de GCG es una señal relativamente débil, una asignación más inteligente del presupuesto de queries puede compensar su ausencia.

---

## 6. Conceptos técnicos necesarios

### 6.1 De GCG a GCQ

**GCG (Greedy Coordinate Gradient)** calcula qué sustituciones de un token parecen prometedoras usando gradientes, evalúa los candidatos y conserva el de menor pérdida. Es white-box porque necesita derivar a través del modelo objetivo [§2, p. 3].

El insight de GCQ es dividir esa iteración:

1. **Proponer y filtrar candidatos:** puede hacerlo un modelo proxy local.
2. **Evaluar los finalistas:** se consulta la pérdida del modelo objetivo.

La transferencia pura confía en que el ataque terminado generalice. GCQ, en cambio, **corrige su trayectoria con feedback del objetivo en cada iteración**.

### 6.2 La función de pérdida

Si el prompt adversarial es `x` y el string objetivo es `y = (y₁, …, yₜ)`, la pérdida es:

```text
ℓ(x) = -Σᵢ log p(yᵢ | x, y₁, …, yᵢ₋₁)
     = -log P(y | x)
```

Minimizarla aumenta la probabilidad conjunta del string objetivo. El paper considera exitoso el ataque si ese string se genera con decodificación greedy [§3.1, p. 4].

**Conexión con clase:** es la misma lógica de *next-token prediction* y entropía cruzada usada para entrenar un LM. Cambia qué se optimiza: en entrenamiento se modifican los pesos; en el ataque se buscan tokens de entrada.

### 6.3 GCQ paso a paso

Entradas principales del Algoritmo 1 [p. 3]: vocabulario `V`, longitud `m`, loss real `ℓ`, loss proxy `ℓp`, iteraciones `T`, cantidad de candidatos locales `bp`, cantidad de queries al objetivo `bq` y tamaño de buffer `B`.

```text
Inicializar un buffer con B prompts
             ↓
Tomar el mejor prompt conocido
             ↓
Generar muchos vecinos cambiando un token
             ↓
Ordenarlos con la loss del proxy local
             ↓
Consultar al objetivo solo para los mejores bq
             ↓
Reemplazar los peores elementos del buffer si hubo mejoras
             ↓
Repetir y devolver el prompt de menor loss
```

El buffer convierte la búsqueda en algo parecido a *best-first search*: conserva varias alternativas buenas en lugar de una sola trayectoria.

### 6.4 ¿Para qué sirve el buffer?

- Mantiene diversidad entre soluciones prometedoras.
- Evita depender por completo de una sola rama greedy.
- Da un umbral: si la pérdida parcial de un candidato ya supera a la del peor elemento del buffer, se puede cortar la evaluación.

Si `B = 1`, el método se acerca a *hill climbing*: solo conserva el mejor estado actual y tiene más riesgo de estancarse.

### 6.5 Short-circuit de la pérdida

Cada término `-log p` es no negativo. Por eso la suma parcial de la pérdida solo puede crecer. Si ya superó `ℓ(b_worst)`, terminar el cálculo no cambiaría que el candidato será descartado. Esta optimización reduce el costo aproximadamente **30 %** [§3.2.2, p. 4].

### 6.6 Inicialización

En vez de comenzar con tokens aleatorios, repiten el string objetivo tantas veces como entra en el prompt y truncan por la izquierda. Con prompts de 20 tokens, esta inicialización ya resuelve **161 de 574 casos, alrededor del 28 %**, sin ejecutar la búsqueda [§§3.2.3 y 4.3].

Esto explota la tendencia autoregresiva del LM a continuar patrones presentes en el contexto.

### 6.7 Cómo estiman logprobs a través de una API

OpenAI había dejado de devolver directamente los logprobs de tokens incluidos en el prompt. Los autores combinan `logit bias` con los cinco logprobs principales para llevar un token de interés al top-5 y luego corregir matemáticamente la distorsión [§3.2.1 y Ap. B].

Si `b` es el bias aplicado y `p_biased` es la probabilidad observada:

```text
p_true = p_biased / (eᵇ(1 - p_biased) + p_biased)
```

Eligen inicialmente un bias cercano a `-log(p̂_true)` usando el score del prompt padre; si falla, hacen búsqueda binaria. Según el paper, la primera estimación funciona más del 99 % de las veces [Ap. B, p. 12].

**Límite importante:** un cambio posterior de la API volvió obsoleta esta técnica concreta para OpenAI; la alternativa de búsqueda binaria seguía siendo posible, pero más costosa. El aporte conceptual sobre ataques por queries permanece, aunque la implementación depende de la interfaz.

### 6.8 Tokenización

Una secuencia de IDs encontrada durante la optimización puede no coincidir con la tokenización canónica del string concatenado. Además, proxy y objetivo pueden usar tokenizers distintos. Los autores vuelven a tokenizar antes de consultar cada modelo [Ap. C, p. 12].

### 6.9 Variante sin proxy

La versión query-only no usa gradientes ni modelo local. Primero prueba una sustitución en cada posición, identifica la posición que produjo la mayor mejora y concentra allí `B'` intentos adicionales. Así asigna mejor las queries y reduce aproximadamente a la mitad las necesarias frente a GCG optimizado [§3.3, pp. 4-5].

---

## 7. Metodología experimental

| Experimento | Objetivo | Proxy / acceso | Métrica y presupuesto |
|---|---|---|---|
| Modelos abiertos, Fig. 1 | Vicuna 1.3 de 7B/13B/33B y Llama 2 Chat 7B | White-box o gradientes de otro Vicuna + queries | ASR acumulado; batch 512; 500 iteraciones |
| Comparación, Tabla 1 | Varios modelos abiertos | GCQ, transferencia pura y AutoDAN | Coincidencia exacta del string |
| GPT-3.5, Figs. 2-3 | `gpt-3.5-turbo-instruct-0914` | Mistral 7B base como proxy | ASR vs costo e iteraciones; máximo USD 1 por objetivo |
| Sin proxy, Fig. 4 | Vicuna 7B | Solo queries | ASR vs cantidad de evaluaciones de loss |
| Moderación OpenAI, Figs. 5 y 6a | `text-moderation-007` | Solo queries | Éxito = ninguna categoría marcada |
| Moderación Llama Guard, Fig. 6b | Llama Guard 7B | Solo queries | ASR vs queries |

### Datos

Usan los **574 harmful strings** introducidos por Zou et al. [36]. Para el ataque universal al moderador de OpenAI, mezclan el dataset, usan 20 strings para optimizar y los 554 restantes como validación [§4.5, p. 8].

### Métricas

- **Generación:** el modelo debe producir exactamente el target bajo greedy decoding.
- **Moderación:** el texto con el sufijo no debe activar ningún flag.
- **Loss de moderación:** suma de los scores de categorías. Evita necesitar los umbrales privados, aunque la condición final de éxito sí se comprueba con los flags.

### Cómputo

Los experimentos de §4.1 usan entre 2 y 8 GPUs A100 durante varios días. Los demás usan una A40 durante varios días [Ap. A, p. 12].

---

## 8. Resultados principales y qué demuestran

### 8.1 Modelos abiertos

- Vicuna se vuelve más difícil de atacar al aumentar su tamaño.
- Llama 2 Chat 7B es mucho más resistente que incluso Vicuna 33B.
- Entre Vicunas, 13B y 33B parecen más similares entre sí que respecto de 7B [Fig. 1, p. 5].

**Qué sugiere:** la robustez no depende solo de la escala; el modelo base y el ajuste de alineamiento importan.

**Qué no demuestra:** no aísla causalmente si la diferencia proviene de arquitectura, datos, fine-tuning o alineamiento.

### 8.2 Comparación con transferencia y AutoDAN

En la Tabla 1 [p. 6]:

- GCG por transferencia pura obtiene **0,000** en todos los pares reportados.
- GCQ obtiene **0,388** con proxy Vicuna 13B y objetivo Vicuna 7B.
- GCQ obtiene **0,791** cuando proxy y objetivo son Vicuna 7B.
- AutoDAN obtiene **1/574** en modo GA y **0/574** en HGA.

**Qué demuestra:** lograr un jailbreak general no equivale a controlar una salida exacta; el feedback del objetivo cambia fuertemente el resultado.

**Matiz:** AutoDAN fue diseñado para jailbreaks, no para harmful strings exactos, y se usó con parámetros por defecto. Es una comparación ilustrativa, no una evaluación exhaustiva de su mejor versión posible.

### 8.3 GPT-3.5 Turbo Instruct

- 161/574 objetivos, aproximadamente **28 %**, salen solo con la inicialización.
- **79,6 %** de ASR con un gasto máximo de USD 0,10 por objetivo.
- **86,0 %** con USD 0,20 por objetivo.
- Costo total de los 574 ataques: **USD 80**.
- Parámetros: prompt de 20 tokens, batch de queries 32, batch proxy 8192 y buffer 128 [§4.3, p. 6].

**Evidencia fuerte:** el proxy es Mistral 7B base, no alineado. No sería un buen sustituto para transferencia pura, pero sí sirve para filtrar candidatos.

### 8.4 Efecto de la longitud

Para objetivos de hasta 20 tokens, el ASR es **97,9 %**. El éxito cae con fuerza cuando el objetivo se acerca o supera la longitud del prompt [Fig. 3, p. 7].

Los autores proponen dos explicaciones:

1. ya no entra una copia completa del objetivo en la inicialización;
2. faltan grados de libertad en el prompt para codificar un target más largo.

Reejecutan los 39 fallos largos con prompts de 40 tokens y presupuesto máximo de USD 2: consiguen **100 %**, con costo medio de USD 0,41.

**Lectura crítica:** el experimento confirma que aumentar la longitud ayuda, pero no separa las dos explicaciones, porque mejora al mismo tiempo la inicialización y los grados de libertad.

### 8.5 Importancia de la inicialización

Con inicialización aleatoria, solo 2 de los primeros 20 strings tienen éxito. La inicialización es crucial en el régimen barato [§4.3, p. 7].

**Límite:** la ablación usa una muestra chica y no aleatoria: “los primeros 20”.

### 8.6 Ataque sin proxy

- La variante optimizada con gradientes necesita aproximadamente **2× menos queries** que GCG original para alcanzar un ASR comparable.
- Sin gradientes, la variante totalmente black-box todavía supera al GCG original aproximadamente en **30 %**, según los autores [Fig. 4, pp. 7-8].

**Interpretación:** asignar bien el presupuesto de búsqueda puede ser tan importante como disponer de una señal de gradiente.

### 8.7 Moderación de OpenAI

- Sin ataque, **197/574 ≈ 34 %** de los strings ya no son marcados.
- Sufijo universal de 20 tokens: **99,2 %** de ASR en validación después de 100 iteraciones / 2000 requests.
- Sufijo universal de 5 tokens: **94,8 %** después de 2000 iteraciones / 40.000 requests.
- Ataques no universales tras 10 iteraciones: **83,8 %** con 5 tokens y **91,4 %** con 20 [§§4.5, pp. 8-9].

Siempre hay que comparar contra la línea base del 34 %: el ataque no parte de cero.

### 8.8 Llama Guard

Tras 320 queries, el ASR es **59 %** para sufijos de 5 tokens y **87 %** para 20 tokens. En OpenAI, con presupuesto comparable, es 84 % y 91 %. La brecha se reduce con más iteraciones [§4.6, p. 9].

### 8.9 No determinismo

Los logprobs de GPT-3.5 no son estables aun con seed fija y temperatura 0. Para una pareja fija de prompt y target, la desviación estándar observada de la loss es 0,068; la diferencia típica entre el mejor y peor elemento del buffer es al menos 3. Los autores consideran que el ruido no altera demasiado la búsqueda [Ap. E, p. 13].

Al reevaluar los prompts exitosos, **519, alrededor del 90 %, vuelven a producir el target**. Por lo tanto, “éxito una vez” no equivale a éxito perfectamente reproducible.

---

## 9. Contribución novedosa frente al trabajo previo

1. Lleva los ataques dirigidos desde un escenario white-box o de transferencia a uno basado en queries contra un modelo remoto.
2. Muestra que el modelo local no necesita ser un buen sustituto; basta con que ayude a ordenar candidatos.
3. Propone una optimización de GCG que reduce las queries y habilita una variante sin proxy ni gradientes.
4. Resuelve obstáculos prácticos: estimación de logprobs, short-circuit, inicialización y diferencias de tokenización.
5. Demuestra que el ataque se extiende de generación a clasificadores de moderación y puede producir sufijos universales.

### Comparación rápida

| Método | Acceso | Resultado que busca | Limitación principal |
|---|---|---|---|
| GCG white-box | Pesos y gradientes | String exacto | Poco realista contra modelos cerrados |
| GCG por transferencia | Modelo local; una prueba final en objetivo | Jailbreak transferible | Mal desempeño dirigido; necesita sustituto parecido |
| AutoDAN | Black-box heurístico/genético | Jailbreak legible | Casi nulo en harmful strings en esta evaluación |
| GCQ | Queries + proxy opcional | String exacto | Necesita una señal de loss o scores y muchas consultas |

---

## 10. Supuestos, limitaciones y amenazas a la validez

### Supuestos del ataque

- El atacante conoce el string objetivo.
- Puede realizar muchas consultas parecidas sin ser bloqueado.
- La API devuelve logprobs, scores continuos o una señal equivalente que permita ordenar candidatos.
- Los costos y capacidades de la API corresponden al momento del experimento.

### Limitaciones reconocidas por los autores

- Los sufijos contienen tokens aparentemente aleatorios; un filtro de perplejidad podría detectarlos [Ap. D].
- La técnica concreta para recuperar logprobs quedó afectada por cambios de API.
- Un black-box más estricto, que solo devuelve texto y no probabilidades, queda abierto.
- El éxito depende mucho de la inicialización y de la relación entre longitud del prompt y del target.
- Los ataques de NLP siguen siendo frágiles frente a los de visión.
- Existe no determinismo y solo alrededor del 90 % de reproducción.

### Amenazas adicionales a la validez

#### Validez interna

- Muchas curvas no muestran intervalos de confianza ni resultados de varias semillas.
- La ablación de inicialización usa solo 20 ejemplos.
- Una sola evaluación puede sobreestimar el éxito estable por el no determinismo.

#### Comparación con baselines

- AutoDAN se evalúa fuera de su objetivo original y con configuración por defecto.
- En la Tabla 1 faltan varias celdas GCQ para los objetivos donde solo se reporta transferencia pura.
- GCQ 7B→7B usa como proxy una copia del propio objetivo abierto; es un escenario especialmente favorable.

#### Validez externa

- Solo se ataca un modelo cerrado generativo y es una versión *instruct completion*.
- No se prueban GPT-4, Claude, modelos de chat modernos ni modelos de razonamiento.
- Se usa un único dataset de harmful strings.
- No se estudia un sistema agéntico real donde el string producido ejecute una acción.

#### Validez de constructo

- Emitir un string exacto no siempre equivale a causar daño; el riesgo depende del sistema que consume la salida.
- En moderación, 34 % ya evade el clasificador sin sufijo.
- La suma de scores es un proxy de la decisión binaria, aunque el éxito final sí se verifica con los flags.

### Erratas e inconsistencias útiles para mostrar lectura atenta

- En el Algoritmo 1, el primer loop usa `bq`, pero por la descripción debería generar `bp` vecinos; además reutiliza la variable `i`.
- La Fig. 1b dice “transfer attacks”, aunque el texto aclara que se conservan queries al objetivo; no es transferencia pura.
- La leyenda de la Fig. 2 usa “Q-GC” en lugar de GCQ.
- En la Fig. 5, el panel (b) dice “10 token suffix”, mientras que el texto y el epígrafe dicen 20.
- La Fig. 6a habla de “prompt” en la leyenda donde el experimento describe un sufijo.

No conviene basar una crítica central en estas erratas, pero sirven para aclarar posibles confusiones.

---

## 12. Qué afirmaciones están bien respaldadas y cuáles no tanto

| Afirmación | Evidencia | Juicio crítico |
|---|---|---|
| Las queries superan a la transferencia pura para strings exactos | Tabla 1; GPT-3.5 | Bien respaldada en los modelos probados |
| Un proxy base puede servir aunque no sea buen surrogate | Mistral 7B → GPT-3.5 | Evidencia fuerte, pero es un solo par principal |
| El ataque puede ser barato | Curva costo-ASR de Fig. 2 | Válido para precios e interfaz de esa API |
| Puede evadir moderadores casi al 100 % | Figs. 5-6 | Fuerte, pero el baseline ya era 34 % y no hay defensa adaptativa |
| El largo del prompt causa la caída con targets largos | Reejecución con 40 tokens | Muestra que el largo ayuda, no cuál mecanismo causal explica la mejora |
| Los ataques query-only son prácticos en general | Vicuna y moderadores | Prometedor, pero “práctico” depende de rate limits, señales disponibles y detección |
| Las defensas contra transferencia son insuficientes | Éxito optimizando contra el target | Es una conclusión sólida del diseño del ataque |

---

## 13. Posición crítica para sostener en el debate

### Lo más valioso

- Cambia el modelo de amenaza: una API cerrada sigue siendo atacable si expone feedback rico.
- Separa claramente la calidad del proxy de la transferibilidad del ataque.
- Evalúa una condición fuerte: coincidencia exacta, no solo ausencia de rechazo.
- Incluye detalles prácticos y reconoce el impacto de cambios de API y no determinismo.

### Lo más cuestionable

- La dependencia de logprobs o scores continuos reduce la generalidad del ataque.
- Parte importante del éxito viene de una inicialización diseñada con el target.
- La evaluación de modelos cerrados se concentra en un único modelo de completion.
- Faltan múltiples semillas, intervalos de confianza y más ablaciones.
- La comparación con AutoDAN no es completamente simétrica.

### Nuestra conclusión equilibrada

> El paper demuestra de forma convincente que el feedback del objetivo puede convertir un ataque débil por transferencia en uno dirigido y efectivo. No demuestra que cualquier API moderna sea vulnerable con el mismo costo, porque depende de señales de pérdida, una interfaz particular y prompts poco naturales. Su aporte más durable es el cambio de perspectiva sobre el modelo de amenaza, más que la receta exacta de 2024.

---

## 14. Defensas: qué funcionaría y con qué límites

### Contra esta versión del ataque

- retirar o limitar logprobs y `logit bias`;
- limitar tasa y detectar secuencias de queries casi idénticas;
- filtrar inputs de perplejidad anormal;
- normalizar, retokenizar o parafrasear entradas;
- validar estructuralmente toda llamada a herramienta;
- usar múltiples capas de moderación y monitoreo.

### Por qué ninguna es una solución absoluta

- Un ataque adaptativo puede optimizar también contra el filtro de perplejidad.
- Quitar logprobs encarece el ataque, pero no prueba que lo vuelva imposible.
- Parafrasear puede degradar pedidos legítimos.
- Un segundo clasificador también puede tener ejemplos adversariales.
- El *rate limiting* cambia el costo, no la vulnerabilidad subyacente.

La conclusión de producción es **defensa en profundidad**, no confiar en una sola barrera.

---

## 15. Experimentos adicionales que propondríamos

1. **Separar longitud e inicialización:** usar prompts largos con inicialización aleatoria y prompts cortos con otra inicialización informada.
2. **Black-box estricto:** atacar APIs que solo devuelven texto o un flag binario.
3. **Chat y reasoning:** repetir en modelos con system prompt, chat template y razonamiento no visible.
4. **Ataque adaptativo a defensas:** incorporar perplejidad o legibilidad a la función objetivo.
5. **Reproducibilidad:** optimizar la loss esperada sobre varias consultas y reportar éxito sostenido.
6. **Más modelos y datasets:** probar distintos proveedores, idiomas y clases de targets.
7. **Impacto real:** medir si un string exacto logra atravesar un parser o activar una herramienta en un agente con permisos limitados.
8. **Ablaciones completas:** variar `B`, `bp`, `bq`, longitud del prompt, proxy y estrategia de selección con varias semillas.

---

## 16. Preguntas probables y respuestas sugeridas

### 1. ¿Cuál es la contribución principal?

> GCQ usa feedback del modelo objetivo durante la búsqueda, en vez de confiar en que un ataque optimizado localmente se transfiera. Eso permite ataques dirigidos contra modelos remotos y reduce la necesidad de un sustituto fiel. La evidencia principal es el contraste con transferencia pura en la Tabla 1 y el 79,6 % de éxito en GPT-3.5 con hasta USD 0,10 por objetivo.

### 2. ¿Cuál es la diferencia entre jailbreak y harmful string?

> En un jailbreak basta con alguna respuesta indebida o con que desaparezca el rechazo. En harmful strings la salida debe coincidir exactamente con un target. Es mucho más difícil porque hay que controlar cada token. AutoDAN casi no logra éxitos en esta tarea aunque sea eficaz como jailbreak.

### 3. ¿Por qué GCQ no es simplemente transferencia?

> Porque consulta al objetivo durante cada iteración y usa su pérdida para decidir qué candidatos conservar. En transferencia, el objetivo no guía la construcción; solo recibe el prompt final.

### 4. ¿Por qué sirve un proxy no alineado?

> El proxy no tiene que compartir exactamente la vulnerabilidad. Solo debe rankear candidatos mejor que el azar. La decisión final la toma la loss del objetivo. Mistral 7B base ilustra esa diferencia.

### 5. ¿Qué función cumple el gradiente en GCG?

> Aproxima qué sustituciones de tokens podrían reducir la loss. Como el espacio textual es discreto, la aproximación no es exacta y GCG igual evalúa candidatos. GCQ reemplaza o elimina esa señal y compensa con queries bien asignadas.

### 6. ¿Por qué es válido el short-circuit?

> La NLL acumulada es una suma de términos no negativos. Si la suma parcial ya es peor que el umbral del buffer, completar el resto no puede volverla mejor.

### 7. ¿La inicialización no hace todo el trabajo?

> Hace una parte importante: resuelve 28 % de entrada. Pero GCQ lleva el éxito a 79,6-86 % con bajo costo. Aun así, la dependencia es una limitación real y los autores muestran que con inicialización aleatoria el rendimiento cae mucho.

### 8. ¿Qué demuestra el experimento de longitud?

> Que un prompt más largo recupera el éxito para targets largos. No permite distinguir si ocurre por una mejor repetición inicial o por más grados de libertad; ambas variables cambian juntas.

### 9. ¿Qué tan justa es la comparación con AutoDAN?

> Es útil para mostrar que jailbreak y string exacto son tareas distintas, pero no es una comparación completamente justa: AutoDAN no fue diseñado para este criterio y se ejecutó con parámetros por defecto.

### 10. ¿Qué pasa si la API no devuelve logprobs?

> Esta versión pierde su principal señal de optimización. Podrían usarse muestreo repetido, una señal binaria o ataques basados en decisiones, pero serían más caros. El propio paper lo deja como trabajo futuro.

### 11. ¿Por qué un filtro de perplejidad puede defender?

> Los sufijos encontrados parecen secuencias de tokens aleatorios y suelen tener baja probabilidad bajo lenguaje normal. El filtro puede detectarlos, aunque un atacante adaptativo podría agregar legibilidad o la pérdida del filtro a su objetivo.

### 12. ¿El paper demuestra que RLHF no sirve?

> No. Demuestra que el comportamiento alineado no es robusto ante ciertas entradas adversariales. RLHF puede mejorar mucho la conducta promedio, pero no ofrece una garantía para todo el espacio de prompts.

### 13. ¿Cómo se relaciona con Goodhart?

> La seguridad se representa mediante señales imperfectas, como un reward o score de moderación. Cuando un atacante optimiza directamente esas señales, puede explotar la brecha entre la métrica y el objetivo real de seguridad.

### 14. ¿Por qué es grave evadir un moderador si el modelo principal sigue alineado?

> Porque los sistemas reales dependen de varias capas. Si una capa de monitoreo es evadible, disminuye la defensa en profundidad. Además, el contenido podría venir de otro modelo, un usuario o una inyección indirecta.

### 15. ¿La emisión exacta de un string siempre es dañina?

> No. El impacto depende del consumidor de esa salida. En un chat puede ser principalmente reputacional; en un agente, un string puede interpretarse como comando o llamada a herramienta. El paper motiva el riesgo con ese segundo escenario, pero no lo evalúa end-to-end.

### 16. ¿Cuál es la amenaza más importante a la validez?

> La generalización: solo hay un modelo cerrado generativo, una API de una época concreta y un dataset. Los resultados prueban viabilidad, pero no permiten afirmar el mismo costo o éxito para todas las APIs modernas.

### 17. ¿Qué experimento agregarían primero?

> Separaríamos los dos mecanismos confundidos en el estudio de longitud: variar independientemente longitud e inicialización y repetir con varias semillas. Eso permitiría saber si GCQ necesita capacidad de “codificación” o principalmente una buena semilla.

### 18. ¿Qué afirmación del paper defenderían con más seguridad?

> Que las defensas basadas únicamente en impedir transferencia son insuficientes. GCQ consulta directamente al objetivo, por lo que no necesita que el ataque final generalice desde otro modelo.

---

## 17. Números para memorizar

| Dato | Valor |
|---|---:|
| Dataset | 574 strings |
| Resueltos solo por inicialización | 161/574 ≈ 28 % |
| GPT-3.5 con ≤ USD 0,10 | 79,6 % |
| GPT-3.5 con ≤ USD 0,20 | 86,0 % |
| Targets de hasta 20 tokens | 97,9 % |
| 39 targets largos con prompt de 40 tokens | 100 %, costo medio USD 0,41 |
| Ahorro del short-circuit | ≈ 30 % |
| Moderación sin ataque | 197/574 ≈ 34 % no marcados |
| Universal OpenAI, 20 tokens | 99,2 % en validación |
| Universal OpenAI, 5 tokens | 94,8 % en validación |
| Llama Guard, 320 queries, 5 / 20 tokens | 59 % / 87 % |
| Reproducción en segunda evaluación | 519 ≈ 90 % |


