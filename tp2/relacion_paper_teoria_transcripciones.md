# Relación del paper con la teoría explicada en clase

## *Query-Based Adversarial Prompt Generation*

Este apunte amplía únicamente la relación entre el paper y la materia. Está construido principalmente a partir de las transcripciones de las clases, no de las diapositivas.

Las transcripciones fueron generadas automáticamente y deforman algunos términos (`fine-tuning`, `RLHF`, `chain of thought`, etc.). En este archivo los términos aparecen normalizados. Las referencias indican el archivo y el timestamp aproximado de la explicación.

---

## 1. La conexión general en una sola cadena

Todo el paper puede conectarse con el recorrido de la materia de esta manera:

```text
Un transformer decoder produce una distribución sobre el próximo token
                              ↓
el pretraining aprende la distribución general del lenguaje
                              ↓
SFT / RLHF / DPO ajustan esa distribución para obtener un asistente alineado
                              ↓
ese ajuste funciona bien en muchos inputs, pero no garantiza robustez universal
                              ↓
GCQ busca tokens de entrada que aumenten la probabilidad de un target dañino exacto
                              ↓
el ataque encuentra una zona del espacio de prompts donde falla la conducta alineada
                              ↓
si además evade al moderador, falla también la capa de monitoreo
                              ↓
si la salida controla una herramienta, el problema pasa a ser de seguridad sistémica
```

Esta cadena integra casi todas las clases relevantes: Transformers, tokenización, transfer learning, fine-tuning, producción y AI Safety.

---

# Clase 1: Transformers

Fuentes principales: [`transformers.VTT`](docs/transformers.VTT), especialmente 01:01:34-01:08:01.

## 2. Autoregresión: por qué el ataque puede controlar un string token por token

En la clase se explicó que la salida del decoder es **autoregresiva**: el token generado vuelve a entrar como parte del contexto para producir el siguiente. Eugenia lo resume diciendo que “la salida se va a usar como entrada” y que esto se repite para cada token (`transformers.VTT`, 01:01:34-01:03:12).

Eso es exactamente lo que usa la loss del paper. Si el target es:

```text
y = (y₁, y₂, ..., yₜ)
```

su probabilidad se descompone como:

```text
P(y | x) = P(y₁ | x)
         · P(y₂ | x, y₁)
         · ...
         · P(yₜ | x, y₁, ..., yₜ₋₁)
```

Por eso GCQ no necesita tratar la salida como un bloque indivisible. Puede medir cuánto favorece un prompt `x` a cada token del target y sumar esas contribuciones mediante la log-probabilidad acumulada.

### Conexión concreta con el paper

- El modelo genera el target de izquierda a derecha.
- La loss es la suma de las pérdidas de los tokens del target.
- El ataque modifica tokens del prompt para que, en cada paso autoregresivo, el siguiente token deseado tenga mayor probabilidad.
- La longitud del target importa porque controlar más pasos autoregresivos es más difícil y más costoso.

### Una forma clara de explicarlo en el oral

> La clase mostró que el decoder genera un token, lo reincorpora al contexto y recién entonces genera el siguiente. El paper aprovecha esa factorización: calcula qué tan probable es cada token del string objetivo dado el prompt y el prefijo correcto. Por eso puede definir una loss acumulada para un target exacto.

## 3. Logits y softmax: qué señal necesita GCQ

En la transcripción se explica que la salida numérica del decoder pasa por una capa lineal que produce un vector de **logits**, un puntaje por token del vocabulario. Luego esos puntajes se transforman en probabilidades (`transformers.VTT`, 01:03:21-01:04:59).

El paper se conecta de forma directa con esa explicación:

1. El modelo produce un logit para cada posible próximo token.
2. Softmax transforma los logits en probabilidades.
3. GCQ necesita conocer o estimar la probabilidad de los tokens del target.
4. La API no siempre muestra esas probabilidades directamente.
5. Los autores usan `logit bias` para alterar temporalmente el logit de un token, hacerlo aparecer entre los resultados visibles y luego corregir matemáticamente el efecto del bias.

El `logit bias` no cambia los pesos ni reentrena el modelo. Modifica el puntaje antes del softmax durante esa consulta.

### Por qué esta conexión es importante

El ataque no funciona solamente observando el texto final. En su versión contra GPT-3.5 necesita una señal más rica: la probabilidad que el modelo asigna a tokens que todavía no eligió. Esa señal sale precisamente de la cadena explicada en clase:

```text
estado interno → capa lineal → logits → softmax → probabilidades
```

### Qué no conviene decir

No conviene decir que GCQ “ataca el mecanismo de atención”. El ataque usa el comportamiento completo del transformer y su distribución de salida, pero no modifica directamente las matrices de atención ni propone una vulnerabilidad específica de self-attention.

## 4. La loss del ataque y la loss de entrenamiento

En la clase, al explicar el entrenamiento, se dijo que se compara la distribución obtenida con la deseada y se usa backpropagation para aumentar la probabilidad del token correcto (`transformers.VTT`, 01:06:33-01:08:01).

El paper utiliza una función objetivo casi idéntica: la probabilidad negativa del target. La diferencia fundamental está en **qué variable se modifica**.

| Situación | Se mantiene fijo | Se optimiza |
|---|---|---|
| Entrenamiento del LM | Datos de entrenamiento | Pesos del modelo |
| GCG white-box | Pesos del modelo | Tokens del prompt usando gradientes |
| GCQ | Pesos y gradientes internos del objetivo | Tokens del prompt usando queries |

Esta es una conexión muy buena para el oral: GCQ convierte la generación de un prompt en un problema de optimización. No está entrenando el modelo; está “entrenando” o buscando la entrada.

## 5. Greedy decoding y criterio de éxito

La clase explica la selección del token a partir de la distribución: en el ejemplo, el token con 0,9 es el elegido (`transformers.VTT`, 01:04:04-01:04:44). El paper define éxito si el target exacto aparece bajo **greedy decoding**, es decir, si en cada paso el token objetivo es el de mayor probabilidad.

Una loss baja ayuda, pero el criterio final es más concreto: cada token deseado debe ganar su paso de generación.

## 6. Por qué el short-circuit es posible

Como la loss total es una suma token por token y cada término `-log p` es no negativo, la pérdida parcial solo puede aumentar. Si durante la evaluación ya es peor que el peor prompt conservado en el buffer, no hace falta calcular los tokens restantes.

Esta optimización del paper se entiende gracias a la naturaleza autoregresiva vista en clase: la loss se puede computar incrementalmente siguiendo el mismo orden de generación.

---

# Clase 2: Tokenización y embeddings

Fuentes principales: [`embeddings_2.VTT`](docs/embeddings_2.VTT), especialmente 00:00:22-00:04:11 y 00:10:15-00:16:10; también [`embeddings_1.VTT`](docs/embeddings_1.VTT).

## 7. Tokenizar no es hacer embeddings

Marina distingue las dos operaciones de manera explícita:

- el tokenizer divide el texto en unidades y las asocia a IDs;
- el embedding representa cada unidad mediante un vector denso (`embeddings_2.VTT`, 00:00:22-00:01:03).

Esta diferencia es central en el paper porque GCQ debe terminar produciendo **tokens válidos**, no vectores arbitrarios del espacio de embeddings.

Trabajos anteriores podían optimizar en un espacio continuo de embeddings y después intentar proyectar el resultado a texto. El problema es que un punto continuo conveniente para la loss puede no corresponder a ningún token real. GCQ busca directamente sobre secuencias discretas del vocabulario.

### Conexión concreta

Si el vocabulario tiene `|V|` tokens y el sufijo tiene longitud `m`, el espacio de búsqueda es:

```text
V^m
```

No significa que GCQ enumere todo ese espacio. Significa que cada coordenada del prompt solo puede tomar valores discretos del vocabulario. Por eso usa sustituciones de un token y búsqueda greedy en vez de descenso por gradiente continuo convencional.

## 8. BPE y tokenización canónica

En la clase se explica que BPE construye el vocabulario a partir de unidades frecuentes (`embeddings_2.VTT`, 00:04:12 en adelante). Como resultado, una misma cadena visible puede dividirse de maneras distintas según el tokenizer.

El Apéndice C del paper muestra un problema muy específico: la optimización podría encontrar una lista de IDs que, al concatenarse como texto y enviarse nuevamente a la API, sea retokenizada de otra manera. Entonces el modelo no recibe realmente la secuencia optimizada.

Por eso los autores:

- convierten la secuencia a string;
- vuelven a tokenizarla con el tokenizer del objetivo;
- también la tokenizan por separado con el tokenizer del proxy.

Esta no es una precaución superficial. La unidad real que procesa el transformer es el token, no el carácter visible ni necesariamente la palabra.

## 9. Vocabularios diferentes entre proxy y objetivo

La clase insiste en que el vocabulario es parte del diseño del tokenizer. Dos modelos pueden segmentar el mismo string de manera diferente.

Eso explica una sutileza de GCQ: Mistral puede servir como proxy de GPT-3.5 aunque no comparta su tokenizer. El proxy puntúa la versión del string que él mismo tokeniza; la API objetivo recibe su propia tokenización. El vínculo entre ambos no está en compartir IDs exactos, sino en que el proxy pueda ordenar strings candidatos de una manera útil.

### Consecuencia conceptual

El proxy no necesita reproducir internamente al objetivo. Solo necesita aportar una señal correlacionada con qué strings naturales o secuencias pueden favorecer el target. Después, las queries corrigen cualquier desacuerdo.

## 10. Glitch tokens y zonas poco entrenadas

La transcripción explica que algunos tokens pueden estar presentes en el vocabulario pero aparecer muy poco en el corpus de entrenamiento del modelo. Eso deja regiones poco exploradas del espacio de embeddings y puede provocar comportamientos extraños (`embeddings_2.VTT`, 00:13:52-00:15:52).

La relación con el paper es conceptual:

- los sufijos adversariales contienen combinaciones de tokens poco naturales;
- esas combinaciones pueden llevar al modelo a regiones muy alejadas de los prompts normales;
- en esas regiones, la conducta aprendida durante el entrenamiento de seguridad puede generalizar mal;
- por la misma rareza, un filtro de perplejidad puede detectar muchos de estos sufijos.

### Matiz importante

No hay que afirmar que GCQ funciona específicamente “gracias a glitch tokens”. El paper no lo demuestra. Los glitch tokens son un antecedente de la misma idea general: la tokenización crea entradas discretas raras donde el comportamiento puede ser impredecible.

## 11. Tokens especiales, herramientas y riesgo de strings exactos

En la clase se explica que los tokens especiales no solo separan sistema, usuario y asistente: también pueden delimitar acciones de herramientas, llamadas a APIs, ejecución de scripts o lectura de documentos (`embeddings_2.VTT`, 00:01:09-00:04:11).

Esto vuelve mucho más clara la motivación del paper. En un chatbot común, forzar un string exacto puede causar contenido ofensivo. En un sistema agéntico, un string exacto puede tener sintaxis operacional:

```text
texto generado → parser → llamada a herramienta → acción externa
```

Por eso el ataque dirigido es más grave que un jailbreak genérico: puede intentar controlar una secuencia que otro componente interpreta como comando.

---

# Clase 3: Transfer learning, fine-tuning y alineamiento

Fuente principal: [`finetuning.VTT`](docs/finetuning.VTT), especialmente 00:31:20-00:35:06, 00:45:06-00:52:19, 01:05:00-01:09:49, 01:28:37-01:40:59 y 01:56:04-01:59:31.

## 12. Transfer learning y transferibilidad adversarial: relacionadas, pero no idénticas

La clase define transfer learning como reutilizar conocimiento aprendido por un modelo en una tarea o dataset para mejorar otro modelo o tarea (`finetuning.VTT`, 00:31:48-00:33:16).

El paper habla principalmente de **transferibilidad de ejemplos adversariales**: un prompt optimizado para engañar a un modelo también engaña a otro.

No son exactamente el mismo fenómeno:

- **Transfer learning:** el desarrollador transfiere conocimiento para construir un modelo mejor.
- **Transferibilidad adversarial:** el atacante aprovecha que modelos diferentes comparten comportamientos o vulnerabilidades.

La conexión está en que ambos dependen de estructura aprendida que no es totalmente específica de un solo modelo.

## 13. Por qué la similitud entre distribuciones importa

En la clase se explica que transferir funciona mejor cuando la tarea y la distribución nueva son parecidas a las del modelo original. Si cambian demasiado, el conocimiento previo puede dejar de ser útil (`finetuning.VTT`, 00:51:04-00:52:19).

El paper encuentra algo análogo en sus ataques:

- Vicuna 13B transfiere relativamente bien hacia Vicuna 33B;
- 7B transfiere peor hacia escalas mayores;
- Vicuna es un sustituto muy malo para Claude en el antecedente citado.

Los autores interpretan que Vicuna 13B y 33B son más parecidos entre sí que respecto de 7B.

### Qué se puede concluir

La similitud entre modelos parece favorecer la transferencia del ataque.

### Qué no se puede concluir

El paper no identifica qué componente produce esa similitud: podría ser la escala, los datos, el fine-tuning, la arquitectura o una combinación.

## 14. Surrogate y proxy: la diferencia conceptual más importante

Un **surrogate** de transferencia debe ser suficientemente parecido al objetivo para que el prompt final funcione sin correcciones.

El **proxy** de GCQ tiene una tarea más limitada: preseleccionar candidatos. Después, la loss real del objetivo determina cuáles se conservan.

Esta diferencia explica por qué Mistral 7B base puede servir como proxy para GPT-3.5 aunque sea inadecuado para transferencia pura.

Una analogía con la clase:

- en transfer learning importa que el conocimiento reutilizado sea pertinente para la nueva distribución;
- en GCQ alcanza con que el proxy aporte una prior útil, porque el feedback del objetivo corrige el resto.

## 15. Knowledge distillation y acceso a probabilidades

La clase explica que, en knowledge distillation, entrenar con la distribución completa de probabilidades del modelo grande aporta más información que usar solamente el texto o una hard label (`finetuning.VTT`, 01:05:00-01:07:27).

El paper no realiza distillation, pero aparece el mismo principio informacional:

- observar solo el texto final da una señal pobre;
- acceder a logprobs permite saber si un candidato mejoró incluso cuando todavía no genera el target;
- esa señal continua vuelve posible una búsqueda eficiente.

También aparece la misma restricción señalada en la clase: trabajar con distribuciones de tokens depende del acceso a probabilidades y de cómo se define el vocabulario. Cuando OpenAI limita los logprobs, el ataque se vuelve mucho más caro.

### Distinción que conviene decir

> GCQ no destila el modelo remoto. La relación con distillation es que ambos se benefician de ver la distribución probabilística, que contiene mucha más información que una salida final discreta.

## 16. Del foundation model al asistente alineado

En la clase se distingue el foundation model, entrenado con next-token prediction sobre un corpus enorme, del asistente obtenido después mediante etapas de fine-tuning. El modelo base aprende la distribución del corpus, pero todavía no está orientado a comportarse como asistente (`finetuning.VTT`, alrededor de 01:14:00-01:16:00).

Esto ayuda a entender dos resultados del paper:

1. **Mistral base sirve como proxy.** Aunque no tenga alineamiento de chat, sí modela la probabilidad del lenguaje y puede ayudar a ordenar candidatos.
2. **La capacidad de producir strings dañinos no desaparece necesariamente con el alineamiento.** El post-training cambia qué respuestas son favorecidas en condiciones normales, pero el modelo conserva gran parte de lo aprendido durante pretraining.

## 17. RLHF como ajuste de probabilidades

La clase describe RLHF así:

- el estado es el prompt más los tokens generados;
- la acción es generar el próximo token;
- un reward model asigna una señal a la respuesta;
- el entrenamiento ajusta la política para aumentar respuestas bien recompensadas (`finetuning.VTT`, 01:29:11-01:34:34).

GCQ actúa sobre el mismo objeto —la distribución de próximos tokens— pero desde afuera:

- RLHF modifica los pesos para que respuestas dañinas tengan menor probabilidad en muchos prompts;
- GCQ busca un prompt excepcional que vuelva a aumentar la probabilidad de un target dañino sin tocar los pesos.

Por eso el paper debe interpretarse como una prueba de **robustez del comportamiento alineado**, no como un método que deshace o revierte el entrenamiento.

## 18. Penalización KL y persistencia del modelo base

La clase explica que la penalización KL evita que el modelo fine-tuneado se aleje demasiado de la distribución del modelo de referencia. Se busca conservar capacidades útiles y hacer que los cambios de post-training sean menores que los del pretraining (`finetuning.VTT`, 01:38:26-01:40:59).

Esto ofrece una intuición posible para el paper:

- el modelo alineado sigue anclado en capacidades del modelo base;
- el ajuste de seguridad modifica preferencias, pero no elimina todo comportamiento posible del pretraining;
- un prompt adversarial puede encontrar una región donde reaparece una continuación no deseada.

### Cuidado

Es una **interpretación apoyada en la teoría de clase**, no un mecanismo demostrado por los experimentos de GCQ. El paper no mide la KL ni compara distintos coeficientes de penalización.

## 19. Violation Rate, False Refusal Rate y ASR

En la clase se discuten dos métricas de alineamiento:

- **Violation Rate:** el modelo responde cuando debería rechazar.
- **False Refusal Rate:** el modelo rechaza cuando podría responder de manera segura.

La transcripción además muestra que estas métricas no generalizan igual entre idiomas (`finetuning.VTT`, 01:56:15-01:58:24).

El ASR del paper puede pensarse como una medición adversarial mucho más estricta que Violation Rate:

- una violación común puede ser cualquier respuesta insegura;
- GCQ exige una respuesta exacta elegida por el atacante.

Esto también sirve para discutir defensas. Un filtro extremadamente agresivo podría bajar el ASR, pero aumentar False Refusal Rate y degradar la utilidad para usuarios legítimos.

## 20. “¿Es RLHF robusto?”

Al cierre de la clase se dice que RLHF generaliza sorprendentemente bien, pero sigue siendo un ajuste relativamente pequeño y parecido a un “parche” sobre un modelo sin esas restricciones originales (`finetuning.VTT`, 01:58:25-01:59:31).

El paper es una respuesta experimental directa a esa pregunta:

- el alineamiento funciona frente a prompts normales;
- GCQ encuentra prompts construidos específicamente para hacerlo fallar;
- por lo tanto, alineamiento conductual y robustez adversarial no son equivalentes.

No corresponde concluir “RLHF no sirve”. La conclusión correcta es: **RLHF no constituye por sí solo una garantía de seguridad ante un adversario adaptativo**.

---

# Clase 4: Reasoning models y producción

Fuente principal: [`reasoningmodels.VTT`](docs/reasoningmodels.VTT), especialmente 00:00:33-00:05:10, 00:10:09-00:19:36, 00:31:48-00:35:30 y 00:45:46-00:47:34.

## 21. El paper no estudia reasoning models

La clase define los reasoning models como modelos que generan tokens adicionales usados como pasos intermedios de planificación o verificación. También explica que en modelos propietarios la cadena de pensamiento suele permanecer oculta (`reasoningmodels.VTT`, 00:00:33-00:02:54).

El paper ataca `gpt-3.5-turbo-instruct`, un modelo de completion, no un reasoning model moderno. Esta diferencia importa porque:

- la API de un reasoning model puede ocultar los tokens internos;
- puede no exponer sus logprobs;
- el target visible aparece después de una trayectoria interna adicional;
- la propia deliberación podría detectar el prompt adversarial o, por el contrario, abrir nuevas superficies de ataque.

Por eso “probar GCQ en reasoning models” es una extensión válida, no un resultado del paper.

## 22. Inference-time scaling y costo del ataque

La clase explica que generar más tokens o varias respuestas puede mejorar resultados, pero aumenta mucho el costo de inferencia (`reasoningmodels.VTT`, 00:10:09-00:15:45).

El paper también está atravesado por un trade-off de inferencia:

- más queries permiten explorar mejor;
- targets más largos requieren más evaluaciones de tokens;
- prompts más largos aumentan el costo de cada query;
- la reconstrucción de logprobs hace que el costo crezca superlinealmente con la longitud del target.

La diferencia es el objetivo: en la clase se gasta más inferencia para resolver mejor una tarea; en GCQ el atacante gasta inferencia para optimizar mejor una entrada adversarial.

## 23. Tool use: por qué una salida exacta puede convertirse en una acción

La clase presenta *agentic tool use* como dar al modelo acceso a herramientas para resolver partes de un problema de forma más robusta (`reasoningmodels.VTT`, 00:16:12-00:17:17).

Esa misma integración aumenta el impacto potencial del paper. Si una aplicación interpreta cierta salida como:

- llamada a una API;
- consulta a una base de datos;
- envío de un correo;
- ejecución de código;
- transferencia o pago;

entonces controlar exactamente la secuencia generada puede ser más peligroso que conseguir una respuesta genéricamente indebida.

La herramienta no causa la vulnerabilidad de GCQ, pero transforma una falla de generación en un riesgo operacional.

## 24. Pipeline de post-training y superficie de ataque

La clase describe un pipeline con varias etapas reutilizadas:

```text
modelo base → SFT para asistente → RLHF/DPO para seguridad → entrenamiento de razonamiento
```

(`reasoningmodels.VTT`, 00:33:12-00:35:30).

El paper ataca el comportamiento que emerge al final del pipeline, pero su proxy puede ser un modelo base. Esto sugiere que las distintas etapas no reemplazan totalmente lo aprendido antes; agregan comportamiento y preferencias sobre una base compartida.

## 25. Evaluación, grados de libertad y evidencia

En la clase se remarca que un experimento de ML exige elegir preprocesamiento, modelo, parámetros, métricas y cantidad de corridas, y que los papers a veces afirman capacidades con evidencia menos robusta de lo que parece (`reasoningmodels.VTT`, 00:45:46-00:47:34).

Esta advertencia sirve para leer críticamente GCQ:

- la ablación de inicialización usa solo los primeros 20 targets;
- muchas curvas no reportan variación entre semillas;
- AutoDAN se usa con parámetros por defecto en una tarea distinta de su objetivo original;
- se evalúa un único modelo cerrado de generación;
- los costos dependen de una API y precios concretos;
- el no determinismo hace que alrededor de 10 % de los éxitos no se reproduzca en una segunda prueba.

La actitud pedida en clase no es descartar el resultado, sino calibrar la afirmación:

> El paper demuestra viabilidad y aporta evidencia fuerte en los escenarios evaluados; no demuestra que todas las APIs modernas puedan atacarse con el mismo costo y éxito.

---

# Clase 6: AI Safety

Fuente principal: [`ai_safety.VTT`](docs/ai_safety.VTT), especialmente 00:07:14-00:19:44 y 00:35:59-00:44:57.

## 26. Jailbreak: tensión de objetivos y generalización inadecuada

La clase define jailbreak como eludir restricciones impuestas al modelo para obtener comportamiento indeseado. Propone dos causas generales: **tensión de objetivos** y **generalización inadecuada** (`ai_safety.VTT`, 00:07:14-00:08:53).

GCQ encaja principalmente como un caso de generalización inadecuada:

- el entrenamiento de seguridad cubre una parte limitada del espacio de inputs;
- GCQ busca secuencias extrañas optimizadas específicamente;
- esas entradas pueden quedar lejos de la distribución de prompts normales;
- allí la política de rechazo no generaliza de forma robusta.

Puede existir también tensión entre ser útil y ser seguro, pero el mecanismo distintivo de GCQ es la búsqueda adversarial de un input donde falla la generalización.

## 27. GCQ es más estricto que un jailbreak común

Un jailbreak común tiene éxito si el modelo deja de rechazar o produce alguna respuesta dañina. GCQ tiene éxito cuando produce un string exacto.

La diferencia se puede expresar así:

```text
jailbreak no dirigido: existe alguna salida insegura aceptable para el atacante
ataque dirigido: solo sirve la salida exacta elegida previamente
```

Esto explica por qué AutoDAN puede ser fuerte como jailbreak y casi no obtener éxitos en la evaluación de harmful strings.

## 28. El antecedente visto literalmente en clase

La clase de AI Safety menciona ataques transferibles entre modelos y explica que un prompt adversarial puede funcionar en modelos entrenados de manera distinta (`ai_safety.VTT`, 00:18:06-00:19:29). Ese es el antecedente directo de Zou et al. que usa el paper.

La progresión es:

```text
GCG white-box
    ↓
prompts universales y transferibles
    ↓
limitación: mala transferencia dirigida y necesidad de un buen surrogate
    ↓
GCQ agrega feedback del modelo objetivo
```

Por eso el nuevo paper no contradice al trabajo visto en clase. Lo extiende para cubrir el escenario donde la transferencia sola no alcanza.

## 29. Prompt injection no es lo mismo que GCQ

La clase define prompt injection como introducir instrucciones que anulan o escapan las instrucciones preexistentes, incluso mediante contenido oculto en páginas o imágenes (`ai_safety.VTT`, 00:12:04-00:14:06).

GCQ es distinto:

- no necesita que el sufijo sea una instrucción comprensible;
- optimiza tokens usando una función de pérdida;
- puede producir secuencias aparentemente aleatorias;
- su objetivo es aumentar la probabilidad de una salida exacta.

Pueden combinarse: una inyección indirecta podría transportar un sufijo adversarial. Pero “prompt injection”, “jailbreak” y “ataque query-based” describen dimensiones diferentes.

## 30. Goodhart: optimizar la métrica y no el objetivo real

En la clase se explica la ley de Goodhart: cuando se optimiza intensamente una métrica concreta, las acciones pueden mejorar esa métrica sin cumplir la meta real. También se conecta con reward hacking (`ai_safety.VTT`, 00:35:59-00:40:11).

Hay dos conexiones con el paper.

### 30.1 La loss como objetivo del atacante

El objetivo real del proveedor es algo complejo: que el modelo no genere daño. GCQ no necesita modelar ese objetivo humano. Optimiza una cantidad concreta y accesible: la NLL de un target.

Al bajar esa métrica, encuentra un prompt que fuerza una conducta contraria a la intención del sistema.

### 30.2 Los scores del moderador como proxy de seguridad

Para atacar al moderador, los autores minimizan la suma de scores de categorías. Esos scores son un proxy cuantitativo de “contenido seguro”. El ataque encuentra strings dañinos cuyo score es bajo.

Eso es una ilustración muy directa de la brecha de Goodhart:

```text
objetivo real: detectar contenido dañino
proxy medido: scores de categorías
optimización adversarial: reducir los scores sin quitar el daño
```

### Matiz

No es reward hacking realizado autónomamente por el LLM. Es un atacante externo explotando una métrica. La estructura del problema es análoga, pero el agente optimizador es distinto.

## 31. Las cuatro categorías de AI Safety aplicadas al paper

La clase organiza problemas abiertos en **robustez, monitoreo, alineamiento y seguridad sistémica** (`ai_safety.VTT`, 00:41:08-00:44:57). El paper toca las cuatro.

### Robustez

La clase define robustez como resistir inputs adversariales y situaciones de baja probabilidad. GCQ es literalmente un método para construir inputs adversariales. Su éxito demuestra que el comportamiento seguro no es robusto en todo el espacio de prompts.

### Monitoreo

La clase describe monitoreo como detectar cuándo ocurre algo inesperado. OpenAI Moderation y Llama Guard cumplen ese rol: observan contenido y deberían señalarlo. El paper demuestra que también pueden recibir ejemplos adversariales.

La lección no es “monitorear no sirve”, sino que el monitor es otro modelo y, por lo tanto, otra superficie que debe evaluarse adversarialmente.

### Alineamiento

El modelo está ajustado para rechazar pedidos dañinos. GCQ muestra una diferencia entre:

- estar alineado en la distribución habitual;
- ser robustamente alineado frente a un adversario que busca el peor input.

El paper cuestiona la segunda propiedad, no niega completamente la primera.

### Seguridad sistémica

La clase ubica aquí la seguridad de la organización y del sistema completo en producción. El impacto final de GCQ depende de:

- qué permisos tiene el modelo;
- si la salida se interpreta como código o llamada a herramienta;
- si existen validaciones determinísticas;
- si hay rate limits y detección de queries anómalas;
- si una acción sensible requiere confirmación humana.

El paper estudia el modelo y los moderadores, no todo ese sistema. Por eso una limitación de validez externa es que no evalúa un ataque end-to-end contra un agente real.

## 32. La distribución de entrenamiento y los inputs de baja probabilidad

La clase explica que los modelos aprenden muy bien lo frecuente en sus grandes corpus, mientras que las situaciones de baja probabilidad tienen comportamiento más difícil de predecir (`ai_safety.VTT`, 00:41:25-00:42:17).

Los sufijos de GCQ son un caso extremo: no buscan parecerse a prompts humanos típicos; buscan minimizar una loss. Por eso pueden caer en regiones raras donde el entrenamiento de seguridad tuvo poca cobertura.

Esta conexión también explica la defensa por perplejidad:

- si el prompt tiene probabilidad anormalmente baja bajo lenguaje normal, puede marcarse;
- pero un atacante adaptativo podría agregar la perplejidad a su función objetivo y buscar sufijos más naturales.

---

# Conexiones transversales

## 33. Cadena técnica: de logits a ataque dirigido

```text
El decoder produce logits
        ↓
softmax produce probabilidades por token
        ↓
la probabilidad del target se factoriza autoregresivamente
        ↓
la NLL resume qué tan lejos está el modelo de emitir ese target
        ↓
GCQ usa esa NLL como señal para elegir sustituciones de tokens
```

Esta cadena conecta directamente Clase 1 con el método del paper.

## 34. Cadena de entrenamiento: de foundation model a fallo adversarial

```text
Pretraining aprende lenguaje y capacidades generales
        ↓
SFT convierte el modelo en asistente
        ↓
RLHF/DPO ajustan preferencias y seguridad
        ↓
KL limita cuánto se aleja del modelo de referencia
        ↓
el comportamiento seguro generaliza bien, pero no a todos los inputs
        ↓
GCQ busca sistemáticamente una excepción
```

Esta cadena conecta Clases 3 y 6.

## 35. Cadena de producción: defensa en profundidad

```text
input del usuario
        ↓
filtros y detección de anomalías
        ↓
modelo generativo alineado
        ↓
moderador de output
        ↓
parser / herramientas / acciones
```

El paper muestra fallos en el modelo generativo y en el moderador. La teoría de producción y seguridad sistémica indica que las acciones críticas no deberían depender únicamente de esas dos capas probabilísticas.

## 36. Cadena de información: por qué los logprobs son tan valiosos

```text
solo texto final          → señal escasa: éxito o fracaso
scores / logprobs         → señal continua de progreso
pesos y gradientes        → acceso white-box completo
```

GCQ ocupa la zona intermedia. No necesita pesos, pero su ataque es mucho más eficiente porque la API revela más información que una simple respuesta textual.

---

# Preguntas de teoría que podrían hacer

## 37. “¿Cómo se relaciona la loss de GCQ con el entrenamiento de un transformer?”

> En ambos casos se trabaja con la probabilidad del token correcto. En entrenamiento se ajustan los pesos mediante backpropagation para aumentar esa probabilidad. En GCQ los pesos del objetivo quedan fijos y se optimiza la secuencia de entrada mediante queries.

## 38. “¿Por qué el texto discreto complica el ataque?”

> Porque no se puede mover un token una cantidad infinitesimal. Cada cambio salta a otro elemento del vocabulario. Los gradientes solo pueden sugerir sustituciones; después hay que evaluarlas como tokens reales.

## 39. “¿Qué tiene que ver la tokenización con la seguridad?”

> Define las unidades que realmente procesa el modelo y crea regiones poco entrenadas. Además, proxy y objetivo pueden segmentar distinto el mismo texto. El paper debe retokenizar cada candidato y sus sufijos raros pueden explotar generalización deficiente.

## 40. “¿Transfer learning y ataque por transferencia son lo mismo?”

> No. Transfer learning reutiliza conocimiento para entrenar o adaptar modelos. Un ataque transferible es un input que conserva su efecto al pasar de un modelo a otro. Se relacionan porque ambos dependen de estructura compartida entre modelos.

## 41. “¿Por qué Mistral base sirve como proxy de un modelo alineado?”

> Porque el proxy no decide el ataque final. Solo ayuda a ordenar candidatos según la probabilidad lingüística del target. Las queries al objetivo corrigen las diferencias de alineamiento y arquitectura.

## 42. “¿El paper demuestra que el alineamiento fue eliminado?”

> No. Los pesos no cambian. El modelo sigue rechazando muchos prompts normales. El ataque encuentra entradas específicas donde el comportamiento alineado no generaliza de manera robusta.

## 43. “¿Qué relación hay con la penalización KL?”

> La KL mantiene al modelo fine-tuneado cerca del modelo base para conservar capacidades. Eso da una intuición de por qué siguen presentes comportamientos del pretraining, pero el paper no mide KL ni demuestra que sea la causa del ataque.

## 44. “¿Dónde aparece Goodhart?”

> El moderador aproxima seguridad con scores. GCQ optimiza esos scores y encuentra contenido dañino que parece seguro para la métrica. Es la brecha entre el proxy medido y el objetivo real.

## 45. “¿Esto es un problema de robustez o de alineamiento?”

> De ambos, pero la contribución experimental está principalmente en robustez: parte de un modelo conductualmente alineado y construye un input adversarial que hace fallar esa conducta. También muestra que el alineamiento no es una garantía universal.

## 46. “¿Qué relación tiene el ataque al moderador con monitoreo?”

> El moderador es una capa de monitoreo. Al evadirlo, el paper muestra que observar la salida con otro modelo no garantiza detección. Hace falta evaluar también al monitor bajo ataques adaptativos.

## 47. “¿Por qué importa tool use?”

> Porque una salida exacta puede dejar de ser solo texto y convertirse en una llamada estructurada. El daño depende de los permisos y validaciones del sistema completo, lo que conecta el paper con seguridad sistémica.

## 48. “¿Funcionaría igual en un reasoning model?”

> No está evaluado. Los tokens internos ocultos, la ausencia de logprobs y una trayectoria de razonamiento adicional cambian el modelo de amenaza. Es una línea futura, no algo que pueda afirmarse a partir del paper.

---

# Errores conceptuales a evitar en el oral

- **“GCQ modifica los pesos.”** No: modifica el prompt.
- **“Ataca la atención.”** Usa la distribución final del modelo; no identifica una falla específica de self-attention.
- **“Transfer learning es lo mismo que transferir un jailbreak.”** Son fenómenos relacionados, no iguales.
- **“Los sufijos son glitch tokens.”** Pueden ser raros, pero el paper no demuestra que dependa de glitch tokens.
- **“RLHF no sirve.”** Sí mejora la conducta normal; lo que no ofrece es robustez adversarial garantizada.
- **“Jailbreak y prompt injection son sinónimos.”** El primero elude restricciones; el segundo introduce instrucciones que compiten con las originales.
- **“El moderador garantiza seguridad.”** Es otra capa probabilística y también puede recibir ejemplos adversariales.
- **“El paper prueba un ataque real contra agentes.”** Lo motiva, pero no lo evalúa end-to-end.
- **“La relación con KL está demostrada.”** Es una interpretación teórica, no un resultado experimental.
- **“Un ASR alto generaliza a cualquier API.”** Depende del acceso, el modelo, el tokenizer, los límites y la época de la API.

---

# Resumen final para memorizar

1. **Transformers:** GCQ optimiza la probabilidad autoregresiva de un target token por token.
2. **Logits y softmax:** los logprobs dan una señal continua para guiar la búsqueda sin acceder a los pesos.
3. **Tokenización:** el espacio de entrada es discreto, enorme y dependiente del tokenizer; por eso hay que retokenizar.
4. **Embeddings:** los prompts raros pueden llevar al modelo a regiones poco cubiertas, pero GCQ no es simplemente un ataque de glitch tokens.
5. **Transferibilidad:** un ataque puede pasar entre modelos parecidos, pero la transferencia dirigida es débil.
6. **Proxy:** no necesita copiar al objetivo; solo filtra candidatos antes de consultar la loss real.
7. **Fine-tuning:** el alineamiento cambia preferencias de salida, no borra todas las capacidades del modelo base.
8. **RLHF:** GCQ no lo revierte; encuentra un input donde su conducta no es robusta.
9. **Producción:** una salida exacta es especialmente riesgosa si controla herramientas.
10. **Goodhart:** minimizar scores del moderador puede producir contenido dañino que la métrica considera seguro.
11. **Robustez:** el paper construye inputs adversariales que rompen la conducta segura.
12. **Monitoreo:** los safety classifiers también son vulnerables.
13. **Seguridad sistémica:** permisos, validaciones y confirmación humana determinan el daño final.

La idea unificadora para responder en el oral es:

> El paper toma el mecanismo probabilístico del transformer visto al comienzo de la materia y lo usa para poner a prueba las capas de alineamiento y monitoreo vistas al final. Su aporte es mostrar que un modelo puede estar alineado para inputs normales y, aun así, no ser robusto frente a alguien que optimiza directamente contra su distribución de salida.
