# Guía de defensa oral — *Query-Based Adversarial Prompt Generation*

**Paper:** Hayase, Borevkovic, Carlini, Tramèr, Nasr — University of Washington, ETH Zurich, Google DeepMind. NeurIPS 2024 (arXiv 2402.12329v2).
**TP2 · 73.69 Large Language Models · 2026** — defensa en formato debate, ~30 min, todos responden sobre todo el paper.

> **Cómo leer las referencias**
> - `[§4.3 · p.6]` = sección y página del PDF del paper. También `[Fig. 2]`, `[Tabla 1]`, `[Alg. 1]`, `[Ap. B · p.12]`, `[nota 1 · p.1]`.
> - `[C3 · diap. 49]` = clase y número de diapositiva (C1 Transformers, C2 Embeddings, C3 Transfer Learning & Fine-tuning, C4 Reasoning/Producción, C6 AI Safety).
> - **💭 Análisis nuestro** = no lo dice el paper; es interpretación para el debate. En el oral conviene presentarlo como tal ("nuestra lectura es…").
>
> **Tip:** marcá en tu PDF cada frase que se referencia acá. En el oral, decir "en la sección 4.3, página 6, reportan…" y señalar la frase suma muchísimo en *interpretación de la evidencia*.

---

## 0. El paper en 60 segundos

Hasta este trabajo, para romper un LLM alineado había dos caminos: ataques **white-box** (GCG de Zou et al., necesita los pesos) o ataques **por transferencia** (optimizo el sufijo en un modelo abierto como Vicuna y lo reuso en GPT-4). La transferencia tiene dos problemas: depende de tener un buen modelo sustituto y **no sirve para ataques dirigidos**, es decir, para forzar al modelo a emitir *exactamente* un string dado [§2 · p.2-3].

La observación central es que cada iteración de GCG tiene dos etapas: (1) filtrar muchos candidatos con el gradiente y (2) elegir el mejor evaluando la loss, y **la etapa (2) solo necesita poder consultar la loss** [§1 · p.2]. Entonces reemplazan (1) por un modelo *proxy* local y hacen (2) consultando la API del modelo objetivo. Eso es **GCQ (Greedy Coordinate Query)**. Además proponen una variante que no necesita proxy ni gradientes [§3.3 · p.4].

Resultados: GPT-3.5 Turbo Instruct emite strings dañinos exactos con **79.6% de éxito gastando ≤ USD 0.10 por string** [§4.3 · p.6]; la moderación de OpenAI se evade con un sufijo universal en **99.2%** de los casos de validación [§4.5 · p.8]; también cae Llama Guard [§4.6 · p.9]. Mensaje de fondo: las defensas que solo apuestan a romper la transferibilidad no alcanzan [§5 · p.9].

---

## 1. Mapa del paper y números para memorizar

### 1.1 Qué hay en cada página

| Página | Contenido |
|---|---|
| p.1 | Abstract, introducción, motivación (agentes, plugins), resultados de transferencia de Zou et al., contribuciones, nota 1 (otros ataques black-box) |
| p.2 | Observación de las dos etapas de GCG, variante sin proxy, caso moderación; §2 Background (visión, black-box, transfer, NLP) |
| p.3 | Limitaciones de la transferencia, GCG, heurísticos, **Algoritmo 1** y §3.1 Método |
| p.4 | Definición de loss y criterio de éxito; §3.2 consideraciones prácticas (logprobs, short-circuit, inicialización); §3.3 variante sin proxy |
| p.5 | §4 Evaluación; §4.1 modelos abiertos (**Fig. 1**); inicio §4.2 |
| p.6 | **Tabla 1**; AutoDAN; §4.3 GPT-3.5 (**Fig. 2**) |
| p.7 | Costo vs iteraciones, longitud del target (**Fig. 3**), ablación de inicialización, §4.4 (**Fig. 4**) |
| p.8 | §4.5 moderación OpenAI, ataque universal (**Fig. 5**) |
| p.9 | No universal, §4.6 Llama Guard (**Fig. 6**), §5 Conclusión y trabajo futuro |
| p.12-13 | Ap. A cómputo, **Ap. B truco de logit bias**, Ap. C tokenización, Ap. D defensas, **Ap. E no determinismo (Fig. 7)** |

### 1.2 Números clave

| Dato | Valor | Ref. |
|---|---|---|
| Strings del dataset *harmful strings* (de Zou et al.) | 574 | [§4.3 · p.6] |
| Resueltos solo con la inicialización (prompt de 20 tokens) | 161/574 ≈ 28% | [§3.2.3 · p.4], [§4.3 · p.6] |
| ASR en GPT-3.5 con ≤ USD 0.10 por target | 79.6% | [§4.3 · p.6] |
| ASR con ≤ USD 0.20 por target | 86.0% | [§4.3 · p.6] |
| Costo total de los 574 strings / tope por target | USD 80 / USD 1 | [§4.3 · p.6] |
| ASR para targets de ≤ 20 tokens | 97.9% | [§4.3 · p.7] |
| 39 fallidos largos, re-corridos con prompt de 40 tokens | 100%, costo medio USD 0.41 (tope USD 2) | [§4.3 · p.7] |
| Inicialización aleatoria (primeros 20 strings) | 2 éxitos de 20 | [§4.3 · p.7] |
| Ahorro por *short-circuit* | ≈ 30% del costo | [§3.2.2 · p.4] |
| GCG optimizado vs GCG original | ≈ 2× menos queries | [§3.3 · p.4], [§4.4 · p.7] |
| Ataque 100% black-box vs GCG original (white-box) | lo supera en ≈ 30% | [§4.4 · p.7-8] |
| Tabla 1: GCQ 13B→7B / GCQ 7B→7B | 0.388 / 0.791 | [Tabla 1 · p.6] |
| Tabla 1: GCG transferencia pura (todos los pares) | 0.000 | [Tabla 1 · p.6] |
| AutoDAN GA / HGA (sufijos de ~70 tokens) | 1/574 / 0/574 | [§4.2 · p.6] |
| Moderación: strings que ya **no** se marcan sin ataque | 197/574 ≈ 34% | [§4.5 · p.8] |
| Universal, sufijo 20 tokens | 99.2% en validación, 100 iteraciones (2.000 requests) | [§4.5 · p.8] |
| Universal, sufijo 5 tokens | 94.8% en validación, 2.000 iteraciones (40.000 requests) | [§4.5 · p.8] |
| No universal, 10 iteraciones (5 / 20 tokens) | 83.8% / 91.4% | [§4.5 · p.9] |
| Llama Guard tras 320 queries (5 / 20 tokens) | 59% / 87% (OpenAI: 84% / 91%) | [§4.6 · p.9] |
| Hiperparámetros GPT-3.5 | m = 20, batch 32, proxy batch 8192, buffer 128 | [§4.3 · p.6] |
| Modelos abiertos (§4.1) | batch 512, 500 iteraciones, ≈ 400× más queries que en cerrados | [§4.1 · p.5] |
| Desvío estándar de la loss por no determinismo | 0.068 (1.000 muestras) | [Ap. E · p.13] |
| Reproducción en una segunda evaluación | 519 prompts (~90%) | [Ap. E · p.13] |

---

## 2. Glosario mínimo

- **Adversarial example:** input diseñado por un adversario para que el modelo se comporte mal [§2 · p.2].
- **White-box:** el atacante tiene los pesos y puede usar gradientes [§2 · p.2].
- **Black-box por transferencia:** se construye el ataque en un modelo local y se "reproduce" en el objetivo; se apoya en que los adversarial examples tienden a funcionar en otros modelos [§2 · p.2].
- **Black-box por queries:** se consulta el modelo objetivo para optimizar el ataque; en visión llegan a ~100% de éxito dirigido en ImageNet con unas miles de queries [§2 · p.2].
- **Surrogate vs proxy:** el paper usa *surrogate* para el modelo de un ataque por transferencia y *proxy* para el modelo local que guía la búsqueda. 💭 **Distinción clave para el debate:** el surrogate tiene que *compartir la vulnerabilidad*; el proxy solo tiene que *ordenar candidatos razonablemente*, porque la decisión final la toma la loss real del objetivo. Por eso un modelo base no alineado (Mistral 7B) sirve de proxy aunque no serviría de surrogate [§4.3 · p.6].
- **Ataque dirigido (targeted) vs no dirigido:** forzar una salida específica vs lograr que el modelo "acceda" a un pedido. La transferencia logra lo segundo pero no lo primero [§2 · p.3].
- **Harmful string vs jailbreak:** en harmful strings, el éxito exige que la generación coincida exactamente con el target [§4.2 · p.6]; en el jailbreak estilo AutoDAN, el éxito se define por la ausencia de frases de rechazo tipo "I'm sorry" [§4.2 · p.5].
- **ASR (attack success rate):** fracción de targets atacados con éxito; en casi todas las figuras se reporta acumulado vs iteraciones, costo o queries.
- **Loss ℓ:** logprob acumulado negativo del target dado el prompt; **proxy loss ℓp:** lo mismo calculado en el modelo local [§3.1 · p.4].
- **Greedy sampling:** el éxito se mide generando con greedy [§3.1 · p.4].
- **Logit bias:** vector que la API suma a los logits antes del log-softmax [Ap. B · p.12].
- **Sufijo universal / no universal:** uno solo para muchos strings vs uno a medida para cada string [§4.5 · p.8-9].
- **Buffer:** los B mejores prompts encontrados, implementado como *min-max heap* [§3.1 · p.3].

---

## 3. Eje 1 — Problema de investigación, relevancia y contexto

**El problema.** Los modelos alineados rechazan pedidos dañinos, pero existen ataques que los hacen acceder o incluso **emitir strings dañinos exactos**, por ejemplo la invocación de un plugin malicioso [§1 · p.1]. El daño va de lo reputacional a algo más serio cuando el modelo **actúa en nombre del usuario** (pagos, leer y mandar mails) [§1 · p.1].

**Por qué los ataques existentes no alcanzan:**
- GCG es white-box: necesita acceso completo al modelo, algo que no se tiene con los modelos de producción más grandes [§1 · p.1].
- La transferencia funciona "a medias": Zou et al. engañaron a GPT-4 y Bard con 46% y 66% de éxito transfiriendo desde Vicuna [§1 · p.1], pero:
  - requiere un surrogate de calidad: Vicuna es mal surrogate para Claude, con ~2% de éxito [§2 · p.3];
  - no logra ataques dirigidos (strings exactos) [§2 · p.3]. Esto es un problema conocido desde visión: incluso en ImageNet los ataques dirigidos por transferencia son difíciles [§2 · p.2].

**Contexto histórico (cómo lo enmarcan).** En visión la secuencia fue white-box → transferencia → ataques por queries, que alcanzan ~100% de éxito dirigido y se pueden combinar con un modelo local para gastar menos queries [§2 · p.2]. En NLP el texto es **discreto**, así que optimizar con gradiente es más difícil [§2 · p.2]; los primeros ataques cambiaban caracteres o palabras, después se optimizó en el espacio continuo de embeddings y se proyectó a tokens, pero no alcanzaba para transformers grandes, hasta que GCG combinó varias ideas [§2 · p.2].
💭 *Lectura nuestra:* el paper replica en LLMs el camino que visión ya recorrió, y lo lleva al paso "query-based".

**Relevancia.** 💭 Los modelos de frontera están detrás de APIs: si alcanza con acceso por queries, **cerrar los pesos no protege**. Los autores lo dicen como conclusión: defensas que dependen exclusivamente de romper la transferibilidad no van a ser efectivas [§5 · p.9].

**Contexto de autores.** 💭 Carlini y Nasr también firman GCG [ref. 36, p.11]: es el mismo grupo extendiendo su propio ataque. Carlini además es coautor de *Unsolved Problems in ML Safety*, bibliografía de la clase de AI Safety [C6 · diap. 67].

---

## 4. Eje 2 — Hipótesis, preguntas y objetivos

El paper **no enuncia hipótesis formales**; conviene decir esto si preguntan y luego reconstruirlas.

**Objetivo explícito:** diseñar un ataque de optimización que construya adversarial examples **directamente sobre un modelo remoto**, sin depender de la transferibilidad [§1 · p.1], con dos beneficios declarados: ataques dirigidos y ataques sin surrogate [§1 · p.1-2].

**Preguntas de investigación (reconstruidas 💭) y dónde se responden:**

| # | Pregunta | Dónde |
|---|---|---|
| P1 | ¿Alcanza el acceso por queries (+ proxy) para forzar strings exactos en un modelo de producción, donde la transferencia falla? | §4.2, §4.3 |
| P2 | ¿Qué tan parecido tiene que ser el proxy al objetivo? | §4.1 (Fig. 1b), §4.3 (proxy Mistral base) |
| P3 | ¿Se puede prescindir del proxy y del gradiente? ¿A qué costo en queries? | §3.3, §4.4 |
| P4 | ¿Aplica a clasificadores (moderación) sin tener un clasificador local? | §4.5, §4.6 |
| P5 | ¿Qué factores determinan el éxito? | §4.3 (longitud del target, inicialización) |

**Hipótesis implícitas:**
- H1: la selección de candidatos de GCG solo requiere queries, así que se puede separar del cálculo de gradiente [§1 · p.2].
- H2: el gradiente es una señal débil (por eso GCG lo combina con búsqueda greedy) [§3.3 · p.4], así que un ataque solo con queries podría ser viable.

---

## 5. Eje 3 — Conceptos teóricos y técnicos fundamentales

### 5.1 Modelos de amenaza

| Tipo | Qué acceso tiene el atacante | Ejemplo | ¿Dirigido? |
|---|---|---|---|
| White-box | Pesos y gradientes | GCG [36] | Sí |
| Transferencia | Un modelo local; al objetivo solo se le manda el resultado final | GCG transferido a GPT-4/Bard | En la práctica no [§2 · p.3] |
| Query-based | Consultas al objetivo (con logprobs), opcionalmente un proxy | **GCQ (este paper)**, PAL [29] | Sí |
| LLM-based / heurísticos | Un LLM que refina prompts, búsqueda greedy, prompts manuales | PAIR [8], TAP [22], [2] | No llegan a strings exactos [nota 1 · p.1] |

### 5.2 La loss y el criterio de éxito

Siguiendo a Zou et al., la loss es el logprob acumulado negativo del target condicionado al prompt, y el ataque es exitoso si el target se genera con greedy [§3.1 · p.4]:

```
ℓ(x) = − Σ_{i=1..t} log p(y_i | x, y_1 … y_{i−1})  =  − log P(y | x)
```

💭 Tres observaciones para el oral:
1. Es la **misma cross-entropy** del entrenamiento de un LM (C1: comparar la distribución predicha con la deseada y hacer backprop [C1 · diap. 47]). La diferencia es *qué* se optimiza: en entrenamiento los pesos, en el ataque los tokens del input.
2. `e^(−ℓ)` es la probabilidad de generar exactamente el target muestreando a temperatura 1.
3. El criterio greedy es más exigente que "ℓ baja": cada token del target tiene que ser el argmax en su paso.

### 5.3 GCG (el predecesor directo)

GCG extiende AutoPrompt: calcula gradientes para todas las sustituciones posibles de un token, selecciona candidatos prometedores, los evalúa y se queda con el de menor loss. Supera a AutoPrompt porque considera todas las coordenadas en vez de elegir una de antemano [§2 · p.3].
💭 Detalle técnico (viene de Zou et al.): el gradiente se toma respecto del vector one-hot de cada token del sufijo. Eso da una aproximación lineal de cuánto cambiaría la loss al cambiar ese token por cada palabra del vocabulario; se toman los top-k por posición, se samplean candidatos y se evalúan con forward passes exactos.

**La descomposición en dos etapas** es el insight del paper: (1) filtrar con gradiente, (2) elegir con la loss exacta. La etapa 2 solo requiere queries [§1 · p.2].

### 5.4 Por qué el texto es más difícil que las imágenes

El texto es discreto, así que la optimización directa por gradiente es más difícil [§2 · p.2]. 💭 El espacio de búsqueda es V^m: con un vocabulario del orden de 100k tokens y m = 20, son ~10^100 sufijos posibles. El gradiente es una aproximación lineal en un espacio donde solo hay "saltos" discretos, por eso predice mal el efecto real de un cambio de token.

---

## 6. Eje 4 — Metodología

### 6.1 GCQ paso a paso [Alg. 1 · p.3], [§3.1 · p.3-4]

Entradas: vocabulario V, largo de secuencia m, loss ℓ, proxy loss ℓp, iteraciones T, *proxy batch size* bp, *query batch size* bq, tamaño de buffer B.

```
buffer (B mejores prompts, min-max heap)
   │  tomar el mejor
   ▼
generar bp vecinos: cambiar 1 token (posición y token al azar)
   │
   ▼
rankear con la proxy loss ℓp (modelo local)  ──►  quedarse con los top-bq
   │
   ▼
evaluar la loss real ℓ en la API del objetivo (con short-circuit)
   │
   ▼
si ℓ(candidato) ≤ ℓ(peor del buffer): sacar el peor, meter el candidato
   │
   └──► repetir T veces; devolver el mejor del buffer
```

- La búsqueda es parecida a *best-first search*: cada nodo es un sufijo, el buffer guarda los B mejores nodos no explorados y en cada iteración se expande el mejor [§3.1 · p.3].
- 💭 **Para qué sirve el buffer:** (a) memoria de alternativas buenas, que permite "volver atrás" si una rama se estanca; (b) da el umbral ℓ(b_worst) que habilita el short-circuit, que según los autores fue la motivación principal para introducirlo [§3.2.2 · p.4].
- 💭 Ojo: en el Alg. 1 los candidatos se filtran con la **proxy loss** (forward passes del proxy), no con el gradiente del proxy. En §4.1, en cambio, sí usan **gradientes** del proxy [§4.1 · p.5]. Son dos formas de usar el modelo local.

### 6.2 GCG vs GCQ

| Aspecto | GCG (white-box) | GCQ |
|---|---|---|
| Generación/filtrado de candidatos | Gradiente del objetivo | Muestreo aleatorio + ranking con proxy local |
| Evaluación exacta | Forward pass local | Queries a la API del objetivo |
| Estado de la búsqueda | Un único sufijo actual | Buffer de los B mejores (best-first) |
| Ahorro de costo | — | Short-circuit (~30%) |
| Inicialización | Genérica | Target repetido (28% resuelto de entrada) |

### 6.3 Consideraciones prácticas (lo "ingenieril" también es contribución)

**(a) Reconstruir logprobs con logit bias + top-5** [§3.2.1 · p.4], [Ap. B · p.12]

- Cronología: en septiembre 2023 OpenAI dejó de devolver logprobs de los tokens del prompt; los autores reconstruyen la loss combinando funciones de la API, con una técnica parecida a la de Morris et al. [23] [§3.2.1 · p.4]. En marzo 2024 OpenAI cambió la API para que el logit bias no afecte los top logprobs; a mayo 2024 todavía se podía inferir con la búsqueda binaria de [23], pero más caro [§3.2.1 · p.4].
- Idea: la API devuelve los top-5 logprobs de cada token generado y permite sumar un bias a los logits. Con el bias "suben" el token de interés al top-5, leen su logprob y corrigen la distorsión con una fórmula [Ap. B · p.12].

```
Softmax con bias b sumado al logit del token:   p_b = e^b·p / (e^b·p + (1 − p))
Despejando p (fórmula del Ap. B):               p = p_b / (e^b·(1 − p_b) + p_b)
```

- Elección del bias: si es muy grande, p_b ≈ 1 y se pierde precisión numérica; si es muy chico, el token no entra al top-5. Usan como bias `−log p̂`, donde p̂ es la estimación del padre (que difiere en un solo token); si falla, búsqueda binaria. La primera elección funciona en más del 99% de los casos [Ap. B · p.12].
- 💭 **Por qué `−log p̂` funciona:** si p̂ ≈ p, entonces e^b·p ≈ 1 y `p_b ≈ 1/(2 − p) ≈ 0.5`. Con probabilidad ~0.5, como mucho *otro* token puede superarlo, así que queda seguro en el top-2, y está lejos de 1 (sin problema de precisión).
- Limitación de la API: un único logit bias por generación, así que puntúan el target token por token agregando al prompt los primeros i tokens del target. Costo: p·t tokens de prompt y t(t+1)/2 de completion para un prompt de p tokens y un target de t [Ap. B · p.12]. Por eso el costo de la loss crece **super-linealmente** con el largo del target [§4.3 · p.6-7].

**(b) Short-circuit** [§3.2.2 · p.4]
Como la loss se computa token por token, se puede cortar el cálculo si ya es suficientemente mala: cualquier prompt con loss mayor que ℓ(b_worst) se va a descartar igual. Ahorra ≈ 30% del costo.
💭 **Por qué es válido:** cada término −log p ≥ 0, así que la suma parcial es monótona creciente; si ya superó el umbral, el valor final también lo supera. También explica por qué el costo fluctúa de forma impredecible [§4.3 · p.7].

**(c) Inicialización** [§3.2.3 · p.4]
En vez de prompts aleatorios, inicializan repitiendo el target tantas veces como entre en el prompt, truncando por izquierda. Con m = 20 eso ya resuelve el 28% de los strings sin correr el algoritmo, por la tendencia del modelo a continuar repeticiones [§4.3 · p.6].
💭 Relación con C1: con atención enmascarada, cada token mira los anteriores; los LMs copian patrones del contexto con mucha facilidad.

### 6.4 Variante sin proxy (query-only) [§3.3 · p.4-5]

- Punto de partida: en GCG el gradiente es una señal débil; se podría ignorar y hacer un ataque greedy puro con reemplazos aleatorios, pero eso sería carísimo [§3.3 · p.4].
- Mejora a GCG: en GCG original cada iteración evalúa B candidatos con un reemplazo en una posición aleatoria, o sea ~B/l intentos por posición para un sufijo de largo l. En la nueva variante primero se prueba **un** reemplazo en cada posición, se elige la posición donde más bajó la loss y se prueban B′ reemplazos más **solo ahí**, con B′ ≪ B sin perder éxito [§3.3 · p.4-5]. Reduce las queries ~2× [§3.3 · p.4].
- 💭 Intuición: en vez de repartir el presupuesto entre todas las posiciones, primero "sondea" cuál es la coordenada más sensible y concentra el presupuesto ahí. Es asignar mejor las queries.

### 6.5 Tokenización [Ap. C · p.12]

- La secuencia de tokens que encuentra la optimización puede no ser la tokenización que el tokenizer produciría para ese string (ejemplo del paper: dos tokens que, concatenados, el tokenizer parte distinto). Por eso re-tokenizan los strings antes de mandarlos a la API; no notaron impacto en el éxito [Ap. C · p.12].
- El proxy no usa el tokenizer de OpenAI (no hay modelos abiertos grandes con ese tokenizer), así que re-tokenizan con el tokenizer del proxy para calcular ℓp [Ap. C · p.12].

### 6.6 Diseño experimental

| Exp. | Sección | Objetivo (target) | Proxy | Datos | Métrica | Presupuesto |
|---|---|---|---|---|---|---|
| White-box (línea base) | §4.1, Fig. 1a | Vicuna 1.3 7B/13B/33B, Llama 2 7B Chat | — | harmful strings | ASR acumulado vs iteraciones | batch 512, 500 iter. |
| Proxy entre escalas | §4.1, Fig. 1b | Vicuna 1.3 | Vicuna 1.3 de otro tamaño (gradientes) | idem | idem | idem |
| Comparación | §4.2, Tabla 1 | Vicuna 7B/13B, Mistral 7B Instruct v0.3, Gemma 2 2B | Vicuna 7B/13B | 574 strings | ASR | AutoDAN default ≈ 128 iter. de GCQ |
| Modelo de producción | §4.3, Fig. 2-3 | gpt-3.5-turbo-instruct-0914 | Mistral 7B **base** | 574 strings | ASR vs USD y vs iteraciones | USD 1 por target (USD 2 con m = 40) |
| Sin proxy, abierto | §4.4, Fig. 4 | Vicuna 7B | ninguno | harmful strings | ASR vs queries de loss | hasta 50.000 |
| Moderación OpenAI | §4.5, Fig. 5-6a | text-moderation-007 | ninguno | 574 (20 train / 554 val. en universal) | ASR = sin flags | requests (32 textos por request) |
| Llama Guard | §4.6, Fig. 6b | Llama Guard 7B | ninguno | idem | ASR vs queries | ~10^4 queries |

Cómputo: 2 a 8 A100 durante varios días para §4.1; una A40 durante varios días para el resto [Ap. A · p.12].

### 6.7 Métricas y criterios de éxito

- **Harmful strings:** éxito = el target aparece exacto con greedy [§3.1 · p.4].
- **Moderación:** éxito = el string con sufijo no recibe ningún flag. Como loss usan la **suma de los scores** de todas las categorías, así no necesitan conocer los umbrales de cada flag, que no son públicos y pueden cambiar [§4.5 · p.8].
- **Universal:** loss promedio sobre 20 strings de entrenamiento; se mide en los 554 restantes [§4.5 · p.8].

---

## 7. Eje 5 — Resultados principales y evidencia

### 7.1 Modelos abiertos [§4.1 · p.5], [Fig. 1]
- White-box: Vicuna es más difícil de atacar a medida que crece; el Llama 2 más chico es bastante más resistente que el Vicuna más grande [§4.1 · p.5].
- Entre escalas (gradientes de proxy + queries al objetivo): 7B transfiere mal hacia arriba; 13B→33B pierde poco; 13B→7B transfiere mal. Concluyen que 13B y 33B se parecen más entre sí que a 7B [§4.1 · p.5].
- 💭 Ojo con la palabra "transfer" en la Fig. 1b: **no** es transferencia pura, porque se mantiene acceso por queries al objetivo [§4.1 · p.5]. La transferencia pura aparece en la Tabla 1.

### 7.2 Comparación con otros ataques [§4.2 · p.5-6], [Tabla 1]
- AutoDAN adaptado a harmful strings: 1/574 (GA) y 0/574 (HGA), con sufijos de ~70 tokens contra 20 de GCQ; con parámetros por defecto equivale a ~128 iteraciones de GCQ [§4.2 · p.6]. Lo incluyen para mostrar que los harmful strings son más difíciles que los jailbreaks [§4.2 · p.5].
- GCG transferencia pura: 0.000 en todos los pares, incluso hacia Mistral 7B Instruct v0.3 y Gemma 2 2B; GCQ: 0.388 (13B→7B) y 0.791 (7B→7B) [Tabla 1 · p.6].
- Conclusión de los autores: es muy difícil forzar strings específicos con bajo grado de acceso [§4.2 · p.6].

### 7.3 GPT-3.5 Turbo Instruct [§4.3 · p.6-7], [Fig. 2]
- 161 de 574 (~28%) se resuelven solo con la inicialización [§4.3 · p.6] → el escalón inicial de la Fig. 2 y la línea punteada "Initialization only".
- El ASR sube rápido: 79.6% con ≤ USD 0.10 por target y 86.0% con USD 0.20; costo total USD 80 [§4.3 · p.6].
- Iteraciones y costo escalan distinto: el costo de la loss vía API crece super-linealmente con el largo del target, mientras que el cómputo del proxy es constante [§4.3 · p.6-7].
- 💭 Dato llamativo: el proxy es un modelo **base no alineado** que no serviría para transferencia pura [§4.3 · p.6], y aun así funciona como filtro.

### 7.4 Longitud del target [§4.3 · p.7], [Fig. 3]
- El éxito cae fuerte cuando el target se acerca o supera el largo del prompt; para targets de ≤ 20 tokens el éxito es 97.9% [§4.3 · p.7].
- Dos explicaciones posibles: (1) la inicialización se debilita si no entra ni una copia del target; (2) faltan grados de libertad para "codificar" el target [§4.3 · p.7].
- Verificación: re-corren los 39 fallidos de más de 20 tokens con un prompt de 40 tokens (presupuesto USD 2): 100% de éxito, costo medio USD 0.41 [§4.3 · p.7].
- 💭 **Buen punto crítico:** el experimento confirma que el problema es el largo del prompt, pero **no separa** las dos explicaciones, porque un prompt más largo arregla ambas a la vez.

### 7.5 Ablación de la inicialización [§4.3 · p.7]
- Con inicialización aleatoria en los primeros 20 strings, solo 2 éxitos. Concluyen que una buena inicialización es crucial en el régimen de bajo costo [§4.3 · p.7].
- 💭 Muestra chica (n = 20) y no aleatoria (los "primeros" 20).

### 7.6 Ataque sin proxy en modelos abiertos [§4.4 · p.7-8], [Fig. 4]
- GCG optimizado: ~2× más eficiente en queries que GCG original con gradientes. Sin gradientes (reemplazos al azar), el ataque black-box igual supera al GCG original en ~30% [§4.4 · p.7-8].
- Concluyen que los ataques black-box solo con queries pueden ser prácticos para forzar strings [§4.4 · p.8].
- 💭 El "30%" es ambiguo: no queda claro si es mejora relativa del ASR o de eficiencia en queries. A ojo en la Fig. 4, a 50k queries GCG queda cerca de 0.4 y el black-box cerca de 0.47.

### 7.7 Moderación de OpenAI [§4.5 · p.8-9], [Fig. 5, 6a]
- Objetivo: text-moderation-007, presentado por OpenAI como su modelo de moderación más robusto [§4.5 · p.8].
- Sin ataque, 197/574 (~34%) ya no se marcan [§4.5 · p.8].
- Universal, 20 tokens: 99.2% en validación tras 100 iteraciones (2.000 requests). 5 tokens: 94.8% tras 2.000 iteraciones (40.000 requests) [§4.5 · p.8].
- No universal, 10 iteraciones: 83.8% (5 tokens) y 91.4% (20 tokens) [§4.5 · p.9]. Cada request de OpenAI son 32 queries [Fig. 6 · p.9].
- El universal es más difícil por dos razones: cada evaluación cuesta el tamaño del set de entrenamiento y tiene que generalizar a strings no vistos [§4.5 · p.8].

### 7.8 Llama Guard 7B [§4.6 · p.9], [Fig. 6b]
- Mismo setting no universal. Tras 320 queries: 59% (5 tokens) y 87% (20 tokens), contra 84% y 91% en OpenAI; la brecha se achica con más iteraciones [§4.6 · p.9].
- 💭 El paper no explica por qué Llama Guard resiste más al principio; si preguntan, decir que es una observación sin explicación en el texto y ofrecer hipótesis (arquitectura generativa como clasificador, otro entrenamiento, otra línea base sin ataque en la Fig. 6b).

### 7.9 No determinismo [Ap. E · p.13], [Fig. 7]
- Los logprobs de GPT-3.5 Turbo Instruct no son estables en el tiempo aun con seed fija; las generaciones no siempre se reproducen con temperatura 0; no saben la causa [Ap. E · p.13].
- Consecuencias: no saben si un prompt exitoso lo seguirá siendo, y los scores son muestras de un proceso aleatorio [Ap. E · p.13].
- Magnitud: desvío de 0.068 en la loss (1.000 muestras); en comparación, la diferencia entre el mejor y el peor del buffer suele ser de al menos 3. En moderación, media 0.02 y desvío 4×10⁻⁴. Por eso no implementan mitigaciones [Ap. E · p.13].
- Re-evaluación: 519 prompts (~90%) vuelven a producir el target [Ap. E · p.13].
- 💭 519/574 ≈ 90.4%. Conviene tener claro por si preguntan: el ASR reportado de una sola evaluación podría estar sobreestimado en ~10% si pensamos en "éxito reproducible".

---

## 8. Eje 6 — Contribución novedosa frente a trabajos previos

| Trabajo | Qué hace | Qué le falta, según el paper |
|---|---|---|
| GCG white-box (Zou et al.) [36] | Optimiza el sufijo con gradientes | Requiere acceso completo al modelo [§1 · p.1] |
| GCG por transferencia [36] | Optimiza en Vicuna y reusa en GPT-4/Bard | Necesita buen surrogate; no logra strings dirigidos [§2 · p.3]; 0.000 en Tabla 1 |
| AutoPrompt [28] | Cambia una coordenada elegida de antemano | GCG lo supera al considerar todas las coordenadas [§2 · p.3] |
| PAIR [8], TAP [22], búsqueda greedy [2] | Un LLM refina ataques / búsqueda aleatoria | Más débiles; no logran strings exactos [nota 1 · p.1] |
| AutoDAN [19] | Algoritmo genético con prompts legibles | 1/574 en harmful strings [§4.2 · p.6] |
| PAL, Sitawarin et al. [29] (concurrente) | Ataque guiado por proxy muy similar | No lo evalúan para strings dirigidos [§2 · p.3] |
| Visión: ZOO [9], Brendel [5], Cheng [10] | Ataques por queries, con o sin prior local | Mismo espíritu en imágenes; este paper lo lleva a LLMs [§2 · p.2] |

**Contribuciones, en una lista para decir de corrido:**
1. Primer ataque por queries que fuerza **strings dañinos exactos** en un modelo de producción (GPT-3.5), algo que la transferencia no logra [§1 · p.1], [§4.3].
2. El proxy no tiene que ser un buen surrogate: sirve un modelo base no alineado [§4.3 · p.6].
3. Una mejora a GCG (~2× menos queries) que habilita un ataque **sin proxy ni gradiente** [§3.3], [§4.4].
4. Ingeniería que lo vuelve práctico: reconstrucción de logprobs vía logit bias, short-circuit, inicialización [§3.2], [Ap. B].
5. Evasión casi total de clasificadores de moderación sin clasificador local (OpenAI y Llama Guard) [§4.5-4.6].

💭 **La novedad conceptual más fuerte** es la descomposición "filtrar con algo barato / decidir con la loss real". Explica por qué un proxy mediocre alcanza y por qué el gradiente importa menos de lo que se creía.

---

## 9. Eje 7 — Supuestos, limitaciones y amenazas a la validez

### 9.1 Supuestos del modelo de amenaza
- El atacante conoce el string objetivo exacto.
- La API permite estimar logprobs de tokens arbitrarios (acá, vía logit bias + top-5) [§3.2.1 · p.4], [Ap. B · p.12]. Los autores lo reconocen como requisito [Ap. D · p.13].
- El modelo objetivo es de *completion* (acepta el prompt crudo): gpt-3.5-turbo-instruct [§4.3 · p.6]. 💭 No hay chat template ni system prompt de por medio.
- Presupuesto acotado en dólares o requests; el costo depende de los precios de 2024 [§4.3], [§4.5].
- 💭 Supone que el proveedor no detecta ni limita el patrón de queries (muchas consultas casi idénticas).

### 9.2 Limitaciones que reconocen los autores
- Cambios en la API vuelven obsoleta la técnica de Ap. B, al menos para OpenAI, o la encarecen [§3.2.1 · p.4], [Ap. B · p.12].
- Los sufijos tienen muchos tokens de aspecto aleatorio: un **filtro de perplejidad** los detectaría. Técnicas para evadir esos filtros podrían dar un ataque adaptativo [Ap. D · p.12-13].
- Ataques sin acceso a logprobs quedan como trabajo futuro [Ap. D · p.13].
- No determinismo y reproducción ~90% [Ap. E · p.13].
- El éxito depende mucho del largo del target [§4.3 · p.7] y de la inicialización [§4.3 · p.7], [§5 · p.9].
- Los ataques en NLP siguen siendo débiles comparados con visión [§5 · p.9].

### 9.3 Amenazas a la validez que podemos señalar nosotros 💭

**Validez interna** (¿los resultados miden lo que dicen?)
- No determinismo: el éxito se mide una vez; con ~10% de no reproducción, el ASR "reproducible" es menor (los autores lo reportan, lo cual suma transparencia).
- La mayoría de las curvas no tienen intervalos de confianza ni varias semillas (la Fig. 3 sí tiene barras).
- La ablación de inicialización usa solo los primeros 20 strings.

**Comparaciones y líneas de base**
- AutoDAN se usa en una tarea para la que no fue diseñado y con parámetros por defecto: riesgo de *baseline nerfing* [C4 · diap. 45]. Atenuante: los autores declaran que lo incluyen para mostrar la diferencia entre jailbreak y harmful string [§4.2 · p.5].
- En la Tabla 1 falta el par GCQ 7B→13B (hay transferencia pura 7B→13B pero no GCQ), y no hay GCQ contra Mistral Instruct ni Gemma 2. Las filas en 0.000 muestran que la transferencia falla, **no** que GCQ funcione en esos pares.
- GCQ 7B→7B (0.791) usa proxy = objetivo: es casi un escenario white-box.

**Validez externa** (¿generaliza?)
- Un solo modelo cerrado de generación, de completion y no de chat; no se prueba GPT-4 ni modelos con system prompt.
- Un solo dataset de targets.
- Los costos en USD y la técnica de logprobs dependen del estado de la API en 2024.

**Validez de constructo** (¿la métrica captura el concepto?)
- ¿Emitir un string exacto es "daño"? Depende del despliegue: importa mucho si la salida se ejecuta (plugins, agentes), que es como lo motivan [§1 · p.1].
- En moderación, ~34% del dataset ya no se marca sin ataque [§4.5 · p.8]: los ASR hay que leerlos contra esa línea base.
- La loss de moderación (suma de scores) es un proxy del objetivo binario "sin flags", aunque el éxito final sí se mide con los flags.

**Validez de conclusión**
- "Supera a GCG en ~30%" es ambiguo en su métrica [§4.4 · p.8].
- No se evalúan defensas adaptativas (perplejidad, paráfrasis), aunque se mencionan [Ap. D].

### 9.4 Detalles e inconsistencias del texto (mostrar lectura atenta, sin sobredimensionar)
- **Alg. 1** [p.3]: el primer loop interno itera sobre [bq], pero el texto dice que se generan bp vecinos; además reutiliza la variable `i` en loops anidados. Probable errata.
- **Alg. 1 vs texto**: el texto habla de buffer de nodos "no explorados" [§3.1 · p.3], pero el pseudocódigo nunca saca del buffer el nodo que expande.
- **Fig. 2** [p.6]: la leyenda dice "Q-GC" en lugar de GCQ.
- **Fig. 5** [p.8]: el epígrafe habla de sufijos de 5 y 20 tokens, pero el panel (b) dice "10 token suffix"; el texto dice 20.
- **Fig. 6a** [p.9]: la leyenda dice "Prompt" donde el texto habla de sufijo.
- **Ap. E** [p.13]: dice que el número de re-evaluación se reporta "en el Apéndice E", o sea, en sí mismo.
- **Fig. 1b** se titula "Transfer attacks" pero incluye queries al objetivo [§4.1 · p.5].

---

## 10. Eje 8 — Relación con los contenidos de la materia

### C1 — Transformers
| Concepto de clase | Conexión con el paper |
|---|---|
| Decoder-only, autoregresivo [C1 · diap. 40, 52] | El target se genera token a token; el éxito se mide con greedy [§3.1 · p.4]. La loss se descompone por token, y eso habilita el short-circuit [§3.2.2]. |
| LINEAR → logits → Softmax [C1 · diap. 41-42] | El logit bias se suma a los logits **antes** del (log-)softmax; la corrección del Ap. B es despejar el softmax [Ap. B · p.12]. |
| Entrenamiento: comparar distribución y backprop [C1 · diap. 47] | 💭 La loss del ataque es la misma NLL; GCG hace backprop hasta el input (one-hot) en lugar de actualizar pesos. |
| Atención enmascarada: cada token mira los anteriores | 💭 El sufijo condiciona toda la generación; la inicialización por repetición aprovecha la tendencia a copiar el contexto [§4.3 · p.6]. |

### C2 — Tokenización y embeddings
| Concepto de clase | Conexión con el paper |
|---|---|
| Vocabulario y tokenización [C2 · diap. 7] | El espacio de búsqueda es V^m [Alg. 1]; la tokenización también define costos, y acá el costo de la loss en tokens es super-lineal [Ap. B]. |
| BPE / tiktoken [C2 · diap. 11-17] | Ap. C: secuencias de tokens no canónicas, re-tokenización para la API y para el proxy (tokenizers distintos) [Ap. C · p.12]. |
| Embedding: lo discreto en un espacio continuo [C2 · diap. 31] | Los ataques en espacio de embeddings + proyección a tokens no alcanzaron en transformers grandes [§2 · p.2]; el problema es que el input real es discreto. |
| Glitch tokens y "riesgos de seguridad asociados a tokenización" [C2 · diap. 18-20, 63] | La propia clase recomienda el paper de GCG (Zou et al.) en esa sección [C2 · diap. 63]. 💭 Los sufijos adversariales son secuencias raras de tokens, por eso un filtro de perplejidad los detectaría [Ap. D]. |

### C3 — Transfer learning y fine-tuning
| Concepto de clase | Conexión con el paper |
|---|---|
| Transferability [C3 · diap. 13-32] | Mismo término: transferibilidad de adversarial examples [§2 · p.2]. 💭 Raíz común: redes distintas aprenden features parecidas [C3 · diap. 32], lo que genera vulnerabilidades compartidas. La Fig. 1b muestra que transfieren mejor los modelos más parecidos (13B↔33B) [§4.1]. |
| Knowledge distillation [C3 · diap. 19-22] | 💭 Un proxy destilado del objetivo podría rankear mejor los candidatos (extensión posible). |
| Desventajas del modelo fundacional y HHH [C3 · diap. 38, 40] | El ataque viola *harmless*; el proxy es justamente un modelo base (Mistral 7B) [§4.3 · p.6]. |
| SFT / RLHF / DPO [C3 · diap. 44-55] | Vicuna es un fine-tune de Llama 1; Llama 2 Chat resiste mucho más [§4.1 · p.5]. 💭 El método de alineamiento pesa más que la escala. |
| Penalización KL [C3 · diap. 49] | 💭 La política alineada se mantiene cerca del modelo base; la capacidad de producir esos strings sigue ahí, "tapada". Eso ayuda a entender por qué un modelo base sirve de proxy. |
| "¿Es RLHF robusto?" (shoggoth con máscara) [C3 · diap. 60] | 💭 El sufijo adversarial busca inputs donde la "máscara" de RLHF se corre. |
| Violation Rate / False Refusal Rate [C3 · diap. 56] | 💭 El ASR es como un VR bajo presión adversarial, pero más estricto (string exacto). Defensas como filtros de perplejidad pueden subir el FRR. |

### C4 — Reasoning models y producción
| Concepto de clase | Conexión con el paper |
|---|---|
| RLHF con 4 modelos / RLVR [C4 · diap. 19-22] | El alineamiento se entrena contra un reward model, que es un proxy de "preferencia humana". 💭 El ataque explota los huecos de esa generalización. |
| Agentic tool use [C4 · diap. 18] | La motivación del paper: si la salida dispara acciones (pagos, mails, plugins), un string exacto es un comando [§1 · p.1]. |
| Productos: clasificador / juez [C4 · diap. 36] | Llama Guard y la moderación son LLMs usados como clasificadores/jueces; el paper los ataca [§4.5-4.6]. |
| Scaling laws [C4 · diap. 40] | Vicuna es más difícil de atacar con escala, pero Llama 2 7B supera a Vicuna 33B [§4.1 · p.5]: la escala no es toda la historia. |
| Prácticas cuestionables en evaluación [C4 · diap. 43-45] | 💭 Baseline nerfing (AutoDAN), reproducibilidad (no determinismo, cambios de API). |
| Self-consistency: sampling [C4 · diap. 16] | 💭 El ataque asume greedy; muestrear varias salidas y agregar podría funcionar como defensa (a evaluar). |
| MoE [C4 · diap. 48-49] | 💭 Hipótesis externa, **no** del paper: el no determinismo de APIs suele atribuirse a batching / ruteo en MoE; el paper dice que no conoce la causa [Ap. E]. |
| Cuantización y precisión [C4 · diap. 60] | La elección del bias está limitada por la precisión numérica [Ap. B · p.12]. |
| Reasoning models [C4 · diap. 4-11] | No evaluados. 💭 Con CoT oculto, la salida final queda "detrás" del razonamiento y muchas APIs no exponen sus logprobs: buena línea de extensión. |

### C6 — AI Safety (la conexión más fuerte)
| Concepto de clase | Conexión con el paper |
|---|---|
| Jailbreaks: tensión de objetivos vs generalización inadecuada [C6 · diap. 3-6] | 💭 Un sufijo optimizado de tokens raros es más **generalización inadecuada** (input fuera de la distribución del entrenamiento de seguridad) que tensión de objetivos. El ejemplo de la clase, obligar a GPT-4 a empezar con una frase afirmativa [C6 · diap. 5], es primo del objetivo de GCG; este paper exige todo el string. |
| "¿Es RLHF robusto?": BadLlama y GCG [C6 · diap. 12-14] | BadLlama necesita los pesos (fine-tuning); GCG [C6 · diap. 14] es la referencia [36] del paper. Este trabajo es su continuación directa: saca la necesidad de pesos o de transferencia y ataca modelos cerrados por API. |
| Prompt injection [C6 · diap. 8-9] | El paper cita inyección indirecta de prompts [13] entre los riesgos de modelos que actúan [§1 · p.1]. |
| Problemas sin resolver: Robustez → *Adversaries* [C6 · diap. 26-27] | Es literalmente un paper del pilar robustez adversarial. |
| Monitoreo / seguridad sistémica [C6 · diap. 26-30] | 💭 La moderación y Llama Guard son **monitores**; evadirlos rompe la "defensa en profundidad". |
| Ley de Goodhart [C6 · diap. 24] | 💭 El alineamiento optimiza un proxy; un adversario optimiza justo en la brecha entre proxy y objetivo real. También la loss de moderación (suma de scores) es un proxy. |
| Pseudo-alineamiento, distributional shift [C6 · diap. 37-44] | 💭 Un modelo alineado "en distribución" puede no estarlo bajo un shift adversarial construido a propósito. |

---

## 11. Eje 9 — Lo más relevante, cuestionable e interesante (posiciones para el grupo)

**Lo más relevante**
- Cerrar los pesos no alcanza: con acceso a una API rica (logprobs), el ataque dirigido es barato [§4.3], [§5].
- Proxy ≠ surrogate: un modelo local mediocre alcanza para filtrar [§4.3 · p.6].
- Los clasificadores de moderación caen casi al 100% con un sufijo universal [§4.5].

**Lo más cuestionable**
- Depende de una superficie de API que el proveedor ya restringió [§3.2.1], [Ap. B].
- Mucha parte del éxito viene de la inicialización (28% gratis; 2/20 sin ella) [§4.3].
- Un único modelo cerrado, de completion; comparación con AutoDAN poco equitativa; faltan celdas en la Tabla 1.
- No determinismo: éxito medido en una sola corrida [Ap. E].

**Lo más interesante**
- El ataque sin gradientes le gana a GCG white-box original [§4.4]. 💭 Sugiere que el gradiente aporta menos que asignar bien las queries.
- La relación largo del target vs largo del prompt (grados de libertad) [§4.3 · p.7].
- El contraste con visión: allá la inicialización casi no importa y cambiar semillas mejora poco; acá la inicialización es decisiva [§5 · p.9].

---

## 12. Eje 10 — Extensiones y nuevas líneas

1. **Sin logprobs** (solo texto): estimar la loss por muestreo o usar la señal binaria "¿salió el target?", al estilo de los ataques basados en decisión de visión [5]. Los autores lo marcan como trabajo futuro [Ap. D · p.13].
2. **Sufijos legibles** que pasen filtros de perplejidad (combinar con ideas de AutoDAN) [Ap. D · p.13].
3. **Modelos de chat y de razonamiento**, con system prompt y CoT oculto.
4. **Robustez al no determinismo:** optimizar la loss esperada y re-evaluar varias veces; mejorar la tasa de reproducción es trabajo futuro para los autores [Ap. E · p.13].
5. **Mejores inicializaciones**, que los autores ven como un área con mucho potencial [§5 · p.9].
6. **Mejores proxies:** destilar un proxy con salidas del objetivo (C3).
7. **Escenarios agénticos reales:** que el target sea una llamada a herramienta dentro de un agente, o que el sufijo llegue por inyección indirecta (C6).
8. **Defensas a evaluar:** limitar o sacar logprobs y logit bias, detectar patrones de queries casi idénticas, rate limiting, filtros de perplejidad, entrenamiento adversarial, re-tokenización o paráfrasis del input (varias listadas en [16], según [Ap. D · p.12]). 💭 Fuera del paper: *randomized smoothing* para LLMs (p. ej. SmoothLLM).

---

## 13. Posibles preguntas de las profesoras (con respuesta sugerida)

> Estructura recomendada para responder: **afirmación → evidencia (sección/figura) → límite o matiz**.

### A. Comprensión general

**1. Resuman el paper en dos minutos.**
→ Usar la sección 0. Cerrar con la conclusión: las defensas que dependen solo de romper la transferibilidad no sirven [§5 · p.9].

**2. ¿Qué diferencia hay entre un jailbreak y un harmful string? ¿Por qué el segundo es más difícil?**
→ Jailbreak: basta con que el modelo no rechace. Harmful string: la salida tiene que coincidir exactamente [§4.2 · p.5-6]. Evidencia: AutoDAN, un buen atacante de jailbreaks, logra 1/574 [§4.2 · p.6]. 💭 Es un ataque dirigido: hay que controlar cada token.

**3. ¿Qué es un ataque white-box, uno por transferencia y uno por queries? ¿Dónde se ubica este paper?**
→ Tabla 5.1. Query-based, con proxy opcional.

**4. ¿Por qué la transferencia no sirve para ataques dirigidos?**
→ El paper lo toma de la literatura de visión: los ataques dirigidos por transferencia son difíciles incluso en ImageNet [§2 · p.2], y en LLMs no logran strings específicos [§2 · p.3]; la Tabla 1 da 0.000. 💭 Transferir "que el modelo se porte mal" solo requiere compartir una debilidad general; transferir "que diga exactamente esto" requiere que la distribución de salida de ambos modelos coincida en detalle.

### B. Método

**5. Expliquen GCG. ¿Qué rol cumple el gradiente y por qué no alcanza?**
→ Sección 5.3. El gradiente sobre el one-hot es una aproximación lineal en un espacio discreto; los autores lo llaman una señal débil, y por eso GCG lo combina con búsqueda greedy [§3.3 · p.4].

**6. ¿Cuál es la observación que permite pasar de GCG a GCQ?**
→ Las dos etapas: filtrar con gradiente y seleccionar con queries; reemplazan el filtro por un proxy y consultan al objetivo [§1 · p.2].

**7. ¿Para qué sirve el buffer? ¿Qué pasa si B = 1?**
→ Best-first, mantener alternativas, habilitar el short-circuit [§3.1], [§3.2.2]. 💭 Con B = 1 queda un hill-climbing greedy (se acepta un candidato solo si mejora al actual): sin memoria para volver atrás, aunque el short-circuit sigue funcionando contra el actual.

**8. ¿Qué es la loss? ¿Cómo se relaciona con la probabilidad de generar el target?**
→ Logprob acumulado negativo [§3.1 · p.4]. 💭 e^(−ℓ) es la probabilidad de generarlo con muestreo a temperatura 1; el criterio greedy es más estricto.

**9. ¿Por qué es válido cortar el cálculo antes (short-circuit)?**
→ 💭 Suma de términos no negativos: monótona. Si ya pasó ℓ(b_worst), el candidato se descartaría igual [§3.2.2 · p.4]. Ahorra ~30%.

**10. ¿Cómo obtienen logprobs si la API no los da para el prompt?**
→ Logit bias + top-5 + fórmula de corrección [Ap. B · p.12]. Explicar la derivación (sección 6.3) y por qué `−log p̂` deja al token en ~0.5. Mencionar que OpenAI cambió la API en marzo 2024 [§3.2.1 · p.4].

**11. ¿Por qué el costo crece super-linealmente con el largo del target?**
→ Un solo logit bias por generación obliga a puntuar token por token, prefijando el target: p·t + t(t+1)/2 tokens [Ap. B · p.12], [§4.3 · p.6-7].

**12. ¿Por qué usan un modelo base no alineado (Mistral 7B) como proxy? ¿No debería parecerse al objetivo?**
→ No sirve para transferencia pura [§4.3 · p.6], pero acá solo filtra; la decisión la toma la loss real. 💭 Además, un modelo base modela bien "qué texto es probable", que es justamente lo que mide la loss.

**13. ¿Cómo funciona la variante sin proxy? ¿Cómo puede ganarle a GCG sin gradientes?**
→ Sondear una sustitución por posición, elegir la mejor posición, concentrar B′ ≪ B intentos ahí [§3.3 · p.4-5]. 💭 GCG reparte ~B/l intentos por posición; asignar mejor las queries compensa perder el gradiente.

**14. La inicialización con el target repetido, ¿no es "hacer trampa"?**
→ No: el atacante conoce el target, así que es información legítima. Pero relativizar: 28% sale gratis [§4.3 · p.6] y sin ella solo 2/20 [§4.3 · p.7]. El algoritmo lleva de 28% a ~80-86% con poco presupuesto. Los autores reconocen que la dependencia de la inicialización muestra que los ataques en NLP aún son débiles [§5 · p.9].

**15. ¿Qué problemas de tokenización aparecieron?**
→ Tokenizaciones no canónicas y tokenizers distintos entre proxy y objetivo; re-tokenizan [Ap. C · p.12].

### C. Resultados y evidencia

**16. Interpreten la Fig. 2. ¿Por qué hay un salto al principio?**
→ Es la inicialización (28%, línea punteada). Después sube rápido y se aplana: 79.6% a USD 0.10, 86% a USD 0.20 [§4.3 · p.6].

**17. ¿Por qué cae el éxito con targets largos? ¿Cómo lo verifican?**
→ Dos explicaciones y el experimento con 40 tokens [§4.3 · p.7]. 💭 Señalar que el experimento no distingue entre ambas explicaciones.

**18. ¿Qué nos dice la Fig. 1 sobre escala y alineamiento?**
→ Vicuna más grande resiste más; Llama 2 7B resiste más que Vicuna 33B [§4.1 · p.5]. 💭 El tipo de alineamiento importa más que la escala (conexión con RLHF, C3).

**19. ¿Por qué usan la suma de scores como loss en moderación?**
→ No hace falta conocer los umbrales por categoría, que no son públicos y pueden cambiar [§4.5 · p.8]. 💭 Es un proxy (Goodhart), pero el éxito se verifica con los flags reales.

**20. ¿Cómo cambia la lectura sabiendo que 34% ya no se marca sin ataque?**
→ La mejora real es de ~34% a ~99% (universal, 20 tokens) [§4.5 · p.8]. Siempre comparar con la línea "No attack" de las figuras.

**21. Universal vs no universal: ¿cuál le conviene a un atacante?**
→ El universal es más caro de encontrar y tiene que generalizar [§4.5 · p.8], pero se hace una vez y sirve para cualquier texto. El no universal es barato por string (83.8-91.4% en 10 iteraciones) [§4.5 · p.9].

**22. ¿Cómo afecta el no determinismo a las conclusiones?**
→ Ap. E: el ruido (0.068) es chico frente a la diferencia típica en el buffer (≥ 3), así que la optimización no se ve afectada; pero ~10% de los éxitos no se reproducen [Ap. E · p.13].

### D. Pensamiento crítico

**23. ¿Qué supuestos hace el modelo de amenaza? ¿Siguen valiendo hoy?**
→ Sección 9.1. La técnica de logprobs ya no funciona igual en OpenAI [§3.2.1], [Ap. B]. 💭 El argumento conceptual (query access alcanza si hay señal de loss) se mantiene; la receta concreta depende de la API.

**24. ¿Es justa la comparación con AutoDAN?**
→ Parcialmente: tarea distinta a la de diseño, parámetros por defecto y sufijos más largos [§4.2 · p.6]. Los autores declaran que lo usan para ilustrar la dificultad de los harmful strings [§4.2 · p.5]. Relacionar con baseline nerfing [C4 · diap. 45].

**25. ¿Qué le falta a la Tabla 1?**
→ GCQ 7B→13B y GCQ contra Mistral Instruct / Gemma 2; 7B→7B es casi white-box.

**26. ¿Qué defensas funcionarían y cuáles no?**
→ No: romper solo la transferibilidad [§5 · p.9]. Sí, contra esta versión: filtros de perplejidad [Ap. D · p.13] y quitar la estimación de logprobs [Ap. D · p.13]. 💭 También: detectar patrones de queries, rate limiting, entrenamiento adversarial; siempre pensar en ataques adaptativos.

**27. Hipotético: la API solo devuelve texto, sin logprobs. ¿Qué harían?**
→ Es trabajo futuro para los autores [Ap. D · p.13]. 💭 Estimar probabilidades muestreando muchas veces (caro) o usar una señal binaria de éxito, como los ataques basados en decisión de visión [5].

**28. Hipotético: el proveedor usa temperatura > 0. ¿Protege?**
→ 💭 Parcialmente: el éxito está definido con greedy [§3.1]. Pero si ℓ es muy baja, e^(−ℓ) es alta y el target igual sale con alta probabilidad. Se podría optimizar directamente la probabilidad bajo sampling.

**29. ¿Emitir un string exacto es realmente dañino?**
→ Depende del despliegue. En un chat, el daño es reputacional; con agentes y plugins, la salida es una acción [§1 · p.1]. 💭 Conectar con tool use [C4 · diap. 18] y prompt injection [C6 · diap. 8-9].

**30. ¿El mérito es del algoritmo o de la inicialización?**
→ De ambos; ver pregunta 14. El algoritmo agrega ~50-60 puntos sobre la inicialización con poco presupuesto [§4.3 · p.6].

### E. Relación con la materia

**31. ¿Cómo se relaciona con "¿Es RLHF robusto?" de la clase?**
→ La clase muestra BadLlama (necesita pesos) y GCG [C6 · diap. 12-14]. Este paper demuestra que ni siquiera hacen falta pesos ni un buen surrogate. Shoggoth [C3 · diap. 60].

**32. ¿Es un jailbreak por tensión de objetivos o por generalización inadecuada?**
→ 💭 Generalización inadecuada: los sufijos son inputs fuera de la distribución del entrenamiento de seguridad [C6 · diap. 3-6].

**33. ¿Qué tiene que ver con transfer learning?**
→ Misma idea de transferibilidad (features compartidos, C3 · diap. 32); la Fig. 1b muestra que transfiere mejor entre modelos parecidos [§4.1 · p.5].

**34. ¿Qué pasaría con un modelo de razonamiento?**
→ No se evalúa. 💭 El CoT oculto y las APIs sin logprobs dificultan medir la loss; por otro lado, el razonamiento podría "detectar" o no el input raro. Buena extensión.

**35. ¿Dónde aparece Goodhart?**
→ 💭 En el alineamiento como proxy de objetivos humanos y en la loss de moderación (suma de scores) [C6 · diap. 24].

**36. ¿Qué relación tiene con la tokenización?**
→ Espacio discreto V^m, re-tokenización [Ap. C], costo en tokens [Ap. B], y la clase recomienda GCG en "riesgos de seguridad asociados a tokenización" [C2 · diap. 63].

### F. Ética y extensiones

**37. ¿Está bien publicar un ataque así?**
→ Los autores reconocen que se puede usar para dañar, pero esperan que motive cautela e investigación en robustez [§5 · p.9]. 💭 Argumento clásico de seguridad: los defensores necesitan conocer los ataques; la técnica de logprobs ya estaba mitigada cuando salió la versión final. No encontramos en el texto una mención explícita a un proceso de *disclosure* con OpenAI; si preguntan, decirlo con cautela.

**38. ¿Qué experimento agregarían?**
→ Elegir 2 o 3 de la sección 12 y justificar qué pregunta responderían (p. ej. separar las dos explicaciones del largo del target: probar prompts largos **sin** la inicialización por repetición).

---

## 14. Checklist para el día del oral

- [ ] Todos pueden dar el pitch de la sección 0 sin leer.
- [ ] Todos pueden dibujar el loop de GCQ (sección 6.1) y explicar buffer, proxy y short-circuit.
- [ ] Al menos dos saben derivar la fórmula del logit bias y explicar `−log p̂`.
- [ ] Todos saben de memoria: 574 / 28% / 79.6% a USD 0.10 / 97.9% ≤ 20 tokens / 34% línea base en moderación / 99.2% universal / ~90% reproducción.
- [ ] Cada integrante tiene 2 críticas propias listas (sección 9.3) y 1 extensión (sección 12).
- [ ] Todos conectan al menos una idea con cada clase (sección 10), sobre todo C6 diap. 12-14 y C2 diap. 63.
- [ ] Separar siempre "lo que dice el paper" de "nuestra interpretación".
- [ ] Si no sabemos algo: decirlo, dar la hipótesis más razonable y cómo lo verificaríamos. Reconocer incertidumbre también se evalúa.
