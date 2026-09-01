# Experimentos — Ejercicio 2

Registro de cada experimento: arquitectura, justificación, resultados y qué se decide cambiar para el siguiente. Las decisiones de diseño más generales (split, encoding de features, por qué Encoder-only, por qué fusión tardía) están en [`Notas.md`](Notas.md) — acá solo se repite lo que hace falta para justificar la config concreta de cada corrida.

Convención: cómputo en `train.py`/`model.py`/`run_experiment<n>.py` (guardan CSV crudo en `output/`), gráficos en `plot_experiment<n>.py` (leen esos CSV, nunca reentrenan) — ver regla de separación cómputo/gráficos en el `CLAUDE.md` del TP.

## Experimento 1 — Transformer de texto puro, config mínima

### Alcance

Predecir `bought` usando **únicamente** `title`+`description` (tokenizados, ver `ejercicio1/Notas.md`) a través de un Transformer Encoder-only. Sin features tabulares todavía — decisión explícita del equipo para aislar el comportamiento del Transformer antes de complicar la arquitectura con la fusión (eso queda para un experimento posterior, no es este).

### Arquitectura ([`model.py`](model.py))

| Pieza | Valor | Por qué |
|---|---|---|
| Embedding | `d_model=16` | Bien por debajo de 100 (sugerencia de la consigna) y del vocabulario (412 tokens: 410 palabras + `<PAD>` + `<UNK>`) — fuerza al modelo a comprimir agresivamente y probar si la atención sola alcanza a encontrar señal (ej. el tag de reputación en `title` identificado en el EDA de `ejercicio1`) sin apoyo tabular. |
| `n_heads` | 1 | El mínimo que sigue siendo "multi-head attention" (con 1 head es self-attention simple). Se deja explícitamente afuera de este experimento para que 2/4/8 heads sea la variable a aislar en el próximo. |
| `n_layers` (encoders apilados) | 1 | Un solo bloque Encoder — self-attention + residual + LayerNorm, MLP feed-forward + residual + LayerNorm (`nn.TransformerEncoderLayer` de PyTorch, sin nada hecho a mano). Es la unidad más chica que la clase define como "Encoder"; apilar más queda para otro experimento. |
| `dim_feedforward` | 64 | Sigue la proporción ~4x `d_model` del paper original (512/128≈4x) pero en una escala mucho más chica. También queda como dial para variar después. |
| Positional encoding | Senoidal | La que se explica en el material de cátedra (Clase 1) para el Transformer original. La profesora nombra explícitamente "qué positional encoding pruebo" como un dial de ablación (`consigna.VTT`) — comparar contra otra variante queda pendiente, no se decide acá. |
| Pooling de salida | Mean-pooling sobre tokens no-pad | **Chequeado contra el material y no está enseñado tal cual** (ni `[CLS]` ni mean-pooling aparecen como técnica de pooling para clasificación en `transformers.VTT`/`embeddings_*.VTT`). Se optó por mean-pooling porque se apoya en algo que sí se explicó: Marina describe la atención misma como "un promedio ponderado" de tokens (`embeddings_1.VTT`, ~L889-930) — mean-pooling es el caso de pesos uniformes de esa misma idea. `[CLS]` en cambio solo aparece atado a cómo se arma el input de pre-entrenamiento de BERT (NSP+MLM, `embeddings_2.VTT` ~L1229), no a "usar su embedding para clasificar" — eso hubiese sido importar una convención de uso de BERT no explicada en clase. Además, mean-pooling no requiere tocar el vocabulario/tokenización ya cerrado de `ejercicio1`. |
| Cabeza de salida | `Linear(16 → 1)` directo | Sin MLP intermedio — el pooled vector va directo a la predicción, la versión más mínima posible. Si hace falta más capacidad ahí es un cambio a probar más adelante. |
| Dropout | 0.1 | Regularización liviana estándar, no es un dial que interese variar por ahora. |
| Parámetros totales | 9.889 | Confirma que el modelo es efectivamente chico. |

### Configuración de entrenamiento ([`train.py`](train.py), [`run_experiment1.py`](run_experiment1.py))

Adam (`lr=1e-3`), `batch_size=128`, 20 épocas, **3 semillas (0, 1, 2)** promediadas — según la aclaración de `consigna.VTT` de no reportar una sola corrida. Métricas evaluadas sobre train (en modo eval, sin dropout) y valid en cada época, para poder comparar ambas curvas y diagnosticar over/underfitting (test no se toca, ver `Notas.md`). Sin threshold (PR-AUC/ROC-AUC), según lo indicado en la consigna.

### Resultados

Mejor época de cada semilla (`output/experiment1_results.csv`):

| seed | best_epoch | valid PR-AUC | valid ROC-AUC | train PR-AUC | gap PR-AUC (train-valid) |
|---|---|---|---|---|---|
| 0 | 19 | 0,660 | 0,940 | 0,775 | 0,114 |
| 1 | 15 | 0,706 | 0,962 | 0,730 | 0,024 |
| 2 | 18 | 0,697 | 0,960 | 0,734 | 0,037 |
| **media ± std** | — | **0,688 ± 0,024** | **0,954 ± 0,012** | — | **0,058 ± 0,049** |

Curvas de entrenamiento (media ± desvío sobre las 3 semillas), `output/experiment1_curves.png`:

![Experimento 1 — curvas de entrenamiento](output/experiment1_curves.png)

### Análisis

- **Hay señal fuerte en el texto solo.** PR-AUC de valid (0,688) está muy por encima de 0,130 (la prevalencia de `bought` — lo que daría un clasificador que ignora completamente el input) y ROC-AUC (0,954) muy por encima de 0,5. Esto confirma con evidencia empírica la hipótesis de `ejercicio1`: el tag de reputación embebido en `title` (y en general el contenido de `title`/`description`) es predictivo del comportamiento de compra, y el Transformer lo está aprovechando incluso sin ninguna feature tabular.
- **Hay overfitting leve, pero desigual entre semillas.** El gap de PR-AUC train-valid en la mejor época varía mucho (0,114 en seed 0 vs. 0,024-0,037 en seeds 1 y 2) — el desvío del gap (0,049) es casi tan grande como su media (0,058). En el gráfico de PR-AUC se ve que las curvas de train y valid se separan de forma sostenida a partir de la época ~5, mientras que en ROC-AUC casi no hay separación visible. Esto es consistente con que ROC-AUC es menos sensible al desbalance de clases (13% positivos) que PR-AUC — reafirma la decisión de `Notas.md` de priorizar PR-AUC como métrica de diagnóstico principal.
- **No hay una época de corte clara.** La mejor época de valid varía por semilla (19, 15, 18) sin un plateau limpio — con este dataset chico (7012 filas de train) y un modelo de ~10K parámetros, 20 épocas ya alcanzan para empezar a sobreajustar en al menos una semilla.
- El modelo es chico a propósito (9.889 parámetros) — cumple con "arrancar chico" antes de escalar `d_model`/heads/layers.

### Qué cambió respecto al Experimento 1

Se subió **`n_heads` de 1 a 2**, manteniendo todo lo demás igual (`d_model=16`, `n_layers=1`, `dim_feedforward=64`, mismo pooling/positional encoding/cabeza de salida) — para aislar el efecto de multi-head attention de forma controlada, sin mezclarlo con un cambio de capacidad (`d_model`) o de profundidad (`n_layers`).

## Experimento 2 — `n_heads`: 1 → 2

### Arquitectura ([`model.py`](model.py))

Idéntica a la del Experimento 1 (ver tabla arriba) salvo `n_heads=2`. Con `d_model=16`, esto reparte la atención en 2 subespacios de 8 dimensiones cada uno en vez de uno solo de 16 — la pregunta que responde este experimento es si separar distintos "tipos" de relación entre palabras (ej. el patrón del tag de reputación vs. el resto del texto) ayuda a capturar mejor la señal, o si con un texto tan corto (`MAX_LEN=45`, vocabulario de 410 palabras) un único head ya alcanza.

**Dato relevante para la ablación**: el conteo de parámetros no cambió (9.889, igual que el Experimento 1). Esto es esperable, no un error: las proyecciones Q/K/V/salida de la atención tienen tamaño `d_model × d_model` sin importar en cuántos heads se reparta esa dimensión — heads no es un dial que agregue parámetros, es un dial que **reparte** la misma capacidad ya existente. Es una distinción importante para poder explicar la arquitectura en la presentación.

### Configuración de entrenamiento

Misma que el Experimento 1: Adam (`lr=1e-3`), `batch_size=128`, 20 épocas, 3 semillas (0, 1, 2).

### Resultados

Mejor época de cada semilla (`output/experiment2_results.csv`):

| seed | best_epoch | valid PR-AUC | valid ROC-AUC | train PR-AUC | gap PR-AUC (train-valid) |
|---|---|---|---|---|---|
| 0 | 20 | 0,695 | 0,952 | 0,793 | 0,098 |
| 1 | 20 | 0,689 | 0,959 | 0,769 | 0,080 |
| 2 | 19 | 0,680 | 0,960 | 0,749 | 0,069 |
| **media ± std** | — | **0,688 ± 0,007** | **0,957 ± 0,004** | — | **0,082 ± 0,015** |

Comparación directa contra el Experimento 1:

| | Exp. 1 (`n_heads=1`) | Exp. 2 (`n_heads=2`) |
|---|---|---|
| valid PR-AUC (media ± std) | 0,688 ± 0,024 | 0,688 ± 0,007 |
| valid ROC-AUC (media ± std) | 0,954 ± 0,012 | 0,957 ± 0,004 |
| gap PR-AUC (media ± std) | 0,058 ± 0,049 | 0,082 ± 0,015 |

Curvas de entrenamiento, `output/experiment2_curves.png`:

![Experimento 2 — curvas de entrenamiento](output/experiment2_curves.png)

### Análisis

- **La performance pico no cambió.** PR-AUC de valid queda prácticamente idéntico (0,688 en ambos experimentos) y ROC-AUC sube apenas 0,003 (dentro del ruido). Con `d_model=16` en un texto corto, repartir la atención en 2 heads de 8 dimensiones en vez de 1 de 16 no le agregó capacidad predictiva al modelo — coherente con que heads no suma parámetros, solo reparte los que ya había.
- **Lo que sí cambió es la estabilidad entre semillas.** El desvío estándar de PR-AUC bajó de 0,024 a 0,007, y el de ROC-AUC de 0,012 a 0,004 — con 2 heads, las 3 corridas convergen a resultados mucho más parecidos entre sí. En el Experimento 1 una sola semilla (seed 0) se alejaba bastante del resto (gap de 0,114 vs. 0,02-0,04 en las otras dos); acá las 3 semillas quedan agrupadas en gaps de 0,07-0,10. Una hipótesis razonable: con un solo head, la inicialización aleatoria de esos mismos 16×16 parámetros puede caer en soluciones más o menos afortunadas de cómo repartir la atención; con 2 heads, forzar de entrada una partición en subespacios más chicos reduce esa variabilidad entre semillas (aunque esto queda como hipótesis a confirmar, no una conclusión cerrada con solo 3 semillas).
- **El overfitting es levemente peor en promedio (0,058 → 0,082), pero mucho más consistente.** Ya no hay una semilla que se dispare sola (std del gap bajó de 0,049 a 0,015) — las 3 corridas ahora sobreajustan de forma pareja. Esto sugiere que el gap de overfitting depende más de la inicialización/semilla que de si hay 1 o 2 heads en este rango tan chico de `d_model`.
- **Conclusión de este dial**: en esta configuración (`d_model=16`, texto corto), `n_heads` no es el cuello de botella de performance — es más un dial de estabilidad de entrenamiento que de capacidad. Los próximos dials a probar (`n_layers`, `d_model`) son candidatos más prometedores para mover el PR-AUC pico.

### Qué cambió respecto al Experimento 1

Se barrieron varios valores de **`n_layers`** (encoders apilados) en una sola tanda, en vez de probar un valor por experimento, para acelerar la iteración: **2, 4 y 8** (valores que la profesora menciona explícitamente en `transformers.VTT` — "apilan N encoders... paper original 6; prueban 2/4/8 también"), más el valor **1** reusado sin reentrenar del Experimento 1 como referencia. El resto de la arquitectura queda fijo en la base del Experimento 1 (`n_heads=1`, `d_model=16`, `dim_feedforward=64`).

**Cambio de metodología a partir de acá**: pedido explícito de correr varios valores y quedarse con el mejor, en vez de aislar cada dial contra la base fija del Experimento 1 uno por uno. A partir del Experimento 4, cada nuevo dial se prueba sobre **la mejor configuración encontrada hasta el momento** (búsqueda greedy / coordinate-ascent: se fija lo ya decidido y se optimiza el siguiente dial), no contra la base original en paralelo. Esto es más rápido y da directamente una arquitectura final, a costa de que los dials ya no quedan perfectamente aislados entre sí (ej. si el mejor `d_model` resultara distinto partiendo de `n_layers=1` en vez de `n_layers=2`, no lo vamos a ver) — trade-off consciente entre rigurosidad del estudio de ablación y tiempo disponible, a explicar así en la presentación.

## Experimento 3 — barrido de `n_layers`: 1 (Exp. 1) / 2 / 4 / 8

### Arquitectura ([`model.py`](model.py))

Igual que el Experimento 1 salvo `n_layers` variable. Apilar encoders es distinto de agregar heads: cada capa nueva agrega un set completo de proyecciones Q/K/V/salida + feed-forward propio (no reparte parámetros existentes, los suma) — por eso acá sí se espera que cambie tanto la capacidad como el riesgo de overfitting, a diferencia de lo que se vio con `n_heads` en el Experimento 2.

### Configuración de entrenamiento

Misma que los experimentos anteriores: Adam (`lr=1e-3`), `batch_size=128`, 20 épocas, 3 semillas (0, 1, 2) por valor de `n_layers`.

### Resultados

Media ± std sobre 3 semillas, por valor de `n_layers` (`output/experiment3_results.csv`):

| `n_layers` | n_params | valid PR-AUC | valid ROC-AUC | gap PR-AUC |
|---|---|---|---|---|
| 1 (Exp. 1) | 9.889 | 0,688 ± 0,024 | 0,954 ± 0,012 | 0,058 ± 0,049 |
| **2** | 13.169 | **0,696 ± 0,018** | 0,953 ± 0,012 | **0,028 ± 0,042** |
| 4 | 19.729 | 0,651 ± 0,025 | 0,950 ± 0,006 | 0,065 ± 0,032 |
| 8 | 32.849 | 0,650 ± 0,053 | 0,947 ± 0,009 | 0,055 ± 0,035 |

Barrido completo, `output/experiment3_sweep.png` (círculo = mejor valor por PR-AUC de valid):

![Experimento 3 — barrido de n_layers](output/experiment3_sweep.png)

### Análisis

- **`n_layers=2` gana en PR-AUC de valid** (0,696, contra 0,688 de la base) **y además generaliza mejor**: tiene el gap de overfitting más chico de los 4 valores probados (0,028, la mitad que la base). No es solo "el mejor número" — es una mejora consistente en dos frentes a la vez, lo cual la hace una elección sólida.
- **Matiz**: en ROC-AUC, `n_layers=1` queda apenas arriba (0,954 vs. 0,953) — diferencia mínima, dentro del solapamiento de los desvíos estándar. No contradice la elección de `n_layers=2` (PR-AUC sigue siendo la métrica de diagnóstico principal por el desbalance de clases, ver discusión de `Notas.md`), pero vale aclararlo: no es que `n_layers=2` gane en todo, gana en la métrica que más importa acá.
- **Más profundidad no ayudó — de hecho, empeoró.** Con `n_layers=4` y `8` tanto PR-AUC como ROC-AUC bajan de forma monótona, y la variabilidad entre semillas crece fuerte en `n_layers=8` (std de PR-AUC 0,053, más del doble que en `n_layers=2`). Con un dataset chico (7.012 filas de train) y secuencias cortas (`MAX_LEN=45`), apilar más encoders agrega parámetros (hasta 32.849 en `n_layers=8`, 3,3x la base) sin que haya más señal para aprovecharlos — el resultado es un modelo más difícil de optimizar de forma estable en pocas épocas, no uno con más capacidad útil.
- **Conclusión de este dial**: `n_layers=2` queda como la configuración ganadora. A partir de acá, la arquitectura base para seguir ablacionando es `n_heads=1`, `n_layers=2`, `d_model=16`, `dim_feedforward=64`.

### Qué cambió respecto al Experimento 3

Con `n_heads` (Exp. 2, sin efecto en el pico) y `n_layers` (Exp. 3, ganador `n_layers=2`) ya explorados, quedan los dos dials de capacidad: **`d_model`** (ancho del stream que atraviesa todo el bloque -- embeddings, Q/K/V, entrada/salida de cada sub-capa) y **`dim_feedforward`** (ancho interno del MLP posicional de cada encoder, ver discusión de por qué están relacionados más abajo). Se corrieron **por separado** (Experimento 4: solo `d_model`; Experimento 5: solo `dim_feedforward`) para no mezclar sus efectos, con el plan de recién combinarlos en un tercer experimento si la evidencia lo pide -- ver el análisis conjunto al final.

Base fija para ambos (la ganadora del Experimento 3): `n_heads=1`, `n_layers=2`.

## Experimento 4 — barrido de `d_model`: 8 / 16 (Exp. 3) / 32 / 64

`dim_feedforward=64` fijo (el valor de la base). Valores probados dentro del límite `<100` que sugiere la consigna, subiendo de a poco desde el mínimo (8) explorado hasta ahora.

Media ± std sobre 3 semillas (`output/experiment4_results.csv`):

| `d_model` | n_params | valid PR-AUC | valid ROC-AUC | gap PR-AUC |
|---|---|---|---|---|
| 8 | 6.137 | 0,665 ± 0,016 | 0,947 ± 0,003 | 0,004 ± 0,039 |
| 16 (Exp. 3) | 13.169 | 0,696 ± 0,018 | 0,953 ± 0,011 | 0,028 ± 0,042 |
| 32 | 30.305 | 0,710 ± 0,022 | **0,964 ± 0,001** | 0,044 ± 0,047 |
| **64** | 76.865 | **0,724 ± 0,021** | 0,962 ± 0,004 | 0,012 ± 0,014 |

![Experimento 4 — barrido de d_model](output/experiment4_sweep.png)

**PR-AUC sube de forma monótona en todo el rango probado** -- todavía no se ve una meseta ni una caída, `d_model=64` es el mejor de los 4 y el gap de overfitting ahí es chico (0,012, comparable al de `d_model=8`). Esto sugiere que el modelo venía short de capacidad de representación en `d_model=16` y que **el techo de este dial probablemente esté más allá de 64** -- diferente de lo que pasó con `n_layers`, donde más no ayudaba. En ROC-AUC el pico está en `d_model=32` (0,964) y baja levemente en 64 (0,962), aunque la diferencia es chica.

## Experimento 5 — barrido de `dim_feedforward`: 16 / 32 / 64 (Exp. 3) / 128

`d_model=16` fijo (la base, no la ganadora de d_model del Experimento 4 -- ver nota metodológica abajo).

Media ± std sobre 3 semillas (`output/experiment5_results.csv`):

| `dim_feedforward` | n_params | valid PR-AUC | valid ROC-AUC | gap PR-AUC |
|---|---|---|---|---|
| 16 | 10.001 | 0,680 ± 0,051 | **0,954 ± 0,007** | 0,064 ± 0,044 |
| 32 | 11.057 | 0,674 ± 0,035 | 0,954 ± 0,007 | 0,063 ± 0,027 |
| **64 (Exp. 3)** | 13.169 | **0,696 ± 0,018** | 0,953 ± 0,011 | 0,028 ± 0,042 |
| 128 | 17.393 | 0,688 ± 0,017 | 0,953 ± 0,006 | 0,049 ± 0,058 |

![Experimento 5 — barrido de dim_feedforward](output/experiment5_sweep.png)

**Acá sí hay un pico interior**, no una tendencia monótona: `dim_feedforward=64` (la proporción 4x sobre `d_model=16` del paper original) gana, y tanto 32 (2x, por debajo) como 128 (8x, por encima) quedan peor. ROC-AUC prácticamente no se mueve en todo el rango (0,953-0,954, dentro del ruido). Esto confirma la intuición de la sección anterior de `Notas.md`: `dim_feedforward` no es "cuanto más mejor" de forma libre, tiene un punto óptimo relacionado con `d_model` -- consistente con la idea de que un feed-forward angosto es un cuello de botella y uno demasiado ancho agrega parámetros sin beneficio claro (y algo de riesgo de overfitting: el gap sube a 0,049 en `dim_feedforward=128`).

### Análisis conjunto -- ¿hace falta un Experimento 6 combinando ambos?

Este era justamente el punto que se planteó antes de correr el 4 y el 5: si los dos muestran que sus valores óptimos dependen uno del otro, hay que combinarlos.

La evidencia apunta a que sí, por lo siguiente: en el Experimento 5 (con `d_model=16` fijo) el óptimo de `dim_feedforward` fue 64 -- exactamente la proporción 4x. Pero el Experimento 4 mantuvo `dim_feedforward=64` fijo mientras subía `d_model`, así que en el punto ganador (`d_model=64`) la proporción real terminó siendo **1x**, no 4x -- y aun así fue el mejor resultado de todo lo corrido hasta ahora (PR-AUC 0,724). Dos lecturas posibles, y no podemos distinguirlas con lo que corrimos:

1. La proporción 4x importa, y `d_model=64` con `dim_feedforward=256` (manteniendo el 4x) daría un resultado todavía mejor que 0,724.
2. La proporción no es tan determinante como sugería el Experimento 5 a `d_model=16` -- lo que importa es la capacidad absoluta de `d_model`, y `dim_feedforward=64` ya alcanza como "suficiente" ancho de MLP en el rango que probamos.

No hay forma de saber cuál es cierta sin probar directamente `d_model=64` con `dim_feedforward` escalado. **Conclusión: sí hace falta el Experimento 6.**

### Qué cambió respecto a los Experimentos 4 y 5

Se probó puntualmente si escalar `dim_feedforward` junto con `d_model` (manteniendo la proporción 4x) mejora los mejores puntos del Experimento 4, en vez de un grid completo (4x4 combinaciones x 3 semillas = 48 corridas habría sido demasiado para lo que responde esta pregunta puntual).

## Experimento 6 — ¿escalar `dim_feedforward` junto con `d_model`?

Dos combinaciones nuevas, manteniendo `n_heads=1`, `n_layers=2`: `d_model=32` con `dim_feedforward=128` (4x), y `d_model=64` con `dim_feedforward=256` (4x). Comparadas contra las mismas filas de `d_model=32` y `64` del Experimento 4 (`dim_feedforward=64` fijo ahí, o sea 2x y 1x respectivamente).

Media ± std sobre 3 semillas (`output/experiment6_results.csv`):

| `d_model` | `dim_feedforward` | proporción | n_params | valid PR-AUC | valid ROC-AUC | gap PR-AUC |
|---|---|---|---|---|---|---|
| 32 | 64 (Exp. 4) | 2x | 30.305 | 0,710 ± 0,022 | 0,964 ± 0,001 | 0,044 ± 0,047 |
| 32 | 128 | 4x | 38.625 | 0,713 ± 0,025 | 0,962 ± 0,004 | 0,049 ± 0,004 |
| 64 | 64 (Exp. 4) | 1x | 76.865 | **0,724 ± 0,021** | 0,962 ± 0,004 | **0,012 ± 0,014** |
| 64 | 256 | 4x | 126.401 | 0,721 ± 0,025 | 0,963 ± 0,003 | 0,099 ± 0,070 |

![Experimento 6 — comparación](output/experiment6_comparison.png)

### Análisis

- **PR-AUC casi no cambia al escalar `dim_feedforward`** en ninguno de los dos `d_model` (32: 0,710→0,713; 64: 0,724→0,721) — las diferencias están completamente dentro del desvío estándar de cada punto (~0,02). El panel izquierdo del gráfico lo muestra claro: las barras son prácticamente idénticas en altura.
- **Donde sí hay un efecto real y grande es en el overfitting**, y va en la dirección contraria a lo esperado: en `d_model=64`, pasar de `dim_feedforward=64` a `256` casi **octuplica** el gap de generalización (0,012 → 0,099), con además mucha más varianza entre semillas (std 0,070). Son 126.401 parámetros contra 76.865 filas efectivas de entrenamiento un orden de magnitud menor — el feed-forward más ancho agrega capacidad que el modelo termina memorizando en vez de generalizar, sin que la performance de valid lo compense.
- **Conclusión: gana la hipótesis 2.** La proporción 4x que fue determinante en el Experimento 5 (a `d_model=16` fijo) **no se sostiene** al escalar `d_model` — ahí lo que pesa es la capacidad absoluta de `d_model`, y `dim_feedforward=64` ya alcanza como ancho de MLP en todo el rango probado (16 a 64). Escalarlo junto con `d_model` no ayuda y en el punto más grande directamente perjudica la generalización.
- **Este dial queda cerrado**: `d_model=64`, `dim_feedforward=64` es la configuración ganadora (PR-AUC valid 0,724 ± 0,021, el mejor resultado de todos los experimentos corridos hasta ahora). La arquitectura base para seguir es `n_heads=1`, `n_layers=2`, `d_model=64`, `dim_feedforward=64`.

### Qué cambiar en el próximo experimento (propuesta a confirmar)

Con `n_heads`, `n_layers`, `d_model` y `dim_feedforward` ya explorados sobre el Transformer de texto puro, los dials que quedan de la lista de `Notas.md` son de otro tipo: el **positional encoding** (senoidal vs. otra variante, o sacarlo) y el **pooling** (mean-pooling vs. otras opciones) -- ambos dials que quedaron pendientes desde el Experimento 1. Alternativamente, ya se agotó lo razonable del Transformer solo con texto y correspondería pasar a la **fusión tardía con las features tabulares** (la arquitectura completa que se va a entregar, ver `Notas.md`), usando esta config ganadora como la rama de texto.

Como siempre, esto es una propuesta — lo confirmamos antes de seguir.
