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

### Qué cambiar en el próximo experimento (propuesta a confirmar)

Para mantener el estudio de ablación limpio (poder atribuir cada cambio de métrica a un solo factor), la propuesta es seguir variando **un dial a la vez desde la misma base del Experimento 1** (`n_heads=1`, `d_model=16`, `n_layers=1`, `dim_feedforward=64`) en vez de encadenar cambios sobre el Experimento 2 — así el Experimento 3 sigue siendo comparable directamente contra el Experimento 1, no contra una combinación de heads+otro cambio.

**Propuesta: `n_layers` de 1 a 2** (volviendo a `n_heads=1`, manteniendo `d_model=16`, `dim_feedforward=64`), para probar si apilar un segundo bloque Encoder (permitiendo una segunda ronda de atención + feed-forward sobre la salida ya contextualizada del primero) mueve el PR-AUC pico más que lo que movió `n_heads` — a diferencia de heads, apilar capas sí agrega parámetros nuevos (un segundo set completo de Q/K/V/feed-forward), así que también hay que vigilar si el overfitting empeora más rápido.

Como siempre, esto es una propuesta — si prefieren probar `d_model` o `dim_feedforward` primero, o combinar heads=2 con otro cambio en vez de volver a la base, lo charlamos antes de correr el Experimento 3.
