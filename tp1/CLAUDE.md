# CLAUDE.md — TP1

Contexto específico del **Trabajo Práctico 1**. Ver el `CLAUDE.md` de la raíz para las reglas generales del monorepo (no coautor, no técnicas fuera de clase, idioma/estilo).

## Contexto del TP

Trabajo Práctico 1 — **Large Language Models (73.69), 2026** — cátedra "Transformers".

**Objetivo:** predecir el **Buy Through Rate (BTR)** (productos comprados / productos impresos en resultados de búsqueda) para un e-commerce de supermercado, usando un sistema que incluya al menos un modelo basado en la arquitectura **Transformer**.

Dataset: `docs/supermarket_products.csv` — eventos históricos de búsqueda (título, descripción, precio, categoría, timestamp, filtros de búsqueda, `cart`/`bought`, marca, tamaño de envase, ingredientes, alérgenos, `nutrition_score`, país de origen, etc.). Ver el enunciado completo en `docs/DeepLearningTP0.pdf` para la definición exacta de cada campo.

Consigna oficial: `docs/DeepLearningTP0.pdf`. Tres ejercicios:

1. **EDA y formulación del problema**: definir la variable objetivo (BTR), caracterizar los datos, elegir features, y decidir el preprocesamiento/encoding de cada una.
2. **Desarrollo del sistema**: diseñar e implementar el modelo (con Transformer incluido), justificando dónde y por qué se usa esa arquitectura. Incluye split train/valid/test, diseño de experimentos (arquitectura chica primero, `d_model < 100`, escalar según cómputo disponible), evaluación con PR-AUC/ROC-AUC (sin necesidad de threshold), y un **estudio de ablación** de los módulos de la arquitectura.
3. **Personalización** (teórico): una slide explicando cómo incorporar personalización de usuario al BTR.

Entrega: repo con `README.md` + hash de commit + presentación (25-30 min) por Campus.

## Material de cátedra

En `docs/` — revisar antes de tomar decisiones de arquitectura:

- `Clase 1 - Introducción a Transformers.pdf` + `transformers.VTT`
- `Clase 2 - Transformación de Texto a Embeddings(1).pdf` + `embeddings_1.VTT`, `embeddings_2.VTT`
- `Clase 2 - Demo.pdf` + `demo_transformers.VTT`
- `Copia de Clase 3 - Transfer Learning & Finetuning.pdf` + `finetuning.VTT`
- `consigna.VTT` — clase donde se presentó el TP (aclaraciones del profesor sobre el enunciado)
- `DeepLearningTP0.pdf` — enunciado oficial

Si el enunciado sugiere algo (ej. "investigar one-hot encoding") es una sugerencia, no una obligación de exclusividad — pero no reemplazar por técnicas no vistas sin avisar.

**No es necesario fine-tuning/transfer learning.** Ni el PDF ni `consigna.VTT` lo exigen — el único requisito es incluir al menos un modelo Transformer. La Clase 3 (`Copia de Clase 3 - Transfer Learning & Finetuning.pdf`, `finetuning.VTT`) es contenido teórico general del curso, sin conexión explícita con el TP1; solo sería relevante si decidimos usarlo como técnica puntual (ej. embeddings pre-entrenados para `title`/`description`), y en ese caso avisar antes.

### Aclaraciones que solo están en el audio de `consigna.VTT` (no en el PDF)

- No enroscarse con el tokenizador — el foco es la arquitectura del Transformer.
- Achicar también el dataset al principio (no solo `d_model`), para no quedar limitados por cómputo.
- **Nada de "vibe coding"**: toda decisión debe estar justificada y entendida por el grupo — no alcanza con "esto lo sugirió Claude y no lo entiendo, pero está implementado". Usar asistentes de código está permitido, pero cada elección tiene que poder explicarse.
- Si no hay GPU propia, usar Google Colab (T4/TPU).
- Promediar varias corridas (o cross-validation) en vez de reportar una sola ejecución.
- No hace falta que el BTR prediga "perfecto" — se evalúa el abordaje y la iteración, no el resultado final.
- Ejercicio 3 (personalización) debe ser una respuesta propia del grupo, no "hecha por Claude".
- No hay una única solución correcta — se fomenta discutir alternativas entre el equipo.

## Separación cómputo / gráficos

Todo resultado de EDA, entrenamiento o evaluación (ablación, métricas por configuración, embeddings, etc.) se guarda primero como **CSV** con los datos crudos. Los gráficos se generan en un script separado que lee esos CSV. Nunca mezclar en un mismo script el procesamiento pesado (cargar el dataset, entrenar, evaluar) con el ploteo: así se puede rehacer o ajustar un gráfico sin reprocesar ni reentrenar nada.

## Idioma y estilo

- No definir un threshold para la predicción de BTR — la consigna dice explícitamente que no es necesario.

## Pitfalls / cosas a recordar

- El foco es la **comprensión de la arquitectura Transformer**, no solo lograr buena performance. No optimizar a ciegas: cada decisión (por qué esta cantidad de heads, por qué este `d_model`, por qué este tipo de embedding) debe poder explicarse en la presentación.
- Arrancar con arquitectura chica (`d_model < 100`) antes de escalar, como sugiere la consigna — no ir directo a un modelo grande.
- Hace falta un **estudio de ablación** — diseñar los experimentos desde el principio pensando en qué módulos se van a poder desactivar/comparar.
