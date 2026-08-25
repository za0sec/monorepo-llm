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

## Idioma y estilo

- No definir un threshold para la predicción de BTR — la consigna dice explícitamente que no es necesario.

## Pitfalls / cosas a recordar

- El foco es la **comprensión de la arquitectura Transformer**, no solo lograr buena performance. No optimizar a ciegas: cada decisión (por qué esta cantidad de heads, por qué este `d_model`, por qué este tipo de embedding) debe poder explicarse en la presentación.
- Arrancar con arquitectura chica (`d_model < 100`) antes de escalar, como sugiere la consigna — no ir directo a un modelo grande.
- Hace falta un **estudio de ablación** — diseñar los experimentos desde el principio pensando en qué módulos se van a poder desactivar/comparar.
