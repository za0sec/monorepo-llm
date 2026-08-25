# CLAUDE.md

Guidance for Claude Code (claude.ai/code) on this repo.

Monorepo de todos los TPs de la materia **Large Language Models (73.69), 2026** — cátedra "Transformers". Cada TP vive en su propia carpeta (`tp1/`, `tp2/`, ...) y tiene su propio `CLAUDE.md` con el contexto específico.

## NO PONER CLAUDE COAUTOR

Nunca agregar `Co-Authored-By: Claude` (ni ninguna variante) en los mensajes de commit. Los commits van **sin** trailer de co-autoría.

## Regla principal: no usar técnicas que no se vieron en clase

Lo único que importa es que no metas arquitecturas, mecanismos o trucos de Transformers/NLP que no estén en el material de cátedra. Si una idea (positional encoding alternativo, tipo de atención, técnica de regularización, librería de alto nivel, etc.) no aparece en las clases o no es de sentido común para EDA/preprocesamiento tabular estándar, avisame antes de usarla — no la metas por tu cuenta aunque la conozcas de otro lado.

El material de cátedra de cada TP está en su carpeta `docs/`. Antes de implementar cualquier pieza de la arquitectura (embeddings, atención, positional encoding, cabezas de salida, fine-tuning/transfer learning), releer las clases correspondientes y ceñirse a lo que ahí se explica.

## Idioma y estilo

- El TP, los apuntes y la documentación están en **español**. Responder y comentar en español salvo pedido contrario.
- Justificar toda decisión de diseño (features, preprocesamiento, arquitectura, hiperparámetros) por escrito — se evalúa la justificación y la comparación de alternativas, no solo que el modelo funcione.
