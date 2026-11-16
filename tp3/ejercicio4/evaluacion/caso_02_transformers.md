# Caso 02 · Autoatención en Transformers

Piloto del ejercicio 4. Pauta anotada por el asistente a partir del material docente y de la transcripción, antes de consultar Chroma o generar informes. Queda abierta a revisión del grupo.

## Entrada del estudiante

- **Objetivo.** «Quiero entender cómo decide un Transformer a qué tokens prestarles atención, cómo se calcula esa atención y qué cambia cuando usa varias cabezas».
- **Audio.** «¿Qué es un TRANSFORMER La Red Neuronal que lo cambió TODO!», provisto por el grupo. Duración aproximada de 18:40. El MP3 local está en [data/casos/transformers/audio.mp3](../data/casos/transformers/audio.mp3). SHA-256 `f68126fe7a24a2f5971a88b40d89b7bc90320486ca2b9919cd81c47c1359e429`.

El nivel introductorio lo fija la aplicación. Se preparó el [VTT local](../data/casos/transformers/transcripcion.vtt) con `whisper-1` en la API directa de OpenAI, idioma `es`, temperatura `0` y formato `vtt`, usando el mismo código de transcripción que el ejercicio 2. El lector validó 219 segmentos entre `00:00:00.000` y `00:18:35.719`. SHA-256 del VTT `f5438c38896ec984e96c2a5cc3afff89da5deb8c79b9c1339f5e1fa8ab6183d2`. Se reutilizará el mismo VTT en ambos brazos.

## Pauta de evaluación

La clase fija lo que se espera aprender. Los estados de la tabla se refieren a lo que se escucha en el audio, según el VTT. La pauta queda fuera de la consulta, la expansión y el prompt del generador.

| Criterio | Evidencia académica necesaria | Estado esperado del video | Evidencia en el VTT del video |
| --- | --- | --- | --- |
| A1 · Roles de Q, K y V | Q representa lo que busca el token, K permite estimar la compatibilidad y V aporta la información que se combina. Apunte p. 1 y diapositivas pp. 27–29. VTT docente 41:40–42:38. | Cubierto | La analogía de la cerradura y la llave explica Q y K (08:52–09:17, 10:14–10:29). El tercer vector se presenta entre 12:49 y 13:09 y se usa en la mezcla entre 13:43 y 14:14. |
| A2 · Cálculo de la atención escalada | Producto escalar de Q y K, división por la raíz de la dimensión de las claves, softmax y suma ponderada de V. Apunte p. 1 y diapositivas pp. 30–33. VTT docente 43:55–45:07 y 47:28–48:09, con transcripción imprecisa en el escalado. | Parcial | Explica el producto escalar para obtener compatibilidades (10:43–11:28) y cómo se mezclan los valores (13:43–14:14). En el VTT no se explica de forma explícita la división por la raíz ni softmax. |
| A3 · Atención con varias cabezas | Cada cabeza tiene sus propias proyecciones Q, K y V, puede dar peso a posiciones diferentes y se combina con las demás. Apunte p. 1 y diapositivas pp. 38–39. VTT docente 58:43–01:00:58 y 01:24:33–01:24:54. | Sin evidencia en el video | No se encontró una explicación de múltiples cabezas en los 219 segmentos. Entre 15:56 y 17:07 el video cierra la introducción a la atención y anuncia una segunda parte para otros mecanismos de Transformers. |

[Apunte del piloto, p. 1](../apuntes/apuntes_transformers.pdf#page=1) · [VTT docente de Transformers](../../../docs/transformers.VTT) · [Diapositivas de Transformers](../../../docs/Clase%201%20-%20Introducci%C3%B3n%20a%20Transformers.pdf)

**Conclusión de referencia.** El video sirve para entender la intuición de la autoatención, los roles de los tres vectores y la combinación ponderada de valores. Para completar el objetivo, el estudiante necesita un pasaje que explique el escalado y softmax, y otro que explique qué aportan las múltiples cabezas. El informe debe recomendar pasajes efectivamente recuperados. Si falta material académico para un criterio, corresponde indicar «Respaldo de clase insuficiente» y registrar el faltante en recuperación.

## Comparación prevista

Comparar `openai_i4` (623 chunks de VTT) con `openai_e4_apuntes_ampliado` (los mismos VTT y 15 chunks de los tres PDFs). Buscar en cada colección completa, sin filtrar por archivo. Congelar la consulta original y sus tres subconsultas, cinco candidatos por consulta, el CrossEncoder de G4, `k=7`, el VTT del video y el prompt y modelo del generador. Las diapositivas se consultaron para preparar la referencia académica y quedan fuera de las dos colecciones de este piloto.

- **Context Recall** antes del reranking y en el top 7. Contar cuántos de los tres criterios respalda por completo el contexto académico recuperado. Registrar si el PDF completa evidencia, la repite o desplaza pasajes útiles.
- **Context Precision@7.** Medir la posición de los chunks que aportan evidencia directa, incluida la evidencia parcial.
- **Informe.** Comparar coincidencia de estados, citas localizables que respaldan lo escrito y Answer Faithfulness. Registrar por separado si faltó evidencia académica, si se omitió una fila o si el estado contradice la transcripción.

Estado de la prueba. Audio y referencia preparados. Expansión, recuperación y generación pendientes.
