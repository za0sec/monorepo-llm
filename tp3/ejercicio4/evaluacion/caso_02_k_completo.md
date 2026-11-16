# Caso 02 con el corpus completo

Se compara `k=7` con `k=8` en `openai_e4_completo`. Se reutilizaron el objetivo, la expansión de cuatro consultas, sus embeddings y el VTT de 219 segmentos del piloto del [Caso 02](caso_02_transformers.md). Se pidieron diez resultados por consulta. La deduplicación por ID dejó 23 candidatos y el CrossEncoder de G4 los ordenó una vez. Ambos informes usaron el mismo prompt, el mismo snapshot de GPT‑4.1, temperatura 0 y un máximo de 1.000 tokens de salida.

| Indicador | k=7 | k=8 |
| --- | ---: | ---: |
| Criterios con respaldo académico completo | 0/3 | 0/3 |
| Estados que coinciden con la pauta del video | 2/3 | 1/3 |
| Estados acertados con respaldo académico completo en los fragmentos recuperados | 0/3 | 0/3 |
| Tokens de entrada a GPT‑4.1 | 6.303 | 6.429 |
| Tokens de salida de GPT‑4.1 | 595 | 647 |

Los informes se corrigieron contra el VTT y la pauta manual del caso. Los estados que coinciden con el video se contabilizan aunque falte respaldo académico. Los dos informes terminaron con `stop` y todos los IDs de las citas fueron válidos.

| Criterio | Estado de referencia | k=7 | k=8 |
| --- | --- | --- | --- |
| A1 · Roles de Q, K y V | Cubierto | Cubierto. Describe qué función cumple cada vector. | Parcial. Exige que el video muestre la proyección por matrices, un requisito adicional al criterio evaluado. |
| A2 · Cálculo de atención escalada | Parcial | Marca el cálculo como cubierto. Omite el escalado y softmax. | Evalúa por separado el producto escalar y la mezcla de valores. Omite el estado del cálculo escalado. |
| A3 · Atención con varias cabezas | Sin evidencia en el video | Sin evidencia en el video. | Sin evidencia en el video. |

El puesto 8, `vtt-8e3231e21599172c-90`, introduce una cabeza de atención en una demo docente (38:31.360–39:07.719). Su texto termina antes de explicar qué busca y no describe el mecanismo de varias cabezas. El informe de `k=8` lo cita para A3, pero esa cita carece de la explicación académica que exige el criterio. El fragmento C1 de ambas variantes describe el origen de los vectores en atención cruzada. En `k=7` se cita para las funciones de Q, K y V y el informe de `k=8` lo recomienda para sus proyecciones. La cita respalda el origen de los vectores en ese tipo de atención. Las funciones de cada vector quedan sin respaldo completo en la cita.

La definición de Q, K y V para autoatención quedó décima y la fórmula escalada quedó duodécima. El puesto 18 explica cómo se combinan las salidas de las cabezas y los puestos 21 y 22 describen sus juegos de Q, K y V. El siguiente contraste podría usar el mismo ranking para comparar un corte que incluya la fórmula con uno que alcance la evidencia de varias cabezas. La mejora de `k=8` que se observó en el Caso 01 no se repitió con este objetivo.

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/comparar_k_completo.py \
  --caso 02 --vecinos 10 --salida tp3/ejercicio4/data/experimentos/caso_02_k_completo
```

[Ranking común](../data/experimentos/caso_02_k_completo/ranking.json) · [Informe k=7](../data/experimentos/caso_02_k_completo/k7/gpt41/informe.md) · [Informe k=8](../data/experimentos/caso_02_k_completo/k8/gpt41/informe.md)
