# Casos 03 a 06 del ejercicio 4

Los cuatro videos amplían la evaluación a otros objetivos de la materia. Las pautas se redactaron a partir del material docente indexado y de lo que se escucha según cada VTT. Los estados son referencias manuales para revisar con el grupo. La expansión, la búsqueda y el generador reciben el objetivo, con la pauta reservada para la corrección.

| Caso | MP3 aportado | Transcripción | Segmentos | Última marca |
| --- | --- | --- | ---: | --- |
| 03 · Transfer learning | [Transfer learning.mp3](../data/casos/transfer_learning/audio.mp3) | [VTT](../data/casos/transfer_learning/transcripcion.vtt) | 98 | 08:28.600 |
| 04 · Tokenización | [Tokenización.mp3](../data/casos/tokenizacion/audio.mp3) | [VTT](../data/casos/tokenizacion/transcripcion.vtt) | 55 | 05:13.119 |
| 05 · Modelos de razonamiento | [Modelos de razonamiento.mp3](../data/casos/razonamiento/audio.mp3) | [VTT](../data/casos/razonamiento/transcripcion.vtt) | 19 | 01:53.800 |
| 06 · AI Safety | [AI Safety.mp3](../data/casos/ai_safety/audio.mp3) | [VTT](../data/casos/ai_safety/transcripcion.vtt) | 147 | 08:14.880 |

Los MP3 y VTT se guardaron en carpetas locales de `ejercicio4/data/casos/`, con un `asr.json` por caso. El registro contiene el modelo, los hashes y la cantidad de segmentos. Se usó Whisper con detección automática de idioma y temperatura 0. Los cuatro VTT están en español. Las páginas docentes de las tablas corresponden a la numeración original de los PDF.

## Caso 03 · Transfer learning

**Objetivo del alumno.** «Quiero entender cuál es la diferencia entre usar un modelo preentrenado para extraer características y hacerle fine-tuning, qué pesos se actualizan en cada caso y cuándo conviene cada enfoque».

| Criterio | Evidencia académica que debería recuperar | Estado esperado | Evidencia del video |
| --- | --- | --- | --- |
| TL1 · Extracción de características | Explicar qué se reutiliza del modelo preentrenado y que se entrena la nueva cabeza con el extractor congelado. Apunte de feature extraction, p. 1, y clase 3, pp. 14 y 16. | **Cubierto** | De 05:09 a 06:06 explica que conserva las capas que extraen características, reemplaza la salida y actualiza solo sus pesos en el ejemplo de flores. |
| TL2 · Fine-tuning | Explicar que se parte de pesos preentrenados y se ajusta parte o todo el modelo para la tarea nueva. Apunte, p. 1, clase 3, p. 17, y VTT docente desde 43:20. | **Cubierto** | De 06:06 a 07:01 explica el ajuste de capas superiores y de toda la red. Distingue esos ajustes del entrenamiento desde una inicialización aleatoria. |
| TL3 · Cuándo usar cada enfoque | Vincular la similitud entre tareas y el desempeño con la decisión de congelar o adaptar capas. Explicar el ahorro de entrenar menos pesos. Apunte, p. 1, clase 3, p. 23, y VTT docente desde 43:20. | **Cubierto** | De 05:09 a 05:43 recomienda congelar si los datos se parecen y solo hacen falta pocas iteraciones. De 06:20 a 06:42 propone ajustar más capas si la salida sola resulta insuficiente. |

[Apunte de feature extraction y fine-tuning](../apuntes/feature_extraction_y_fine_tuning.pdf) · [Diapositivas de la clase 3](../slides/clase_03_fine_tuning.pdf) · [VTT docente](../transcripciones/finetuning.VTT)

## Caso 04 · Tokenización

**Objetivo del alumno.** «Quiero entender cómo pasa un texto de ser una cadena de caracteres a tokens, IDs y embeddings, y qué representa cada paso».

| Criterio | Evidencia académica que debería recuperar | Estado esperado | Evidencia del video |
| --- | --- | --- | --- |
| TK1 · Segmentación | Explicar qué unidad genera el tokenizador y mostrar que una palabra puede dividirse en más de un token. Clase 2, pp. 7 y 8. | **Cubierto** | De 00:32 a 00:58 define los tokens como palabras, partes de palabras, signos o espacios y da un ejemplo de segmentación. |
| TK2 · IDs y vocabulario | Explicar que el tokenizador asigna a cada token un identificador entero de su vocabulario. Clase 2, pp. 7 a 9, que muestran los IDs asignados a los segmentos de «mate». | **Sin evidencia en el video** | Entre 00:49 y 01:28 pasa de segmentar a representar con vectores. A 02:12 menciona el vocabulario al generar un token. Queda sin explicar el mapeo entre token e ID. |
| TK3 · Embedding | Distinguir la segmentación del paso que representa cada unidad de texto como un vector de números. Clase 2, pp. 5, 9 y 31. | **Cubierto** | De 01:16 a 01:36 explica que después de tokenizar cada token se convierte en un vector de alta dimensión que entra al Transformer. |

[Diapositivas de la clase 2](../slides/clase_02_embeddings.pdf)

## Caso 05 · Modelos de razonamiento

**Objetivo del alumno.** «Quiero entender qué hacen los modelos de razonamiento, qué ganamos al dedicar más cómputo a una respuesta y qué técnicas permiten hacerlo durante la inferencia».

| Criterio | Evidencia académica que debería recuperar | Estado esperado | Evidencia del video |
| --- | --- | --- | --- |
| RZ1 · Pasos intermedios | Explicar que los modelos de razonamiento pueden dedicar tokens adicionales a planificar, verificar o descomponer un problema antes de la respuesta final. Clase 4, p. 4, y VTT docente 00:42–01:05. | **Cubierto** | Entre 00:24 y 00:43 describe dividir el problema, probar un enfoque, revisarlo y corregir errores antes de responder. En 01:25 explica que esto usa más tokens. |
| RZ2 · Cómputo en inferencia | Relacionar más trabajo al producir una respuesta con una posible mejora y un costo en cómputo o tiempo. Clase 4, p. 12. | **Cubierto** | De 01:12 a 01:44 explica que se puede elegir cuánto cómputo dedicar, que más tokens agregan tiempo y costo y que la respuesta aún puede ser errónea. |
| RZ3 · Técnicas durante la inferencia | Explicar una técnica concreta de las que presenta la clase, con su procedimiento. Por ejemplo, generar varias respuestas y elegir por acuerdo o revisar una respuesta candidata y refinarla. Clase 4, p. 18. | **Parcial** | De 00:24 a 00:43 menciona revisar y corregir pasos en general. De 00:49 a 01:06 desarrolla el aprendizaje por refuerzo como entrenamiento. Queda sin explicar un método concreto de uso durante la inferencia y cómo se aplicaría. |

[Diapositivas de la clase 4](../slides/clase_04_reasoning_y_produccion.pdf) · [VTT docente](../transcripciones/reasoningmodels.VTT)

## Caso 06 · AI Safety

**Objetivo del alumno.** «Quiero entender la diferencia entre prompt injection y data poisoning en modelos de lenguaje y en qué parte del sistema actúa cada riesgo».

| Criterio | Evidencia académica que debería recuperar | Estado esperado | Evidencia del video |
| --- | --- | --- | --- |
| AS1 · Prompt injection | Explicar qué ocurre cuando una instrucción ajena altera la tarea que el sistema iba a ejecutar. Clase 6, p. 9, y VTT docente 12:05–13:39. | **Cubierto** | De 01:12 a 01:18 define el riesgo como cambiar la tarea de la IA. De 04:48 a 05:03 muestra conceptualmente cómo puede confundirse una instrucción externa con una instrucción del sistema. |
| AS2 · Data poisoning | Explicar que contaminar los datos de entrenamiento puede alterar el comportamiento que aprende el modelo. Clase 6, p. 10, y VTT docente 13:58–14:25. | **Sin evidencia en el video** | El video desarrolla la inyección de instrucciones y la evaluación de defensas. Menciona el reentrenamiento como defensa a 01:22 y 04:17. Queda sin explicar la contaminación de los datos de entrenamiento. |
| AS3 · Dónde actúa cada riesgo | Distinguir instrucciones o contenido externo durante el uso del sistema de alteraciones en los datos que el modelo aprende durante el entrenamiento. Clase 6, pp. 9 y 10. | **Parcial** | El video explica el lado de las instrucciones durante el uso (01:12–01:18 y 04:48–05:03). Su contraste inicial es con los jailbreaks. Falta el lado de los datos de entrenamiento. |

[Diapositivas de la clase 6](../slides/clase_06_ai_safety.pdf) · [VTT docente](../transcripciones/ai_safety.VTT)

## Uso para comparar rankings

Mantener cada objetivo y VTT fijos en todos los cortes. Medir primero cuántos criterios respalda por completo el contexto académico. Después comparar los estados y sus citas en los informes. Los casos 03 a 06 se prepararon antes de consultar Chroma y de generar informes con ellos.

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/transcribir_casos.py \
  tp3/ejercicio4/data/casos/{transfer_learning,tokenizacion,razonamiento,ai_safety}
```
