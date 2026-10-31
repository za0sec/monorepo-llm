# Caso 01: cobertura de embeddings en un video

## Estado y propósito

Ejemplo de respuesta elaborado por el asistente mediante revisión de las
transcripciones. El 31/10/2026 el usuario confirmó haber revisado el caso como
primer paso del plan de implementación. Se conserva como referencia manual de
desarrollo para concretar la salida de la aplicación y comprobar la recuperación.
No es un resultado de un pipeline RAG ni constituye por sí solo un conjunto de
evaluación independiente.

Esta revisión utiliza los archivos locales y no requiere llamadas adicionales
a la API. La confirmación de revisión del caso no documenta una auditoría de la
transcripción contra el audio original.

## Entrada del caso

- **Video:** `¿Qué son los EMBEDDINGS Grandes Modelos de Lenguaje.mp3`, de unos
  10 minutos y 9 segundos. El archivo está en `data/audios/`.
- **Objetivo de estudio propuesto para este ejemplo:** entender qué son los
  embeddings y cómo se compara la similitud entre textos. Este objetivo es una
  suposición de trabajo del asistente, que el grupo puede ajustar.
- **Nivel:** introductorio, con explicación del criterio de comparación. El
  entrenamiento detallado de Word2Vec y BERT queda fuera de este objetivo.

Fuentes utilizadas:

- [Transcripción de la clase](../../../docs/embeddings_2.VTT).
- [Transcripción automática del video](../data/transcripciones/¿Qué%20son%20los%20EMBEDDINGS%20Grandes%20Modelos%20de%20Lenguaje.vtt).

Ambas fuentes están excluidas de Git por la configuración del repositorio y
deben estar disponibles localmente. Los tiempos del video provienen de Whisper
y los de la clase, del VTT original; permiten localizar la evidencia, pero pueden
ser aproximados.

## Criterios de cobertura

El material de la materia establece los contenidos esperados para el objetivo.
La transcripción del video aporta evidencia para clasificarlos.

- **Cubierto:** desarrolla el criterio al nivel solicitado y aporta una
  explicación o un ejemplo suficiente. Puede usar palabras diferentes a la clase.
- **Parcial:** desarrolla una parte del criterio, pero deja otra sin explicar.
  La justificación debe indicar ambas partes.
- **Sin evidencia:** la transcripción disponible no aporta una explicación del
  criterio. Esto requiere revisar el contenido, además de buscar términos.
  Una falta de coincidencia literal por sí sola no demuestra una omisión.

| ID | Contenido esperado | Condición para considerarlo cubierto | Referencia de la clase |
| --- | --- | --- | --- |
| E1 | Qué es un embedding de texto | Explicar que es una representación numérica vectorial que busca representar información del texto. | Entradas 123 y 181: 18:32.360-18:50.210 y 25:22.170-25:37.590. |
| E2 | Papel del contexto | Distinguir una representación fija por palabra de una que cambia según el contexto y explicar la diferencia con un ejemplo. | Entradas 191-197: 27:47.050-28:44.829. |
| E3 | Cómo comparar textos mediante vectores | Explicar que se representan ambos textos y explicitar un criterio para comparar sus vectores, como la dirección o el ángulo. | Entradas 137-139: 20:32.610-21:00.349. |
| E4 | Similitud coseno | Explicar su cálculo como producto interno dividido por el producto de las normas de los vectores. | Entrada 222: 31:55.350-32:14.820. |
| E5 | Limitación relacionada con la magnitud | Explicar que el coseno no conserva diferencias de magnitud y dar un caso donde esto importa. | Entradas 182-188: 25:37.950-27:13.979. |

E3 evalúa la explicación del procedimiento general; E4, el desarrollo de una
medida específica. Ambos están relacionados. Este ejemplo no agrega las
etiquetas en un porcentaje de cobertura.

## Ejemplo de respuesta al usuario

### Contenidos cubiertos

**Qué es un embedding.** El video presenta los embeddings como representaciones
vectoriales de los datos y aplica esa idea al texto. En 02:54.820-03:03.820
explica la representación vectorial aprendida por una red neuronal; en
09:03.580-09:18.580 retoma la idea de un arreglo de números que captura información
del texto. Esto cubre E1 al nivel introductorio solicitado.

**La importancia del contexto.** En 03:19.740-05:23.580 contrasta la representación
fija por palabra con representaciones que incorporan el contexto. El ejemplo de
"banco" en frases diferentes permite entender por qué una misma palabra puede
necesitar representaciones diferentes. Esto cubre E2 conceptualmente, aunque el
video no use las mismas etiquetas de "estático" y "dinámico" que la clase.

### Contenidos tratados parcialmente

**Cómo comparar textos mediante embeddings.** En 05:44.580-06:35.580 relaciona
significados similares con representaciones vectoriales similares. En
07:25.580-07:54.580 describe una búsqueda semántica donde se representan el texto
y la consulta con embeddings y se comparan sus vectores. Sin embargo, no explica
qué operación permite decidir cuánto se parecen. E3 queda parcial: desarrolla la
idea y el esquema de uso, pero falta el criterio concreto de comparación.

### Contenidos sin evidencia en la transcripción del audio

**El cálculo de la similitud coseno.** La lectura de la transcripción completa no
aporta una explicación del coseno, del ángulo entre vectores ni del cálculo con
producto interno y normas. La referencia genérica a una "comparación vectorial"
en 07:46.580-07:50.580 no alcanza para cubrir E4.

**El efecto de la magnitud al usar similitud coseno.** Tampoco encontramos una
explicación de por qué vectores con distinta magnitud pueden tener la misma
similitud coseno, ni de cuándo esa propiedad es conveniente o problemática.
E5 queda sin evidencia.

### Material de la materia recomendado

Para completar el procedimiento de comparación de textos y la similitud coseno:

1. Revisar la clase `embeddings_2`, **20:32.610-21:00.349**, donde se relaciona
   el ángulo entre vectores con su similitud y se distingue de la magnitud.
2. Continuar en **31:55.350-32:14.820**, donde se explica el producto interno
   y la normalización por las normas de los vectores.

Para entender las limitaciones de esa comparación:

3. Revisar **25:37.950-27:13.979**, especialmente el ejemplo de preferencias
   por series. La clase compara vectores con la misma dirección y distinta
   magnitud para discutir qué información se pierde al usar el coseno.

### Conclusión sobre la utilidad del video

El video sirve como introducción a los embeddings y al papel del contexto.
También permite entender para qué se pueden usar representaciones similares
en una búsqueda semántica. Para el objetivo propuesto, hace falta complementarlo
con el material de la materia sobre la medida de comparación: cómo funciona
la similitud coseno y qué limitaciones tiene.

## Límites de esta revisión y uso posterior

- "Sin evidencia" se refiere a la transcripción automática disponible. La
  revisión del audio puede revelar una omisión o un error del transcriptor.
  El contenido que aparezca solo en pantalla queda fuera del alcance basado
  exclusivamente en audio.
- Se conservan posibles errores de transcripción, por ejemplo "1,72 cm" al
  describir una altura. No se atribuyen al autor sin revisar el audio original.
- Los nombres de modelos y las dimensiones mencionadas por el video no forman
  parte de estos criterios. Este análisis de cobertura no verifica todas sus
  afirmaciones técnicas ni su vigencia.
- El caso ya fue revisado por el usuario como referencia para el primer paso.
  Cualquier corrección posterior del objetivo, criterios o evidencias debe
  documentarse.
- El próximo paso será intentar recuperar automáticamente los fragmentos de
  clase necesarios para el objetivo y comprobarlos contra estas referencias.
  Este caso servirá para desarrollar y depurar; harán falta otros casos
  separados para evaluar las mejoras sin ajustar todo a un único video.
