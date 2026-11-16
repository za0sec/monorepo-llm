# Ajuste del prompt y revisión de citas

Se comparó el prompt de G1 con una variante que pide una cita de video para «Cubierto» o «Parcial» y que solicita comprobar los IDs antes de entregar el informe. Se compararon los dos prompts en Razonamiento, donde el original citó un ID inexistente, y en AI Safety, donde la etiqueta de data poisoning contradecía la explicación. [Prompt original](../../ejercicio2/g1/prompt.md) · [Variante](../prompt_informe.md) · [Pautas de los casos](casos_03_06.md)

Se reutilizaron las entradas de `k=7` de ambos casos. Solo se cambió el mensaje de sistema. Se mantuvieron el objetivo, el VTT, los fragmentos de clase, el orden y el mensaje de usuario. Se usó `gpt-4.1-2025-04-14`, temperatura 0 y un máximo de 1.000 tokens de salida. Se comparó una respuesta nueva por caso con su control guardado. Los cuatro informes terminaron con `stop`.

| Caso | Estados correctos, G1 → variante | IDs inexistentes, G1 → variante | Tokens de entrada, G1 → variante |
| --- | --- | --- | --- |
| 05 · Razonamiento | 3/3 → 3/3 | C14 → ninguno | 1.478 → 1.600 |
| 06 · AI Safety | 2/3 → 1/3 | Ninguno → ninguno | 3.509 → 3.631 |

En Razonamiento se mantienen «Cubierto» para pasos intermedios y cómputo en inferencia y «Parcial» para técnicas de inferencia. Las referencias de la variante existen. Los tramos citados para técnicas de inferencia se centran en entrenamiento por refuerzo, por lo que la cita sigue aportando poco respaldo al estado. La existencia de un ID y su respaldo se evalúan por separado. [Informe original](../data/experimentos/caso_05_k_completo/k7/gpt41/informe.md) · [Variante](../data/experimentos/ajuste_prompt_informe/caso_05/gpt41/informe.md)

En AI Safety, la variante reconoce que el video omite la definición de data poisoning. Cambia de «Cubierto» a «Parcial» en prompt injection, aunque el video da una definición y un ejemplo. Cambia de «Parcial» a «Sin evidencia» en la comparación, aunque el video sí desarrolla uno de los dos riesgos. El acierto se contabiliza con la misma pauta manual de la comparación de `k`. [Informe original](../data/experimentos/caso_06_k_completo/k7/gpt41/informe.md) · [Variante](../data/experimentos/ajuste_prompt_informe/caso_06/gpt41/informe.md)

El generador ahora agrega «Requiere revisión» al informe con referencias ubicadas si encuentra un ID ausente en la entrada. Se verificó el aviso con el informe de control de Razonamiento y con casos de citas válidas e IDs inexistentes de clase y video. El texto crudo y el registro de IDs se conservan. [Código de validación](../../ejercicio2/generar_g1.py)

**Decisión.** Conservar el aviso y mantener G1 como prompt de referencia. La variante queda disponible como resultado experimental. Mejoró los dos errores puntuales y obtuvo 4/6 estados en conjunto, frente a 5/6 del control. El siguiente paso es verificar el flujo con el corpus completo desde un objetivo y un MP3.
