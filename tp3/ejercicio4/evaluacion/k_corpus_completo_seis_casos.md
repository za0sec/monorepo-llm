# Comparación de k con el corpus completo y seis objetivos

Se usaron los casos 01 y 02 como controles de desarrollo y los casos 03 a 06 para comprobar si la configuración funciona en otros temas. Se reutilizaron los cuatro informes del Caso 01 y los de `k=7` y `k=8` del Caso 02. Se generaron 18 informes nuevos para completar la comparación. [Casos nuevos y pautas](casos_03_06.md) · [Control 01](caso_01_k_completo.md#cortes-del-ranking-ampliado) · [Control 02](caso_02_k_completo.md)

Se buscó en `openai_e4_completo` con `text-embedding-3-small`, distancia coseno y diez resultados por consulta. Para cada objetivo se usaron la consulta original y tres subconsultas. El CrossEncoder de G4 ordenó los candidatos deduplicados. Los cuatro cortes son prefijos del mismo ranking de cada caso. Se verificó que cada caso conservó su VTT completo, el prompt de G1 y GPT‑4.1 a temperatura 0 con un máximo de 1.000 tokens de salida. Los cuatro casos nuevos heredaron el nivel introductorio de la prueba de control y sus pautas quedaron fuera de las entradas. Los 24 informes terminaron con `stop`. [Prompt del generador](../../ejercicio2/g1/prompt.md)

**Corrección manual.** Context Recall cuenta qué criterios quedan completamente respaldados por los chunks del corte. Context Precision promedia la precisión acumulada en las posiciones que aportan evidencia directa, incluso parcial. Puede dar 100 % con un solo fragmento relevante en el primer puesto. El acierto de estado se toma de la tabla del informe y se compara con la pauta del video. Una contradicción entre la etiqueta y su explicación cuenta como error de estado. Las dos primeras métricas evalúan el contexto y la tercera evalúa el informe.

## Resultados de los cuatro casos nuevos

| Indicador | k=7 | k=8 | k=10 | k=12 |
| --- | ---: | ---: | ---: | ---: |
| Context Recall, criterios completos | 8/12 | 8/12 | 8/12 | 9/12 |
| Context Precision, promedio por caso | 89,1 % | 85,5 % | 84,5 % | 80,1 % |
| Estados correctos del video | 10/12 | 10/12 | 9/12 | 8/12 |
| Tokens de entrada promedio por informe | 2.675 | 2.766 | 2.984 | 3.151 |
| Informes con algún ID de cita inexistente | 1/4 | 1/4 | 1/4 | 1/4 |

Cada celda de la tabla siguiente muestra los criterios respaldados y los estados correctos, en ese orden.

| Caso | Candidatos únicos | k=7 | k=8 | k=10 | k=12 |
| --- | ---: | --- | --- | --- | --- |
| 03 · Transfer learning | 15 | 2/3 · 2/3 | 2/3 · 2/3 | 2/3 · 2/3 | 2/3 · 2/3 |
| 04 · Tokenización | 23 | 3/3 · 3/3 | 3/3 · 3/3 | 3/3 · 2/3 | 3/3 · 2/3 |
| 05 · Razonamiento | 22 | 0/3 · 3/3 | 0/3 · 3/3 | 0/3 · 3/3 | 1/3 · 2/3 |
| 06 · AI Safety | 19 | 3/3 · 2/3 | 3/3 · 2/3 | 3/3 · 2/3 | 3/3 · 2/3 |

### Evidencia académica en el ranking

Los valores de la tercera columna son los puestos relevantes para Context Precision entre los 12 primeros. Los IDs `C#` de los informes corresponden a esos mismos puestos.

| Caso | Primer respaldo completo por criterio | Puestos relevantes |
| --- | --- | --- |
| 02 · Transformers | A1 en 10 y A2 en 12. A3 sigue incompleto en 12. | 1, 10, 11, 12 |
| 03 · Transfer learning | TL1 en 6 y TL2 en 1. TL3 sigue parcial. El VTT docente en 1 aporta el costo y el ajuste necesario. El apunte en 12 agrega cómo comparar los enfoques, sin justificar cuándo cambiar según la similitud entre tareas. | 1, 3, 4, 5, 6, 7, 12 |
| 04 · Tokenización | TK1 y TK2 entre 1 y 3. TK3 en 2. El puesto 9 suma otro ejemplo explícito de asignación de IDs que refuerza TK2. La descripción de C2 confunde un par de IDs con el vector, por eso el mapeo se comprueba con el VTT en 1 y el texto seleccionable en 3. | 1, 2, 3, 4, 8, 9, 11, 12 |
| 05 · Razonamiento | RZ1 queda parcial. Hay referencias a secuencias más largas y al contraste entre cadena de pensamiento y respuesta, con poca explicación de la función de los pasos. RZ2 queda parcial. La correlación con la longitud se observa en una discusión de entrenamiento. RZ3 se completa con los puestos 4 y 12, que juntos explican cómo se delega un subproblema a una herramienta durante la inferencia. | 1, 3, 4, 5, 8, 12 |
| 06 · AI Safety | AS1 en 4, AS2 en 1 y AS3 con 1, 4 y 7. La distinción se apoya en las dos fuentes académicas. | 1, 2, 3, 4, 7, 8, 11 |

### Estados que emitieron los informes

| Criterio | Pauta | k=7 | k=8 | k=10 | k=12 |
| --- | --- | --- | --- | --- | --- |
| TL1 · Extracción | Cubierto | Cubierto | Cubierto | Cubierto | Cubierto |
| TL2 · Fine-tuning | Cubierto | Cubierto | Cubierto | Cubierto | Cubierto |
| TL3 · Cuándo usar cada enfoque | Cubierto | Parcial | Parcial | Parcial | Parcial |
| TK1 · Segmentación | Cubierto | Cubierto | Cubierto | Cubierto | Cubierto |
| TK2 · IDs y vocabulario | Sin evidencia | Sin evidencia | Sin evidencia | Parcial | Parcial |
| TK3 · Embedding | Cubierto | Cubierto | Cubierto | Cubierto | Cubierto |
| RZ1 · Pasos intermedios | Cubierto | Cubierto | Cubierto | Cubierto | Cubierto |
| RZ2 · Cómputo en inferencia | Cubierto | Cubierto | Cubierto | Cubierto | Parcial |
| RZ3 · Técnicas de inferencia | Parcial | Parcial | Parcial | Parcial | Parcial |
| AS1 · Prompt injection | Cubierto | Cubierto | Cubierto | Cubierto | Cubierto |
| AS2 · Data poisoning | Sin evidencia | Cubierto | Cubierto | Cubierto | Cubierto |
| AS3 · Dónde actúa cada riesgo | Parcial | Parcial | Parcial | Parcial | Parcial |

- **Transfer learning.** El informe usó C6 para justificar ambos enfoques con los cortes 7, 8 y 10. Ese fragmento se corta antes del fine-tuning. C1 ya lo respaldaba y, con `k=12`, el informe cita C12, que explica explícitamente las capas que se ajustan. La fila sobre cuándo ajustar más capas siguió como «Parcial», aunque el video da dos condiciones concretas.
- **Tokenización.** Con `k=10` y `k=12`, la fila de IDs usa la mención de vocabulario durante la predicción o la explicación de embeddings como evidencia parcial del mapeo. Esos tramos carecen del paso que se evalúa. El ejemplo académico adicional en C9 ya está disponible en ambas corridas.
- **Razonamiento.** Los cuatro informes citaron IDs académicos fuera de su top-k. Se registraron C14 con `k=7`, C13 y C14 con `k=8`, C14 y C15 con `k=10`, y C13, C14 y C15 con `k=12`. Acertar un estado del video puede coexistir con respaldo académico incompleto o citas inexistentes. Los otros casos usaron IDs que estaban en sus entradas.
- **AI Safety.** La fila que define data poisoning quedó «Cubierto» en las cuatro corridas, con la fuente de video vacía y una explicación que dice que el video omite el tema. Hay material académico desde C1. El error está en la etiqueta del informe.

## Comparación de los seis objetivos

| Indicador | k=7 | k=8 | k=10 | k=12 |
| --- | ---: | ---: | ---: | ---: |
| Context Recall, criterios completos | 11/20 | 11/20 | 12/20 | 14/20 |
| Context Precision, promedio por caso | 92,7 % | 90,4 % | 81,3 % | 74,3 % |
| Estados correctos del video | 15/20 | 15/20 | 14/20 | 14/20 |
| Tokens de entrada promedio por informe | 3.391 | 3.475 | 3.693 | 3.879 |

En el Caso 02, `k=10` suma las funciones de Q, K y V y `k=12` incorpora la fórmula escalada. El informe pasa a marcar la fórmula «Parcial», que es el estado correcto, y marca «Parcial» la explicación de Q, K y V por exigir además las matrices de proyección. El total queda en 2/3 en ambos cortes. Distintos valores de `k` mejoran casos distintos. Los dos criterios adicionales en el contexto de Transformers y el de razonamiento explican la mejora agregada de Context Recall de `k=12` frente a `k=7`.

**Decisión.** Mantener `k=7` como valor de referencia del flujo. Empata con `k=8` en respaldo y acierto de los cuatro casos nuevos y requiere menos tokens. `k=12` mejora el respaldo académico en dos objetivos, pero en esta muestra el informe pierde aciertos al agregar contexto. La muestra tiene seis objetivos con una corrida por corte, así que la elección de `k=7` sigue siendo provisional. El siguiente contraste puede aislar la validación de citas y la separación entre evidencia del curso y cobertura del video con el ranking fijo.

```bash
export PYTHONPATH=tp3/ejercicio2
for caso in 02 03 04 05 06; do
  .venv/bin/python tp3/ejercicio4/comparar_k_completo.py --caso "$caso" --vecinos 10 \
    --salida "tp3/ejercicio4/data/experimentos/caso_${caso}_k_completo"
done
```

[Artefactos del Caso 01](../data/experimentos/caso_01_vecinos10/) · [Caso 02](../data/experimentos/caso_02_k_completo/) · [Caso 03](../data/experimentos/caso_03_k_completo/) · [Caso 04](../data/experimentos/caso_04_k_completo/) · [Caso 05](../data/experimentos/caso_05_k_completo/) · [Caso 06](../data/experimentos/caso_06_k_completo/)
