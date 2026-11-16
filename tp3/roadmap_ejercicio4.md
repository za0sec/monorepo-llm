# TP3 · Plan del ejercicio 4

## Objetivo de la iteración

Medir cuánto aporta agregar **apuntes de estudiantes y diapositivas de clase** al corpus de VTT del ejercicio 2. En el Caso 01, la configuración G4 llevó al generador respaldo completo para 3 de 5 criterios. Algunos faltantes provenían de la selección de candidatos y el generador también omitió un criterio con evidencia disponible. El corpus ampliado permite comprobar si nuevas fuentes recuperables aportan definiciones, fórmulas o ejemplos y si el generador los usa en el informe. La consigna pide documentar la limitación, la modificación y los resultados antes y después. [G4](ejercicio2/experimentos.md#g4--expansión-con-reranking-de-los-candidatos-de-g3) · [Consigna, p. 3](../docs/RAG/tp3.pdf)

## 1. Seleccionar las fuentes

| Fuente | Disponible | Uso previsto |
| --- | --- | --- |
| Apuntes de prueba | [Apuntes de Transformers](ejercicio4/apuntes/apuntes_transformers.pdf), dos páginas elaboradas a partir de la clase. | La página 1 trata atención y la página 2, codificación posicional y máscara causal. |
| Apunte sobre el encoder | [El bloque del encoder](ejercicio4/apuntes/bloque_del_encoder.pdf), una página. | Autoatención, red feed-forward y Add & Norm. Fuente docente, pp. 22 y 36–37. |
| Apunte sobre el decodificador | [Del decodificador al próximo token](ejercicio4/apuntes/decodificador_y_prediccion.pdf), una página. | Atención cruzada, logits, softmax y generación autorregresiva. Fuente docente, pp. 22 y 40–44. |
| Diapositivas de Transformers | [Clase 1 · Introducción a Transformers](<../docs/Clase 1 - Introducción a Transformers.pdf>), 70 páginas, 2026. | Primera comparación de diapositivas sobre la misma unidad. |
| Diapositivas de Fine-tuning | [Clase 3 · Transfer Learning & Finetuning](<../docs/Clase 3 - Transfer Learning & Finetuning.pdf>), 77 páginas, 2026. | Siguiente unidad con diapositivas disponibles. Registrar si para esa unidad se consiguen apuntes preexistentes. |
| Diapositivas de Embeddings | [Clase 2 · Transformación de Texto a Embeddings](<../docs/Clase 2 - Transformación de Texto a Embeddings(1).pdf>), 65 páginas. La portada dice 73.64 y 2025. | Verificar si corresponde al material usado en 73.69 antes de incluirlo en la comparación principal. |

Transformers es el primer tema porque tenemos un apunte de prueba y diapositivas de la misma clase. El PDF derivado sobre embeddings de `output/pdf/` se preparó con el Caso 01 ya conocido y queda como antecedente del prototipo. El apunte de Transformers se preparó a partir de la clase, antes de definir las consultas del piloto.

## 2. Incorporar PDFs con páginas citables

1. Registrar cada PDF en un manifiesto con título, tipo de fuente, ruta y SHA‑256. Mantener los VTT e índices del ejercicio 2 como referencia.
2. Extraer solo texto por página con `pypdf`, como en el notebook de la cátedra. Guardar el número de página del archivo desde 1 antes de descartar páginas sin texto. El contenido gráfico queda fuera del índice. [Notebook del reporte ONU, celda 8](../docs/RAG/UN_Report_Advanced_RAG.ipynb)
3. Aplicar el splitter recursivo de G4 dentro de cada página, con un máximo de 160 tokens medidos por MiniLM y 10 de solapamiento. La metadata de cada chunk PDF contiene `fuente`, `tipo_fuente` y `pagina`. Cada chunk tiene su propio ID y texto. Los chunks VTT conservan sus tiempos.
4. Guardar cada variante en su propia colección Chroma, con el mismo `text-embedding-3-small`, 1.536 dimensiones y distancia coseno. Verificar que los chunks de VTT coincidan entre variantes y registrar el número de chunks por tipo de fuente.
5. Resolver las referencias del generador desde los metadatos. Una cita PDF muestra el nombre del documento y su página. Una cita VTT muestra el nombre y el intervalo. Verificar con un ejemplo de cada tipo que el localizador permite encontrar la evidencia.

### Carga del apunte de Transformers

La colección `openai_e4_apuntes` se guardó en la misma base Chroma de I4 y se reabrió para verificarla. Contiene **632 chunks**, con **623 de VTT copiados de `openai_i4` y 9 del PDF**. Chroma conserva ambas colecciones. En el apunte se guardaron **4 chunks de la página 1 y 5 de la página 2**. La fórmula de atención quedó asociada a la primera página y la regla de máscara causal a la segunda. Cada chunk del PDF tiene `fuente`, `tipo_fuente` y `pagina`.

Esta primera carga usó solo el apunte de dos páginas. Sus dos borradores de una página quedaron fuera. Al trasladar el apunte se actualizó `fuente` en los nueve chunks y en el manifiesto. Los IDs, el texto, las páginas y los vectores coincidieron antes y después.

Se reutilizaron el texto, la metadata y los embeddings de los VTT. La lectura posterior a la copia de los vectores mostró diferencias de redondeo de hasta `1,49e-8` por componente. La verificación usa una tolerancia absoluta de `1e-7`. El primer intento se revirtió al detectar la diferencia con la igualdad exacta, así que se repitió la llamada de embeddings del PDF. La carga final reportó **971 tokens de entrada** para los nueve chunks con `text-embedding-3-small`. [Script de carga](ejercicio4/indexar_apuntes.py) · [Manifiesto local](ejercicio4/data/carga_apuntes.json)

Para reproducir desde la raíz con las [dependencias instaladas](ejercicio4/requirements.txt) y la colección de I4 disponible, usar un nombre y un manifiesto nuevos si la colección ya existe.

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/indexar_apuntes.py \
  --coleccion openai_e4_apuntes_repeticion \
  --salida tp3/ejercicio4/data/carga_apuntes_repeticion.json
```

### Carga de los tres apuntes

Se incorporaron los dos PDFs adicionales a `openai_e4_apuntes_ampliado`. Cada uno aportó 3 chunks de su primera página. La colección tiene **638 chunks**: 623 de VTT y 15 de los tres PDFs. Reutilizó los vectores de los VTT y los 9 del PDF original. Se enviaron a embeddings solo los 6 chunks nuevos, con 664 tokens de entrada. Al reabrir Chroma se verificaron el conteo, los IDs y la metadata de cada uno de los 15 chunks PDF. `openai_i4` y `openai_e4_apuntes` siguen disponibles para la comparación. [Manifiesto local](ejercicio4/data/carga_apuntes_ampliado.json)

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/indexar_apuntes.py \
  --desde-manifiesto tp3/ejercicio4/data/carga_apuntes.json \
  --pdf tp3/ejercicio4/apuntes/bloque_del_encoder.pdf \
  --pdf tp3/ejercicio4/apuntes/decodificador_y_prediccion.pdf \
  --coleccion openai_e4_apuntes_ampliado \
  --salida tp3/ejercicio4/data/carga_apuntes_ampliado.json
```

## 3. Comparación controlada

| Variante | Corpus | Pregunta |
| --- | --- | --- |
| Base | VTT de Embeddings, Transformers y Fine-tuning. | ¿Cuánta evidencia recupera G4 con el corpus del ejercicio 2? |
| Con apuntes | Mismos VTT más los tres PDFs de Transformers. | ¿Qué criterios adicionales se respaldan al incorporar todos los apuntes disponibles? |
| Con apuntes y diapositivas | Mismos VTT y apuntes más diapositivas. | ¿Qué aportan las diapositivas sobre el corpus con apuntes? |

Por objetivo se congela el audio transcripto, la consulta original y sus tres subconsultas. Las variantes usan la misma expansión, cinco candidatos por consulta, deduplicación, el mismo CrossEncoder y siete chunks finales. Se mantienen el prompt y el snapshot de GPT‑4.1 de G4, temperatura 0 y hasta 1.000 tokens de salida. La única variable experimental es el conjunto de fuentes disponibles. El formato de las citas se resuelve por código con el mismo criterio en cada variante. [Configuración del ejercicio 2](ejercicio2/README.md)

**Casos.** Usar el Caso 01 como control de integración. Empezar la comparación de corpus con un par nuevo de objetivo y video sobre Transformers, y sumar otro par en una prueba posterior para revisar si el resultado se repite. Para cada caso, registrar el criterio, la evidencia académica y el estado esperado del video antes de ejecutar. Usar los mismos criterios y denominadores para todas las variantes de ese caso. Registrar qué evidencia existe en cada corpus para distinguir falta de material, fallo de búsqueda y fallo del generador.

## 4. Evaluación y registro

| Nivel | Qué medir |
| --- | --- |
| Indexación | Documentos y páginas leídos, páginas con extracción deficiente, cantidad de chunks por fuente y trazabilidad de las citas. |
| Recuperación | Context Recall y Context Precision sobre los chunks finales. Registrar qué tipos de fuente entran y si aportan evidencia nueva o repetida. |
| Informe | Acierto de estados, citas localizables y con respaldo, Answer Faithfulness, relevancia manual y completitud. |

Guardar por caso las fuentes y sus hashes, los rankings, los siete chunks enviados, el informe y la evaluación. Para cada criterio que cambia, mostrar el pasaje que se ganó o se perdió y si el informe lo usó. Priorizar que aumente el Context Recall y revisar si los criterios adicionales se traducen en estados y citas más sólidos. Registrar el costo adicional de indexar y la longitud real del contexto.

**Primer piloto completado.** El [Caso 02 de Transformers](ejercicio4/evaluacion/caso_02_transformers.md) comparó `openai_i4` con `openai_e4_apuntes_ampliado`, ambos con `k=7`, la misma transcripción y búsqueda sin filtro por fuente. La propuesta inicial sumaba un PDF para medir su aporte individual. La prueba sumó los tres apuntes para medir el efecto del corpus ampliado. Context Recall@7 pasó de 0/3 a 1/3 y Context Precision@7 quedó en 14,3 % en ambas variantes. [Resultados y auditoría](ejercicio4/evaluacion/resultado_caso_02_apuntes.md)

**Siguiente contraste.** Comparar `k=7` con `k=10` sobre los mismos 11 candidatos reordenados de la colección mixta. La fórmula quedó octava. Después, verificar el resultado en otro objetivo y evaluar las diapositivas en una comparación separada.
