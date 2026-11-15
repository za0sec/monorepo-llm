# TP3 · Plan del ejercicio 4

## Objetivo de la iteración

Medir cuánto aporta agregar **apuntes de estudiantes y diapositivas de clase** al corpus de VTT del ejercicio 2. En el Caso 01, la configuración G4 llevó al generador respaldo completo para 3 de 5 criterios. Algunos faltantes provenían de la selección de candidatos y el generador también omitió un criterio con evidencia disponible. El corpus ampliado permite comprobar si nuevas fuentes recuperables aportan definiciones, fórmulas o ejemplos y si el generador los usa en el informe. La consigna pide documentar la limitación, la modificación y los resultados antes y después. [G4](ejercicio2/experimentos.md#g4--expansión-con-reranking-de-los-candidatos-de-g3) · [Consigna, p. 3](../docs/RAG/tp3.pdf)

## 1. Seleccionar las fuentes

| Fuente | Disponible | Uso previsto |
| --- | --- | --- |
| Apuntes de prueba | [Apuntes de Transformers](../output/pdf/ejercicio4/apuntes_transformers.pdf), dos páginas elaboradas a partir de la clase. | Primer PDF del piloto. La página 1 trata atención y la página 2, codificación posicional y máscara causal. |
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

## 3. Comparación controlada

| Variante | Corpus | Pregunta |
| --- | --- | --- |
| Base | VTT de Embeddings, Transformers y Fine-tuning. | ¿Cuánta evidencia recupera G4 con el corpus del ejercicio 2? |
| Con apuntes | Mismos VTT más el PDF de prueba de Transformers. | ¿Qué criterios adicionales se respaldan al incorporar apuntes? |
| Con apuntes y diapositivas | Mismos VTT y apuntes más diapositivas. | ¿Qué aportan las diapositivas sobre el corpus con apuntes? |

Por objetivo se congela el audio transcripto, la consulta original y sus tres subconsultas. Las variantes usan la misma expansión, cinco candidatos por consulta, deduplicación, el mismo CrossEncoder y siete chunks finales. Se mantienen el prompt y el snapshot de GPT‑4.1 de G4, temperatura 0 y hasta 1.000 tokens de salida. La única variable experimental es el conjunto de fuentes disponibles. El formato de las citas se resuelve por código con el mismo criterio para las tres variantes. [Configuración del ejercicio 2](ejercicio2/README.md)

**Casos.** Usar el Caso 01 para verificar que una corrida con VTT sigue funcionando. Para comparar corpus, preparar al menos dos pares nuevos de objetivo y video. Empezar con Transformers y dejar por escrito qué criterio se espera, dónde lo respalda la clase y qué estado le corresponde según el video antes de ejecutar. Usar los mismos criterios y denominadores en las tres variantes. Registrar también qué evidencia existe en cada corpus para distinguir falta de material, fallo de búsqueda y fallo del generador.

## 4. Evaluación y registro

| Nivel | Qué medir |
| --- | --- |
| Indexación | Documentos y páginas leídos, páginas con extracción deficiente, cantidad de chunks por fuente y trazabilidad de las citas. |
| Recuperación | Context Recall en el conjunto de candidatos y en el top 7. Context Precision en el top 7. Registrar qué tipos de fuente entran y si aportan evidencia nueva o repetida. |
| Informe | Acierto de estados, citas localizables y con respaldo, Answer Faithfulness, relevancia manual y completitud. |

Guardar por caso las fuentes y sus hashes, los rankings, los siete chunks enviados, el informe y la evaluación. Para cada criterio que cambia, mostrar el pasaje que se ganó o se perdió y si el informe lo usó. Priorizar que aumente el Context Recall y revisar si los criterios adicionales se traducen en estados y citas más sólidos. Registrar el costo adicional de indexar y la longitud real del contexto.

**Primer piloto.** Se verificó que el apunte de dos páginas permite extraer por separado los dos temas y sus fórmulas. El siguiente paso es cargar ese PDF y comparar los VTT solos contra los VTT con el apunte, ambos con `k=7`. Después se compara `k=7` con `k=10` sobre un mismo ranking ampliado, si hay al menos diez candidatos distintos. Las diapositivas quedan para una comparación posterior.
