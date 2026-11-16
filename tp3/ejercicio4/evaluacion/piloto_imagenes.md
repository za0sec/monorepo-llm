# Piloto de indexación de imágenes en diapositivas

Se compara el texto seleccionable con dos representaciones del contenido gráfico, OCR y descripción generada. El piloto reutiliza el índice textual del TP como diseño y usa tres colecciones de prueba con las mismas ocho páginas. Las diapositivas viables se incorporarán al corpus completo en una etapa posterior.

## Entradas congeladas

| Consulta literal | Fuente esperada | Evidencia completa |
| --- | --- | --- |
| ¿Cómo se calcula la similitud coseno entre dos embeddings? | Embeddings, página original 32. | El numerador es el producto escalar de los dos vectores. El denominador es el producto de sus normas euclídeas. |
| ¿Qué bloque conecta el encoder con el decoder? | Transformers, página original 22. | La salida del encoder alimenta el bloque intermedio de atención multi-cabeza del decoder (atención cruzada), situado encima de la autoatención enmascarada. |

Las páginas 31 y 33 de Embeddings y 25, 27, 30 y 36 de Transformers actúan como fuentes de temas cercanos en las tres colecciones. Se conserva el texto seleccionable de las ocho páginas. Los dos recortes se procesan con OCR y con el modelo que genera la descripción. Para indexar el texto de cada recorte, usar un documento aparte con el mismo PDF, la página original y el método.

La variante es la forma de representar los dos recortes. Fijar el recursivo de 160 tokens de MiniLM y 10 de solapamiento, `text-embedding-3-small`, coseno, consulta literal y el CrossEncoder de G4. En este piloto reducido usamos una sola consulta por ejemplo, obtenemos hasta 12 candidatos, reordenamos y evaluamos `k=7` final en las tres variantes.

## Evaluación manual

- **Extracción.** Cotejar los dos textos que genera cada método con el recorte. Registrar qué operandos y relaciones se recuperan y si aparecen errores de símbolos, etiquetas o relaciones inventadas.
- **Context Recall@7.** Contar cada consulta como cubierta si los chunks finales, en conjunto, respaldan toda su evidencia. Informar el resultado por consulta y sobre las dos en conjunto.
- **Context Precision@7.** Dar relevancia a un chunk si respalda directamente alguna parte de la evidencia esperada. El título de la página o una lista de términos sin un paso de la fórmula o sin una relación entre bloques quedan fuera de ese criterio. Promediar la precisión acumulada en las posiciones relevantes. Si ninguna aporta evidencia directa, asignar cero.
- **Procedencia.** Registrar la posición, el documento y la página original de cada fragmento que aporta la evidencia. La prueba termina en recuperación.

## Ejecución y resultados

Se renderizaron las figuras con Poppler y se revisaron los recortes antes de compararlos. El OCR se hizo con Apple Vision en macOS 27.2.1, reconocimiento preciso en español e inglés y corrección de lenguaje desactivada. El primer intento en el sandbox no pudo usar Apple Vision. El reintento con acceso al servicio local produjo las dos lecturas. Se describieron los mismos PNG con `gpt-4.1-2025-04-14`, temperatura 0, detalle alto y límite de 450 tokens. El prompt pidió transcribir fórmulas e indicar el origen y destino de las flechas sin agregar información externa. Se reutilizaron los embeddings del texto base y de las dos consultas entre las variantes. Las tres colecciones tienen 8, 10 y 12 chunks, respectivamente. Se reabrieron los índices para verificar IDs, vectores, textos y páginas.

### Texto obtenido de los recortes

| Figura | OCR | Descripción |
| --- | --- | --- |
| Embeddings, p. 32 | Leyó `A•B`, confundió `θ` con `0` y no reconstruyó el denominador ni la fracción completa. | Transcribió completos el producto escalar y el producto de normas, también en notación expandida. Describió mal la dirección del vector A, que en la figura apunta hacia arriba y a la derecha. |
| Transformers, p. 22 | Leyó las etiquetas, con un error en “Add & Norm”, y no expresó las relaciones de las flechas. | Reconoció que la salida del encoder llega al decoder y enumeró los módulos. Omitió que la flecha diagonal llega específicamente al bloque intermedio de atención multi-cabeza. |

### Evidencia en el top 7 final

| Variante | Recall fórmula | Recall diagrama | Context Recall@7 total | Context Precision@7 promedio |
| --- | ---: | ---: | ---: | ---: |
| Texto seleccionable | 0/1 | 0/1 | 0/2 | 0 % |
| Texto + OCR | 0/1 | 0/1 | 0/2 | 0 % |
| Texto + descripción | 0/1 | 0/1 | 0/2 | 50 % |

En la consulta de coseno, las tres variantes recuperaron el título de la p. 32 en el puesto 1. En la variante con descripción quedó sexta la parte que anuncia la fórmula. El splitter dejó la fórmula correcta en el siguiente chunk, que quedó octavo después del reranking.

En la consulta sobre el diagrama, la descripción de la conexión general quedó primera. Se anotó como evidencia parcial porque identifica que la salida del encoder llega al decoder. La lista de módulos quedó segunda y ninguno de los dos fragmentos identifica el bloque que recibe esa salida. La precision para esta consulta fue 1/1 y para la consulta de fórmula fue 0, con promedio de 50 %. El recall requiere que quede respaldado el destino específico de la flecha.

**Decisión.** La descripción de la fórmula mostró que se puede recuperar información visual correcta, pero en este piloto ninguna variante agregó evidencia completa al top 7. Para fórmulas, queda justificado probar título y fórmula en un mismo chunk antes de repetir la búsqueda. Para diagramas, queda pendiente evaluar una descripción que explicite el destino de cada flecha importante. La colección principal `openai_e4_texto` conserva sus 1.163 chunks.

### Reproducción

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/comparar_imagenes.py preparar \
  --salida tp3/ejercicio4/data/experimentos/piloto_imagenes_repeticion
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/comparar_imagenes.py comparar \
  --salida tp3/ejercicio4/data/experimentos/piloto_imagenes_repeticion
```

El primer paso usa `pdftoppm`, `swiftc` y Apple Vision. El segundo necesita la clave de OpenAI y el CrossEncoder local de G4. La ejecución original consumió 1.134 tokens de entrada en embeddings y 2.062 de entrada y 327 de salida en visión. [Preparación y OCR](../data/experimentos/piloto_imagenes/preparacion.json) · [Rankings finales](../data/experimentos/piloto_imagenes/resultados.json) · [Descripción de la fórmula](../data/experimentos/piloto_imagenes/embeddings-32-descripcion.json) · [Descripción del diagrama](../data/experimentos/piloto_imagenes/transformers-22-descripcion.json)

## Comparación de recorte y diapositiva completa

La variante es la imagen que recibe GPT‑4.1. Las dos colecciones conservan las mismas ocho páginas de texto seleccionable y agregan dos descripciones, una de la fórmula y otra del diagrama. Se regeneran las descripciones de los recortes con el mismo prompt que se usa para las páginas completas. Se pasa de «recorte» a «imagen de material académico» en la primera frase. Se renderizan las páginas a la misma escala del piloto y se reutilizan sus recortes exactos. Se mantienen las dos consultas, la pauta, el modelo, los parámetros de visión, el splitter, los embeddings, el reranker y el top 7.

### Resultados

| Variante | Recall fórmula | Recall diagrama | Context Recall@7 total | Context Precision@7 promedio |
| --- | ---: | ---: | ---: | ---: |
| Texto + descripción del recorte | 0/1 | 0/1 | 0/2 | 50 % |
| Texto + descripción de la diapositiva completa | 0/1 | 0/1 | 0/2 | 50 % |

**Embeddings, página 32.** Ambas descripciones transcribieron bien la fórmula. Con la diapositiva completa el modelo también incluyó el título e identificó explícitamente la similitud coseno. El chunk que anuncia la fórmula subió del puesto 6 al 1. El chunk que contiene la fórmula pasó del puesto 11 al 8. El splitter separó la fórmula de su introducción en ambos formatos y ninguno la llevó al top 7.

**Transformers, página 22.** Las dos descripciones reconocieron el vínculo general entre encoder y decoder y enumeraron las capas. La página completa agregó el ejemplo de traducción y describió cada capa seguida de Add & Norm, de forma coherente con el flujo de abajo hacia arriba del diagrama. Ambas omitieron el destino concreto de la flecha diagonal. El vínculo general quedó primero en las dos variantes y se contó como evidencia parcial, con Context Precision 1 para esta consulta y 0 para la de coseno.

Cada colección tiene 12 chunks. Al reabrirlas se verificaron los textos, los metadatos y los vectores. Los dos formatos compartieron los embeddings de texto seleccionable y de las consultas. El total fue de 1.412 tokens de embeddings. Las dos descripciones de páginas usaron 2.392 tokens de entrada y 337 de salida, y las de recortes, 2.052 de entrada y 320 de salida.

**Decisión.** Probar el efecto de conservar el título y la fórmula juntos en un fragmento sobre el mismo corpus piloto. Este contraste mostró posiciones mejores para la fórmula con contexto de página y dejó igual la evidencia que efectivamente llega al top 7. La lectura del destino de flechas sigue pendiente.

```bash
PYTHONPATH=tp3/ejercicio2 .venv/bin/python tp3/ejercicio4/comparar_imagenes.py comparar-pagina \
  --salida tp3/ejercicio4/data/experimentos/piloto_diapos_completas_repeticion
```

[Rankings de ambos formatos](../data/experimentos/piloto_diapos_completas/resultados.json) · [Descripción de la página de coseno](../data/experimentos/piloto_diapos_completas/embeddings-32-pagina-descripcion.json) · [Descripción de la página de Transformers](../data/experimentos/piloto_diapos_completas/transformers-22-pagina-descripcion.json)
