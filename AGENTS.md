# Repository Guidelines

## Prioridad y contexto actual

**Estamos trabajando en el generador del ejercicio 2 del TP3. El próximo experimento es G1, la comparación de prompts.** Para retomar, leer el [registro experimental](tp3/ejercicio2/experimentos.md#g0--informe-básico) y el [roadmap](tp3/roadmap_ejercicio2.md). Antes de implementar, consultar [la guía](docs/RAG/GUIA_TP3.md), las consignas originales y las clases pertinentes. La guía reúne el material académico y señala contradicciones de fechas que deben confirmarse con la cátedra.

## Estado del TP3 al 14/11/2026

- **Producto.** Evaluar qué cubre el contenido hablado de un video para un objetivo de estudio. El material académico establece qué debería aprenderse. El VTT del video aporta evidencia de lo que se explica y entra al generador por separado del corpus académico. El informe para el estudiante explica la cobertura, recomienda pasajes de clase y concluye para qué le sirve el video. Mantener intacto el ejercicio 1, que ya fue corregido.
- **Indexación y recuperación.** El corpus de desarrollo tiene los VTT docentes de Embeddings, Transformers y Fine-tuning. Se eligió el splitter recursivo de hasta 160 tokens, contados con el tokenizer de MiniLM, y 10 de solapamiento. Se congelaron 623 chunks con fuente, IDs y tiempos para comparar MiniLM y `text-embedding-3-small`. El modelo de OpenAI, con 1.536 dimensiones, quedó como referencia de desarrollo en la colección persistente `openai_i4` de Chroma con distancia coseno. Con `k=7` se obtuvieron 9/14 criterios completos en las diez consultas, con 66,7% de Context Recall y 60,6% de Context Precision. Priorizar cobertura y, a igualdad de criterios completos, comparar la posición de la evidencia y el contexto requerido. Variar la dimensión del vector queda fuera de la prioridad actual. [Experimentos I1–I4 y R1](tp3/ejercicio2/experimentos.md)
- **Audio del Caso 01.** La transcripción completa de Whisper tiene 99 segmentos y se mantiene fija entre variantes de generación. En seis WAV con referencia humana, Whisper obtuvo WER 5,87% y CER 3,92%, frente a 14,09% y 3,07% de Wav2Vec2. Las métricas de los recortes se registran aparte de la evaluación del informe completo. El caso y la recuperación congelada de G0 indican sus archivos de entrada. [Registro de transcripción](tp3/ejercicio2/transcripcion/README.md)
- **Generación G0.** La corrida vigente está documentada en el [registro de G0](tp3/ejercicio2/experimentos.md#g0--informe-básico) y se reproduce con [generar_g0.py](tp3/ejercicio2/generar_g0.py) y [caso_g0_embeddings.json](tp3/ejercicio2/caso_g0_embeddings.json). Usó Qwen2.5-1.5B-Instruct local, el mismo video y dos conjuntos de referencias de clase, seis pasajes seleccionados a mano y siete recuperados para el objetivo literal. El informe con pasajes manuales obtuvo 0/5 estados correctos y el recuperado 1/5. Ambos quedaron sin citas verificables en el formato pedido. El recuperado inventó IDs y agotó los 1.000 tokens de salida. Hubo ensayos preliminares en prosa y tabla; para continuar usar la corrida vigente `informes_estudiante`.
- **Regla de diseño del generador.** Los criterios salen del objetivo de estudio, el nivel y los pasajes de clase. La pauta E1–E5 del [caso manual](tp3/archivo/ejercicio2_anterior/evaluacion/caso_01_embeddings.md) se usa para corregir la salida y queda fuera del prompt. La falta de respaldo académico debe quedar explícita y distinguirse de la falta de evidencia en la transcripción. Identificar las citas de clase con `[C#]` y las de video con `[V#]`, y verificar qué afirmaciones respalda cada fuente.
- **Siguiente paso.** Probar G1 con una pauta explícita para derivar los contenidos, asignar estados, citar evidencia y recomendar pasajes concretos. Conservar modelo, caso, VTT, referencias y presupuesto para comparar con G0; reutilizar la recuperación guardada. Medir estados correctos, Answer Faithfulness, citas verificables y completitud. Validar el enfoque luego con objetivos de otras unidades. Para completar el corpus del ejercicio 2, la intención es sumar apuntes PDF preexistentes con procedencia y páginas. Reservar las diapositivas para una comparación antes/después en el ejercicio 4.

## Estructura y arquitectura

- `tp1/`: predicción de BTR con PyTorch; `experiments/` procesa, entrena y evalúa; `plots/` grafica CSV de `output/`; `data/` contiene splits preparados.
- `tp2/`: paper, consigna y apuntes de defensa; sin aplicación propia.
- `tp3/ejercicio1/`: propuesta y diagrama del evaluador de cobertura de videos.
- `tp3/ejercicio2/`: código activo, configuración de casos y bitácora experimental. `transcripcion/` reúne scripts y pruebas de ASR. Los índices, corridas y demás salidas locales se conservan en carpetas `data/` ignoradas por Git.
- `tp3/archivo/ejercicio2_anterior/`: código, datos y resultados históricos del prototipo. Consultar el [inventario del archivo](tp3/archivo/README.md) antes de reutilizar una pieza. La guía local conserva una fotografía anterior del estado del código.
- `docs/`: material local de cátedra, excluido de Git; respetar las minúsculas para portabilidad.

## Instalación y ejecución

Usar `.venv/bin/python` desde la raíz. Las dependencias están en [requirements.txt](tp3/ejercicio2/requirements.txt) y los comandos por prueba en [experimentos.md](tp3/ejercicio2/experimentos.md). Para reproducir G0 en este checkout, usar `tp3/ejercicio2/data/experimentos/g0/recuperacion.json` y una carpeta de salida nueva. La generación usa los pesos locales de Qwen en la caché del archivo. En otro checkout, verificar que estén disponibles los datos ignorados antes de intentar reproducir. Para pruebas de API, usar el destino y el alcance autorizados en la sesión, con credenciales en `tp3/.env`. No imprimirlas ni versionarlas. Para comandos del TP1, consultar la guía (§9).

## Código y formatos

Usar español en documentación, comentarios y justificaciones. Seguir Python existente: cuatro espacios, funciones/variables `snake_case`, clases `PascalCase`, constantes `UPPER_SNAKE_CASE`. No hay formatter ni linter configurados. Conservar fuentes, IDs y tiempos; separar cómputo y gráficos según [CLAUDE.md del TP1](tp1/CLAUDE.md).

## Testing y validación

Validar las entradas y salidas de la etapa que se cambia. Para generación, comprobar que el VTT llega completo, que los criterios manuales quedan fuera del prompt y que las referencias y parámetros permanecen fijos en cada comparación de prompts. Guardar las salidas crudas, el contexto y los motivos de truncamiento. Distinguir pilotos de desarrollo, anotaciones del asistente y resultados revisados por el grupo. Para indexación y recuperación, verificar la trazabilidad del VTT, los límites de tokens y la persistencia del índice. No hay CI. `evaluate_test.py` reentrena y evalúa TP1.

## Commits y pull requests

Hacer commits solo cuando el usuario los pida. El historial usa mensajes breves, mayormente en español. Describir cambio, motivo y validación en cada PR. Respetar [CLAUDE.md](CLAUDE.md) y omitir trailers de coautoría.

## Instrucciones para agentes

Actualizar [tp3/ejercicio2/experimentos.md](tp3/ejercicio2/experimentos.md) en cada avance significativo. Registrar el cambio, motivo, configuración, pruebas realizadas, resultados y decisión. Mantener el próximo paso actualizado en el [roadmap](tp3/roadmap_ejercicio2.md). Conservar las decisiones anteriores y sus reemplazos en orden cronológico. El trabajo activo queda en `tp3/ejercicio2/` y el material anterior en `tp3/archivo/`.

Preservar cambios ajenos y originales docentes. Citar páginas, tiempos o celdas; separar obligaciones, ejemplos y propuestas. Justificar decisiones; avisar antes de incorporar técnicas ajenas al material. Completar el pipeline básico antes de Advanced RAG y extras. No presentar el caso manual como resultado automático. La guía bajo `docs/` requiere inclusión explícita para compartirse por Git.
