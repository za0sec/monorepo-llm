# Repository Guidelines

## Prioridad y contexto actual

**Estamos trabajando en el TP3. `docs/RAG` (referido también como `Docs/RAG`) es la principal fuente académica y técnica de sus requisitos.** Antes de implementar funcionalidades del TP3, leer [GUIA_TP3.md](docs/RAG/GUIA_TP3.md), las consignas originales y las clases correspondientes. La guía identifica contradicciones de fechas que deben confirmarse con la cátedra.

## Estructura y arquitectura

- `tp1/`: predicción de BTR con PyTorch; `experiments/` procesa, entrena y evalúa; `plots/` grafica CSV de `output/`; `data/` contiene splits preparados.
- `tp2/`: paper, consigna y apuntes de defensa; sin aplicación propia.
- `tp3/ejercicio1/`: propuesta y diagrama del evaluador de cobertura de videos.
- `tp3/ejercicio2/`: carpeta del reinicio, reservada para la implementación nueva. El [roadmap experimental](tp3/roadmap_ejercicio2.md) define etapas, comparaciones y métricas.
- `tp3/archivo/ejercicio2_anterior/`: código, datos y resultados históricos del prototipo. Consultar el [inventario del archivo](tp3/archivo/README.md) antes de reutilizar una pieza. La guía local conserva una fotografía anterior del estado del código.
- `docs/`: material local de cátedra, excluido de Git; respetar las minúsculas para portabilidad.

## Instalación y ejecución

El ejercicio 2 nuevo aún no tiene comandos ni dependencias. Documentarlos al implementar cada etapa. Si una futura prueba de ASR usa la API, configurar las credenciales localmente en `tp3/.env`. No imprimirlas ni versionarlas. Para comandos del TP1, dependencias y directorios de ejecución, consultar la guía (§9).

## Código y formatos

Usar español en documentación, comentarios y justificaciones. Seguir Python existente: cuatro espacios, funciones/variables `snake_case`, clases `PascalCase`, constantes `UPPER_SNAKE_CASE`. No hay formatter ni linter configurados. Conservar fuentes, IDs y tiempos; separar cómputo y gráficos según [CLAUDE.md del TP1](tp1/CLAUDE.md).

## Testing y validación

El ejercicio 2 nuevo aún no tiene pruebas. Validar la trazabilidad del VTT, la integridad de los chunks y la persistencia del índice al desarrollar esas etapas. Aplicar los controles y métricas por módulo del roadmap antes de comparar configuraciones. No hay CI. `evaluate_test.py` reentrena/evalúa TP1.

## Commits y pull requests

El historial usa mensajes breves, mayormente en español; no impone Conventional Commits. Describir cambio, motivo y validación en cada PR; es una pauta propuesta, sin plantilla existente. Respetar [CLAUDE.md](CLAUDE.md): commits sin trailers de coautoría.

## Instrucciones para agentes

Actualizar [tp3/presentación.md](tp3/presentación.md) en cada avance significativo del TP3: registrar en orden cronológico el cambio, su motivo, alcance/configuración, pruebas realizadas, resultados y siguiente paso. Conservar decisiones anteriores y sus reemplazos; distinguir implementaciones, resultados manuales y propuestas pendientes. Es la bitácora compartida para ponerse al día con el proyecto.

Preservar cambios ajenos y originales docentes. Citar páginas, tiempos o celdas; separar obligaciones, ejemplos y propuestas. Justificar decisiones; avisar antes de incorporar técnicas ajenas al material. Completar el pipeline básico antes de Advanced RAG y extras. No presentar el caso manual como resultado automático. La guía bajo `docs/` requiere inclusión explícita para compartirse por Git.
