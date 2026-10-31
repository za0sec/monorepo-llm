# Repository Guidelines

## Prioridad y contexto actual

**Estamos trabajando en el TP3. `docs/RAG` (referido también como `Docs/RAG`) es la principal fuente académica y técnica de sus requisitos.** Antes de implementar funcionalidades del TP3, leer [GUIA_TP3.md](docs/RAG/GUIA_TP3.md), las consignas originales y las clases correspondientes. La guía identifica contradicciones de fechas que deben confirmarse con la cátedra.

## Estructura y arquitectura

- `tp1/`: predicción de BTR con PyTorch; `experiments/` procesa, entrena y evalúa; `plots/` grafica CSV de `output/`; `data/` contiene splits preparados.
- `tp2/`: paper, consigna y apuntes de defensa; sin aplicación propia.
- `tp3/ejercicio1/`: propuesta y diagrama del evaluador de cobertura de videos.
- `tp3/ejercicio2/`: preparación de VTT, transcripción con Whisper y caso manual de evaluación. **El RAG automático todavía no está implementado.** Chroma, MiniLM y los generadores son propuestas/referencias; distinguirlas de dependencias instaladas.
- `docs/`: material local de cátedra, excluido de Git; respetar las minúsculas para portabilidad.

## Instalación y ejecución

Desde la raíz, Python 3.9+ para preparar VTT:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r tp3/ejercicio2/requirements.txt
python3 tp3/ejercicio2/preparar_transcripcion.py
python3 tp3/ejercicio2/transcribir_audio.py "ruta/al/audio.mp3"
```

La transcripción puede consumir API; requiere audio y configuración de `tp3/.env`. No imprimir ni versionar credenciales. Para comandos del TP1, dependencias y directorios de ejecución, consultar la guía (§9).

## Código y formatos

Usar español en documentación, comentarios y justificaciones. Seguir Python existente: cuatro espacios, funciones/variables `snake_case`, clases `PascalCase`, constantes `UPPER_SNAKE_CASE`. No hay formatter ni linter configurados. Conservar fuentes, IDs y tiempos; separar cómputo y gráficos según [CLAUDE.md del TP1](tp1/CLAUDE.md).

## Testing y validación

No hay suite automatizada ni CI configurada. Validar VTT con `python3 tp3/ejercicio2/preparar_transcripcion.py --salida /tmp/tp3-vtt-check` y cotejar segmentos originales. `evaluate_test.py` evalúa el modelo del TP1 y reentrena; no es una suite. Para TP3, validar evidencias y métricas por módulo antes de comparar configuraciones.

## Commits y pull requests

El historial usa mensajes breves, mayormente en español; no impone Conventional Commits. Describir cambio, motivo y validación en cada PR; es una pauta propuesta, sin plantilla existente. Respetar [CLAUDE.md](CLAUDE.md): commits sin trailers de coautoría.

## Instrucciones para agentes

Preservar cambios ajenos y originales docentes. Citar páginas, tiempos o celdas; separar obligaciones, ejemplos y propuestas. Justificar decisiones; avisar antes de incorporar técnicas ajenas al material. Completar el pipeline básico antes de Advanced RAG y extras. No presentar el caso manual como resultado automático. La guía bajo `docs/` requiere inclusión explícita para compartirse por Git.
