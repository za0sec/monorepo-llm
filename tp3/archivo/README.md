# Archivo histórico del TP3

El [ejercicio 2 anterior](ejercicio2_anterior/) conserva la carpeta completa al 14/11/2026 (base Git `51ce249`): código, pruebas, transcripciones, chunks, índice, cachés, resultados y documentación. Los 77 archivos previamente versionados se trasladaron junto con 98 archivos locales ignorados. El `.gitignore` original conserva la separación entre los artefactos de texto versionables y los audios, índices, cachés, logs y otros archivos locales.

Candidatos para recuperar después de validar sus entradas y comportamiento:

- [Lector de VTT](ejercicio2_anterior/preparar_transcripcion.py) y [pruebas de chunks](ejercicio2_anterior/tests/test_preparar_chunks.py): cues, límites y procedencia.
- [Caso manual de embeddings](ejercicio2_anterior/evaluacion/caso_01_embeddings.md), apartado «Criterios esperados»: puede ayudar a calibrar la pauta en desarrollo.
- [Transcriptor anterior](ejercicio2_anterior/transcribir_audio.py) y [adaptador del proxy](ejercicio2_anterior/proxy_catedra.py). La comparación de transcripción y sus comandos están en el [ejercicio 2 activo](../ejercicio2/transcripcion/README.md).
- [Plan de reconstrucción del 09/11](plan_evaluacion_experimental_anterior.md). Documenta el recorrido previo al reinicio.
- [Ablación previa de k](ejercicio2_anterior/evaluacion/ablacion_k.md) y [registro de generación](ejercicio2_anterior/evaluacion/generacion_caso01.md): referencias para formular hipótesis y evitar repetir diagnósticos ya conocidos.

La copia conserva los archivos tal como estaban. Algunos enlaces, rutas e instrucciones internas todavía presuponen la ubicación original `tp3/ejercicio2/`. Si se necesita reproducir una corrida histórica, recuperar esa estructura en un checkout separado y aportar los insumos locales que correspondan. Los resultados de ese recorrido pertenecen al prototipo anterior. El [roadmap vigente](../roadmap_ejercicio2.md) define el reinicio.
