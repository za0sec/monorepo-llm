# Ejercicio 3 · Responsible AI, Safety y puesta en producción

Nuestro sistema recibe un objetivo de estudio y el audio de un video. Whisper transcribe el audio, el objetivo se expande en tres subconsultas que buscan en Chroma sobre los materiales de la cátedra, un CrossEncoder reordena los candidatos y GPT‑4.1 recibe los 7 mejores junto con la transcripción. El resultado indica qué contenidos están cubiertos, parciales, sin evidencia en el video o sin respaldo suficiente en la clase.

## 1. Responsible AI y Safety

### 1.1 Principios de Responsible AI

Estos son algunos de los principio que aplicarían a nuestro sistema
| Principio | Riesgo en nuestro sistema | Medidas actuales y propuestas |
| --- | --- | --- |
| Transparencia y supervisión humana | El estudiante puede tomar el informe como definitivo. En la prueba de cierre, todas las citas eran localizables. El informe recomendó una slide para estudiar la secuencia completa de tokenización, IDs y embeddings, aunque el diagrama de esa slide omitía la asignación de IDs. | El informe ya muestra las citas disponibles junto a cada estado y distingue cuándo falta evidencia del video o respaldo de clase. Proponemos agregar un botón por fila para reportar estados incorrectos o citas que no respaldan lo escrito e incorporar esos casos a la evaluación. |
| Equidad y robustez | Whisper puede perder evidencia con ciertos acentos, ruido o música. El reranker, entrenado para ordenar textos en inglés, podría evitar pasajes relevantes en español. | En seis recortes del video de Embeddings, Whisper obtuvo una tasa de error de palabras (WER) del 5,87 %. Ampliar la prueba a otros videos con transcripciones de referencia y comparar el WER y el acierto de los estados por acento y calidad de audio. Limitar la duración del audio. |
| Privacidad de datos | Algunos VTT de clase identifican a quienes hablan. OpenAI recibe el audio para transcribir y la transcripción y los pasajes de clase para generar el informe. | Quitar los nombres antes de indexar y pedir permiso para usar las grabaciones de clase. Informar qué datos se envían y fijar plazos de borrado para el audio y la transcripción, incluso si se guarda para reutilizarla. |

### 1.2 Safety

Un video puede incluir instrucciones habladas que intenten alterar su evaluación. Es un caso de prompt injection. Para reducir el riesgo, el prompt actual presenta la transcripción en un bloque separado e indica que las fuentes se traten como datos. Probaríamos el mismo video con y sin una instrucción ajena al contenido y revisaríamos si cambian los estados o el respaldo de las citas.

También validaríamos la estructura y los estados permitidos. El sistema ya marca el informe con «Requiere revisión» cuando detecta un ID de cita ausente en la entrada. El respaldo que aporta cada cita se comprueba al leer la fuente.

## 2. Estrategias para poner el sistema en producción

El sistema se despliega en AWS y se divide en dos flujos: la indexación, que corre cuando cambia el material, y la consulta, que corre por cada video. Los dos usan la API de OpenAI.

![Figura 1. Arquitectura de despliegue en AWS.](arquitectura_nube.png)

### 2.1 Indexación

El material de la cátedra (transcripciones, apuntes y diapositivas) se guarda en Amazon S3. Cuando se agrega o cambia un archivo, un proceso de indexación lo carga, lo divide con el corte recursivo de 160 tokens y solapamiento 10, describe las diapositivas con GPT‑4.1 y calcula los embeddings con text-embedding-3-small. Los vectores se guardan en Chroma, la misma base de los experimentos, que corre como servidor en AWS con un disco persistente. Por el volumen del corpus no hace falta un servicio administrado, y si la base se pierde se reconstruye desde S3.

### 2.2 Servicio

Una interfaz web, con las mismas funciones que la que probamos de forma local, se publica en un servidor en la nube con acceso solo para estudiantes autenticados. El servidor recibe el objetivo y el MP3, guarda el audio de forma temporal y ejecuta el flujo en segundo plano, porque transcribir un video lleva tiempo. Whisper, los embeddings y GPT‑4.1 se usan por la API de OpenAI, así que no hace falta una GPU propia; el reranker es liviano y corre en CPU en el mismo servidor. Si falla la expansión o el reranker, se usa el modo básico, que trae los 7 chunks más cercanos por coseno.

### 2.3 Costo

Con la configuración final, cada informe usa en promedio unos 2.700 tokens de entrada en GPT‑4.1, a lo que se suman la expansión, los embeddings y los minutos de Whisper. Para bajar el costo guardaríamos la transcripción de cada audio, ya que varios estudiantes pueden pedir el mismo video, y limitaríamos la duración. Un modelo más barato, como gpt-4.1-nano, solo reemplazaría al actual si obtiene las mismas métricas.

### 2.4 Monitoreo y versiones

En cada ejecución registraríamos las consultas, los chunks, los puntajes, los tokens y la latencia, por ejemplo con LangSmith. Habría alertas ante informes incompletos, puntajes de reranking muy bajos o muchos estados marcados como incorrectos. Fijaríamos la versión de cada modelo, y cualquier cambio de prompt, modelo o parámetros se evaluaría antes con los seis videos que usamos en los experimentos. Como son pocos casos, antes de abrir el sistema a usuarios tendríamos que ampliarlos con más objetivos y videos.
