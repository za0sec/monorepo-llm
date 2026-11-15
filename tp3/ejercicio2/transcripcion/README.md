# Comparación de transcripción

La [ficha revisada](data/asr_caso01_20261109/referencia.md) contiene seis recortes del caso 01. La evaluación usa una copia congelada de la referencia revisada. El manifiesto registra los hashes y el origen del borrador que se cotejó contra el audio.

## Piloto ASR01

Se ejecutó Wav2Vec2 en CPU con decodificación CTC greedy. El hash de los
pesos coincidió con el publicado. El recorte dura 23,5 s y la inferencia
duró 0,98 s, con 0,22 s adicionales de carga de modelo y procesador.

| Modelo | WER | CER |
| --- | ---: | ---: |
| Wav2Vec2 | 14,55 % (8/55) | 5,21 % (16/307) |
| Whisper vía proxy | Pendiente (HTTP 429) | Pendiente (HTTP 429) |

Wav2Vec2 unió «comencemos para» y escribió «envedings» por «embeddings».
Las métricas también cuentan como diferencia «72» frente a «setenta y dos»,
porque la normalización conserva los dígitos. Los tiempos CTC por palabra
son aproximados y quedan por validar. [Transcripción y tiempos](data/asr_caso01_20261109/piloto/wav2vec/ASR01.json) · [Métricas](data/asr_caso01_20261109/piloto/ASR01.metricas.json)

Con autorización explícita, se solicitó al proxy de la cátedra la
transcripción de ASR01 con `openai/whisper-1`, idioma español y temperatura
0. El intento y un reintento devolvieron HTTP 429. La respuesta del
reintento indica un límite relacionado con cuota o facturación. El proxy
no devolvió una transcripción, por lo que faltan las métricas de Whisper.
[Diagnóstico del reintento](data/asr_caso01_20261109/piloto/whisper/reintento.json)

El registro del primer intento de Whisper confirmó el mensaje de OpenAI
«You have no credits remaining». El identificador de clave virtual y el
equipo coinciden con los de una llamada exitosa del 04/11.

La prueba de conexión con la guía de LiteLLM devolvió HTTP 200 en el
catálogo autenticado, donde figura `openai/whisper-1`. Las consultas
de texto devolvieron HTTP 404 con `gpt-3.5-turbo` y con
`openai/gpt-3.5-turbo`, y HTTP 429 con `openai/gpt-4.1-2025-04-14`.
La verificación se hizo con un mensaje sintético, sin enviar audio.

## Control del VTT y seis recortes · 14/11/2026

Se midió la transcripción de Whisper que se usará en el generador, y se completaron los seis recortes de Wav2Vec2. El total de referencia es de **511 palabras y 2.602 caracteres** en 179 segundos. Para cada tasa agregada se sumaron los errores y las unidades de referencia de los seis recortes. La normalización se mantuvo igual a la del piloto. Se pasó a minúsculas, se quitó la puntuación, se unificaron los espacios y se mantuvieron tildes y números. CER se calculó sin espacios. [Métricas y textos por recorte](data/asr_caso01_20261109/comparacion_20261114/metricas.json)

| Entrada medida | WER | CER |
| --- | ---: | ---: |
| VTT histórico de Whisper, segmentos con punto medio dentro del recorte | 15,07% (77/511) | 12,80% (333/2.602) |
| VTT histórico de Whisper, todos los segmentos que se cruzan con el recorte | 23,87% (122/511) | 21,79% (567/2.602) |
| Wav2Vec2, los seis WAV exactos | 14,09% (72/511) | 3,07% (80/2.602) |

El VTT tiene tiempos por segmento. Cortar a los tiempos de los recortes puede incluir palabras anteriores o posteriores, o dejar texto de borde afuera. La diferencia entre ambas reglas muestra esa sensibilidad. Estas tasas sirven como diagnóstico del VTT.

En los intervalos revisados, el VTT conservó la definición de embedding, el ejemplo de «banco» y la explicación general de la búsqueda semántica. En ASR05, por ejemplo, transcribió «escribir» por «describir» y preservó la comparación vectorial. La muestra revisa esas evidencias puntuales del Caso 01. El VTT queda como entrada fija para las dos corridas del generador.

### Comparación sobre los mismos seis WAV

Tras la autorización explícita para la API directa de OpenAI, se transcribieron los seis recortes con `whisper-1` en español y temperatura 0. Los hashes de audio coinciden con el manifiesto y con los insumos de Wav2Vec2. Se usaron las mismas referencias y reglas de normalización. [Métricas y textos de la comparación](data/asr_caso01_20261109/comparacion_20261114/metricas_con_whisper.json)

| Recorte | WER Wav2Vec2 | WER Whisper | CER Wav2Vec2 | CER Whisper |
| --- | ---: | ---: | ---: | ---: |
| ASR01 | 14,55% | 0,00% | 5,21% | 0,00% |
| ASR02 | 10,00% | 8,89% | 1,79% | 9,38% |
| ASR03 | 13,41% | 0,00% | 1,21% | 0,00% |
| ASR04 | 13,41% | 10,98% | 2,11% | 8,69% |
| ASR05 | 17,78% | 4,44% | 4,17% | 0,88% |
| ASR06 | 15,18% | 8,04% | 4,17% | 3,44% |
| **Agregado** | **14,09% (72/511)** | **5,87% (30/511)** | **3,07% (80/2.602)** | **3,92% (102/2.602)** |

Whisper redujo el WER en los seis recortes. Su CER agregado fue mayor. En ASR02 omitió siete palabras consecutivas de la referencia y en ASR04 agregó una frase de nueve palabras al final frente al texto revisado. Estas diferencias afectan más caracteres que varios errores cortos. El resultado caracteriza seis recortes conocidos de un video. Para G0 conservamos el VTT completo como entrada fija. Los WAV de la comparación sirven para evaluar transcripción y cubren solo una parte del video.

Para recalcular las medidas desde la raíz con las transcripciones guardadas:

```bash
caso=tp3/ejercicio2/transcripcion/data/asr_caso01_20261109
vtt='tp3/ejercicio2/evaluacion/casos/caso_01/transcripcion.vtt'
.venv/bin/python -m tp3.ejercicio2.transcripcion.evaluar_recortes "$caso/manifiesto.json" \
  --vtt "$vtt" --wav2vec "$caso/comparacion_20261114/wav2vec" \
  --whisper "$caso/comparacion_20261114/whisper" \
  --salida "$caso/comparacion_20261114/metricas_repeticion.json"
```

## Reproducción local

Desde la raíz, con el checkpoint de Wav2Vec2 en `tp3/ejercicio2/transcripcion/data/modelos/`:

```bash
caso=tp3/ejercicio2/transcripcion/data/asr_caso01_20261109
.venv/bin/python tp3/ejercicio2/transcripcion/transcribir_wav2vec.py \
  "$caso/ASR01.wav" --salida "$caso/piloto/wav2vec/ASR01.json"
.venv/bin/python tp3/ejercicio2/transcripcion/medir_asr.py ASR01 \
  --referencias "$caso/referencias_verificadas.json" \
  --wav2vec "$caso/piloto/wav2vec/ASR01.json" \
  --salida "$caso/piloto/ASR01.metricas.json"
```

Usar nombres de salida nuevos para repetir el piloto. Para recalcular métricas alcanza con las transcripciones guardadas.
