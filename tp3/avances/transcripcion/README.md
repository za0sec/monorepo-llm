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

## Reproducción local

Desde la raíz, con el checkpoint de Wav2Vec2 en `tp3/avances/transcripcion/data/modelos/`:

```bash
caso=tp3/avances/transcripcion/data/asr_caso01_20261109
.venv/bin/python tp3/avances/transcripcion/transcribir_wav2vec.py \
  "$caso/ASR01.wav" --salida "$caso/piloto/wav2vec/ASR01.json"
.venv/bin/python tp3/avances/transcripcion/medir_asr.py ASR01 \
  --referencias "$caso/referencias_verificadas.json" \
  --wav2vec "$caso/piloto/wav2vec/ASR01.json" \
  --salida "$caso/piloto/ASR01.metricas.json"
```

Usar nombres de salida nuevos para repetir el piloto. Para recalcular métricas alcanza con las transcripciones guardadas.
