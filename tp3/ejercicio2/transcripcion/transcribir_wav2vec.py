"""Transcribe un recorte WAV con un checkpoint local de Wav2Vec2."""

import argparse
import json
import wave
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, __version__ as version_transformers


MODELO = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
REVISION = "96d7e9b4e4a78af515a3c6d3cee7c0826045d276"
SHA256_PESOS = "41c110e55d2eac8c79486ad87dbe8f9527ed034fe087a6adf03c891eeba914c1"
CACHE = Path(__file__).resolve().parent / "data" / "modelos"
FRECUENCIA = 16000


def transcribir(audio: Path) -> dict:
    with wave.open(str(audio), "rb") as archivo:
        if (archivo.getnchannels(), archivo.getsampwidth(), archivo.getframerate()) != (
            1, 2, FRECUENCIA
        ):
            raise ValueError("Se espera un WAV PCM de 16 bits, mono y 16 kHz.")
        muestras = archivo.readframes(archivo.getnframes())
    if not muestras:
        raise ValueError("El recorte está vacío.")

    checkpoint = CACHE / f"models--{MODELO.replace('/', '--')}" / "snapshots" / REVISION
    with (checkpoint / "pytorch_model.bin").open("rb") as pesos:
        hash_pesos = sha256()
        for bloque in iter(lambda: pesos.read(8 * 1024 * 1024), b""):
            hash_pesos.update(bloque)
    if hash_pesos.hexdigest() != SHA256_PESOS:
        raise ValueError("El hash del checkpoint de Wav2Vec2 no coincide.")

    torch.set_num_threads(4)
    inicio_carga = perf_counter()
    procesador = Wav2Vec2Processor.from_pretrained(checkpoint, local_files_only=True)
    modelo = Wav2Vec2ForCTC.from_pretrained(
        checkpoint, local_files_only=True, use_safetensors=False, weights_only=True,
    ).eval()
    segundos_carga = perf_counter() - inicio_carga

    inicio = perf_counter()
    onda = np.frombuffer(muestras, dtype="<i2").astype(np.float32) / 32768.0
    entrada = procesador(onda, sampling_rate=FRECUENCIA, return_tensors="pt")
    with torch.inference_mode():
        ids = modelo(**entrada).logits[0].argmax(dim=-1)
    resultado = procesador.decode(ids.tolist(), output_word_offsets=True)
    segundos_inferencia = perf_counter() - inicio
    paso = modelo.config.inputs_to_logits_ratio / FRECUENCIA
    return {
        "audio": audio.name,
        "sha256_audio": sha256(audio.read_bytes()).hexdigest(),
        "modelo": MODELO,
        "revision": REVISION,
        "sha256_pesos": SHA256_PESOS,
        "dispositivo": "cpu",
        "hilos": torch.get_num_threads(),
        "torch": torch.__version__,
        "transformers": version_transformers,
        "decodificacion": "CTC greedy",
        "segundos_carga": segundos_carga,
        "segundos_inferencia": segundos_inferencia,
        "texto": resultado.text,
        "palabras_tiempos_ctc_aproximados": [
            {"texto": palabra["word"], "inicio": palabra["start_offset"] * paso,
             "fin": palabra["end_offset"] * paso}
            for palabra in resultado.word_offsets
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    if args.salida.exists():
        parser.error("La salida ya existe. Elegir un archivo nuevo.")
    try:
        resultado = transcribir(args.audio)
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        with args.salida.open("x", encoding="utf-8") as archivo:
            json.dump(resultado, archivo, ensure_ascii=False, indent=2)
            archivo.write("\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"No se pudo transcribir con Wav2Vec2: {error}\n")
    print(f"Transcripción guardada en {args.salida}")


if __name__ == "__main__":
    main()
