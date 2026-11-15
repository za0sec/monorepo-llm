"""Transcribe los recortes verificados del piloto con Whisper vía API directa."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, OpenAI


MODELO = "whisper-1"
API_OPENAI = "https://api.openai.com/v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifiesto", type=Path)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    manifiesto = json.loads(args.manifiesto.read_bytes())
    clave = dotenv_values(Path(__file__).resolve().parents[2] / ".env").get("OPENAI_API_KEY")
    if not clave:
        parser.error("Falta OPENAI_API_KEY en tp3/.env.")
    cliente = OpenAI(base_url=API_OPENAI, api_key=clave, timeout=90, max_retries=0)
    args.salida.mkdir(parents=True, exist_ok=True)

    for recorte in manifiesto["recortes"]:
        clip_id = recorte["clip_id"]
        audio = args.manifiesto.parent / recorte["archivo"]
        hash_audio = sha256(audio.read_bytes()).hexdigest()
        if hash_audio != recorte["sha256"]:
            parser.error(f"El audio de {clip_id} no coincide con el manifiesto.")
        destino = args.salida / f"{clip_id}.json"
        if destino.exists():
            previo = json.loads(destino.read_bytes())
            if (previo["sha256_audio"], previo["modelo"], previo["idioma"], previo["temperatura"]) != (
                hash_audio, MODELO, "es", 0,
            ):
                parser.error(f"La transcripción guardada de {clip_id} usa otra configuración.")
            print(f"{clip_id} reutilizado.", flush=True)
            continue
        inicio = perf_counter()
        try:
            with audio.open("rb") as entrada:
                respuesta = cliente.audio.transcriptions.create(
                    file=entrada, model=MODELO, language="es", temperature=0,
                    response_format="verbose_json", timestamp_granularities=["segment", "word"],
                )
        except APIStatusError as error:
            parser.exit(1, f"{clip_id}: el servicio respondió HTTP {error.status_code}.\n")
        except APIConnectionError:
            parser.exit(1, f"{clip_id}: no se recibió confirmación del servicio.\n")
        resultado = {
            "clip_id": clip_id,
            "audio": audio.name,
            "sha256_audio": hash_audio,
            "modelo": MODELO,
            "servicio": "openai_directo",
            "idioma": "es",
            "temperatura": 0,
            "segundos": perf_counter() - inicio,
            **respuesta.model_dump(),
        }
        with destino.open("x", encoding="utf-8") as archivo:
            json.dump(resultado, archivo, ensure_ascii=False, indent=2)
            archivo.write("\n")
        print(f"{clip_id} guardado.", flush=True)


if __name__ == "__main__":
    main()
