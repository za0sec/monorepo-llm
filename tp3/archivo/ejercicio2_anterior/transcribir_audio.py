"""Transcribe el audio de un video con Whisper y guarda VTT y texto para revisión."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

from preparar_transcripcion import leer_vtt


CARPETA_TP = Path(__file__).resolve().parents[1]


def timestamp(segundos: float) -> str:
    """Convierte los segundos de Whisper al formato temporal de WebVTT."""
    total_ms = round(segundos * 1000)
    if total_ms < 0:
        raise ValueError("La transcripción contiene un tiempo negativo.")
    horas, resto = divmod(total_ms, 3_600_000)
    minutos, resto = divmod(resto, 60_000)
    segundos, milis = divmod(resto, 1000)
    return f"{horas:02}:{minutos:02}:{segundos:02}.{milis:03}"


def generar_vtt(respuesta: dict) -> str:
    """Usa los tiempos originales del transcriptor, sin inventar intervalos."""
    segmentos = respuesta.get("segments")
    if not segmentos:
        raise ValueError("El servicio devolvió una respuesta sin segmentos temporales.")
    bloques = ["WEBVTT"]
    for numero, segmento in enumerate(segmentos, start=1):
        bloques.append(
            f"{numero}\n"
            f"{timestamp(segmento['start'])} --> {timestamp(segmento['end'])}\n"
            f"{segmento['text'].strip()}"
        )
    return "\n\n".join(bloques) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Archivo de audio del video.")
    parser.add_argument(
        "--salida", type=Path,
        default=Path(__file__).parent / "data/transcripciones",
        help="Carpeta donde se guardan las transcripciones.",
    )
    args = parser.parse_args()
    audio = args.audio.resolve()
    if not audio.is_file() or audio.stat().st_size == 0:
        parser.error(f"No existe el audio o está vacío: {audio}")

    vtt_path = args.salida / f"{audio.stem}.vtt"
    txt_path = args.salida / f"{audio.stem}.txt"
    respuesta_path = args.salida / f"{audio.stem}.respuesta.json"
    if vtt_path.exists():
        print(f"Reutilizando la transcripción existente: {vtt_path}", flush=True)
    elif respuesta_path.exists():
        print(f"Reutilizando la respuesta guardada: {respuesta_path}", flush=True)
        vtt_path.write_text(
            generar_vtt(json.loads(respuesta_path.read_text(encoding="utf-8"))),
            encoding="utf-8",
        )
    else:
        if audio.stat().st_size > 25_000_000:
            parser.error("Esta primera versión admite audios de hasta 25 MB.")
        load_dotenv(CARPETA_TP / ".env")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            parser.error("Falta OPENAI_API_KEY en el entorno o en tp3/.env.")
        # Se conserva el proxy del TP. Sus modelos usan el prefijo openai/.
        base_url = os.getenv("OPENAI_BASE_URL", "https://sa-llmproxy.it.itba.edu.ar/")
        model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "openai/whisper-1")
        client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=180, max_retries=0
        )
        print(f"Transcribiendo {audio.name} con {model}...", flush=True)
        try:
            with audio.open("rb") as archivo:
                respuesta = client.audio.transcriptions.with_raw_response.create(
                    file=archivo,
                    model=model,
                    language="es",
                    response_format="verbose_json",
                    temperature=0,
                )
        except APIConnectionError:
            parser.exit(1, "No se pudo conectar al servicio de transcripción.\n")
        except APIStatusError as error:
            parser.exit(
                1, f"El servicio rechazó la transcripción (HTTP {error.status_code}). "
                "Revisar acceso al modelo y configuración del proxy.\n"
            )
        args.salida.mkdir(parents=True, exist_ok=True)
        # Guardar antes de interpretar permite resolver errores sin otra llamada.
        respuesta_path.write_text(respuesta.text, encoding="utf-8")
        vtt_path.write_text(
            generar_vtt(json.loads(respuesta.text)), encoding="utf-8"
        )

    # Son segmentos temporales del transcriptor, todavía no chunks de RAG.
    segmentos = leer_vtt(vtt_path)
    txt_path.write_text(
        f"Audio: {audio.name}\nTranscripción automática, pendiente de revisión.\n\n"
        + "\n\n".join(
            f"[{s['inicio']} --> {s['fin']}] {s['texto']}" for s in segmentos
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Segmentos temporales: {len(segmentos)}")
    print(f"VTT: {vtt_path}")
    print(f"Texto para revisar: {txt_path}")


if __name__ == "__main__":
    main()
