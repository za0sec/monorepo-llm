"""Genera los VTT de los casos nuevos y registra su procedencia."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from generar_g1 import guardar_json
from preparar_transcripcion import leer_vtt
from recuperar_g2 import cliente_openai
from transcripcion.transcribir_whisper import MODELO, transcribir_archivo


def transcribir(carpeta, cliente):
    audio = carpeta / "audio.mp3"
    destino = carpeta / "transcripcion.vtt"
    auditoria = carpeta / "asr.json"
    vtt_temporal = destino.with_suffix(".vtt.tmp")
    auditoria_temporal = auditoria.with_suffix(".json.tmp")
    huella_audio = sha256(audio.read_bytes()).hexdigest()
    if destino.exists() or auditoria.exists():
        registro = auditoria
        if destino.is_file() and not auditoria.exists() and auditoria_temporal.is_file():
            registro = auditoria_temporal
        if not destino.is_file() or not registro.is_file():
            raise ValueError(f"Falta el VTT o el registro de la corrida de {carpeta.name}.")
        previo = json.loads(registro.read_bytes())
        if (previo["sha256_audio"] != huella_audio
                or previo["sha256_vtt"] != sha256(destino.read_bytes()).hexdigest()):
            raise ValueError(f"Los archivos de {carpeta.name} difieren de la corrida registrada.")
        if registro == auditoria_temporal:
            registro.replace(auditoria)
        return carpeta.name, previo["segmentos"]

    vtt = transcribir_archivo(cliente, audio, idioma=None, response_format="vtt").encode("utf-8")
    segmentos = leer_vtt(destino, vtt)
    datos = {
        "modelo": MODELO, "idioma": "autodetección", "temperatura": 0,
        "sha256_audio": huella_audio, "sha256_vtt": sha256(vtt).hexdigest(),
        "segmentos": len(segmentos), "inicio": segmentos[0]["inicio"],
        "fin": max(segmentos, key=lambda s: s["fin_ms"])["fin"],
    }
    vtt_temporal.write_bytes(vtt)
    guardar_json(auditoria_temporal, datos)
    vtt_temporal.replace(destino)
    auditoria_temporal.replace(auditoria)
    return carpeta.name, len(segmentos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("carpetas", type=Path, nargs="+", help="Carpetas que contienen audio.mp3.")
    args = parser.parse_args()
    carpetas = [ruta.resolve() for ruta in args.carpetas]
    if len(set(carpetas)) != len(carpetas) or any(not (p / "audio.mp3").is_file() for p in carpetas):
        parser.error("Cada carpeta debe ser única y contener un audio.mp3.")
    cliente = cliente_openai(timeout=180).with_options(max_retries=2)
    with ThreadPoolExecutor(max_workers=4) as equipo:
        for nombre, cantidad in equipo.map(lambda p: transcribir(p, cliente), carpetas):
            print(f"{nombre}: {cantidad} segmentos.", flush=True)
