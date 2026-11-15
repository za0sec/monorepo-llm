"""Reordena los candidatos congelados de G3 con el CrossEncoder de la cátedra."""

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

from generar_g1 import DATOS, guardar_json
from recuperar_g2 import reemplazar_clase
from recuperar_g3 import intercalar


MODELO = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAX_LONGITUD = 512


def reordenar(objetivo, candidatos, snapshot):
    from sentence_transformers import CrossEncoder

    textos = [chunk["texto"] for chunk in candidatos]
    modelo = CrossEncoder(
        str(snapshot), max_length=MAX_LONGITUD, device="cpu", local_files_only=True,
        model_kwargs={"use_safetensors": True},
    )
    pares = modelo.tokenizer([objetivo] * len(textos), textos, truncation=False)
    longitudes = [len(ids_par) for ids_par in pares["input_ids"]]
    if max(longitudes) > MAX_LONGITUD:
        raise ValueError("Un par consulta–chunk supera el contexto del reranker.")
    inicio = perf_counter()
    scores = modelo.predict([(objetivo, texto) for texto in textos], batch_size=16, show_progress_bar=False)
    segundos = perf_counter() - inicio
    puntajes = [float(score) for score in scores]
    if len(puntajes) != len(candidatos) or not all(map(isfinite, puntajes)):
        raise ValueError("Se necesita un puntaje finito por cada candidato.")
    ranking = [
        {"id": chunk["id"], "puesto_g3": i, "puntaje": puntaje, "tokens_par": longitud}
        for i, (chunk, puntaje, longitud) in enumerate(zip(candidatos, puntajes, longitudes), 1)
    ]
    ranking.sort(key=lambda fila: -fila["puntaje"])
    por_id = {chunk["id"]: chunk for chunk in candidatos}
    return [por_id[fila["id"]] for fila in ranking], {
        "modelo": MODELO, "revision": snapshot.name, "dispositivo": "cpu",
        "max_longitud": MAX_LONGITUD, "consulta": objetivo, "segundos": segundos,
        "ranking": ranking,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=DATOS.parent / "g3/entrada_expandida.json")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--salida", type=Path, default=DATOS.parent / "g4/entrada_rerankeada.json")
    args = parser.parse_args()
    if args.salida.exists():
        parser.error("--salida debe apuntar a un archivo nuevo.")
    if not (args.snapshot / "model.safetensors").is_file():
        parser.error("Se necesita un snapshot local de MiniLM con model.safetensors.")

    base = args.control.read_bytes()
    control = json.loads(base)
    recuperacion = control["recuperacion"]
    candidatos = intercalar([r["resultados"] for r in recuperacion["rankings"]])
    ids = [c["id"] for c in candidatos]
    k = recuperacion["k"]
    if ids != recuperacion["candidatos_distintos"] or candidatos[:k] != list(control["clase"].values()):
        raise ValueError("Los candidatos deben reproducir el orden y el top k de G3.")
    orden, auditoria = reordenar(control["configuracion_experimental"]["objetivo"], candidatos, args.snapshot)
    seleccion = orden[:k]
    entrada = reemplazar_clase(control, seleccion, {
        **recuperacion,
        "reranker": auditoria,
    })
    entrada["sha256_entrada_g3"] = hashlib.sha256(base).hexdigest()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    guardar_json(args.salida, entrada)
    print(f"{len(candidatos)} candidatos puntuados, {len(seleccion)} en el contexto final.")


if __name__ == "__main__":
    main()
