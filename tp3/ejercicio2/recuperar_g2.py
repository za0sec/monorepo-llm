"""Congela los siete resultados de Chroma para el contraste G2."""

import argparse
import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import dotenv_values
from openai import OpenAI

from comparar_embeddings import API_OPENAI, MODELO_OPENAI, obtener_vectores
from generar_g1 import CARPETA, DATOS, REPO, guardar_json


def recuperar(control):
    caso = control["configuracion_experimental"]
    indice = REPO / caso["indice_i4"]
    ranking = (indice / "ranking.json").read_bytes()
    if hashlib.sha256(ranking).hexdigest() != control["sha256_ranking_i4"]:
        raise ValueError("El índice de I4 debe coincidir con el control de G1.")
    i4 = json.loads(ranking)
    por_id = {chunk["id"]: chunk for chunk in i4["chunks"]}
    ruta = indice / "chroma"
    if not (ruta / "chroma.sqlite3").is_file():
        raise ValueError("Falta el índice persistido de I4.")
    chroma = chromadb.PersistentClient(path=str(ruta), settings=Settings(anonymized_telemetry=False))
    coleccion = chroma.get_collection(caso["coleccion"], embedding_function=None)
    if coleccion.count() != len(por_id) or coleccion.configuration["hnsw"]["space"] != i4["distancia"]:
        raise ValueError("Chroma no coincide con el registro de I4.")
    clave = dotenv_values(CARPETA.parent / ".env").get("OPENAI_API_KEY")
    if not clave:
        raise ValueError("Falta OPENAI_API_KEY en tp3/.env.")
    cliente = OpenAI(base_url=API_OPENAI, api_key=clave, max_retries=0, timeout=30)
    vectores, uso = obtener_vectores(cliente, MODELO_OPENAI, [caso["objetivo"]])
    campos = ("modelo_respuesta", "dimensiones")
    if any(uso[campo] != i4["uso_openai"][campo] for campo in campos):
        raise ValueError("El embedding de consulta no coincide con I4.")

    k = len(control["clase"])
    respuesta = coleccion.query(query_embeddings=vectores, n_results=k, include=["documents", "metadatas", "distances"])
    hits = [respuesta[campo][0] for campo in ("ids", "documents", "metadatas", "distances")]
    if any(len(x) != k for x in hits) or len(set(hits[0])) != k:
        raise ValueError("Chroma no devolvió los k chunks distintos esperados.")
    clase = {}
    for puesto, (id_, texto, meta, distancia) in enumerate(zip(*hits), 1):
        original = por_id[id_]
        procedencia = {campo: original[campo] for campo in ("fuente", "inicio", "fin")}
        procedencia["bloques"] = ",".join(original["bloques"])
        if texto != original["texto"] or meta != procedencia:
            raise ValueError(f"El contenido o los metadatos de {id_} difieren de I4.")
        clase[f"C{puesto}"] = {
            "id": id_, "texto": texto, "tokens_minilm": original["tokens"],
            "distancia_coseno": distancia, **meta,
        }

    sistema, usuario = control["mensajes"]
    antes, bloque = usuario["content"].split("\n\nMaterial de clase:\n", 1)
    _, video = bloque.rsplit("\n\nTranscripción del video:\n", 1)
    texto_clase = "\n\n".join(f"[{i}] {c['texto']}" for i, c in clase.items())
    mensajes = [sistema, {**usuario, "content": (
        f"{antes}\n\nMaterial de clase:\n{texto_clase}\n\nTranscripción del video:\n{video}"
    )}]
    return {
        **control, "mensajes": mensajes, "clase": clase,
        "recuperacion": {"consulta": caso["objetivo"], "k": k, "distancia": i4["distancia"], "uso": uso},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=DATOS / "entrada_manual.json")
    parser.add_argument("--salida", type=Path, default=DATOS.parent / "g2/entrada_recuperada.json")
    args = parser.parse_args()
    if args.salida.exists():
        parser.error("--salida debe ser un archivo nuevo.")
    control = args.control.read_bytes()
    entrada = recuperar(json.loads(control))
    entrada["sha256_control"] = hashlib.sha256(control).hexdigest()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    guardar_json(args.salida, entrada)
    print(f"Recuperación congelada en {args.salida}.")


if __name__ == "__main__":
    main()
