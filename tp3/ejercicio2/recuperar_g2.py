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


def abrir_indice(control):
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
    return coleccion, i4, por_id


def cliente_openai(timeout=30):
    clave = dotenv_values(CARPETA.parent / ".env").get("OPENAI_API_KEY")
    if not clave:
        raise ValueError("Falta OPENAI_API_KEY en tp3/.env.")
    return OpenAI(base_url=API_OPENAI, api_key=clave, max_retries=0, timeout=timeout)


def buscar(coleccion, i4, por_id, cliente, consultas, k):
    vectores, uso = obtener_vectores(cliente, MODELO_OPENAI, consultas)
    campos = ("modelo_respuesta", "dimensiones")
    if any(uso[campo] != i4["uso_openai"][campo] for campo in campos):
        raise ValueError("El embedding de consulta no coincide con I4.")
    return consultar_vectores(coleccion, por_id, vectores, k), uso


def consultar_vectores(coleccion, por_id, vectores, k):
    respuesta = coleccion.query(query_embeddings=vectores, n_results=k, include=["documents", "metadatas", "distances"])
    columnas = [respuesta[campo] for campo in ("ids", "documents", "metadatas", "distances")]
    if any(len(columna) != len(vectores) for columna in columnas):
        raise ValueError("Faltan rankings de consulta.")
    rankings = []
    for hits in zip(*columnas):
        if any(len(x) != k for x in hits) or len(set(hits[0])) != k:
            raise ValueError("Chroma no devolvió los k chunks distintos esperados.")
        ranking = []
        for id_, texto, meta, distancia in zip(*hits):
            original = por_id[id_]
            if "metadata" in original:
                procedencia = original["metadata"]
            else:
                procedencia = {campo: original[campo] for campo in ("fuente", "inicio", "fin")}
                procedencia["bloques"] = ",".join(original["bloques"])
            if texto != original["texto"] or meta != procedencia:
                raise ValueError(f"El contenido o los metadatos de {id_} difieren del registro del índice.")
            ranking.append({
                "id": id_, "texto": texto, "tokens_minilm": original["tokens"],
                "distancia_coseno": distancia, **meta,
            })
        rankings.append(ranking)
    return rankings


def reemplazar_clase(control, chunks, recuperacion):
    clase = {f"C{i}": chunk for i, chunk in enumerate(chunks, 1)}
    sistema, usuario = control["mensajes"]
    antes, bloque = usuario["content"].split("\n\nMaterial de clase:\n", 1)
    _, video = bloque.rsplit("\n\nTranscripción del video:\n", 1)
    texto_clase = "\n\n".join(f"[{i}] {c['texto']}" for i, c in clase.items())
    mensajes = [sistema, {**usuario, "content": (
        f"{antes}\n\nMaterial de clase:\n{texto_clase}\n\nTranscripción del video:\n{video}"
    )}]
    return {
        **control, "mensajes": mensajes, "clase": clase,
        "recuperacion": recuperacion,
    }


def recuperar(control):
    consulta = control["configuracion_experimental"]["objetivo"]
    k = len(control["clase"])
    coleccion, i4, por_id = abrir_indice(control)
    rankings, uso = buscar(coleccion, i4, por_id, cliente_openai(), [consulta], k)
    return reemplazar_clase(control, rankings[0], {
        "consulta": consulta, "k": k, "distancia": i4["distancia"], "uso": uso,
    })


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
