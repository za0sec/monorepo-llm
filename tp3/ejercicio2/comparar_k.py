"""Recupera un ranking común para comparar k=2, 4 y 8 sobre el índice de I4."""

import argparse
import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import dotenv_values
from openai import OpenAI

from comparar_embeddings import API_OPENAI, CARPETA, MODELO_OPENAI, obtener_vectores


VALORES_K = (2, 4, 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4", type=Path, default=CARPETA / "data/experimentos/i4")
    parser.add_argument("--consultas", type=Path, default=CARPETA / "consultas_i1_ampliado.json")
    parser.add_argument("--salida", type=Path, required=True, help="Ruta de un JSON nuevo.")
    args = parser.parse_args()
    if args.salida.exists():
        parser.error("--salida debe apuntar a un archivo nuevo.")
    base_i4 = (args.i4 / "ranking.json").read_bytes()
    i4 = json.loads(base_i4)
    uso_i4 = i4["uso_openai"]
    if (i4["servicio_embeddings"], uso_i4["modelo_respuesta"]) != ("openai_directo", MODELO_OPENAI):
        parser.error("Se requiere el índice de text-embedding-3-small de la corrida directa de I4.")
    referencia = args.consultas.read_bytes()
    if hashlib.sha256(referencia).hexdigest() != i4["sha256_consultas"]:
        parser.error("Las consultas deben coincidir con las de I4.")
    consultas = json.loads(referencia)["consultas"]
    por_id = {c["id"]: c for c in i4["chunks"]}
    if not (args.i4 / "chroma" / "chroma.sqlite3").is_file():
        parser.error("No se encontró la colección persistida de I4.")
    chroma = chromadb.PersistentClient(
        path=str(args.i4 / "chroma"), settings=Settings(anonymized_telemetry=False),
    )
    coleccion = chroma.get_collection("openai_i4", embedding_function=None)
    guardados = coleccion.get(include=["documents"])
    if (dict(zip(guardados["ids"], guardados["documents"])) != {i: c["texto"] for i, c in por_id.items()}
            or coleccion.configuration["hnsw"]["space"] != i4["distancia"]):
        parser.error("La colección debe coincidir con los documentos y la métrica de I4.")

    clave = dotenv_values(CARPETA.parent / ".env").get("OPENAI_API_KEY")
    if not clave:
        parser.error("Falta OPENAI_API_KEY en tp3/.env.")
    cliente = OpenAI(base_url=API_OPENAI, api_key=clave, max_retries=0, timeout=60)
    try:
        vectores, uso = obtener_vectores(cliente, MODELO_OPENAI, [q["texto"] for q in consultas])
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")
    if (uso["modelo_respuesta"], uso["dimensiones"]) != (uso_i4["modelo_respuesta"], uso_i4["dimensiones"]):
        parser.error("Los embeddings de consulta no coinciden en modelo y dimensión con los de I4.")

    respuesta = coleccion.query(query_embeddings=vectores, n_results=max(VALORES_K), include=["distances"])
    rankings = {}
    for consulta, ids, distancias in zip(consultas, respuesta["ids"], respuesta["distances"]):
        if len(ids) != max(VALORES_K) or len(set(ids)) != len(ids) or len(distancias) != len(ids):
            parser.error("Chroma no devolvió ocho resultados distintos para cada consulta.")
        ranking = [{"chunk": id_, "distancia_coseno": d} for id_, d in zip(ids, distancias)]
        qid = consulta["id"]
        rankings[qid] = {
            "resultados": ranking,
            "mismo_orden_k4_que_i4": ids[:4] == [hit["chunk"] for hit in i4["ranking_openai"][qid]],
            "tokens_contexto_por_k": {
                k: sum(por_id[id_]["tokens"] for id_ in ids[:k]) for k in VALORES_K
            },
        }
    if len(rankings) != len(consultas):
        parser.error("Faltan rankings de consulta.")
    resultado = {
        "sha256_ranking_i4": hashlib.sha256(base_i4).hexdigest(),
        "sha256_consultas": i4["sha256_consultas"],
        "coleccion": coleccion.name,
        "distancia": i4["distancia"],
        "valores_k": VALORES_K,
        "uso_openai": uso,
        "vectores_consultas": {q["id"]: v for q, v in zip(consultas, vectores)},
        "consultas": rankings,
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("x", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")
    print(f"{len(rankings)} rankings de {max(VALORES_K)} resultados guardados en {args.salida}.")


if __name__ == "__main__":
    main()
