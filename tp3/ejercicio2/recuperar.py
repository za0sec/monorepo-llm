"""Recupera referencias académicas desde un objetivo de estudio; no evalúa el video."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from chromadb.errors import NotFoundError

from config import ARCHIVO_CHUNKS, DATA, K_INICIAL, RUTA_CHROMA
from indexar_material import (
    abrir_cliente, cargar_documento, cargar_modelo, codificar, crear_manifiesto,
    nombre_coleccion, validar_coleccion, validar_documento,
)
from preparar_chunks import hash_json


def recuperar(objetivo: str, documento: dict, ruta: Path = RUTA_CHROMA,
              nombre: str = None, k: int = K_INICIAL, solo_local: bool = False) -> dict:
    objetivo = objetivo.strip()
    if not objetivo or k <= 0:
        raise ValueError("Se necesita un objetivo no vacío y k positivo.")
    validar_documento(documento)
    manifiesto = crear_manifiesto(documento)
    nombre = nombre or nombre_coleccion(manifiesto)
    if not ruta.is_dir():
        raise ValueError("No existe la base local. Ejecutar indexar_material.py primero.")
    try:
        coleccion = abrir_cliente(ruta).get_collection(nombre, embedding_function=None)
    except NotFoundError as error:
        raise ValueError("No existe esa colección. Ejecutar indexar_material.py primero.") from error
    validar_coleccion(coleccion, manifiesto)
    inicio = perf_counter()
    modelo = cargar_modelo(solo_local)
    vector = codificar(modelo, [objetivo])
    segundos_embedding = perf_counter() - inicio
    inicio_busqueda = perf_counter()
    resultados = coleccion.query(
        query_embeddings=vector.tolist(), n_results=min(k, coleccion.count()),
        include=["documents", "metadatas", "distances"],
    )
    segundos_busqueda = perf_counter() - inicio_busqueda
    referencias = []
    for posicion, (cid, texto, metadata, distancia) in enumerate(zip(
        resultados["ids"][0], resultados["documents"][0],
        resultados["metadatas"][0], resultados["distances"][0],
    ), 1):
        metadata = dict(metadata)
        metadata["segmentos_ids"] = json.loads(metadata["segmentos_ids"])
        referencias.append({"posicion": posicion, "chunk_id": cid, "texto": texto,
                            "distancia_coseno": distancia, "similitud_coseno": 1 - distancia,
                            "metadata": metadata})
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(), "objetivo": objetivo,
        "coleccion": nombre, "ruta_chroma": str(ruta.resolve()), "manifiesto": manifiesto,
        "k_solicitado": k, "k_efectivo": len(referencias), "umbral": None,
        "segundos_embedding_incluida_carga": segundos_embedding,
        "segundos_busqueda": segundos_busqueda, "segundos_totales": perf_counter() - inicio,
        "referencias": referencias,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objetivo")
    parser.add_argument("--chunks", type=Path, default=ARCHIVO_CHUNKS)
    parser.add_argument("--base", type=Path, default=RUTA_CHROMA)
    parser.add_argument("--coleccion")
    parser.add_argument("--k", type=int, default=K_INICIAL)
    parser.add_argument("--solo-local", action="store_true")
    parser.add_argument("--salida", type=Path, help="Archivo JSON; por defecto guarda una corrida nueva en data/resultados/.")
    args = parser.parse_args()
    try:
        resultado = recuperar(args.objetivo, cargar_documento(args.chunks), args.base,
                              args.coleccion, args.k, args.solo_local)
        salida = args.salida or DATA / "resultados" / (
            datetime.now(timezone.utc).strftime("retrieval_%Y%m%dT%H%M%S_%f_")
            + hash_json([resultado["objetivo"], resultado["coleccion"], args.k])[:8] + ".json"
        )
        salida.parent.mkdir(parents=True, exist_ok=True)
        salida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"No se pudieron recuperar referencias: {error}\n")
    for r in resultado["referencias"]:
        m = r["metadata"]
        print(f"\n{r['posicion']}. {m['fuente']} [{m['inicio']} --> {m['fin']}]\n"
              f"ID: {r['chunk_id']}\nDistancia coseno: {r['distancia_coseno']:.4f}; "
              f"similitud (1-distancia): {r['similitud_coseno']:.4f}\n{r['texto']}")
    print(f"\nReferencias: {resultado['k_efectivo']}; resultado guardado en {salida}")


if __name__ == "__main__":
    main()
