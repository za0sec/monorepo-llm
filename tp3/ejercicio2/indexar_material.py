"""Indexa chunks académicos con MiniLM y Chroma persistente.

Referencias: Langchain_Virtual_Assitant.ipynb, celda 9, y
UN_Report_Advanced_RAG.ipynb, celdas 18-23. La persistencia y los controles
de configuración adaptan esas demos al diseño local del TP3.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import chromadb
import numpy as np
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from config import ARCHIVO_CHUNKS, CACHE_MODELOS, DIMENSION, MODELO, REVISION, RUTA_CHROMA
from preparar_chunks import hash_json


def cargar_documento(ruta: Path) -> dict:
    documento = json.loads(ruta.read_text(encoding="utf-8"))
    validar_documento(documento)
    return documento


def validar_documento(documento: dict) -> None:
    """Rechaza artefactos incompletos o inconsistentes antes de tocar la base."""
    try:
        config = documento["configuracion"]
        if config["modelo_tokenizer"] != MODELO or config["revision_tokenizer"] != REVISION:
            raise ValueError("Los chunks deben usar el mismo modelo y revisión que el embedder.")
        if documento["id_configuracion"] != hash_json(config):
            raise ValueError("El hash de configuración de chunks no coincide.")
        chunks = documento["chunks"]
        if not chunks or len({c["chunk_id"] for c in chunks}) != len(chunks):
            raise ValueError("Se necesitan chunks no vacíos y sin IDs duplicados.")
        for c in chunks:
            identidad = [documento["fuente"], documento["sha256_fuente"],
                         documento["id_configuracion"], c["inicio_caracter"], c["fin_caracter"]]
            if (c["chunk_id"] != hash_json(identidad)
                    or c["fuente"] != documento["fuente"]
                    or c["id_configuracion"] != documento["id_configuracion"]):
                raise ValueError("Un chunk no corresponde a su fuente o configuración.")
            if (not c["texto"].strip() or not c["segmentos_ids"]
                    or not 0 <= c["inicio_ms"] < c["fin_ms"]
                    or not 0 <= c["inicio_caracter"] < c["fin_caracter"]
                    or len(c["texto"]) != c["fin_caracter"] - c["inicio_caracter"]
                    or not 0 < c["tokens_totales"] <= config["limite_encoder"]):
                raise ValueError("Un chunk tiene texto, referencias o límites inválidos.")
            # Campos necesarios para conservar toda la trazabilidad en Chroma.
            for campo in ("inicio", "fin", "tokens_contenido"):
                if campo not in c:
                    raise ValueError(f"Falta el campo {campo} en un chunk.")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("El JSON no tiene el formato de preparar_chunks.py.") from error


def crear_manifiesto(documento: dict) -> dict:
    return {
        "version": 1,
        "sha256_documento_chunks": hash_json(documento),
        "fuente": documento["fuente"], "sha256_fuente": documento["sha256_fuente"],
        "id_configuracion_chunks": documento["id_configuracion"],
        "cantidad_chunks": len(documento["chunks"]),
        "embedding": {"modelo": MODELO, "revision": REVISION, "dimension": DIMENSION,
                      "dispositivo": "cpu", "normalizar": False},
        "metrica": "cosine",
        "versiones": {p: version(p) for p in
                      ("chromadb", "sentence-transformers", "torch", "transformers", "tokenizers")},
    }


def nombre_coleccion(manifiesto: dict) -> str:
    return "tp3_" + hash_json(manifiesto)[:24]


def abrir_cliente(ruta: Path):
    return chromadb.PersistentClient(
        path=str(ruta.resolve()), settings=Settings(anonymized_telemetry=False),
    )


def validar_coleccion(coleccion, manifiesto: dict, exigir_completa: bool = True) -> None:
    metadata = coleccion.metadata or {}
    if metadata.get("manifiesto") != json.dumps(manifiesto, sort_keys=True, ensure_ascii=False):
        raise ValueError("La colección pertenece a otro corpus, configuración o versión. Usar otra colección.")
    if coleccion.configuration["hnsw"]["space"] != "cosine":
        raise ValueError("La colección no está configurada con distancia coseno.")
    if exigir_completa and (metadata.get("estado") != "lista"
                            or coleccion.count() != manifiesto["cantidad_chunks"]):
        raise ValueError("La indexación está incompleta. Ejecutar indexar_material.py antes de consultar.")


def cargar_modelo(solo_local: bool = False):
    # Carga diferida: repetir una carga completa no descarga ni ejecuta el modelo.
    from sentence_transformers import SentenceTransformer

    modelo = SentenceTransformer(
        MODELO, revision=REVISION, device="cpu", cache_folder=str(CACHE_MODELOS),
        local_files_only=solo_local, trust_remote_code=False,
    )
    if modelo.get_sentence_embedding_dimension() != DIMENSION:
        raise ValueError("La dimensión del modelo no coincide con la configuración.")
    return modelo


def codificar(modelo, textos: list[str]) -> np.ndarray:
    """Impide que MiniLM trunque silenciosamente chunks u objetivos extensos."""
    for texto in textos:
        cantidad = len(modelo.tokenizer.encode(texto, add_special_tokens=True, truncation=False))
        if cantidad > modelo.max_seq_length:
            raise ValueError(f"Entrada de {cantidad} tokens excede el encoder ({modelo.max_seq_length}).")
    vectores = modelo.encode(
        textos, batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=False,
    )
    if (vectores.shape != (len(textos), DIMENSION) or not np.isfinite(vectores).all()
            or (np.linalg.norm(vectores, axis=1) == 0).any()):
        raise ValueError("El embedder produjo vectores inválidos.")
    return vectores


def metadata_chunk(chunk: dict) -> dict:
    metadata = {k: v for k, v in chunk.items() if k not in ("chunk_id", "texto", "segmentos_ids")}
    # JSON conserva la lista completa y funciona en las versiones de Chroma sin listas en metadata.
    metadata["segmentos_ids"] = json.dumps(chunk["segmentos_ids"], ensure_ascii=False)
    return metadata


def indexar(documento: dict, ruta: Path = RUTA_CHROMA, nombre: str = None,
            solo_local: bool = False, tamano_lote: int = 32) -> dict:
    validar_documento(documento)
    if tamano_lote <= 0:
        raise ValueError("El tamaño de lote debe ser positivo.")
    inicio = perf_counter()
    manifiesto = crear_manifiesto(documento)
    nombre = nombre or nombre_coleccion(manifiesto)
    cliente = abrir_cliente(ruta)
    try:
        coleccion = cliente.get_collection(nombre, embedding_function=None)
    except NotFoundError:
        coleccion = cliente.create_collection(
            nombre, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
            metadata={"manifiesto": json.dumps(manifiesto, sort_keys=True, ensure_ascii=False),
                      "estado": "incompleta"},
        )
    validar_coleccion(coleccion, manifiesto, exigir_completa=False)
    existentes = set(coleccion.get(include=[])["ids"])
    esperados = {c["chunk_id"] for c in documento["chunks"]}
    if not existentes <= esperados:
        raise ValueError("La colección contiene IDs ajenos al corpus. Usar otra colección.")
    pendientes = [c for c in documento["chunks"] if c["chunk_id"] not in existentes]
    if pendientes:
        coleccion.modify(metadata={**coleccion.metadata, "estado": "incompleta"})
        modelo = cargar_modelo(solo_local)
        for desde in range(0, len(pendientes), tamano_lote):
            lote = pendientes[desde:desde + tamano_lote]
            vectores = codificar(modelo, [c["texto"] for c in lote])
            coleccion.upsert(
                ids=[c["chunk_id"] for c in lote], embeddings=vectores.tolist(),
                documents=[c["texto"] for c in lote], metadatas=[metadata_chunk(c) for c in lote],
            )
    if set(coleccion.get(include=[])["ids"]) != esperados:
        raise ValueError("La colección no contiene exactamente los chunks esperados.")
    coleccion.modify(metadata={**coleccion.metadata, "estado": "lista"})
    return {"coleccion": nombre, "ruta_chroma": str(ruta.resolve()),
            "chunks_totales": coleccion.count(), "chunks_nuevos": len(pendientes),
            "chunks_reutilizados": len(existentes), "segundos": perf_counter() - inicio,
            "manifiesto": manifiesto}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", nargs="?", type=Path, default=ARCHIVO_CHUNKS)
    parser.add_argument("--base", type=Path, default=RUTA_CHROMA)
    parser.add_argument("--coleccion", help="Por defecto se deriva de la fuente y configuración.")
    parser.add_argument("--solo-local", action="store_true", help="No descargar pesos; usar caché local.")
    args = parser.parse_args()
    try:
        resumen = indexar(cargar_documento(args.entrada), args.base, args.coleccion, args.solo_local)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"No se pudo indexar el material: {error}\n")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
