"""Compara los índices con y sin apuntes usando el mismo Caso 02 y G4."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from openai import APIConnectionError, APIStatusError

from comparar_embeddings import MODELO_OPENAI, obtener_vectores
from evaluar_video import K, SNAPSHOT
from generar_g1 import CARPETA as CARPETA_E2
from generar_g1 import armar_mensajes, generar_gpt41, guardar_informe, guardar_json
from indexar_apuntes import CARPETA, COLECCION_I4, INDICE_I4, TOLERANCIA_VECTOR
from indexar_apuntes import cargar_apuntes_previos, cargar_vtt, registros
from preparar_transcripcion import leer_vtt
from recuperar_g2 import cliente_openai, consultar_vectores
from recuperar_g3 import consultas_expandidas, expandir, intercalar
from rerankear_g4 import reordenar


OBJETIVO = (
    "Quiero entender cómo decide un Transformer a qué tokens prestarles atención, "
    "cómo se calcula esa atención y qué cambia cuando usa varias cabezas"
)
RUTA_CASO = CARPETA / "data/casos/transformers"
MANIFIESTO = CARPETA / "data/carga_apuntes_ampliado.json"
COLECCION_MIXTA = "openai_e4_apuntes_ampliado"
VECINOS = 5


def cargar_indices():
    contenido_i4 = (INDICE_I4 / "ranking.json").read_bytes()
    huella_i4 = sha256(contenido_i4).hexdigest()
    i4 = json.loads(contenido_i4)
    ruta = INDICE_I4 / "chroma"
    if not (ruta / "chroma.sqlite3").is_file():
        raise ValueError("Falta la base persistida de Chroma.")
    cliente = chromadb.PersistentClient(path=str(ruta), settings=Settings(anonymized_telemetry=False))
    base = cliente.get_collection(COLECCION_I4, embedding_function=None)
    nombre, pdfs, chunks_pdf, registro_pdf = cargar_apuntes_previos(
        cliente, MANIFIESTO, huella_i4, i4["uso_openai"],
    )
    if nombre != COLECCION_MIXTA or len(pdfs) != 3:
        raise ValueError("El manifiesto debe corresponder a la colección de los tres apuntes.")
    mixta = cliente.get_collection(nombre, embedding_function=None)
    if any(c.configuration["hnsw"]["space"] != i4["distancia"] for c in (base, mixta)):
        raise ValueError("Ambas colecciones deben usar la misma distancia de I4.")

    vtt_base = cargar_vtt(base, i4)
    mixta_registros = registros(mixta)
    if (len(vtt_base), len(registro_pdf), len(mixta_registros)) != (623, 15, 638):
        raise ValueError("El conteo difiere del corpus congelado del piloto.")
    if set(vtt_base).intersection(registro_pdf) or set(mixta_registros) != (vtt_base.keys() | registro_pdf.keys()):
        raise ValueError("La colección mixta debe contener exactamente los VTT y los tres apuntes.")
    for id_, (texto, meta, vector) in vtt_base.items():
        otro_texto, otra_meta, otro_vector = mixta_registros[id_]
        if (texto, meta) != (otro_texto, otra_meta) or not np.allclose(
            vector, otro_vector, rtol=0, atol=TOLERANCIA_VECTOR,
        ):
            raise ValueError(f"El VTT {id_} cambió en la colección mixta.")

    por_id = {chunk["id"]: chunk for chunk in i4["chunks"]}
    por_id_mixto = {**por_id, **{chunk["id"]: chunk for chunk in chunks_pdf}}
    return [(base, por_id), (mixta, por_id_mixto)], i4, huella_i4


def recuperar(coleccion, por_id, consultas, vectores):
    rankings = consultar_vectores(coleccion, por_id, vectores, VECINOS)
    candidatos = intercalar(rankings)
    detalle = {
        "coleccion": coleccion.name,
        "k": K,
        "resultados_por_consulta": VECINOS,
        "rankings": [{"consulta": q, "resultados": r} for q, r in zip(consultas, rankings)],
        "candidatos_distintos": [c["id"] for c in candidatos],
    }
    if len(candidatos) < K:
        refuerzo, = consultar_vectores(coleccion, por_id, vectores[:1], K)
        por_candidato = {c["id"]: c for c in candidatos}
        for chunk in refuerzo:
            por_candidato.setdefault(chunk["id"], chunk)
        candidatos = list(por_candidato.values())
        detalle["refuerzo_consulta_original"] = refuerzo
        detalle["candidatos_distintos"] = list(por_candidato)
    orden, detalle["reranker"] = reordenar(consultas[0], candidatos, SNAPSHOT)
    return {f"C{i}": c for i, c in enumerate(orden[:K], 1)}, detalle


def comparar(salida):
    indices, i4, huella_i4 = cargar_indices()
    if not (SNAPSHOT / "model.safetensors").is_file():
        raise ValueError(f"Falta el snapshot local de MiniLM en {SNAPSHOT}.")
    ruta_vtt = RUTA_CASO / "transcripcion.vtt"
    vtt = ruta_vtt.read_bytes()
    huella_vtt = sha256(vtt).hexdigest()
    if huella_vtt != "f5438c38896ec984e96c2a5cc3afff89da5deb8c79b9c1339f5e1fa8ab6183d2":
        raise ValueError("El VTT debe coincidir con la referencia congelada del Caso 02.")
    video = {f"V{i}": s for i, s in enumerate(leer_vtt(ruta_vtt, vtt), 1)}
    prompt = (CARPETA_E2 / "g1/prompt.md").read_text(encoding="utf-8").rstrip("\n")
    caso = {
        "objetivo": OBJETIVO,
        "nivel": "introductorio",
        "max_tokens_salida": 1000,
        "gpt41": {"modelo": "gpt-4.1-2025-04-14", "temperatura": 0},
    }
    salida.mkdir(parents=True)
    cliente = cliente_openai(timeout=120)
    expansion = expandir(cliente, caso)
    consultas = consultas_expandidas(caso, expansion)
    ruta_expansion = salida / "expansion.json"
    guardar_json(ruta_expansion, expansion)
    print("Expansión guardada. Obteniendo los embeddings comunes.", flush=True)
    vectores, uso = obtener_vectores(cliente, MODELO_OPENAI, consultas)
    if any(uso[campo] != i4["uso_openai"][campo] for campo in ("modelo_respuesta", "dimensiones")):
        raise ValueError("Los embeddings de consulta deben coincidir con I4.")
    ruta_vectores = salida / "consultas.json"
    guardar_json(ruta_vectores, {"consultas": consultas, "vectores": vectores, "uso": uso})
    comun = {
        "sha256_video": huella_vtt,
        "sha256_ranking_i4": huella_i4,
        "sha256_manifiesto_pdf": sha256(MANIFIESTO.read_bytes()).hexdigest(),
        "sha256_expansion": sha256(ruta_expansion.read_bytes()).hexdigest(),
        "sha256_consultas": sha256(ruta_vectores.read_bytes()).hexdigest(),
        "video": video,
    }
    controles = []
    for coleccion, por_id in indices:
        destino = salida / coleccion.name
        destino.mkdir()
        clase, recuperacion = recuperar(coleccion, por_id, consultas, vectores)
        entrada = {
            **comun,
            "configuracion_experimental": {**caso, "coleccion": coleccion.name},
            "recuperacion": {**recuperacion, "distancia": i4["distancia"]},
            "clase": clase,
            "mensajes": armar_mensajes(caso, prompt, clase, video),
        }
        ruta = destino / "entrada.json"
        guardar_json(ruta, entrada)
        controles.append((destino, ruta, entrada))
        print(f"{coleccion.name}: {len(recuperacion['candidatos_distintos'])} candidatos, {len(clase)} seleccionados.", flush=True)

    incompletos = []
    for destino, ruta, entrada in controles:
        informe, auditoria = generar_gpt41(entrada)
        guardar_informe(destino / "gpt41", entrada, ruta.read_bytes(), informe, auditoria)
        if not informe.strip() or auditoria["motivo_fin"] != "stop":
            incompletos.append(destino.name)
        print(f"{destino.name}: informe guardado ({auditoria['motivo_fin']}).", flush=True)
    if incompletos:
        raise ValueError(f"Informes incompletos: {', '.join(incompletos)}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, required=True, help="Carpeta nueva para la corrida.")
    args = parser.parse_args()
    try:
        comparar(args.salida)
    except APIStatusError as error:
        parser.exit(1, f"OpenAI respondió HTTP {error.status_code}.\n")
    except APIConnectionError:
        parser.exit(1, "No se recibió confirmación de OpenAI.\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
