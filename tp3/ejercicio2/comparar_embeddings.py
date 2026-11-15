"""Compara MiniLM con OpenAI sin volver a cortar el corpus de I3."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import chromadb
from chromadb.config import Settings
from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, OpenAI

from comparar_splitters import CARPETA, K, buscar


PROXY_CATEDRA = "https://sa-llmproxy.it.itba.edu.ar/"
API_OPENAI = "https://api.openai.com/v1"
MODELO_OPENAI = "text-embedding-3-small"
MODELO_PROXY = f"openai/{MODELO_OPENAI}"
TAMANO_LOTE = 32
LIMITE_ENTRADA = 8192


def obtener_vectores(cliente, modelo, textos):
    # En este corpus el tamaño en bytes es una cota conservadora de los tokens.
    if any(not t.strip() or len(t.encode("utf-8")) > LIMITE_ENTRADA for t in textos):
        raise ValueError("Hay un texto vacío o demasiado grande para la API de embeddings.")
    vectores = []
    lotes = []
    modelo_respuesta = None
    dimension = None
    for inicio in range(0, len(textos), TAMANO_LOTE):
        lote = textos[inicio:inicio + TAMANO_LOTE]
        inicio_llamada = perf_counter()
        try:
            http = cliente.embeddings.with_raw_response.create(
                model=modelo, input=lote, encoding_format="float",
            )
        except APIStatusError as error:
            raise RuntimeError(f"El servicio de embeddings respondió HTTP {error.status_code}.") from None
        except APIConnectionError:
            raise RuntimeError("No se pudo conectar con el servicio de embeddings.") from None
        respuesta = http.parse()
        datos = sorted(respuesta.data, key=lambda dato: dato.index)
        if [dato.index for dato in datos] != list(range(len(lote))):
            raise ValueError("La API no devolvió un vector por cada texto en el lote.")
        if modelo_respuesta is None:
            modelo_respuesta = respuesta.model
            dimension = len(datos[0].embedding)
        if respuesta.model != modelo_respuesta:
            raise ValueError("Cambió el identificador del modelo entre lotes.")
        for dato in datos:
            vector = dato.embedding
            if len(vector) != dimension or not any(vector) or not all(map(math.isfinite, vector)):
                raise ValueError("La API devolvió un embedding inválido.")
            vectores.append(vector)
        lotes.append({
            "textos": len(lote),
            "segundos": perf_counter() - inicio_llamada,
            "tokens_entrada": respuesta.usage.prompt_tokens if respuesta.usage else None,
            "costo_reportado_proxy": http.headers.get("x-litellm-response-cost"),
        })
    return vectores, {
        "modelo_respuesta": modelo_respuesta,
        "dimensiones": dimension,
        "lotes": lotes,
    }


def cargar_insumos(ruta_baseline, ruta_consultas):
    contenido = ruta_baseline.read_bytes()
    baseline = json.loads(contenido)
    if (baseline["max_tokens"], baseline["solapamiento"], baseline["distancia"], baseline["k"]) != (
        160, 10, "cosine", K,
    ):
        raise ValueError("La baseline debe ser el recursivo de 160 tokens con 10 de solapamiento.")
    referencia = ruta_consultas.read_bytes()
    if hashlib.sha256(referencia).hexdigest() != baseline["sha256_consultas"]:
        raise ValueError("Las consultas no coinciden con las usadas en la baseline.")
    consultas = json.loads(referencia)["consultas"]
    minilm = baseline["variantes"]["recursivo"]
    if set(minilm["ranking"]) != {q["id"] for q in consultas}:
        raise ValueError("El ranking de MiniLM no cubre las consultas de la comparación.")
    return baseline, minilm, consultas, hashlib.sha256(contenido).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path,
        default=CARPETA / "data/experimentos/i3/solape_10_r1/ranking.json",
    )
    parser.add_argument("--consultas", type=Path, default=CARPETA / "consultas_i1_ampliado.json")
    parser.add_argument("--modelo", choices=(MODELO_PROXY, MODELO_OPENAI))
    parser.add_argument("--openai-directo", action="store_true",
                        help="Usa la API directa de OpenAI con la clave de tp3/.env.")
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    if args.salida.exists():
        parser.error("--salida debe ser un directorio nuevo.")
    if args.openai_directo and args.modelo == MODELO_PROXY:
        parser.error("La API directa requiere el identificador sin el prefijo openai/.")
    modelo = args.modelo or (MODELO_OPENAI if args.openai_directo else MODELO_PROXY)
    baseline, minilm, consultas, hash_baseline = cargar_insumos(args.baseline, args.consultas)
    clave = dotenv_values(CARPETA.parent / ".env").get("OPENAI_API_KEY")
    if not clave:
        parser.error("Falta OPENAI_API_KEY en tp3/.env.")
    base_url = API_OPENAI if args.openai_directo else PROXY_CATEDRA
    cliente = OpenAI(base_url=base_url, api_key=clave, max_retries=0, timeout=60)
    chunks = minilm["chunks"]
    textos = [c["texto"] for c in chunks] + [q["texto"] for q in consultas]
    try:
        vectores, uso = obtener_vectores(cliente, modelo, textos)
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")

    args.salida.mkdir(parents=True, exist_ok=False)
    base = chromadb.PersistentClient(
        path=str(args.salida / "chroma"), settings=Settings(anonymized_telemetry=False),
    )
    ranking = buscar(base, "openai_i4", chunks, vectores[:len(chunks)], vectores[len(chunks):], consultas)
    resultado = {
        "sha256_baseline": hash_baseline,
        "sha256_consultas": baseline["sha256_consultas"],
        "fuentes": baseline["fuentes"],
        "max_tokens": baseline["max_tokens"],
        "solapamiento": baseline["solapamiento"],
        "distancia": baseline["distancia"],
        "k": K,
        "modelo_local_minilm": baseline["modelo_local"],
        "servicio_embeddings": "openai_directo" if args.openai_directo else "proxy_catedra",
        "modelo_openai_solicitado": modelo,
        "uso_openai": uso,
        "chunks": chunks,
        "ranking_minilm": minilm["ranking"],
        "ranking_openai": ranking,
    }
    ruta = args.salida / "ranking.json"
    ruta.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(chunks)} chunks y {len(consultas)} consultas. Resultados en {ruta}.")


if __name__ == "__main__":
    main()
