"""Prepara chunks de una clase VTT conservando texto, intervenciones y tiempos.

Referencias: Langchain_Pinecone.ipynb, celda 15, y
UN_Report_Advanced_RAG.ipynb, celdas 11-14, en docs/RAG.
"""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from preparar_transcripcion import REPO, leer_vtt


MODELO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
VERSION = 1


def hash_json(valor) -> str:
    contenido = json.dumps(valor, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(contenido.encode("utf-8")).hexdigest()


def cargar_tokenizer(revision: str = REVISION, solo_local: bool = False):
    """Carga solo tokenizer y configuración del encoder; no carga sus pesos."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODELO, revision=revision, use_fast=True, local_files_only=solo_local
    )
    archivo = hf_hub_download(
        MODELO, "sentence_bert_config.json", revision=revision,
        local_files_only=solo_local,
    )
    limite = json.loads(Path(archivo).read_text())["max_seq_length"]
    return tokenizer, min(limite, tokenizer.model_max_length)


def unir_segmentos(segmentos: list[dict]) -> tuple[str, list[tuple]]:
    """Registra offsets de caracteres sobre el texto limpio unido por párrafos."""
    partes, posiciones = [], []
    cursor = 0
    for segmento in segmentos:
        texto = segmento["texto"]
        posiciones.append((cursor, cursor + len(texto), segmento))
        partes.append(texto)
        cursor += len(texto) + 2  # Separador entre intervenciones: \n\n.
    return "\n\n".join(partes), posiciones


def ventanas_tokens(texto: str, tokenizer, tamano: int, overlap: int):
    """Corta mediante offsets del tokenizer, sin decodificar ni normalizar texto.

    Al cortar dentro de una palabra, volver a tokenizar puede cambiar el conteo.
    Por eso se verifica cada substring y se achica si excede el presupuesto.
    """
    codificado = tokenizer(
        texto, add_special_tokens=False, truncation=False,
        return_offsets_mapping=True,
    )
    offsets = codificado["offset_mapping"]
    if not offsets:
        raise ValueError("Hay un fragmento sin tokens; revisar el texto original.")
    inicio_token = 0
    while inicio_token < len(offsets):
        fin_token = min(inicio_token + tamano, len(offsets))
        inicio = 0 if inicio_token == 0 else offsets[inicio_token][0]
        while fin_token > inicio_token:
            fin = offsets[fin_token][0] if fin_token < len(offsets) else len(texto)
            fragmento = texto[inicio:fin]
            cantidad = len(tokenizer.encode(fragmento, add_special_tokens=False))
            if cantidad <= tamano:
                break
            fin_token -= 1
        else:
            raise ValueError("No se puede representar un fragmento con ese límite de tokens.")
        yield inicio, fin, cantidad
        if fin_token == len(offsets):
            break
        inicio_token = max(inicio_token + 1, fin_token - overlap)


def crear_chunks(
    segmentos: list[dict], fuente: str, sha256_fuente: str, tokenizer,
    limite_encoder: int, caracteres: int = 1000, overlap_caracteres: int = 100,
    tokens: int = 200, overlap_tokens: int = 20, revision: str = REVISION,
) -> dict:
    if caracteres <= 0 or not 0 <= overlap_caracteres < caracteres:
        raise ValueError("El tamaño por caracteres debe ser positivo y superar al overlap.")
    if tokens <= 0 or not 0 <= overlap_tokens < tokens:
        raise ValueError("El tamaño por tokens debe ser positivo y superar al overlap.")
    especiales = tokenizer.num_special_tokens_to_add(pair=False)
    if tokens + especiales > limite_encoder:
        raise ValueError(
            f"{tokens} tokens + {especiales} especiales exceden el límite "
            f"del encoder ({limite_encoder})."
        )
    if not segmentos:
        raise ValueError("No hay segmentos para preparar chunks.")
    configuracion = {
        "version": VERSION, "modelo_tokenizer": MODELO, "revision_tokenizer": revision,
        "caracteres": caracteres, "overlap_caracteres": overlap_caracteres,
        "tokens": tokens, "overlap_tokens": overlap_tokens,
        "limite_encoder": limite_encoder, "tokens_especiales": especiales,
        "separadores": ["\n\n", "\n", ". ", " ", ""],
    }
    id_configuracion = hash_json(configuracion)
    texto, posiciones = unir_segmentos(segmentos)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=caracteres, chunk_overlap=overlap_caracteres,
        separators=configuracion["separadores"],
    )
    chunks, vistos = [], set()
    anterior_inicio, anterior_fin = -1, 0
    for parte in splitter.split_text(texto):
        # Desambiguar repeticiones: no volver a asignar una aparición anterior.
        desde = max(0, anterior_inicio + 1, anterior_fin - overlap_caracteres)
        base = texto.find(parte, desde)
        if base < 0:
            raise ValueError("No se pudo localizar un fragmento en el texto original.")
        anterior_inicio, anterior_fin = base, base + len(parte)
        for inicio_rel, fin_rel, cantidad in ventanas_tokens(
            parte, tokenizer, tokens, overlap_tokens
        ):
            inicio, fin = base + inicio_rel, base + fin_rel
            if (inicio, fin) in vistos:
                continue
            vistos.add((inicio, fin))
            relacionados = [s for a, b, s in posiciones if a < fin and b > inicio]
            primero = min(relacionados, key=lambda s: s["inicio_ms"])
            ultimo = max(relacionados, key=lambda s: s["fin_ms"])
            fragmento = texto[inicio:fin]
            total = len(tokenizer.encode(fragmento, add_special_tokens=True))
            if cantidad > tokens or total > limite_encoder:
                raise ValueError("Un chunk excede el límite de tokens del encoder.")
            chunks.append({
                "chunk_id": hash_json([fuente, sha256_fuente, id_configuracion, inicio, fin]),
                "fuente": fuente, "id_configuracion": id_configuracion,
                "segmentos_ids": [s["id"] for s in relacionados],
                "inicio": primero["inicio"], "fin": ultimo["fin"],
                "inicio_ms": primero["inicio_ms"], "fin_ms": ultimo["fin_ms"],
                "inicio_caracter": inicio, "fin_caracter": fin,
                "tokens_contenido": cantidad, "tokens_totales": total,
                "texto": fragmento,
            })

    # Garantía de integridad: ningún carácter no blanco puede desaparecer.
    cubiertos = bytearray(len(texto))
    for chunk in chunks:
        a, b = chunk["inicio_caracter"], chunk["fin_caracter"]
        cubiertos[a:b] = b"\x01" * (b - a)
    if any(not cubiertos[i] and not c.isspace() for i, c in enumerate(texto)):
        raise ValueError("El splitting perdió contenido; no se guardarán resultados.")
    return {
        "fuente": fuente, "sha256_fuente": sha256_fuente,
        "configuracion": configuracion, "id_configuracion": id_configuracion,
        "cantidad_segmentos": len(segmentos), "chunks": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", nargs="?", type=Path, default=REPO / "docs/embeddings_2.VTT")
    parser.add_argument("--salida", type=Path, default=Path(__file__).parent / "data/chunks")
    parser.add_argument("--caracteres", type=int, default=1000)
    parser.add_argument("--overlap-caracteres", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--overlap-tokens", type=int, default=20)
    parser.add_argument("--solo-local", action="store_true", help="Usar solo archivos de tokenizer en caché.")
    args = parser.parse_args()
    try:
        ruta = args.entrada.resolve()
        segmentos = leer_vtt(ruta)
        fuente = str(ruta.relative_to(REPO)) if ruta.is_relative_to(REPO) else str(ruta)
        tokenizer, limite = cargar_tokenizer(solo_local=args.solo_local)
        documento = crear_chunks(
            segmentos, fuente, hashlib.sha256(ruta.read_bytes()).hexdigest(),
            tokenizer, limite, args.caracteres, args.overlap_caracteres,
            args.tokens, args.overlap_tokens,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"No se pudieron preparar los chunks: {error}\n")
    args.salida.mkdir(parents=True, exist_ok=True)
    json_path = args.salida / f"{ruta.stem}.chunks.json"
    txt_path = args.salida / f"{ruta.stem}.chunks.txt"
    json_path.write_text(
        json.dumps(documento, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    txt_path.write_text(
        f"Fuente: {fuente}\nConfiguración: {documento['id_configuracion']}\n\n"
        + "\n\n".join(
            f"Chunk {i}: {c['chunk_id']}\n"
            f"[{c['inicio']} --> {c['fin']}] Intervenciones: {', '.join(c['segmentos_ids'])}\n"
            f"Tokens: {c['tokens_contenido']} de contenido, {c['tokens_totales']} con especiales\n"
            f"{c['texto']}"
            for i, c in enumerate(documento["chunks"], 1)
        ) + "\n", encoding="utf-8",
    )
    print(f"Fuente: {fuente}")
    print(f"Segmentos: {len(segmentos)}; chunks: {len(documento['chunks'])}")
    print(f"Máximo de tokens con especiales: {max(c['tokens_totales'] for c in documento['chunks'])}")
    print(f"JSON: {json_path}\nVista de texto: {txt_path}")


if __name__ == "__main__":
    main()
