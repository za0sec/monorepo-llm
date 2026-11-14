"""Compara ventanas fijas y splitting recursivo con búsqueda por coseno."""

import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import chromadb
import torch
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


CARPETA = Path(__file__).resolve().parent
TAMANO = 160
K = 4
SEPARADORES = ["\n\n", "\n", ". ", " ", ""]


def unir_segmentos(segmentos):
    tramos = []
    inicio = 0
    for segmento in segmentos:
        fin = inicio + len(segmento["texto"])
        tramos.append((inicio, fin, segmento))
        inicio = fin + 1
    return "\n".join(s["texto"] for s in segmentos), tramos


def ventanas_fijas(texto, tokenizer, contar, tamano):
    """Usa offsets de MiniLM para conservar el texto original al cortar."""
    offsets = tokenizer(
        texto, add_special_tokens=False, truncation=False,
        return_offsets_mapping=True, verbose=False,
    )["offset_mapping"]
    inicio_token = 0
    inicio = 0
    while inicio_token < len(offsets):
        fin_token = min(inicio_token + tamano, len(offsets))
        while fin_token > inicio_token:
            fin = offsets[fin_token][0] if fin_token < len(offsets) else len(texto)
            # Un corte dentro de una palabra puede cambiar el conteo al retokenizar.
            if inicio < fin and 0 < contar(texto[inicio:fin]) <= tamano:
                break
            fin_token -= 1
        else:
            raise ValueError("No se encontró un corte válido para la ventana fija.")
        yield texto[inicio:fin]
        inicio_token, inicio = fin_token, fin


def ubicar_fragmentos(fragmentos, texto, contar, solapamiento):
    inicio_anterior = -1
    fin_anterior = 0
    for fragmento in fragmentos:
        if solapamiento and fin_anterior:
            # Cada corte avanza en la misma fuente y puede empezar dentro del anterior.
            posibles = []
            inicio = texto.find(fragmento, inicio_anterior + 1)
            while 0 <= inicio <= fin_anterior:
                if inicio + len(fragmento) > fin_anterior:
                    if contar(texto[inicio:fin_anterior]) <= solapamiento:
                        posibles.append(inicio)
                inicio = texto.find(fragmento, inicio + 1)
            if len(posibles) != 1:
                raise ValueError(f"No se pudo ubicar de forma inequívoca el corte tras {fin_anterior}.")
            inicio = posibles[0]
        else:
            inicio = fin_anterior
        fin = inicio + len(fragmento)
        if texto[inicio:fin] != fragmento or fin <= fin_anterior:
            raise ValueError("Los chunks deben cubrir el texto de entrada en orden.")
        yield inicio, fin, fragmento
        inicio_anterior, fin_anterior = inicio, fin
    if fin_anterior != len(texto):
        raise ValueError("El último chunk no llega al final del texto de entrada.")


def crear_chunks(fragmentos, texto, tramos, contar, tamano, fuente, primer_id, solapamiento):
    chunks = []
    cortes = ubicar_fragmentos(fragmentos, texto, contar, solapamiento)
    for numero, (inicio, fin, fragmento) in enumerate(cortes, primer_id):
        tokens = contar(fragmento)
        if not 0 < tokens <= tamano:
            raise ValueError(f"Chunk {numero} fuera del límite: {tokens} tokens.")
        bloques = [s for a, b, s in tramos if a < fin and inicio < b]
        chunks.append({
            "id": f"c{numero:03}",
            "texto": fragmento,
            "fuente": fuente,
            "inicio_caracter": inicio,
            "fin_caracter": fin,
            "bloques": [s["id"] for s in bloques],
            "inicio": bloques[0]["inicio"],
            "fin": max(bloques, key=lambda s: s["fin_ms"])["fin"],
            "tokens": tokens,
        })
    return chunks


def medir_texto_compartido(chunks, ranking, contar):
    """Cuenta tokens de intervalos repetidos de una misma fuente."""
    por_id = {c["id"]: c for c in chunks}
    solapes = [
        contar(actual["texto"][:max(0, previo["fin_caracter"] - actual["inicio_caracter"])])
        for previo, actual in zip(chunks, chunks[1:])
        if previo["fuente"] == actual["fuente"]
    ]
    consultas = {}
    for consulta, hits in ranking.items():
        recuperados = [por_id[hit["chunk"]] for hit in hits]
        fin_por_fuente = {}
        repetidos = 0
        for chunk in sorted(recuperados, key=lambda c: (c["fuente"], c["inicio_caracter"])):
            fuente = chunk["fuente"]
            fin_previo = fin_por_fuente.get(fuente, 0)
            caracteres_repetidos = max(0, fin_previo - chunk["inicio_caracter"])
            repetidos += contar(chunk["texto"][:caracteres_repetidos])
            fin_por_fuente[fuente] = max(fin_previo, chunk["fin_caracter"])
        consultas[consulta] = {
            "tokens_recuperados": sum(c["tokens"] for c in recuperados),
            "tokens_en_intervalos_repetidos": repetidos,
        }
    return {
        "pares_contiguos": len(solapes),
        "pares_con_solapamiento": sum(s > 0 for s in solapes),
        "tokens_compartidos_totales": sum(solapes),
        "tokens_compartidos_maximo": max(solapes, default=0),
        "consultas": consultas,
    }


def codificar(modelo, textos):
    for texto in textos:
        if len(modelo.tokenizer.encode(texto, add_special_tokens=True)) > modelo.max_seq_length:
            raise ValueError("MiniLM tendría que truncar una entrada.")
    vectores = modelo.encode(textos, batch_size=32, show_progress_bar=False)
    if not all(math.isfinite(float(v)) for vector in vectores for v in vector):
        raise ValueError("MiniLM produjo un vector inválido.")
    return vectores.tolist()


def buscar(cliente, nombre, chunks, vectores_chunks, vectores_consultas, consultas):
    coleccion = cliente.create_collection(
        nombre, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
    )
    coleccion.add(
        ids=[c["id"] for c in chunks],
        documents=[c["texto"] for c in chunks],
        embeddings=vectores_chunks,
        metadatas=[{"fuente": c["fuente"], "inicio": c["inicio"], "fin": c["fin"],
                    "bloques": ",".join(c["bloques"])} for c in chunks],
    )
    if coleccion.count() != len(chunks):
        raise ValueError("La colección quedó incompleta.")
    ranking = coleccion.query(
        query_embeddings=vectores_consultas, n_results=K, include=["distances"],
    )
    return {
        consulta["id"]: [
            {"chunk": identificador, "distancia_coseno": distancia}
            for identificador, distancia in zip(ids, distancias)
        ]
        for consulta, ids, distancias in zip(consultas, ranking["ids"], ranking["distances"])
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo-local", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append",
                        help="JSON de un VTT preparado. Repetir para sumar fuentes.")
    parser.add_argument("--consultas", type=Path, default=CARPETA / "consultas_i1.json")
    parser.add_argument("--tamano", type=int, default=TAMANO,
                        help="Máximo de tokens de contenido por chunk.")
    parser.add_argument("--solapamiento", type=int, default=0,
                        help="Máximo de tokens compartidos para el splitter recursivo.")
    parser.add_argument("--solo-recursivo", action="store_true",
                        help="Ejecuta únicamente el splitter recursivo.")
    parser.add_argument("--salida", type=Path, required=True,
                        help="Directorio nuevo para el índice y los resultados.")
    args = parser.parse_args()
    if args.tamano < 1:
        parser.error("--tamano debe ser positivo.")
    if not 0 <= args.solapamiento < args.tamano:
        parser.error("--solapamiento debe ser menor que el tamaño y no negativo.")
    if args.solapamiento and not args.solo_recursivo:
        parser.error("Usar --solo-recursivo al configurar el solapamiento.")
    rutas = args.corpus or [CARPETA / "data/procesados/embeddings_2.json"]
    documentos = []
    fuentes = {}
    for ruta in rutas:
        datos = ruta.read_bytes()
        documento = json.loads(datos)
        fuente = documento["fuente"]
        if fuente in fuentes:
            raise ValueError(f"Fuente repetida: {fuente}")
        fuentes[fuente] = {
            "sha256_fuente": documento["sha256_fuente"],
            "sha256_corpus": hashlib.sha256(datos).hexdigest(),
        }
        documentos.append(documento)
    referencia = args.consultas.read_bytes()
    consultas = json.loads(referencia)["consultas"]

    torch.set_num_threads(1)
    modelo = SentenceTransformer(str(args.modelo_local.resolve()), device="cpu",
                                 local_files_only=True)
    tokenizer = modelo.tokenizer
    max_con_especiales = args.tamano + tokenizer.num_special_tokens_to_add(pair=False)
    if max_con_especiales > modelo.max_seq_length:
        parser.error("--tamano más los tokens especiales excede el límite del encoder.")

    def contar(fragmento):
        return len(tokenizer.encode(fragmento, add_special_tokens=False, verbose=False))

    recursivo = RecursiveCharacterTextSplitter(
        separators=SEPARADORES, length_function=contar, chunk_size=args.tamano,
        chunk_overlap=args.solapamiento, keep_separator="end", strip_whitespace=False,
    )
    variantes = {"recursivo": []} if args.solo_recursivo else {"ventanas_fijas": [], "recursivo": []}
    for documento in documentos:
        texto, tramos = unir_segmentos(documento["segmentos"])
        for nombre in variantes:
            cortes = (recursivo.split_text(texto) if nombre == "recursivo"
                      else ventanas_fijas(texto, tokenizer, contar, args.tamano))
            chunks = variantes[nombre]
            chunks.extend(crear_chunks(
                cortes, texto, tramos, contar, args.tamano, documento["fuente"],
                len(chunks) + 1, args.solapamiento,
            ))
    vectores_consultas = codificar(modelo, [q["texto"] for q in consultas])
    args.salida.mkdir(parents=True, exist_ok=False)
    cliente = chromadb.PersistentClient(
        path=str(args.salida / "chroma"), settings=Settings(anonymized_telemetry=False),
    )
    resultado = {
        "fuentes": fuentes,
        "sha256_consultas": hashlib.sha256(referencia).hexdigest(),
        "modelo_local": str(args.modelo_local.resolve()),
        "limite_encoder": modelo.max_seq_length,
        "max_tokens": args.tamano,
        "solapamiento": args.solapamiento,
        "separadores": SEPARADORES,
        "distancia": "cosine",
        "k": K,
        "python": platform.python_version(),
        "dependencias": {
            paquete: version(paquete)
            for paquete in ("chromadb", "langchain-text-splitters", "sentence-transformers",
                            "torch", "transformers", "tokenizers")
        },
        "variantes": {},
    }
    for nombre, chunks in variantes.items():
        vectores_chunks = codificar(modelo, [c["texto"] for c in chunks])
        ranking = buscar(cliente, nombre, chunks, vectores_chunks, vectores_consultas, consultas)
        resultado["variantes"][nombre] = {
            "chunks": chunks,
            "ranking": ranking,
            "texto_compartido": medir_texto_compartido(chunks, ranking, contar),
        }
        print(f"{nombre}: {len(chunks)} chunks y {len(ranking)} consultas.")
    ruta = args.salida / "ranking.json"
    ruta.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(ruta)


if __name__ == "__main__":
    main()
