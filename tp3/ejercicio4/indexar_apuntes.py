"""Carga apuntes PDF en una colección mixta con los VTT congelados de I4."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from transformers import AutoTokenizer

from comparar_embeddings import MODELO_OPENAI, obtener_vectores
from comparar_splitters import MINILM_LOCAL, SEPARADORES, TAMANO, ubicar_fragmentos
from preparar_transcripcion import REPO
from recuperar_g2 import cliente_openai


CARPETA = Path(__file__).resolve().parent
INDICE_I4 = CARPETA.parent / "ejercicio2/data/experimentos/i4"
PDF_PILOTO = CARPETA / "apuntes/apuntes_transformers.pdf"
COLECCION_I4 = "openai_i4"
SOLAPAMIENTO = 10
TOLERANCIA_VECTOR = 1e-7


def preparar_pdf(ruta):
    contenido = ruta.read_bytes()
    lector = PdfReader(BytesIO(contenido))
    huella = sha256(contenido).hexdigest()
    fuente = str(ruta.relative_to(REPO)) if ruta.is_relative_to(REPO) else str(ruta)
    tokenizer = AutoTokenizer.from_pretrained(str(MINILM_LOCAL), local_files_only=True)

    def contar(texto):
        return len(tokenizer.encode(texto, add_special_tokens=False, verbose=False))

    splitter = RecursiveCharacterTextSplitter(
        separators=SEPARADORES, length_function=contar, chunk_size=TAMANO,
        chunk_overlap=SOLAPAMIENTO, keep_separator="end", strip_whitespace=False,
    )
    chunks = []
    paginas = []
    for numero, pagina in enumerate(lector.pages, 1):
        texto = (pagina.extract_text() or "").strip()
        cortes = splitter.split_text(texto)
        for secuencia, (inicio, fin, fragmento) in enumerate(
            ubicar_fragmentos(cortes, texto, contar, SOLAPAMIENTO), 1,
        ):
            tokens = contar(fragmento)
            if not 0 < tokens <= TAMANO:
                raise ValueError(f"Chunk de la página {numero} fuera del límite de tokens.")
            chunks.append({
                "id": f"pdf-{huella[:16]}-p{numero}-{secuencia}",
                "texto": fragmento,
                "inicio_caracter": inicio,
                "fin_caracter": fin,
                "tokens": tokens,
                "metadata": {"fuente": fuente, "tipo_fuente": "apunte", "pagina": numero},
            })
        paginas.append({"pagina": numero, "chunks": len(cortes)})
    if not chunks:
        raise ValueError("El PDF no tiene texto extraíble.")
    return chunks, {"fuente": fuente, "sha256": huella, "paginas": paginas}


def registros(coleccion):
    datos = coleccion.get(include=["documents", "metadatas", "embeddings"])
    return dict(zip(datos["ids"], zip(datos["documents"], datos["metadatas"], datos["embeddings"])))


def cargar_vtt(coleccion, base):
    guardados = registros(coleccion)
    ids = [c["id"] for c in base["chunks"]]
    if len(ids) != len(guardados) or set(ids) != set(guardados):
        raise ValueError("Los IDs de Chroma no coinciden con el registro de I4.")
    dimension = base["uso_openai"]["dimensiones"]
    for chunk in base["chunks"]:
        texto, meta, vector = guardados[chunk["id"]]
        esperado = {campo: chunk[campo] for campo in ("fuente", "inicio", "fin")}
        esperado["bloques"] = ",".join(chunk["bloques"])
        if texto != chunk["texto"] or meta != esperado or len(vector) != dimension:
            raise ValueError(f"El chunk {chunk['id']} de Chroma no coincide con I4.")
    return {id_: guardados[id_] for id_ in ids}


def cargar_apuntes_previos(cliente, ruta, huella_i4, uso_i4):
    previo = json.loads(ruta.read_bytes())
    uso_previo = previo["uso_embeddings_pdf"]
    campos = ("modelo_respuesta", "dimensiones")
    if previo["sha256_ranking_i4"] != huella_i4 or any(uso_previo[c] != uso_i4[c] for c in campos):
        raise ValueError("El manifiesto previo debe usar la misma baseline y los mismos embeddings.")
    coleccion = cliente.get_collection(previo["coleccion"], embedding_function=None)
    if coleccion.configuration["hnsw"]["space"] != previo["distancia"]:
        raise ValueError("La distancia de la colección previa difiere de su manifiesto.")
    chunks = previo["chunks_pdf"]
    ids = [chunk["id"] for chunk in chunks]
    datos = coleccion.get(ids=ids, include=["documents", "metadatas", "embeddings"])
    guardados = dict(zip(datos["ids"], zip(datos["documents"], datos["metadatas"], datos["embeddings"])))
    if set(ids) != guardados.keys() or len(ids) != len(guardados):
        raise ValueError("Los IDs del apunte previo no coinciden con Chroma.")
    for chunk in chunks:
        texto, metadata, vector = guardados[chunk["id"]]
        if (texto, metadata) != (chunk["texto"], chunk["metadata"]) or len(vector) != uso_i4["dimensiones"]:
            raise ValueError(f"El chunk previo {chunk['id']} no coincide con su manifiesto.")
    pdfs = previo["pdfs"] if "pdfs" in previo else [previo["pdf"]]
    for pdf in pdfs:
        if sha256((REPO / pdf["fuente"]).read_bytes()).hexdigest() != pdf["sha256"]:
            raise ValueError(f"El PDF previo {pdf['fuente']} cambió desde su carga.")
    return previo["coleccion"], pdfs, chunks, {id_: guardados[id_] for id_ in ids}


def escribir_coleccion(cliente, nombre, distancia, esperados, destino, manifiesto):
    nueva = cliente.create_collection(
        nombre, embedding_function=None, configuration={"hnsw": {"space": distancia}},
    )
    try:
        ids = list(esperados)
        lote = cliente.get_max_batch_size()
        for inicio in range(0, len(ids), lote):
            seleccion = ids[inicio:inicio + lote]
            textos, metadatos, vectores = zip(*(esperados[id_] for id_ in seleccion))
            nueva.add(ids=seleccion, documents=list(textos), metadatas=list(metadatos), embeddings=np.asarray(vectores))
        guardados = registros(nueva)
        if guardados.keys() != esperados.keys():
            raise ValueError("Faltan chunks en la colección mixta.")
        diferencia_maxima = 0.0
        for id_, (texto, meta, vector) in esperados.items():
            guardado, metadata, embedding = guardados[id_]
            if (guardado, metadata) != (texto, meta) or embedding.shape != vector.shape:
                raise ValueError(f"El chunk {id_} cambió al guardar.")
            diferencia = float(np.max(np.abs(embedding - vector)))
            if not diferencia <= TOLERANCIA_VECTOR:
                raise ValueError(f"El vector de {id_} cambió más de {TOLERANCIA_VECTOR} al guardar.")
            diferencia_maxima = max(diferencia_maxima, diferencia)
        manifiesto["verificacion"] = {
            "diferencia_maxima_componentes": diferencia_maxima,
            "tolerancia": TOLERANCIA_VECTOR,
        }
        with destino.open("x", encoding="utf-8") as archivo:
            try:
                json.dump(manifiesto, archivo, ensure_ascii=False, indent=2)
                archivo.write("\n")
                archivo.flush()
            except OSError:
                destino.unlink()
                raise
    except Exception:
        cliente.delete_collection(nombre)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, action="append", help="PDF nuevo. Se puede indicar varias veces.")
    parser.add_argument("--desde-manifiesto", type=Path, help="Carga anterior cuyos apuntes se reutilizan.")
    parser.add_argument("--indice", type=Path, default=INDICE_I4)
    parser.add_argument("--coleccion", default="openai_e4_apuntes")
    parser.add_argument("--salida", type=Path, default=CARPETA / "data/carga_apuntes.json")
    args = parser.parse_args()
    if args.desde_manifiesto and not args.pdf:
        parser.error("Indicá al menos un PDF nuevo con --pdf.")
    if args.salida.exists():
        parser.error("--salida debe apuntar a un archivo nuevo.")
    ruta_chroma = args.indice / "chroma"
    if not (ruta_chroma / "chroma.sqlite3").is_file():
        parser.error("Falta la colección persistida de I4.")
    cliente = chromadb.PersistentClient(path=str(ruta_chroma), settings=Settings(anonymized_telemetry=False))
    if args.coleccion in {coleccion.name for coleccion in cliente.list_collections()}:
        parser.error("La colección de destino ya existe. Elegí un nombre nuevo.")
    datos = (args.indice / "ranking.json").read_bytes()
    base = json.loads(datos)
    huella_i4 = sha256(datos).hexdigest()
    original = cliente.get_collection(COLECCION_I4, embedding_function=None)
    if (base["max_tokens"], base["solapamiento"], base["distancia"]) != (TAMANO, SOLAPAMIENTO, "cosine"):
        parser.error("Se necesita la baseline recursiva de I4 de 160 tokens y 10 de solapamiento.")
    if original.configuration["hnsw"]["space"] != base["distancia"]:
        parser.error("La métrica de Chroma difiere del registro de I4.")

    vtt = cargar_vtt(original, base)
    coleccion_previa, pdfs, chunks, apuntes = None, [], [], {}
    if args.desde_manifiesto:
        coleccion_previa, pdfs, chunks, apuntes = cargar_apuntes_previos(
            cliente, args.desde_manifiesto, huella_i4, base["uso_openai"],
        )
    ids = set(vtt) | apuntes.keys()
    huellas = {pdf["sha256"] for pdf in pdfs}
    nuevos = []
    for ruta in args.pdf or [PDF_PILOTO]:
        paginas, pdf = preparar_pdf(ruta.resolve())
        ids_pdf = {chunk["id"] for chunk in paginas}
        if pdf["sha256"] in huellas or ids & ids_pdf:
            parser.error(f"El PDF {pdf['fuente']} ya está incluido o repite IDs.")
        pdfs.append(pdf)
        nuevos.extend(paginas)
        ids.update(ids_pdf)
        huellas.add(pdf["sha256"])

    vectores, uso = obtener_vectores(cliente_openai(), MODELO_OPENAI, [c["texto"] for c in nuevos])
    if len(vectores) != len(nuevos) or any(uso[c] != base["uso_openai"][c] for c in ("modelo_respuesta", "dimensiones")):
        parser.error("Los embeddings del PDF deben coincidir con I4 en modelo y dimensión.")
    apuntes.update({
        c["id"]: (c["texto"], c["metadata"], np.asarray(vector, dtype=np.float32))
        for c, vector in zip(nuevos, vectores)
    })
    manifiesto = {
        "coleccion": args.coleccion,
        "coleccion_vtt": COLECCION_I4,
        "sha256_ranking_i4": huella_i4,
        "distancia": base["distancia"],
        "splitter": {"tamano": TAMANO, "solapamiento": SOLAPAMIENTO, "revision_tokenizer": MINILM_LOCAL.name},
        "pdfs": pdfs,
        "conteo": {"vtt": len(vtt), "pdf": len(apuntes), "total": len(vtt) + len(apuntes)},
        "uso_embeddings_pdf": uso,
        "chunks_pdf": chunks + nuevos,
    }
    if len(pdfs) == 1:
        manifiesto["pdf"] = pdfs[0]
    if coleccion_previa:
        manifiesto["coleccion_apuntes_previa"] = coleccion_previa
        manifiesto["chunks_pdf_reutilizados"] = len(chunks)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    escribir_coleccion(cliente, args.coleccion, base["distancia"], {**vtt, **apuntes}, args.salida, manifiesto)
    print(f"{args.coleccion}: {len(vtt)} chunks VTT y {len(apuntes)} chunks PDF.")


if __name__ == "__main__":
    main()
