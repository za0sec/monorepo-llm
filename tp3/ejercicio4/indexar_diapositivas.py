"""Amplía el corpus de texto con los PDF de diapositivas y sus descripciones."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fcntl import LOCK_EX, LOCK_NB, flock
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile

import chromadb
from chromadb.config import Settings
import numpy as np
from pypdf import PdfReader

from comparar_embeddings import MODELO_OPENAI, TAMANO_LOTE, obtener_vectores
from comparar_imagenes import (
    MODELO_VISION, PILOTO_FORMATOS, PROMPT_FORMATOS, SLIDES, describir, fragmentar, renderizar_pagina,
)
from comparar_splitters import TAMANO
from generar_g1 import guardar_json
from indexar_apuntes import CARPETA, INDICE_I4, REPO, crear_splitter, escribir_coleccion, registros
from recuperar_g2 import cliente_openai


BASE = CARPETA / "data/carga_texto.json"
CACHE = CARPETA / "data/cache_diapositivas"


def leer(ruta):
    return json.loads(ruta.read_bytes())


def preparar_paginas(manifiesto):
    paginas = []
    for seleccion in manifiesto["archivos"]:
        ruta = SLIDES / seleccion["archivo"]
        if sha256(ruta.read_bytes()).hexdigest() != seleccion["sha256_seleccion"]:
            raise ValueError(f"El PDF {ruta.name} no coincide con el manifiesto.")
        lector = PdfReader(ruta)
        originales = seleccion["paginas_originales"]
        if lector.page_labels != list(map(str, originales)) or len(set(originales)) != len(originales):
            raise ValueError(f"La numeración de {ruta.name} no coincide con el manifiesto.")
        clase = ruta.stem.split("_", 2)[2]
        for fisica, original in enumerate(originales, 1):
            crudo = lector.pages[fisica - 1].extract_text() or ""
            limpio = crudo.encode("utf-8", errors="ignore").decode("utf-8")
            paginas.append({
                "id": f"{clase}-{original}",
                "pagina_pdf": fisica,
                "sha256_pdf": seleccion["sha256_seleccion"],
                "glifos_descartados": len(crudo) - len(limpio),
                "texto": limpio.strip(),
                "metadata": {
                    "fuente": seleccion["fuente"], "pagina": original, "tipo_fuente": "diapositiva",
                    "archivo": ruta.relative_to(REPO).as_posix(),
                },
            })
    if len(paginas) != 105 or len({p["id"] for p in paginas}) != len(paginas):
        raise ValueError("Se esperaban las 105 páginas seleccionadas sin duplicados.")
    return paginas


def preparar_descripcion(pagina, cache, cliente):
    id_ = pagina["id"]
    carpeta = cache / pagina["sha256_pdf"]
    carpeta.mkdir(exist_ok=True)
    registro = carpeta / f"{id_}-descripcion.json"
    prefijo = carpeta / f"{id_}-pagina"
    png = prefijo.with_suffix(".png")
    if not png.is_file():
        renderizar_pagina(REPO / pagina["metadata"]["archivo"], pagina["pagina_pdf"], prefijo)
    huella = sha256(png.read_bytes()).hexdigest()
    origen = "cache"
    if not registro.is_file():
        piloto = PILOTO_FORMATOS / f"{id_}-pagina-descripcion.json"
        if piloto.is_file():
            datos = leer(piloto)
            if (datos["sha256"], datos["prompt"], datos["modelo"]) == (huella, PROMPT_FORMATOS, MODELO_VISION):
                copyfile(piloto, registro)
                origen = "piloto"
        if not registro.is_file():
            origen = "api"
    texto = describir(cliente, registro, {"archivo": png.name, "sha256": huella}, PROMPT_FORMATOS)
    datos = leer(registro)
    return texto, {
        "origen": origen, "sha256_imagen": huella, "uso": datos["uso"],
        "ruta": registro.relative_to(REPO).as_posix(),
    }


def reutilizar_vectores(chunks):
    ruta = PILOTO_FORMATOS / "resultados.json"
    if not ruta.is_file():
        return {}
    piloto = leer(ruta)
    uso = piloto["uso_embeddings"]
    if (uso["modelo_respuesta"], uso["dimensiones"]) != (MODELO_OPENAI, 1536):
        return {}
    anteriores = {c["id"]: c for c in piloto["variantes"]["texto_pagina"]["chunks"]}
    vectores = leer(PILOTO_FORMATOS / "embeddings.json")["chunks"]
    reutilizados = {}
    for chunk in chunks:
        anterior = anteriores.get(chunk["id"])
        if (anterior and anterior["texto"] == chunk["texto"]
                and all(chunk["metadata"].get(k) == v for k, v in anterior["metadata"].items())):
            vector = np.asarray(vectores[chunk["id"]], dtype=np.float32)
            if vector.shape == (1536,) and np.isfinite(vector).all() and vector.any():
                reutilizados[chunk["id"]] = vector
    return reutilizados


def completar_vectores(chunks, reutilizados, cache, cliente):
    faltantes = [c for c in chunks if c["id"] not in reutilizados]
    por_id = dict(reutilizados)
    lotes = []
    for inicio in range(0, len(faltantes), TAMANO_LOTE):
        lote = faltantes[inicio:inicio + TAMANO_LOTE]
        textos = [c["texto"] for c in lote]
        huella = sha256(json.dumps([MODELO_OPENAI, textos], ensure_ascii=False).encode()).hexdigest()
        ruta = cache / f"embeddings-{huella}.json"
        previo = ruta.is_file()
        if previo:
            datos = leer(ruta)
            vectores, uso = datos["vectores"], datos["uso"]
        else:
            vectores, uso = obtener_vectores(cliente, MODELO_OPENAI, textos)
        matriz = np.asarray(vectores, dtype=np.float32)
        if ((uso["modelo_respuesta"], uso["dimensiones"]) != (MODELO_OPENAI, 1536)
                or matriz.shape != (len(lote), 1536) or not np.isfinite(matriz).all()
                or not matriz.any(axis=1).all()):
            raise ValueError("El lote no tiene los embeddings esperados.")
        if not previo:
            guardar_json(ruta.with_suffix(".json.tmp"), {"vectores": vectores, "uso": uso})
            ruta.with_suffix(".json.tmp").replace(ruta)
        por_id.update({c["id"]: vector for c, vector in zip(lote, matriz)})
        lotes.extend({**fila, "origen": "cache" if previo else "api"} for fila in uso["lotes"])
    return por_id, lotes


def indexar(nombre, salida, cache):
    if not (INDICE_I4 / "chroma/chroma.sqlite3").is_file():
        raise ValueError("Falta la base persistente de Chroma.")
    base = leer(BASE)
    i4 = (INDICE_I4 / "ranking.json").read_bytes()
    db = chromadb.PersistentClient(
        path=str(INDICE_I4 / "chroma"), settings=Settings(anonymized_telemetry=False),
    )
    if salida.exists() or nombre in {c.name for c in db.list_collections()}:
        raise ValueError("La colección de destino o el manifiesto ya existe.")
    previa = db.get_collection(base["coleccion"], embedding_function=None)
    existentes = registros(previa)
    ids_base = {c["id"] for c in leer_chunks_base(i4, base)}
    if existentes.keys() != ids_base or previa.configuration["hnsw"]["space"] != base["distancia"]:
        raise ValueError("La colección de texto no coincide con su manifiesto.")
    manifiesto = leer(SLIDES / "manifiesto.json")
    paginas = preparar_paginas(manifiesto)
    splitter, contar = crear_splitter()
    cliente = cliente_openai(timeout=120).with_options(max_retries=4)
    documentos = []
    with ThreadPoolExecutor(max_workers=8) as ejecutor:
        pedidos = ejecutor.map(lambda p: preparar_descripcion(p, cache, cliente), paginas)
        for i, (pagina, (descripcion, resultado)) in enumerate(zip(paginas, pedidos), 1):
            meta = pagina["metadata"]
            documentos.extend([
                (pagina["id"], pagina.pop("texto"), {**meta, "metodo": "seleccionable"}),
                (pagina["id"], descripcion, {**meta, "metodo": "descripcion_pagina"}),
            ])
            pagina["descripcion"] = resultado
            if i % 10 == 0 or i == len(paginas):
                print(f"{i}/{len(paginas)} descripciones preparadas.", flush=True)
    chunks = fragmentar(documentos, splitter, contar)
    ids_nuevos = {c["id"] for c in chunks}
    if (len(ids_nuevos) != len(chunks) or ids_nuevos & ids_base
            or any(not 0 < c["tokens"] <= TAMANO for c in chunks)):
        raise ValueError("Los fragmentos de diapositiva tienen IDs o tamaños inválidos.")
    metodos = {(c["metadata"]["archivo"], c["metadata"]["pagina"], c["metadata"]["metodo"]) for c in chunks}
    for documento in documentos:
        _, texto, meta = documento
        if not texto or (meta["archivo"], meta["pagina"], meta["metodo"]) not in metodos:
            raise ValueError(f"Falta contenido de {meta['metodo']} para {meta['fuente']} p. {meta['pagina']}.")
    reutilizados = reutilizar_vectores(chunks)
    print(f"{len(chunks)} chunks de diapositiva. {len(reutilizados)} embeddings reutilizados del piloto.", flush=True)
    vectores, lotes = completar_vectores(chunks, reutilizados, cache, cliente)
    registros_nuevos = {c["id"]: (c["texto"], c["metadata"], vectores[c["id"]]) for c in chunks}
    resultado = {
        "coleccion": nombre,
        "coleccion_base": base["coleccion"],
        "manifiesto_base": BASE.relative_to(REPO).as_posix(),
        "sha256_manifiesto_base": sha256(BASE.read_bytes()).hexdigest(),
        "distancia": base["distancia"], "splitter": base["splitter"],
        "vision": {"modelo": MODELO_VISION, "prompt": PROMPT_FORMATOS},
        "diapositivas": manifiesto["archivos"], "paginas": paginas,
        "conteo": {
            "base": len(existentes), "paginas": len(paginas),
            **Counter(c["metadata"]["metodo"] for c in chunks), "total": len(existentes) + len(chunks),
        },
        "uso_embeddings": {
            "modelo_respuesta": MODELO_OPENAI, "dimensiones": 1536,
            "reutilizados_piloto": len(reutilizados), "lotes": lotes,
        },
        "chunks_diapositivas": chunks,
    }
    escribir_coleccion(db, nombre, base["distancia"], {**existentes, **registros_nuevos}, salida, resultado)
    print(f"{nombre}: {resultado['conteo']['total']} chunks.", flush=True)


def leer_chunks_base(i4, base):
    if sha256(i4).hexdigest() != base["sha256_ranking_i4"]:
        raise ValueError("El registro de I4 cambió desde la carga de texto.")
    return json.loads(i4)["chunks"] + base["chunks_pdf"] + base["chunks_vtt_adicionales"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coleccion", default="openai_e4_completo")
    parser.add_argument("--salida", type=Path, default=CARPETA / "data/carga_completa.json")
    args = parser.parse_args()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / ".lock").open("a") as bloqueo:
        flock(bloqueo, LOCK_EX | LOCK_NB)
        indexar(args.coleccion, args.salida, CACHE)


if __name__ == "__main__":
    main()
