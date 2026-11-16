"""Piloto de recuperación con texto seleccionable, OCR y descripción de figuras."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
from subprocess import run
from tempfile import TemporaryDirectory

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

from comparar_embeddings import MODELO_OPENAI, obtener_vectores
from comparar_splitters import TAMANO, ubicar_fragmentos
from evaluar_video import K, SNAPSHOT
from generar_g1 import guardar_json
from indexar_apuntes import CARPETA, SOLAPAMIENTO, crear_splitter, registros
from recuperar_g2 import cliente_openai, consultar_vectores
from rerankear_g4 import reordenar


SLIDES = CARPETA / "slides"
PILOTO = CARPETA / "data/experimentos/piloto_imagenes"
PILOTO_FORMATOS = CARPETA / "data/experimentos/piloto_diapos_completas"
PAGINAS = {"clase_01_transformers.pdf": [22, 25, 27, 30, 36], "clase_02_embeddings.pdf": [31, 32, 33]}
# Recortes x, y, ancho y alto en píxeles al renderizar a 1800 × 1014.
RECORTES = {"transformers-22": (130, 140, 1518, 658), "embeddings-32": (114, 408, 1576, 442)}
CONSULTAS = [
    "¿Cómo se calcula la similitud coseno entre dos embeddings?",
    "¿Qué bloque conecta el encoder con el decoder?",
]
MODELO_VISION = "gpt-4.1-2025-04-14"
PROMPT_VISION = (
    "Describe en español la información visible en este recorte de una diapositiva académica. "
    "Si hay una fórmula legible, transcríbela en texto y conserva los símbolos, términos y agrupaciones. "
    "Si hay un diagrama, indica qué conectan y hacia dónde apuntan las flechas visibles. "
    "Conserva los nombres de las etiquetas. Usa hasta 140 palabras. "
    "Indica si alguna parte es ilegible y evita agregar información externa."
)
PROMPT_FORMATOS = PROMPT_VISION.replace("este recorte de una diapositiva académica", "esta imagen de material académico")
PROMPT_FLECHAS = PROMPT_FORMATOS.replace(
    "Si hay un diagrama, indica qué conectan y hacia dónde apuntan las flechas visibles.",
    "Si hay un diagrama, seguí las flechas visibles desde el inicio hasta la punta. "
    "Priorizá las que conectan grupos distintos. Escribí esas conexiones en líneas separadas "
    "con el formato «origen → destino». Si la punta llega a un componente dentro de un grupo, "
    "nombrá ese componente con su etiqueta exacta y no solo el grupo. "
    "Si la punta no permite identificar el destino, indicalo.",
)


def renderizar_pagina(archivo, fisica, prefijo, recorte=()):
    opciones = []
    if recorte:
        x, y, ancho, alto = map(str, recorte)
        opciones = ["-x", x, "-y", y, "-W", ancho, "-H", alto]
    run([
        "pdftoppm", "-png", "-singlefile", "-scale-to-x", "1800", "-scale-to-y", "1014",
        "-f", str(fisica), "-l", str(fisica), *opciones, str(archivo), str(prefijo),
    ], check=True)


def preparar(salida):
    manifiesto = json.loads((SLIDES / "manifiesto.json").read_bytes())
    documentos = {d["archivo"]: d for d in manifiesto["archivos"]}
    paginas, recortes = [], {}
    salida.mkdir(parents=True)
    for nombre, seleccion in PAGINAS.items():
        registro = documentos[nombre]
        archivo = SLIDES / nombre
        if sha256(archivo.read_bytes()).hexdigest() != registro["sha256_seleccion"]:
            raise ValueError(f"Cambió la selección {nombre}.")
        lector = PdfReader(archivo)
        clase = nombre.split("_", 2)[2].removesuffix(".pdf")
        for numero in seleccion:
            fisica = registro["paginas_originales"].index(numero) + 1
            if lector.page_labels[fisica - 1] != str(numero):
                raise ValueError("La etiqueta de página difiere del manifiesto.")
            meta = {"fuente": registro["fuente"], "pagina": numero}
            id_ = f"{clase}-{numero}"
            texto = (lector.pages[fisica - 1].extract_text() or "").strip()
            paginas.append((id_, texto, {**meta, "metodo": "seleccionable"}))
            if id_ in RECORTES:
                renderizar_pagina(archivo, fisica, salida / id_, RECORTES[id_])
                png = salida / f"{id_}.png"
                recortes[id_] = {**meta, "archivo": png.name, "sha256": sha256(png.read_bytes()).hexdigest()}

    with TemporaryDirectory() as temporal:
        cache = Path(temporal) / "modules"
        ocr = Path(temporal) / "ocr"
        run([
            "swiftc", "-module-cache-path", str(cache), str(CARPETA / "ocr_recortes.swift"), "-o", str(ocr),
        ], check=True)
        lectura = run([str(ocr), *(str(salida / r["archivo"]) for r in recortes.values())],
                      capture_output=True, text=True)
    if lectura.returncode != 0:
        raise RuntimeError(lectura.stderr.strip() or f"OCR terminó con código {lectura.returncode}.")
    lineas = lectura.stdout.splitlines()
    if len(lineas) != len(recortes):
        raise ValueError("Se esperaba una salida OCR por recorte.")
    for recorte, resultado in zip(recortes.values(), lineas):
        recorte["ocr"] = json.loads(resultado)
    guardar_json(salida / "preparacion.json", {"paginas": paginas, "recortes": recortes})


def describir(cliente, ruta, figura, prompt=PROMPT_VISION):
    imagen = (ruta.parent / figura["archivo"]).read_bytes()
    if sha256(imagen).hexdigest() != figura["sha256"]:
        raise ValueError("La imagen no coincide con la entrada preparada.")
    if ruta.exists():
        previo = json.loads(ruta.read_bytes())
        if (previo["sha256"], previo["prompt"], previo["modelo"]) != (
            figura["sha256"], prompt, MODELO_VISION,
        ):
            raise ValueError("La descripción guardada corresponde a otra entrada.")
        return previo["texto"]
    datos = b64encode(imagen).decode("ascii")
    respuesta = cliente.chat.completions.create(
        model=MODELO_VISION, temperature=0, max_completion_tokens=450, store=False,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{datos}", "detail": "high"}},
        ]}],
    )
    opcion = respuesta.choices[0]
    if opcion.finish_reason != "stop" or not opcion.message.content:
        raise ValueError("La descripción de la imagen quedó incompleta.")
    guardar_json(ruta, {
        "sha256": figura["sha256"], "modelo": respuesta.model, "prompt": prompt,
        "uso": respuesta.usage.model_dump(), "texto": opcion.message.content.strip(),
    })
    return opcion.message.content.strip()


def fragmentar(documentos, splitter, contar):
    return [
        {"id": f"{id_}-{meta['metodo']}-{i}", "texto": fragmento, "tokens": contar(fragmento), "metadata": meta}
        for id_, texto, meta in documentos
        for i, (_, _, fragmento) in enumerate(
            ubicar_fragmentos(splitter.split_text(texto), texto, contar, SOLAPAMIENTO), 1,
        )
    ]


def comparar(salida):
    insumos = json.loads((salida / "preparacion.json").read_bytes())
    if (salida / "chroma").exists():
        raise ValueError("La comparación requiere una carpeta sin un índice previo.")
    cliente = cliente_openai(timeout=120)
    ocr, descripcion = [], []
    for id_, figura in insumos["recortes"].items():
        meta = {campo: figura[campo] for campo in ("fuente", "pagina")}
        ocr.append((id_, "\n".join(figura["ocr"]), {**meta, "metodo": "ocr"}))
        texto = describir(cliente, salida / f"{id_}-descripcion.json", figura)
        descripcion.append((id_, texto, {**meta, "metodo": "descripcion"}))
    evaluar_variantes(salida, insumos["paginas"], {
        "texto": [], "texto_ocr": ocr, "texto_descripcion": descripcion,
    }, cliente)


def comparar_pagina(salida, piloto):
    insumos = json.loads((piloto / "preparacion.json").read_bytes())
    if (salida / "chroma").exists():
        raise ValueError("La comparación requiere una carpeta sin un índice previo.")
    manifiesto = json.loads((SLIDES / "manifiesto.json").read_bytes())
    documentos = {d["fuente"]: d for d in manifiesto["archivos"]}
    salida.mkdir(parents=True, exist_ok=True)
    variantes = {"texto_recorte": [], "texto_pagina": []}
    imagenes = {}
    for id_, recorte in insumos["recortes"].items():
        registro = documentos[recorte["fuente"]]
        archivo = SLIDES / registro["archivo"]
        if sha256(archivo.read_bytes()).hexdigest() != registro["sha256_seleccion"]:
            raise ValueError(f"Cambió la selección {archivo.name}.")
        fisica = registro["paginas_originales"].index(recorte["pagina"]) + 1
        imagen_recorte = salida / f"{id_}-recorte.png"
        copyfile(piloto / recorte["archivo"], imagen_recorte)
        renderizar_pagina(archivo, fisica, salida / f"{id_}-pagina")
        imagen_pagina = salida / f"{id_}-pagina.png"
        meta = {campo: recorte[campo] for campo in ("fuente", "pagina")}
        imagenes[id_] = {
            "recorte": {**meta, "archivo": imagen_recorte.name, "sha256": recorte["sha256"]},
            "pagina": {**meta, "archivo": imagen_pagina.name, "sha256": sha256(imagen_pagina.read_bytes()).hexdigest()},
        }
    guardar_json(salida / "preparacion.json", {"paginas": insumos["paginas"], "imagenes": imagenes})
    cliente = cliente_openai(timeout=120)
    for id_, alternativas in imagenes.items():
        for tipo, figura in alternativas.items():
            texto = describir(cliente, salida / f"{id_}-{tipo}-descripcion.json", figura, PROMPT_FORMATOS)
            meta = {campo: figura[campo] for campo in ("fuente", "pagina")}
            variantes[f"texto_{tipo}"].append((id_, texto, {**meta, "metodo": f"descripcion_{tipo}"}))
    evaluar_variantes(salida, insumos["paginas"], variantes, cliente)


def comparar_prompt(salida, piloto):
    insumos = json.loads((piloto / "preparacion.json").read_bytes())
    if (salida / "chroma").exists():
        raise ValueError("La comparación requiere una carpeta sin un índice previo.")
    salida.mkdir(parents=True, exist_ok=True)
    cliente = cliente_openai(timeout=120)
    variantes = {"prompt_control": [], "prompt_flechas": []}
    imagenes = {}
    for id_, medios in insumos["imagenes"].items():
        figura = medios["pagina"]
        ruta_control = piloto / f"{id_}-pagina-descripcion.json"
        if not ruta_control.is_file():
            raise ValueError(f"Falta la descripción de control de {id_}.")
        control = describir(cliente, ruta_control, figura, PROMPT_FORMATOS)
        copyfile(piloto / figura["archivo"], salida / figura["archivo"])
        nuevo = describir(cliente, salida / f"{id_}-flechas-descripcion.json", figura, PROMPT_FLECHAS)
        meta = {campo: figura[campo] for campo in ("fuente", "pagina")}
        variantes["prompt_control"].append((id_, control, {**meta, "metodo": "descripcion_pagina"}))
        variantes["prompt_flechas"].append((id_, nuevo, {**meta, "metodo": "descripcion_flechas"}))
        imagenes[id_] = {**figura, "sha256_control": sha256(ruta_control.read_bytes()).hexdigest()}
    guardar_json(salida / "preparacion.json", {"paginas": insumos["paginas"], "imagenes": imagenes})
    evaluar_variantes(salida, insumos["paginas"], variantes, cliente)


def evaluar_variantes(salida, paginas, variantes, cliente):
    splitter, contar = crear_splitter()
    base = fragmentar(paginas, splitter, contar)
    adicionales = {variante: fragmentar(docs, splitter, contar) for variante, docs in variantes.items()}
    todos = base + [chunk for chunks in adicionales.values() for chunk in chunks]
    if len(base) < K or any(not 0 < c["tokens"] <= TAMANO for c in todos):
        raise ValueError("Los chunks deben respetar el tamaño y alcanzar el top k en las tres variantes.")
    vectores, uso = obtener_vectores(cliente, MODELO_OPENAI, [c["texto"] for c in todos] + CONSULTAS)
    if (uso["modelo_respuesta"], uso["dimensiones"]) != (MODELO_OPENAI, 1536):
        raise ValueError("Los embeddings deben coincidir con el modelo y la dimensión de I4.")
    por_id = dict(zip((c["id"] for c in todos), vectores))
    guardar_json(salida / "embeddings.json", {"uso": uso, "chunks": por_id, "consultas": vectores[len(todos):]})
    chroma = chromadb.PersistentClient(path=str(salida / "chroma"), settings=Settings(anonymized_telemetry=False))
    resultados = {}
    for nombre, adicional in adicionales.items():
        chunks = base + adicional
        coleccion = chroma.create_collection(nombre, embedding_function=None, configuration={"hnsw": {"space": "cosine"}})
        coleccion.add(
            ids=[c["id"] for c in chunks], documents=[c["texto"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks], embeddings=[por_id[c["id"]] for c in chunks],
        )
        if set(registros(coleccion)) != {c["id"] for c in chunks}:
            raise ValueError(f"El índice {nombre} quedó incompleto.")
        rankings = consultar_vectores(
            coleccion, {c["id"]: c for c in chunks}, vectores[len(todos):], min(12, len(chunks)),
        )
        finales = []
        for consulta, candidatos in zip(CONSULTAS, rankings):
            orden, auditoria = reordenar(consulta, candidatos, SNAPSHOT)
            finales.append({"consulta": consulta, "top": orden[:K], "reranker": auditoria})
        resultados[nombre] = {"chunks": chunks, "rankings": finales}
    guardar_json(salida / "resultados.json", {"k": K, "uso_embeddings": uso, "variantes": resultados})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=("preparar", "comparar", "comparar-pagina", "comparar-prompt"))
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--piloto", type=Path, help="Carpeta del piloto que se usa como control.")
    args = parser.parse_args()
    if args.accion == "comparar-pagina":
        comparar_pagina(args.salida.resolve(), (args.piloto or PILOTO).resolve())
    elif args.accion == "comparar-prompt":
        comparar_prompt(args.salida.resolve(), (args.piloto or PILOTO_FORMATOS).resolve())
    else:
        (preparar if args.accion == "preparar" else comparar)(args.salida.resolve())


if __name__ == "__main__":
    main()
