"""Piloto de recuperación con texto seleccionable, OCR y descripción de figuras."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
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
                x, y, ancho, alto = map(str, RECORTES[id_])
                run([
                    "pdftoppm", "-png", "-singlefile", "-scale-to-x", "1800", "-scale-to-y", "1014",
                    "-f", str(fisica), "-l", str(fisica), "-x", x, "-y", y, "-W", ancho, "-H", alto,
                    str(archivo), str(salida / id_),
                ], check=True)
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


def describir(cliente, ruta, recorte):
    imagen = (ruta.parent / recorte["archivo"]).read_bytes()
    if sha256(imagen).hexdigest() != recorte["sha256"]:
        raise ValueError("El recorte cambió después del OCR.")
    if ruta.exists():
        previo = json.loads(ruta.read_bytes())
        if (previo["sha256"], previo["prompt"], previo["modelo"]) != (
            recorte["sha256"], PROMPT_VISION, MODELO_VISION,
        ):
            raise ValueError("La descripción guardada corresponde a otra entrada.")
        return previo["texto"]
    datos = b64encode(imagen).decode("ascii")
    respuesta = cliente.chat.completions.create(
        model=MODELO_VISION, temperature=0, max_completion_tokens=450, store=False,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT_VISION},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{datos}", "detail": "high"}},
        ]}],
    )
    opcion = respuesta.choices[0]
    if opcion.finish_reason != "stop" or not opcion.message.content:
        raise ValueError("La descripción de la imagen quedó incompleta.")
    guardar_json(ruta, {
        "sha256": recorte["sha256"], "modelo": respuesta.model, "prompt": PROMPT_VISION,
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
    splitter, contar = crear_splitter()
    base, extra_ocr, extra_descripcion = [
        fragmentar(documentos, splitter, contar) for documentos in (insumos["paginas"], ocr, descripcion)
    ]
    todos = base + extra_ocr + extra_descripcion
    if len(base) < K or any(not 0 < c["tokens"] <= TAMANO for c in todos):
        raise ValueError("Los chunks deben respetar el tamaño y alcanzar el top k en las tres variantes.")
    vectores, uso = obtener_vectores(cliente, MODELO_OPENAI, [c["texto"] for c in todos] + CONSULTAS)
    if (uso["modelo_respuesta"], uso["dimensiones"]) != (MODELO_OPENAI, 1536):
        raise ValueError("Los embeddings deben coincidir con el modelo y la dimensión de I4.")
    por_id = dict(zip((c["id"] for c in todos), vectores))
    guardar_json(salida / "embeddings.json", {"uso": uso, "chunks": por_id, "consultas": vectores[len(todos):]})
    chroma = chromadb.PersistentClient(path=str(salida / "chroma"), settings=Settings(anonymized_telemetry=False))
    resultados = {}
    for nombre, adicional in (("texto", []), ("texto_ocr", extra_ocr), ("texto_descripcion", extra_descripcion)):
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
    parser.add_argument("accion", choices=("preparar", "comparar"))
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    (preparar if args.accion == "preparar" else comparar)(args.salida.resolve())


if __name__ == "__main__":
    main()
