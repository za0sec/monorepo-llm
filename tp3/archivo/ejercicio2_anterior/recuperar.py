"""Recupera referencias académicas desde un objetivo de estudio; no evalúa el video."""

import argparse
import json
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from time import perf_counter

from chromadb.errors import NotFoundError

from config import ARCHIVO_CHUNKS, DATA, K_INICIAL, RUTA_CHROMA
from indexar_material import (
    abrir_cliente, cargar_documento, cargar_modelo, codificar, crear_manifiesto,
    nombre_coleccion, validar_coleccion, validar_documento,
)
from preparar_chunks import formatear_ubicacion, hash_json
from proxy_catedra import MODELO_PROXY, MODELOS_PROXY, completar


INSTRUCCIONES_EXPANSION = (
    "Descomponé el objetivo de estudio en hasta cuatro subtemas mínimos, distintos y "
    "pertinentes para una primera comprensión en una materia de modelos de lenguaje. "
    "Por cada subtema, escribí una pregunta breve en español que permita recuperar "
    "una explicación en material de clase. Ordená de lo básico a lo específico. "
    "Devolvé únicamente un objeto JSON con la clave subtemas y una lista de objetos "
    "con las claves tema y consulta. No respondas las preguntas."
)

PROMPTS_EXPANSION = {
    "inicial": INSTRUCCIONES_EXPANSION,
    "concreto": INSTRUCCIONES_EXPANSION + (
        " Cuando el subtema trate de un proceso, mencioná en la pregunta un método "
        "representativo por su nombre y preguntá cómo funciona con un ejemplo. "
        "Para los usos, elegí una aplicación concreta. Mantené las preguntas cortas "
        "y elegí términos técnicos pertinentes para el objetivo."
    ),
}


def planificar_subtemas(objetivo: str, prompt: str = "inicial", modelo: str = MODELO_PROXY) -> dict:
    mensajes = [{"role": "system", "content": PROMPTS_EXPANSION[prompt]},
                {"role": "user", "content": objetivo}]
    plan = completar(mensajes, max_tokens=200, formato_json=True, modelo=modelo)
    return {**plan, "prompt_expansion": prompt}


def leer_subtemas(plan: dict, objetivo: str) -> list[dict]:
    try:
        if plan["finish_reason"] != "stop":
            raise ValueError("El proxy no completó el plan de subtemas.")
        if plan["mensajes"][-1]["content"] != objetivo:
            raise ValueError("El plan guardado corresponde a otro objetivo.")
        subtemas = json.loads(plan["respuesta"])["subtemas"]
        if not isinstance(subtemas, list) or not 1 <= len(subtemas) <= 4:
            raise ValueError("El plan debe contener entre uno y cuatro subtemas.")
        for subtema in subtemas:
            if any(not isinstance(subtema[campo], str) or not subtema[campo].strip()
                   for campo in ("tema", "consulta")):
                raise ValueError("Cada subtema necesita un tema y una consulta.")
        if len({s["consulta"].strip().casefold() for s in subtemas}) != len(subtemas):
            raise ValueError("El plan tiene consultas repetidas.")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("El plan guardado tiene un formato inválido.") from error
    return subtemas


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


def recuperar_expandida(objetivo: str, documento: dict, plan: dict, ruta: Path = RUTA_CHROMA,
                        nombre: str = None, k: int = K_INICIAL, solo_local: bool = False) -> dict:
    subtemas = leer_subtemas(plan, objetivo)
    busquedas = [recuperar(s["consulta"], documento, ruta, nombre, k, solo_local) for s in subtemas]
    vistos = {}
    for puesto in range(busquedas[0]["k_efectivo"]):
        for subtema, busqueda in zip(subtemas, busquedas):
            referencia = busqueda["referencias"][puesto]
            vistos.setdefault(referencia["chunk_id"], {
                **referencia, "subtema": subtema["tema"], "consulta": subtema["consulta"],
                "posicion_subconsulta": referencia["posicion"],
            })
    referencias = [{**r, "posicion": i} for i, r in enumerate(islice(vistos.values(), k), 1)]
    return {
        "objetivo": objetivo, "coleccion": busquedas[0]["coleccion"],
        "manifiesto": busquedas[0]["manifiesto"],
        "seleccion": "por puesto y orden de subtema, omitiendo IDs repetidos",
        "k_por_subconsulta": k, "k_final": len(referencias),
        "candidatos_totales": sum(b["k_efectivo"] for b in busquedas),
        "candidatos_unicos": len(vistos), "plan": plan,
        "subtemas": subtemas, "busquedas": busquedas, "referencias": referencias,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objetivo")
    parser.add_argument("--chunks", type=Path, default=ARCHIVO_CHUNKS)
    parser.add_argument("--base", type=Path, default=RUTA_CHROMA)
    parser.add_argument("--coleccion")
    parser.add_argument("--k", type=int, default=K_INICIAL)
    parser.add_argument("--solo-local", action="store_true")
    expansion = parser.add_mutually_exclusive_group()
    expansion.add_argument("--expandir", action="store_true", help="Generar subtemas por el proxy.")
    expansion.add_argument("--plan", type=Path, help="Reutilizar el JSON de un plan de subtemas.")
    parser.add_argument("--prompt-expansion", choices=PROMPTS_EXPANSION,
                        help="Variante del prompt para --expandir. Por defecto: inicial.")
    parser.add_argument("--modelo-proxy", choices=MODELOS_PROXY,
                        help=f"Modelo para --expandir. Por defecto: {MODELO_PROXY}.")
    parser.add_argument("--salida", type=Path, help="Archivo JSON; por defecto guarda una corrida nueva en data/resultados/.")
    args = parser.parse_args()
    if args.prompt_expansion and not args.expandir:
        parser.error("--prompt-expansion requiere --expandir.")
    if args.modelo_proxy and not args.expandir:
        parser.error("--modelo-proxy requiere --expandir.")
    try:
        objetivo = args.objetivo.strip()
        if not objetivo or args.k <= 0:
            raise ValueError("Se necesita un objetivo no vacío y k positivo.")
        documento = cargar_documento(args.chunks)
        salida = args.salida or DATA / "resultados" / (
            datetime.now(timezone.utc).strftime("retrieval_%Y%m%dT%H%M%S_%f_")
            + hash_json([objetivo, args.coleccion, args.k])[:8] + ".json"
        )
        salida.parent.mkdir(parents=True, exist_ok=True)
        if args.expandir or args.plan:
            if salida.exists():
                raise ValueError(f"Ya existe el resultado: {salida}")
            if args.expandir:
                ruta_plan = salida.with_name(salida.stem + ".plan.json")
                with ruta_plan.open("x", encoding="utf-8") as archivo:
                    try:
                        plan = planificar_subtemas(objetivo, args.prompt_expansion or "inicial",
                                                   modelo=args.modelo_proxy or MODELO_PROXY)
                    except (OSError, ValueError, RuntimeError):
                        ruta_plan.unlink()
                        raise
                    archivo.write(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
            else:
                plan = json.loads(args.plan.read_text(encoding="utf-8"))
            resultado = recuperar_expandida(objetivo, documento, plan, args.base,
                                            args.coleccion, args.k, args.solo_local)
        else:
            resultado = recuperar(objetivo, documento, args.base, args.coleccion,
                                  args.k, args.solo_local)
        modo = "x" if args.expandir or args.plan else "w"
        with salida.open(modo, encoding="utf-8") as archivo:
            archivo.write(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"No se pudieron recuperar referencias: {error}\n")
    for r in resultado["referencias"]:
        m = r["metadata"]
        print(f"\n{r['posicion']}. {m['fuente']} [{formatear_ubicacion(m)}]\n"
              f"ID: {r['chunk_id']}\nDistancia coseno: {r['distancia_coseno']:.4f}; "
              f"similitud (1-distancia): {r['similitud_coseno']:.4f}\n{r['texto']}")
    print(f"\nReferencias: {len(resultado['referencias'])}; resultado guardado en {salida}")


if __name__ == "__main__":
    main()
