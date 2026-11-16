"""Compara distintos cortes de un solo ranking del corpus completo."""

import argparse
from hashlib import sha256
from pathlib import Path

from comparar_apuntes import RUTA_CASO, VECINOS
from comparar_embeddings import MODELO_OPENAI, obtener_vectores
from evaluar_video import SNAPSHOT
from generar_g1 import CARPETA as CARPETA_E2
from generar_g1 import armar_mensajes, generar_gpt41, guardar_informe, guardar_json
from indexar_diapositivas import CARPETA, INDICE_I4, REPO, leer
from preparar_transcripcion import leer_vtt
from recuperar_g2 import abrir_corpus_completo, cliente_openai, consultar_vectores
from recuperar_g3 import consultas_expandidas, expandir, intercalar
from rerankear_g4 import reordenar


CORTES = (7, 8, 10, 12)
MANIFIESTO = CARPETA / "data/carga_completa.json"
CONTROL_CASO_02 = CARPETA / "data/experimentos/caso_02_apuntes"
CASOS_NUEVOS = {
    "03": (
        "transfer_learning",
        "Quiero entender cuál es la diferencia entre usar un modelo preentrenado para extraer "
        "características y hacerle fine-tuning, qué pesos se actualizan en cada caso y cuándo "
        "conviene cada enfoque",
    ),
    "04": (
        "tokenizacion",
        "Quiero entender cómo pasa un texto de ser una cadena de caracteres a tokens, IDs y "
        "embeddings, y qué representa cada paso",
    ),
    "05": (
        "razonamiento",
        "Quiero entender qué hacen los modelos de razonamiento, qué ganamos al dedicar más "
        "cómputo a una respuesta y qué técnicas permiten hacerlo durante la inferencia",
    ),
    "06": (
        "ai_safety",
        "Quiero entender la diferencia entre prompt injection y data poisoning en modelos de "
        "lenguaje y en qué parte del sistema actúa cada riesgo",
    ),
}


def preparar_ranking(salida, caso, coleccion, por_id, m, vecinos_por_consulta, control, consultas_previas):
    ruta = salida / "ranking.json"
    huella = sha256(MANIFIESTO.read_bytes()).hexdigest()
    if control and (control["sha256_manifiesto"], control["configuracion"], control["reranker"]["revision"]) != (
        huella, caso, SNAPSHOT.name,
    ):
        raise ValueError("El control corresponde a otro corpus, objetivo o reranker.")
    expansion_control = consultas_previas["expansion"] if consultas_previas else None
    if control:
        if expansion_control and expansion_control != control["expansion"]:
            raise ValueError("La expansión del control difiere de la expansión del caso.")
        expansion_control = control["expansion"]
    if ruta.exists():
        previo = leer(ruta)
        if (previo["sha256_manifiesto"], previo["configuracion"], previo["reranker"]["revision"]) != (
            huella, caso, SNAPSHOT.name,
        ) or previo["resultados_por_consulta"] != vecinos_por_consulta:
            raise ValueError("El ranking guardado pertenece a otra configuración.")
        if expansion_control and previo["expansion"] != expansion_control:
            raise ValueError("El ranking guardado usa otra expansión.")
        return previo
    ruta_expansion = salida / "expansion.json"
    expansion = expansion_control
    if expansion is None:
        expansion = (leer(ruta_expansion) if ruta_expansion.is_file()
                     else expandir(cliente_openai(timeout=120), caso))
    consultas = consultas_expandidas(caso, expansion)
    guardar_json(ruta_expansion, expansion)
    if consultas_previas:
        datos = consultas_previas["consultas"]
        if datos["consultas"] != consultas:
            raise ValueError("Los vectores guardados corresponden a otras consultas.")
        vectores, uso = datos["vectores"], datos["uso"]
    else:
        vectores, uso = obtener_vectores(cliente_openai(timeout=120), MODELO_OPENAI, consultas)
    if any(uso[c] != m["uso_embeddings"][c] for c in ("modelo_respuesta", "dimensiones")):
        raise ValueError("Los embeddings de consulta deben coincidir con el corpus completo.")
    vecinos = consultar_vectores(coleccion, por_id, vectores, vecinos_por_consulta)
    if control:
        cantidad_control = control["resultados_por_consulta"]
        prefijos = [[c["id"] for c in ranking[:cantidad_control]] for ranking in vecinos]
        ids_control = [[c["id"] for c in r["resultados"]] for r in control["rankings"]]
        if prefijos != ids_control:
            raise ValueError("Los prefijos de Chroma no reprodujeron los resultados del control.")
    candidatos = intercalar(vecinos)
    orden, auditoria = reordenar(caso["objetivo"], candidatos, SNAPSHOT)
    resultado = {
        "configuracion": caso, "sha256_manifiesto": huella, "distancia": m["distancia"],
        "expansion": expansion, "uso_embeddings": uso, "resultados_por_consulta": vecinos_por_consulta,
        "rankings": [{"consulta": q, "resultados": r} for q, r in zip(consultas, vecinos)],
        "reranker": auditoria, "orden_final": orden,
    }
    if consultas_previas:
        resultado["sha256_consultas_reutilizadas"] = consultas_previas["sha256_consultas"]
    guardar_json(ruta, resultado)
    return resultado


def cargar_caso(numero, prompt):
    if numero == "01":
        config = leer(CARPETA_E2 / "g1/caso.json")
        return config, REPO / config["video_vtt"], config["segmentos_video_esperados"], None
    if numero in CASOS_NUEVOS:
        nombre, objetivo = CASOS_NUEVOS[numero]
        carpeta = CARPETA / "data/casos" / nombre
        vtt = carpeta / "transcripcion.vtt"
        registro = leer(carpeta / "asr.json")
        if sha256(vtt.read_bytes()).hexdigest() != registro["sha256_vtt"]:
            raise ValueError(f"El VTT del Caso {numero} difiere del registrado por Whisper.")
        config = leer(CARPETA_E2 / "g1/caso.json")
        config["objetivo"] = objetivo
        return config, vtt, registro["segmentos"], None
    carpeta = CONTROL_CASO_02
    entrada = leer(carpeta / "openai_e4_apuntes_ampliado/entrada.json")
    expansion = carpeta / "expansion.json"
    consultas = carpeta / "consultas.json"
    vtt = RUTA_CASO / "transcripcion.vtt"
    for ruta, campo in ((expansion, "sha256_expansion"), (consultas, "sha256_consultas"), (vtt, "sha256_video")):
        if sha256(ruta.read_bytes()).hexdigest() != entrada[campo]:
            raise ValueError(f"{ruta.name} cambió desde el control del Caso 02.")
    if entrada["mensajes"][0]["content"] != prompt:
        raise ValueError("El prompt difiere del que se usó en el Caso 02.")
    previas = {
        "expansion": leer(expansion), "consultas": leer(consultas),
        "sha256_consultas": entrada["sha256_consultas"],
    }
    return entrada["configuracion_experimental"], vtt, 219, previas


def comparar(salida, vecinos_por_consulta, control, numero):
    coleccion, por_id, m, _ = abrir_corpus_completo(MANIFIESTO, INDICE_I4)
    prompt = (CARPETA_E2 / "g1/prompt.md").read_text(encoding="utf-8").rstrip("\n")
    config, ruta_vtt, cantidad_segmentos, consultas_previas = cargar_caso(numero, prompt)
    caso = {k: config[k] for k in ("objetivo", "nivel", "gpt41", "max_tokens_salida")}
    caso["coleccion"] = coleccion.name
    vtt = ruta_vtt.read_bytes()
    segmentos = leer_vtt(ruta_vtt, vtt)
    if len(segmentos) != cantidad_segmentos:
        raise ValueError(f"El VTT no tiene los segmentos esperados del Caso {numero}.")
    video = {f"V{i}": segmento for i, segmento in enumerate(segmentos, 1)}
    salida.mkdir(parents=True, exist_ok=True)
    ranking = preparar_ranking(salida, caso, coleccion, por_id, m, vecinos_por_consulta, control, consultas_previas)
    orden = ranking["orden_final"]
    cortes = CORTES
    ks = [k for k in cortes if k <= len(orden)]
    if len(orden) < max(cortes) and len(orden) not in ks:
        ks.append(len(orden))
    print(f"{len(orden)} candidatos únicos. Cortes: {ks}.", flush=True)
    huella_ranking = sha256((salida / "ranking.json").read_bytes()).hexdigest()
    for k in ks:
        carpeta = salida / f"k{k}"
        carpeta.mkdir(exist_ok=True)
        clase = {f"C{i}": c for i, c in enumerate(orden[:k], 1)}
        entrada = {
            "configuracion_experimental": caso, "k": k, "sha256_ranking": huella_ranking,
            "sha256_video": sha256(vtt).hexdigest(), "video": video, "clase": clase,
            "mensajes": armar_mensajes(caso, prompt, clase, video),
        }
        ruta = carpeta / "entrada.json"
        destino = carpeta / "gpt41"
        if ruta.exists() and leer(ruta) != entrada:
            raise ValueError(f"La entrada previa de k={k} difiere del ranking o del prompt.")
        if not ruta.exists():
            guardar_json(ruta, entrada)
        if destino.exists():
            if leer(destino / "corrida.json")["sha256_entrada"] != sha256(ruta.read_bytes()).hexdigest():
                raise ValueError(f"El informe previo de k={k} corresponde a otra entrada.")
            continue
        informe, auditoria = generar_gpt41(entrada)
        guardar_informe(destino, entrada, ruta.read_bytes(), informe, auditoria)
        print(f"k={k}: informe guardado ({auditoria['motivo_fin']}).", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--vecinos", type=int, default=VECINOS, help="Resultados de Chroma por consulta.")
    parser.add_argument("--control", type=Path, help="ranking.json con la expansión y los resultados de referencia.")
    parser.add_argument("--caso", choices=("01", "02", *CASOS_NUEVOS), default="01")
    args = parser.parse_args()
    if args.vecinos < 1:
        parser.error("--vecinos debe ser positivo.")
    comparar(args.salida, args.vecinos, leer(args.control) if args.control else None, args.caso)
