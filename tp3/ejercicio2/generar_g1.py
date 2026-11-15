"""Prepara y compara dos generadores sobre la misma entrada manual de G1."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

from preparar_transcripcion import REPO, leer_vtt


CARPETA = Path(__file__).resolve().parent
DATOS = CARPETA / "data/experimentos/g1"
CACHE = CARPETA / "data/modelos"
CITAS = re.compile(r"\[([CV]\d+)\]")


def guardar_json(ruta, datos):
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preparar_entrada():
    import chromadb
    from chromadb.config import Settings

    caso = json.loads((CARPETA / "g1/caso.json").read_bytes())
    prompt = (CARPETA / "g1/prompt.md").read_text(encoding="utf-8").rstrip("\n")
    ruta_vtt = REPO / caso["video_vtt"]
    vtt = ruta_vtt.read_bytes()
    segmentos = leer_vtt(ruta_vtt, vtt)
    if len(segmentos) != caso["segmentos_video_esperados"]:
        raise ValueError("La cantidad de segmentos del VTT cambió.")
    video = {f"V{i}": segmento for i, segmento in enumerate(segmentos, 1)}

    indice = REPO / caso["indice_i4"]
    ranking = (indice / "ranking.json").read_bytes()
    chunks_i4 = {chunk["id"]: chunk for chunk in json.loads(ranking)["chunks"]}
    ruta_chroma = indice / "chroma"
    if not (ruta_chroma / "chroma.sqlite3").is_file():
        raise ValueError("Falta el índice persistido en I4.")
    cliente = chromadb.PersistentClient(
        path=str(ruta_chroma), settings=Settings(anonymized_telemetry=False),
    )
    coleccion = cliente.get_collection(caso["coleccion"], embedding_function=None)
    if coleccion.count() != len(chunks_i4) or coleccion.configuration["hnsw"]["space"] != "cosine":
        raise ValueError("La colección debe coincidir con el índice de I4.")
    ids = caso["chunks_manuales"]
    if len(ids) != 7 or len(set(ids)) != 7:
        raise ValueError("El control de G1 requiere siete IDs distintos de I4.")
    resultado = coleccion.get(ids=ids, include=["documents", "metadatas"])
    por_id = {
        id_: (texto, meta)
        for id_, texto, meta in zip(resultado["ids"], resultado["documents"], resultado["metadatas"])
    }
    if set(por_id) != set(ids):
        raise ValueError("No se encontraron los siete chunks en Chroma.")
    clase = {}
    for i, id_ in enumerate(ids, 1):
        texto, meta = por_id[id_]
        original = chunks_i4[id_]
        procedencia = {campo: original[campo] for campo in ("fuente", "inicio", "fin")}
        procedencia["bloques"] = ",".join(original["bloques"])
        if texto != original["texto"] or meta != procedencia:
            raise ValueError(f"El contenido o los metadatos de {id_} difieren de I4.")
        clase[f"C{i}"] = {"id": id_, "texto": texto, "tokens_minilm": original["tokens"], **meta}

    return {
        "configuracion_experimental": caso,
        "sha256_video": hashlib.sha256(vtt).hexdigest(),
        "sha256_ranking_i4": hashlib.sha256(ranking).hexdigest(),
        "clase": clase,
        "video": video,
        "mensajes": armar_mensajes(caso, prompt, clase, video),
    }


def armar_mensajes(caso, prompt, clase, video):
    texto_clase = "\n\n".join(f"[{i}] {p['texto']}" for i, p in clase.items())
    texto_video = "\n".join(f"[{i}] {p['texto']}" for i, p in video.items())
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": (
            f"Objetivo: {caso['objetivo']}\nNivel: {caso['nivel']}\n\n"
            f"Material de clase:\n{texto_clase}\n\nTranscripción del video:\n{texto_video}"
        )},
    ]


def generar_qwen(entrada):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = entrada["configuracion_experimental"]
    qwen = config["qwen"]
    opciones = {"cache_dir": CACHE, "revision": qwen["revision"], "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(qwen["modelo"], **opciones)
    tokens = tokenizer.apply_chat_template(
        entrada["mensajes"], add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    cantidad = tokens["input_ids"].shape[-1]
    if cantidad > qwen["max_tokens_entrada"]:
        raise ValueError(f"La entrada de Qwen excede el límite: {cantidad} tokens.")
    modelo = AutoModelForCausalLM.from_pretrained(qwen["modelo"], dtype=torch.float32, **opciones).eval()
    inicio = perf_counter()
    with torch.inference_mode():
        salida = modelo.generate(
            **tokens, do_sample=False, max_new_tokens=config["max_tokens_salida"],
            temperature=None, top_p=None, top_k=None,
            repetition_penalty=qwen["penalizacion_repeticion"],
        )[0, cantidad:]
    segundos = perf_counter() - inicio
    fin = "stop" if salida[-1].item() in modelo.generation_config.eos_token_id else "length"
    return tokenizer.decode(salida, skip_special_tokens=True).strip(), {
        **qwen, "dispositivo": "cpu", "dtype": "float32", "do_sample": False,
        "tokens_entrada": cantidad, "tokens_salida": len(salida),
        "motivo_fin": fin, "segundos": segundos,
    }


def generar_gpt41(entrada):
    from dotenv import dotenv_values
    from openai import OpenAI

    config = entrada["configuracion_experimental"]
    gpt41 = config["gpt41"]
    clave = dotenv_values(CARPETA.parent / ".env").get("OPENAI_API_KEY")
    if not clave:
        raise ValueError("Falta OPENAI_API_KEY en tp3/.env.")
    cliente = OpenAI(base_url="https://api.openai.com/v1", api_key=clave, max_retries=0, timeout=60)
    inicio = perf_counter()
    respuesta = cliente.chat.completions.create(
        model=gpt41["modelo"], messages=entrada["mensajes"], temperature=gpt41["temperatura"],
        max_completion_tokens=config["max_tokens_salida"], store=False,
    )
    segundos = perf_counter() - inicio
    opcion = respuesta.choices[0]
    return opcion.message.content or "", {
        **gpt41, "modelo_respuesta": respuesta.model,
        "tokens_entrada": respuesta.usage.prompt_tokens,
        "tokens_salida": respuesta.usage.completion_tokens,
        "motivo_fin": opcion.finish_reason, "segundos": segundos,
    }


def ubicar_citas(informe, entrada):
    fuentes = {**entrada["clase"], **entrada["video"]}

    def ubicar(cita):
        id_ = cita[1]
        if id_ not in fuentes:
            return cita[0]
        fuente = fuentes[id_]
        nombre = f"{Path(fuente['fuente']).stem}, " if id_ in entrada["clase"] else ""
        return f"[{id_}, {nombre}{fuente['inicio']}–{fuente['fin']}]"

    return CITAS.sub(ubicar, informe), sorted(set(CITAS.findall(informe)) - fuentes.keys())


def guardar_informe(destino, entrada, contenido, informe, metadatos):
    informe_ubicado, invalidas = ubicar_citas(informe, entrada)
    destino.mkdir()
    (destino / "informe_crudo.md").write_text(informe + "\n", encoding="utf-8")
    (destino / "informe.md").write_text(informe_ubicado + "\n", encoding="utf-8")
    guardar_json(destino / "corrida.json", {
        **metadatos,
        "sha256_entrada": hashlib.sha256(contenido).hexdigest(),
        "max_tokens_salida": entrada["configuracion_experimental"]["max_tokens_salida"],
        "citas_invalidas": invalidas,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=("preparar", "qwen", "gpt41"))
    parser.add_argument("--carpeta", type=Path, default=DATOS)
    parser.add_argument("--entrada", type=Path, help="JSON congelado para una corrida de generación.")
    args = parser.parse_args()
    ruta_entrada = args.entrada or args.carpeta / "entrada_manual.json"
    if args.accion == "preparar":
        if ruta_entrada.exists():
            parser.error("La entrada ya está congelada. Para otra preparación, usá --carpeta.")
        entrada = preparar_entrada()
        ruta_entrada.parent.mkdir(parents=True, exist_ok=True)
        guardar_json(ruta_entrada, entrada)
        print(f"Entrada congelada en {ruta_entrada}.")
        return

    destino = args.carpeta / args.accion
    if destino.exists():
        parser.error("La corrida ya existe. Para otra ejecución, usá --carpeta.")
    contenido = ruta_entrada.read_bytes()
    entrada = json.loads(contenido)
    generar = generar_qwen if args.accion == "qwen" else generar_gpt41
    informe, metadatos = generar(entrada)
    guardar_informe(destino, entrada, contenido, informe, metadatos)
    print(f"Informe y auditoría en {destino}.")


if __name__ == "__main__":
    main()
