"""Evalúa la cobertura de un video a partir de un objetivo y su audio."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

from openai import APIConnectionError, APIStatusError

from generar_g1 import CARPETA, DATOS, armar_mensajes, generar_gpt41, guardar_informe, guardar_json
from preparar_transcripcion import leer_vtt
from recuperar_g2 import abrir_indice, buscar, cliente_openai
from recuperar_g3 import consultas_expandidas, expandir, intercalar
from rerankear_g4 import MODELO as MODELO_RERANKER
from rerankear_g4 import reordenar
from transcripcion.transcribir_whisper import MODELO as MODELO_ASR
from transcripcion.transcribir_whisper import transcribir_archivo


REVISION_RERANKER = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
SNAPSHOT = (
    DATOS.parent / "g4/modelos" / f"models--{MODELO_RERANKER.replace('/', '--')}"
    / "snapshots" / REVISION_RERANKER
)
K = 7


def recuperar_clase(cliente, indice, caso, modo, snapshot):
    coleccion, i4, por_id = indice
    objetivo = caso["objetivo"]
    detalle = {"modo": modo, "consulta": objetivo, "k": K, "distancia": i4["distancia"]}
    consultas, por_consulta = [objetivo], K
    if modo == "avanzado":
        expansion = expandir(cliente, caso)
        consultas = consultas_expandidas(caso, expansion)
        detalle["expansion"] = expansion
        por_consulta = 5

    rankings, uso = buscar(coleccion, i4, por_id, cliente, consultas, por_consulta)
    candidatos = intercalar(rankings)
    if len(candidatos) < K:
        refuerzo, uso_refuerzo = buscar(coleccion, i4, por_id, cliente, [objetivo], K)
        por_chunk = {chunk["id"]: chunk for chunk in candidatos}
        for chunk in refuerzo[0]:
            por_chunk.setdefault(chunk["id"], chunk)
        candidatos = list(por_chunk.values())
        detalle["refuerzo_consulta_original"] = {"resultados": refuerzo[0], "uso_embeddings": uso_refuerzo}
    detalle["uso_embeddings"] = uso
    detalle["rankings"] = [{"consulta": q, "resultados": r} for q, r in zip(consultas, rankings)]
    detalle["candidatos_distintos"] = [chunk["id"] for chunk in candidatos]
    if modo == "avanzado":
        candidatos, detalle["reranker"] = reordenar(objetivo, candidatos, snapshot)
    return candidatos[:K], detalle


def evaluar(audio, caso, modo, snapshot, salida):
    ruta_indice = Path(caso["indice_i4"]) / "ranking.json"
    huella_indice = sha256(ruta_indice.read_bytes()).hexdigest()
    indice = abrir_indice({"configuracion_experimental": caso, "sha256_ranking_i4": huella_indice})
    prompt = (CARPETA / "g1/prompt.md").read_text(encoding="utf-8").rstrip("\n")
    cliente = cliente_openai(timeout=120)

    print("Transcribiendo audio con Whisper…", flush=True)
    inicio = perf_counter()
    vtt = transcribir_archivo(cliente, audio, response_format="vtt").encode("utf-8")
    asr = {
        "modelo": MODELO_ASR, "idioma": "es", "temperatura": 0,
        "audio": audio.name, "sha256_audio": sha256(audio.read_bytes()).hexdigest(),
        "segundos": perf_counter() - inicio,
    }
    ruta_vtt = salida / "transcripcion.vtt"
    ruta_vtt.write_bytes(vtt)
    video = {f"V{i}": segmento for i, segmento in enumerate(leer_vtt(ruta_vtt, vtt), 1)}

    print(f"Recuperando material de clase ({modo})…", flush=True)
    seleccion, recuperacion = recuperar_clase(cliente, indice, caso, modo, snapshot)
    clase = {f"C{i}": chunk for i, chunk in enumerate(seleccion, 1)}
    entrada = {
        "configuracion_experimental": caso,
        "transcripcion": asr,
        "sha256_video": sha256(vtt).hexdigest(),
        "sha256_ranking_i4": huella_indice,
        "recuperacion": recuperacion,
        "clase": clase,
        "video": video,
        "mensajes": armar_mensajes(caso, prompt, clase, video),
    }
    ruta_entrada = salida / "entrada.json"
    guardar_json(ruta_entrada, entrada)

    print("Generando informe con GPT‑4.1…", flush=True)
    informe, auditoria = generar_gpt41(entrada)
    guardar_informe(salida / "gpt41", entrada, ruta_entrada.read_bytes(), informe, auditoria)
    if not informe.strip() or auditoria["motivo_fin"] != "stop":
        raise ValueError("La generación no terminó normalmente. La salida quedó guardada para revisión.")
    print(f"Informe: {salida / 'gpt41/informe.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objetivo")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--modo", choices=("avanzado", "basico"), default="avanzado")
    parser.add_argument("--indice", type=Path, default=DATOS.parent / "i4")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--salida", type=Path, help="Directorio nuevo para guardar el informe y su auditoría.")
    args = parser.parse_args()
    objetivo = " ".join(args.objetivo.split())
    if not objetivo:
        parser.error("Escribí un objetivo de estudio.")
    if not args.audio.is_file() or args.audio.stat().st_size == 0:
        parser.error("Indicá un archivo de audio existente y no vacío.")
    if args.modo == "avanzado" and not (args.snapshot / "model.safetensors").is_file():
        parser.error(f"Falta el snapshot del reranker en {args.snapshot}.")
    caso = {
        "objetivo": objetivo,
        "nivel": "introductorio",
        "indice_i4": str(args.indice.resolve()),
        "coleccion": "openai_i4",
        "max_tokens_salida": 1000,
        "gpt41": {"modelo": "gpt-4.1-2025-04-14", "temperatura": 0},
    }
    if args.salida is None:
        carpeta = CARPETA / "data/ejecuciones"
        carpeta.mkdir(parents=True, exist_ok=True)
        salida = Path(mkdtemp(prefix="video-", dir=carpeta))
    else:
        salida = args.salida.resolve()
        try:
            salida.mkdir(parents=True)
        except FileExistsError:
            parser.error("--salida debe apuntar a un directorio nuevo.")
    try:
        evaluar(args.audio, caso, args.modo, args.snapshot, salida)
    except APIStatusError as error:
        parser.exit(1, f"OpenAI respondió HTTP {error.status_code}. Artefactos parciales en {salida}.\n")
    except APIConnectionError:
        parser.exit(1, f"No se recibió confirmación de OpenAI. Artefactos parciales en {salida}.\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error} Artefactos en {salida}.\n")


if __name__ == "__main__":
    main()
