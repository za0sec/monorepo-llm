"""Mide WER y CER del VTT histórico y de las transcripciones por recorte."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from tp3.ejercicio2.transcripcion.medir_asr import medir
from tp3.ejercicio2.preparar_transcripcion import leer_vtt


def extraer_segmentos(segmentos, inicio, fin, incluir_bordes):
    if incluir_bordes:
        elegidos = [s for s in segmentos if s["inicio_ms"] < fin and inicio < s["fin_ms"]]
    else:
        elegidos = [s for s in segmentos if inicio * 2 <= s["inicio_ms"] + s["fin_ms"] < fin * 2]
    return " ".join(s["texto"] for s in elegidos), [
        {"id": s["id"], "inicio_ms": s["inicio_ms"], "fin_ms": s["fin_ms"]}
        for s in elegidos
    ]


def acumular(recortes, metodo, metrica):
    valores = [r["resultados"][metodo][metrica] for r in recortes]
    campos = ("sustituciones", "omisiones", "inserciones", "errores", "unidades_referencia")
    total = {campo: sum(v[campo] for v in valores) for campo in campos}
    total["tasa"] = total["errores"] / total["unidades_referencia"]
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifiesto", type=Path)
    parser.add_argument("--vtt", type=Path, required=True)
    parser.add_argument("--wav2vec", type=Path)
    parser.add_argument("--whisper", type=Path)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    if args.salida.exists():
        parser.error("La salida ya existe.")
    contenido_manifiesto = args.manifiesto.read_bytes()
    manifiesto = json.loads(contenido_manifiesto)
    ruta_ref = args.manifiesto.parent / manifiesto["archivo_referencias_verificadas"]
    contenido_ref = ruta_ref.read_bytes()
    hash_ref = sha256(contenido_ref).hexdigest()
    if hash_ref != manifiesto["sha256_referencias_verificadas"]:
        parser.error("Las referencias no coinciden con el manifiesto.")
    referencias = json.loads(contenido_ref)["referencias"]
    contenido_vtt = args.vtt.read_bytes()
    segmentos = leer_vtt(args.vtt, contenido_vtt)

    recortes = []
    for clip in manifiesto["recortes"]:
        clip_id = clip["clip_id"]
        inicio, fin = clip["inicio_ms"], clip["fin_ms"]
        ref = referencias[clip_id]
        resultados = {}
        for metodo, bordes in (("vtt_punto_medio", False), ("vtt_con_bordes", True)):
            texto, bloques = extraer_segmentos(segmentos, inicio, fin, bordes)
            resultados[metodo] = {"segmentos_vtt": bloques, **medir(ref, texto)}
        for metodo, carpeta, campo in (
            ("wav2vec", args.wav2vec, "texto"), ("whisper", args.whisper, "text"),
        ):
            if carpeta:
                fuente = carpeta / f"{clip_id}.json"
                prediccion = json.loads(fuente.read_bytes())
                if prediccion["sha256_audio"] != clip["sha256"]:
                    parser.error(f"La transcripción de {clip_id} con {metodo} usa otro audio.")
                resultados[metodo] = {"archivo": str(fuente), **medir(ref, prediccion[campo])}
        recortes.append({
            "clip_id": clip_id, "inicio_ms": inicio, "fin_ms": fin, "resultados": resultados,
        })

    agregados = {
        metodo: {metrica: acumular(recortes, metodo, metrica) for metrica in ("wer", "cer")}
        for metodo in recortes[0]["resultados"]
    }
    informe = {
        "sha256_manifiesto": sha256(contenido_manifiesto).hexdigest(),
        "referencias": str(ruta_ref),
        "sha256_referencias": hash_ref,
        "vtt": str(args.vtt),
        "sha256_vtt": sha256(contenido_vtt).hexdigest(),
        "recortes": recortes,
        "agregados": agregados,
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("x", encoding="utf-8") as archivo:
        json.dump(informe, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")
    for metodo, tasas in agregados.items():
        print(f"{metodo}: WER {tasas['wer']['tasa']:.2%}, CER {tasas['cer']['tasa']:.2%}")


if __name__ == "__main__":
    main()
