"""Calcula WER y CER por recorte a partir de transcripciones guardadas."""

import argparse
import json
import unicodedata
from hashlib import sha256
from pathlib import Path


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto.lower())
    sin_puntuacion = "".join(
        " " if unicodedata.category(letra).startswith("P") else letra for letra in texto
    )
    return " ".join(sin_puntuacion.split())


def contar_ediciones(referencia: list[str], prediccion: list[str]) -> dict:
    """Alinea por distancia de Levenshtein y cuenta cada tipo de edición."""
    if not referencia:
        raise ValueError("La referencia normalizada está vacía.")
    costos = [list(range(len(prediccion) + 1))]
    for i, origen in enumerate(referencia, 1):
        fila = [i]
        for j, destino in enumerate(prediccion, 1):
            fila.append(min(
                costos[i - 1][j] + 1,
                fila[j - 1] + 1,
                costos[i - 1][j - 1] + (origen != destino),
            ))
        costos.append(fila)

    i, j = len(referencia), len(prediccion)
    sustituciones = omisiones = inserciones = 0
    while i or j:
        if i and j and costos[i][j] == costos[i - 1][j - 1] + (
            referencia[i - 1] != prediccion[j - 1]
        ):
            sustituciones += referencia[i - 1] != prediccion[j - 1]
            i -= 1
            j -= 1
        elif i and costos[i][j] == costos[i - 1][j] + 1:
            omisiones += 1
            i -= 1
        else:
            inserciones += 1
            j -= 1
    errores = costos[-1][-1]
    return {
        "sustituciones": sustituciones,
        "omisiones": omisiones,
        "inserciones": inserciones,
        "errores": errores,
        "unidades_referencia": len(referencia),
        "tasa": errores / len(referencia),
    }


def medir(referencia: str, prediccion: str) -> dict:
    ref = normalizar(referencia)
    pred = normalizar(prediccion)
    return {
        "referencia_normalizada": ref,
        "prediccion_normalizada": pred,
        "wer": contar_ediciones(ref.split(), pred.split()),
        "cer": contar_ediciones(list(ref.replace(" ", "")), list(pred.replace(" ", ""))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_id")
    parser.add_argument("--referencias", type=Path, required=True)
    parser.add_argument("--wav2vec", type=Path)
    parser.add_argument("--whisper", type=Path)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    if not (args.wav2vec or args.whisper):
        parser.error("Indicar al menos un archivo de transcripción.")

    try:
        contenido = args.referencias.read_bytes()
        referencia = json.loads(contenido)["referencias"][args.clip_id]
        informe = {
            "clip_id": args.clip_id,
            "referencias": str(args.referencias),
            "sha256_referencias": sha256(contenido).hexdigest(),
            "resultados": {},
        }
        for modelo, archivo, campo in (
            ("wav2vec", args.wav2vec, "texto"), ("whisper", args.whisper, "text")
        ):
            if archivo:
                prediccion = json.loads(archivo.read_text(encoding="utf-8"))[campo]
                informe["resultados"][modelo] = {
                    "archivo": str(archivo), **medir(referencia, prediccion),
                }
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        with args.salida.open("x", encoding="utf-8") as archivo:
            json.dump(informe, archivo, ensure_ascii=False, indent=2)
            archivo.write("\n")
    except (KeyError, OSError, ValueError, TypeError) as error:
        parser.exit(1, f"No se pudieron calcular las métricas: {error}\n")
    for modelo, resultado in informe["resultados"].items():
        print(f"{modelo}: WER {resultado['wer']['tasa']:.2%}, CER {resultado['cer']['tasa']:.2%}")


if __name__ == "__main__":
    main()
