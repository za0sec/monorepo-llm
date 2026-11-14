"""Convierte un VTT de clase a segmentos JSON y una vista de texto con tiempos."""

import argparse
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TIMESTAMP = r"(?:\d{2,}:)?[0-5]\d:[0-5]\d\.\d{3}"
INTERVALO = re.compile(
    rf"^(?P<inicio>{TIMESTAMP}) --> (?P<fin>{TIMESTAMP})(?:[ \t]+.*)?$"
)


def milisegundos(timestamp: str) -> int:
    """Acepta los formatos WebVTT HH:MM:SS.mmm y MM:SS.mmm."""
    partes = timestamp.split(":")
    horas, minutos, segundos = partes if len(partes) == 3 else ["0", *partes]
    segundos, milis = segundos.split(".")
    return ((int(horas) * 60 + int(minutos)) * 60 + int(segundos)) * 1000 + int(milis)


def leer_vtt(ruta: Path) -> list[dict]:
    """Lee VTT de texto de la cátedra; falla ante bloques que no reconoce."""
    contenido = ruta.read_text(encoding="utf-8-sig").strip()
    lineas_archivo = contenido.splitlines()
    if not lineas_archivo or re.fullmatch(r"WEBVTT(?:[ \t].*)?", lineas_archivo[0]) is None:
        raise ValueError(f"{ruta}: falta la cabecera WEBVTT.")
    bloques = re.split(r"\n[ \t]*\n", contenido)

    segmentos = []
    ids = set()
    for numero, bloque in enumerate(bloques[1:], start=2):
        lineas = bloque.splitlines()
        if lineas[0] == "NOTE" or lineas[0].startswith(("NOTE ", "NOTE\t")):
            continue
        indice = 0 if "-->" in lineas[0] else 1
        intervalo = INTERVALO.fullmatch(lineas[indice]) if len(lineas) > indice else None
        if intervalo is None:
            raise ValueError(f"{ruta}: intervalo no reconocido en el bloque {numero}.")
        inicio, fin = intervalo.group("inicio", "fin")
        inicio_ms, fin_ms = milisegundos(inicio), milisegundos(fin)
        texto = re.sub(r"\s+", " ", " ".join(lineas[indice + 1:])).strip()
        if not texto or fin_ms <= inicio_ms:
            raise ValueError(f"{ruta}: texto vacío o intervalo inválido en el bloque {numero}.")
        if segmentos and inicio_ms < segmentos[-1]["inicio_ms"]:
            raise ValueError(f"{ruta}: tiempos fuera de orden en el bloque {numero}.")
        cue_id = lineas[0] if indice else str(len(segmentos) + 1)
        if cue_id in ids:
            raise ValueError(f"{ruta}: identificador repetido {cue_id!r}.")
        ids.add(cue_id)
        segmentos.append({
            "id": cue_id,
            "inicio": inicio,
            "fin": fin,
            "inicio_ms": inicio_ms,
            "fin_ms": fin_ms,
            "texto": texto,
        })
    if not segmentos:
        raise ValueError(f"{ruta}: no hay segmentos de texto.")
    return segmentos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "entrada", nargs="?", type=Path, default=REPO / "docs/embeddings_2.VTT"
    )
    parser.add_argument(
        "--salida", type=Path, default=Path(__file__).parent / "data/procesados"
    )
    args = parser.parse_args()
    ruta = args.entrada.resolve()
    segmentos = leer_vtt(ruta)
    # La fuente relativa permite compartir los resultados dentro del repositorio.
    fuente = str(ruta.relative_to(REPO)) if ruta.is_relative_to(REPO) else str(ruta)
    documento = {"fuente": fuente, "segmentos": segmentos}
    args.salida.mkdir(parents=True, exist_ok=True)
    json_path = args.salida / f"{ruta.stem}.json"
    txt_path = args.salida / f"{ruta.stem}.txt"
    json_path.write_text(
        json.dumps(documento, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    txt_path.write_text(
        f"Fuente: {fuente}\n\n" + "\n\n".join(
            f"[{s['inicio']} --> {s['fin']}] {s['texto']}" for s in segmentos
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Fuente: {fuente}")
    print(f"Segmentos: {len(segmentos)}")
    print(f"Intervalo: {segmentos[0]['inicio']} --> {segmentos[-1]['fin']}")
    print(f"JSON: {json_path}")
    print(f"Vista de texto: {txt_path}")


if __name__ == "__main__":
    main()
