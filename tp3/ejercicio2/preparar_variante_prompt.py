"""Congela una variante que solo cambia el mensaje de sistema."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("prompt", type=Path)
    parser.add_argument("salida", type=Path)
    args = parser.parse_args()
    control = args.control.read_bytes()
    variante = json.loads(control)
    sistema, usuario = variante["mensajes"]
    prompt = args.prompt.read_text(encoding="utf-8").rstrip("\n")
    if sistema["role"] != "system" or usuario["role"] != "user" or not prompt.strip():
        parser.error("Se necesitan un mensaje de sistema, uno de usuario y un prompt nuevo.")
    if prompt == sistema["content"]:
        parser.error("El prompt debe cambiar respecto del control.")

    variante["mensajes"][0] = {**sistema, "content": prompt}
    variante["sha256_control_del_prompt"] = hashlib.sha256(control).hexdigest()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("x", encoding="utf-8") as archivo:
        json.dump(variante, archivo, ensure_ascii=False, indent=2)
        archivo.write("\n")
    print(f"Variante de prompt congelada en {args.salida}.")


if __name__ == "__main__":
    main()
