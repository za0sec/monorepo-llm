"""Interfaz local para el evaluador del ejercicio 4."""

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote, unquote


CARPETA = Path(__file__).resolve().parent
EVALUADOR = CARPETA.parent / "ejercicio2/evaluar_video.py"
LIMITE_AUDIO = 25_000_000
TIPOS_FUENTE = {
    "apuntes": (".pdf", "Apuntes PDF"),
    "transcripciones": (".vtt", "Transcripciones VTT"),
    "slides": (".pdf", "Diapositivas"),
}


def fuentes_corpus():
    manifiesto = json.loads((CARPETA / "slides/manifiesto.json").read_text(encoding="utf-8"))
    diapositivas = {item["archivo"]: item for item in manifiesto["archivos"]}
    fuentes = {}
    for carpeta, (extension, grupo) in TIPOS_FUENTE.items():
        directorio = CARPETA / carpeta
        for archivo in sorted(directorio.iterdir()):
            if (archivo.suffix.lower() != extension or not archivo.is_file()
                    or archivo.resolve().parent != directorio):
                continue
            if carpeta == "slides" and archivo.name not in diapositivas:
                continue
            diapositiva = diapositivas.get(archivo.name, {})
            fuentes[f"{carpeta}/{archivo.name}"] = {
                "nombre": Path(diapositiva.get("fuente", archivo.name)).name,
                "grupo": grupo,
                "paginas": diapositiva.get("paginas_originales"),
            }
    return fuentes


def enlaces_citas(clase, fuentes):
    enlaces = {}
    for id_, chunk in clase.items():
        nombre = Path(chunk.get("archivo") or chunk["fuente"]).name
        carpeta = {"diapositiva": "slides"}.get(chunk.get("tipo_fuente"))
        if carpeta is None:
            carpeta = "transcripciones" if Path(nombre).suffix.lower() == ".vtt" else "apuntes"
        clave = f"{carpeta}/{nombre}"
        if clave not in fuentes:
            continue
        url = "/fuentes/" + quote(clave)
        pagina = chunk.get("pagina")
        if pagina is not None:
            paginas = fuentes[clave]["paginas"]
            if paginas is None:
                url += f"#page={pagina}"
            elif pagina in paginas:
                url += f"#page={paginas.index(pagina) + 1}"
        enlaces[id_] = url
    return enlaces


def pagina_principal():
    grupos = {}
    for clave, fuente in fuentes_corpus().items():
        url = "/fuentes/" + quote(clave)
        nombre = escape(fuente["nombre"])
        grupos.setdefault(fuente["grupo"], []).append(
            f'<li><a href="{url}" target="_blank" rel="noopener">{nombre}</a></li>'
        )
    listado = "".join(
        f"<div><h3>{escape(grupo)}</h3><ul>{''.join(items)}</ul></div>"
        for grupo, items in grupos.items()
    )
    plantilla = (CARPETA / "ui_local.html").read_text(encoding="utf-8")
    return plantilla.replace("<!-- LISTADO_FUENTES -->", listado).encode()


class Interfaz(BaseHTTPRequestHandler):
    def responder(self, estado, tipo, cuerpo):
        self.send_response(estado)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(cuerpo)

    def json(self, estado, **resultado):
        self.responder(estado, "application/json; charset=utf-8", json.dumps(resultado).encode())

    def do_GET(self):
        if self.path.startswith("/fuentes/"):
            clave = unquote(self.path[len("/fuentes/"):])
            if clave not in fuentes_corpus():
                return self.json(404, error="Fuente desconocida.")
            ruta = CARPETA / clave
            tipo = "application/pdf" if ruta.suffix.lower() == ".pdf" else "text/plain; charset=utf-8"
            return self.responder(200, tipo, ruta.read_bytes())
        if self.path != "/":
            return self.json(404, error="Página desconocida.")
        self.responder(200, "text/html; charset=utf-8", pagina_principal())

    def do_POST(self):
        origen = f"http://127.0.0.1:{self.server.server_port}"
        if self.path != "/evaluar" or self.headers.get("Origin") != origen:
            return self.json(403, error="Abrí la interfaz desde la dirección local indicada.")
        objetivo = unquote(self.headers.get("X-Objetivo", "")).strip()
        try:
            longitud = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            longitud = 0
        if not objetivo or len(objetivo) > 1000 or "\x00" in objetivo:
            return self.json(400, error="Ingresá un objetivo de hasta 1.000 caracteres.")
        if not 0 < longitud <= LIMITE_AUDIO:
            return self.json(400, error="Seleccioná un MP3 de hasta 25 MB.")

        try:
            with TemporaryDirectory(prefix="tp3-ui-") as temporal:
                carpeta = Path(temporal)
                audio = carpeta / "audio.mp3"
                datos = self.rfile.read(longitud)
                if len(datos) != longitud:
                    return self.json(400, error="El audio llegó incompleto.")
                audio.write_bytes(datos)
                salida = carpeta / "resultado"
                corrida = subprocess.run(
                    [sys.executable, str(EVALUADOR), "--corpus", "completo", "--salida", str(salida),
                     "--", objetivo, str(audio)],
                    capture_output=True, encoding="utf-8", errors="replace", timeout=360,
                )
                informe = salida / "gpt41/informe.md"
                if corrida.returncode == 0:
                    entrada = json.loads((salida / "entrada.json").read_text(encoding="utf-8"))
                    return self.json(
                        200, informe=informe.read_text(encoding="utf-8"),
                        citas=enlaces_citas(entrada["clase"], fuentes_corpus()),
                    )
                print(corrida.stderr, file=sys.stderr, flush=True)
                return self.json(502, error="El evaluador no pudo terminar. Revisá el detalle en la terminal.")
        except subprocess.TimeoutExpired:
            return self.json(504, error="La evaluación excedió el tiempo de espera.")
        except OSError as error:
            print(error, file=sys.stderr, flush=True)
            return self.json(500, error="No se pudieron escribir o leer los archivos de la evaluación.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puerto", type=int, default=8765)
    args = parser.parse_args()
    with HTTPServer(("127.0.0.1", args.puerto), Interfaz) as servidor:
        print(f"Abrí http://127.0.0.1:{servidor.server_port}", flush=True)
        try:
            servidor.serve_forever()
        except KeyboardInterrupt:
            pass
