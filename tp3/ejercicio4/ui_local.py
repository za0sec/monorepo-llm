"""Interfaz local para el evaluador del ejercicio 4."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import unquote


CARPETA = Path(__file__).resolve().parent
EVALUADOR = CARPETA.parent / "ejercicio2/evaluar_video.py"
LIMITE_AUDIO = 25_000_000


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
        if self.path != "/":
            return self.json(404, error="Página desconocida.")
        self.responder(200, "text/html; charset=utf-8", (CARPETA / "ui_local.html").read_bytes())

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
                    return self.json(200, informe=informe.read_text(encoding="utf-8"))
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
