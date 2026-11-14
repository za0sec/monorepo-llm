"""Verifica las pruebas de expansión e informe sin consumir el proxy."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generar_informe
import proxy_catedra
import recuperar


class InformeProxyTest(unittest.TestCase):
    def test_llamada_registra_mensajes_modelo_y_costo(self):
        mensajes = [{"role": "user", "content": "Objetivo de prueba"}]
        entorno = {"OPENAI_API_KEY": "clave-de-prueba"}
        respuesta = SimpleNamespace(
            model="gpt-4.1-2025-04-14", usage=None,
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="{}"))],
        )
        http = SimpleNamespace(parse=lambda: respuesta,
                               headers={"x-litellm-response-cost": "0.001"})
        with patch.object(proxy_catedra, "dotenv_values", return_value=entorno), \
             patch.object(proxy_catedra, "OpenAI") as cliente:
            crear = cliente.return_value.chat.completions.with_raw_response.create
            crear.return_value = http
            auditoria = proxy_catedra.completar(mensajes, max_tokens=200, formato_json=True)
        crear.assert_called_once_with(
            model="openai/gpt-4.1-2025-04-14", messages=mensajes, temperature=0, store=False,
            max_completion_tokens=200, response_format={"type": "json_object"},
        )
        self.assertEqual(auditoria["mensajes"], mensajes)
        self.assertEqual(auditoria["costo_reportado_proxy"], "0.001")

    def test_planificacion_usa_prompt_inicial_y_gpt41(self):
        with patch.object(recuperar, "completar", return_value={}) as completar:
            plan = recuperar.planificar_subtemas("Objetivo de prueba")
        self.assertEqual(plan["prompt_expansion"], "inicial")
        self.assertEqual(completar.call_args.args[0][0]["content"], recuperar.INSTRUCCIONES_EXPANSION)
        self.assertEqual(completar.call_args.kwargs["modelo"], "openai/gpt-4.1-2025-04-14")

    def test_expansion_deduplica_por_puesto_y_conserva_subtema(self):
        objetivo = "Objetivo de prueba"
        subtemas = [{"tema": "Concepto", "consulta": "Pregunta A"},
                    {"tema": "Uso", "consulta": "Pregunta B"}]
        plan = {"finish_reason": "stop", "mensajes": [{"content": objetivo}],
                "respuesta": json.dumps({"subtemas": subtemas})}
        busquedas = [{"coleccion": "prueba", "manifiesto": {}, "k_efectivo": 2,
                      "referencias": [{"chunk_id": cid, "posicion": pos}
                                      for pos, cid in enumerate(ids, 1)]}
                     for ids in (("a", "b"), ("a", "c"))]
        with patch.object(recuperar, "recuperar", side_effect=busquedas):
            resultado = recuperar.recuperar_expandida(objetivo, {}, plan, k=2)
        self.assertEqual(resultado["candidatos_unicos"], 3)
        self.assertEqual([r["chunk_id"] for r in resultado["referencias"]], ["a", "b"])
        self.assertEqual(resultado["referencias"][1]["posicion_subconsulta"], 2)

    def test_cli_guarda_auditoria_y_muestra_tiempos_en_detalle(self):
        objetivo = "Objetivo de prueba"
        respuesta = {
            "cobertura": [{"tema_id": "T1", "estado": "parcial", "explicacion": "Falta el ejemplo.",
                           "citas_clase": ["C1"], "citas_video": ["V1"]}],
            "material_recomendado": [{"tema_id": "T1", "que_estudiar": "Ejemplo", "cita_clase": "C1"}],
            "conclusion": "El video sirve como introducción.",
        }
        with tempfile.TemporaryDirectory() as directorio:
            carpeta = Path(directorio)
            video, entrada, salida = (carpeta / n for n in ("video.vtt", "recuperacion.json", "informe.json"))
            video.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\nUna definición.\n", encoding="utf-8")
            entrada.write_text(json.dumps({
                "objetivo": objetivo, "subtemas": [{"tema": "Concepto", "consulta": "¿Qué es?"}],
                "referencias": [{"texto": "Un ejemplo.", "metadata": {
                    "fuente": "Apunte.pdf", "pagina_inicio": 3, "pagina_fin": 3,
                }}],
            }), encoding="utf-8")
            modelo = "openai/gpt-4.1-2025-04-14"
            argv = ["generar_informe.py", objetivo, str(video), "--backend", "proxy",
                    "--prompt", "estudiante", "--recuperacion", str(entrada), "--salida", str(salida)]
            with patch.object(sys, "argv", argv), patch("builtins.print"), \
                 patch.object(generar_informe, "completar", return_value={
                     "finish_reason": "stop", "respuesta": json.dumps(respuesta),
                 }) as completar:
                generar_informe.main()
            self.assertIn("Un ejemplo.", completar.call_args.args[0][1]["content"])
            self.assertEqual(completar.call_args.kwargs["modelo"], modelo)
            self.assertEqual(json.loads(salida.read_text())["informe_estructurado"], respuesta)
            visible, evidencia = salida.with_suffix(".md").read_text().split("<details>", 1)
            self.assertIn("Apunte · pág. 3", visible)
            self.assertIn("00:00:00.000–00:00:02.000", evidencia)
