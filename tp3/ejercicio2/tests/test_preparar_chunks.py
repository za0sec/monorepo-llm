"""Integridad de chunks con el tokenizer real de MiniLM (previamente en caché)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preparar_chunks import cargar_tokenizer, crear_chunks
from preparar_transcripcion import REPO, leer_vtt


def segmentos_de(textos):
    return [
        {"id": str(i), "texto": texto,
         "inicio": f"00:00:{i:02}.000", "fin": f"00:00:{i+1:02}.000",
         "inicio_ms": i * 1000, "fin_ms": (i + 1) * 1000}
        for i, texto in enumerate(textos, 1)
    ]


class PrepararChunksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.limite = cargar_tokenizer(solo_local=True)

    def preparar(self, segmentos, **opciones):
        return crear_chunks(
            segmentos, "clase.VTT", "hash-fuente", self.tokenizer, self.limite,
            **opciones,
        )

    def comprobar_integridad(self, segmentos, documento):
        original = "\n\n".join(s["texto"] for s in segmentos)
        posiciones, cursor = [], 0
        for s in segmentos:
            posiciones.append((cursor, cursor + len(s["texto"]), s))
            cursor += len(s["texto"]) + 2
        cubiertos = set()
        ids = set()
        for c in documento["chunks"]:
            a, b = c["inicio_caracter"], c["fin_caracter"]
            self.assertEqual(c["texto"], original[a:b])
            self.assertTrue(c["texto"].strip())
            self.assertNotIn(c["chunk_id"], ids)
            ids.add(c["chunk_id"])
            cubiertos.update(range(a, b))
            esperados = [s for inicio, fin, s in posiciones if inicio < b and fin > a]
            self.assertEqual(c["segmentos_ids"], [s["id"] for s in esperados])
            self.assertEqual(c["inicio_ms"], min(s["inicio_ms"] for s in esperados))
            self.assertEqual(c["fin_ms"], max(s["fin_ms"] for s in esperados))
            conteo = len(self.tokenizer.encode(c["texto"], add_special_tokens=True))
            self.assertEqual(c["tokens_totales"], conteo)
            self.assertLessEqual(conteo, self.limite)
            self.assertLessEqual(c["tokens_contenido"], documento["configuracion"]["tokens"])
        self.assertTrue(all(i in cubiertos or c.isspace() for i, c in enumerate(original)))

    def test_conserva_mayusculas_acentos_y_texto(self):
        segmentos = segmentos_de(["Marina: ¡ÁRBOL, atención y similitud coseno!", "Ejemplo: Ñandú."])
        d = self.preparar(segmentos)
        self.comprobar_integridad(segmentos, d)
        self.assertEqual(len(d["chunks"]), 1)

    def test_repeticiones_no_reutilizan_tiempos_anteriores(self):
        segmentos = segmentos_de(["See the table below.", "See the table below.", "Summary"])
        d = self.preparar(segmentos, caracteres=30, overlap_caracteres=20)
        self.comprobar_integridad(segmentos, d)
        repetidos = [c for c in d["chunks"] if c["texto"] == "See the table below."]
        self.assertEqual([c["segmentos_ids"] for c in repetidos], [["1"], ["2"]])

    def test_segunda_pasada_y_overlap_sin_perdida(self):
        segmentos = segmentos_de(["electroencefalográficamente ÁRBOL " * 35])
        d = self.preparar(segmentos, caracteres=5000, overlap_caracteres=0,
                          tokens=20, overlap_tokens=4)
        self.comprobar_integridad(segmentos, d)
        self.assertGreater(len(d["chunks"]), 1)
        self.assertLess(d["chunks"][1]["inicio_caracter"], d["chunks"][0]["fin_caracter"])

    def test_intervalos_temporales_superpuestos(self):
        segmentos = segmentos_de(["Primera intervención larga.", "Segunda intervención."])
        segmentos[0].update(fin_ms=9000, fin="00:00:09.000")
        d = self.preparar(segmentos)
        self.comprobar_integridad(segmentos, d)
        self.assertEqual(d["chunks"][0]["fin_ms"], 9000)

    def test_identidad_estable_y_separacion_de_configuraciones(self):
        segmentos = segmentos_de(["Texto con embeddings y similitud coseno."])
        d = self.preparar(segmentos)
        self.assertEqual(d, self.preparar(segmentos))
        otro = self.preparar(segmentos, tokens=128)
        self.assertNotEqual(d["chunks"][0]["chunk_id"], otro["chunks"][0]["chunk_id"])

    def test_rechaza_parametros_invalidos_y_exceso_del_encoder(self):
        segmentos = segmentos_de(["Texto."])
        for opciones in [dict(caracteres=0), dict(overlap_caracteres=1000),
                         dict(tokens=20, overlap_tokens=20), dict(overlap_tokens=-1),
                         dict(tokens=256)]:
            with self.subTest(opciones=opciones), self.assertRaises(ValueError):
                self.preparar(segmentos, **opciones)
        with self.assertRaises(ValueError):
            self.preparar([])

    def test_clase_real_completa(self):
        segmentos = leer_vtt(REPO / "docs/embeddings_2.VTT")
        d = self.preparar(segmentos)
        self.comprobar_integridad(segmentos, d)
        self.assertEqual(
            {s["id"] for s in segmentos},
            {sid for c in d["chunks"] for sid in c["segmentos_ids"]},
        )


if __name__ == "__main__":
    unittest.main()
