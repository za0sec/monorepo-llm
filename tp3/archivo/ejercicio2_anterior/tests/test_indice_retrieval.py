"""Integración con Chroma real y vectores controlados, sin descargar pesos.

La pertinencia semántica de MiniLM se revisa aparte sobre la clase real.
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import indexar_material as indice
import recuperar as retrieval
from config import DIMENSION
from preparar_chunks import cargar_tokenizer, crear_chunks
from test_preparar_chunks import segmentos_de


class IndiceRetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.limite = cargar_tokenizer(solo_local=True)

    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.ruta = Path(self.temporal.name) / "chroma"
        self.segmentos = segmentos_de([
            "Marina: Similitud coseno y dirección de los vectores.",
            "Marina: El contexto modifica el embedding de palabras.",
            "Marina: El producto interno también mide similitud.",
        ])
        self.documento = self.preparar()
        self.assertEqual(len(self.documento["chunks"]), 3)
        self.textos = [c["texto"] for c in self.documento["chunks"]]
        self.modelo = SimpleNamespace(
            tokenizer=self.tokenizer, max_seq_length=self.limite,
            encode=Mock(side_effect=self.vectores),
        )
        self.patch_indice = patch.object(indice, "cargar_modelo", return_value=self.modelo).start()
        self.patch_retrieval = patch.object(retrieval, "cargar_modelo", return_value=self.modelo).start()
        self.addCleanup(patch.stopall)

    def preparar(self, tokens=40):
        return crear_chunks(self.segmentos, "clase.VTT", "hash-prueba", self.tokenizer,
                            self.limite, caracteres=60, overlap_caracteres=0,
                            tokens=tokens, overlap_tokens=0)

    def vectores(self, textos, **opciones):
        vectores = np.zeros((len(textos), DIMENSION), dtype=np.float32)
        for i, texto in enumerate(textos):
            vectores[i, self.textos.index(texto) if texto in self.textos else 0] = 1
        return vectores

    def test_persiste_en_otro_proceso_y_repetir_no_recalcula(self):
        primera = indice.indexar(self.documento, self.ruta)
        self.assertEqual(primera["chunks_nuevos"], 3)
        # Otro proceso abre el almacenamiento: no basta con reutilizar el cliente en memoria.
        codigo = (
            "import chromadb,sys; from chromadb.config import Settings; "
            "c=chromadb.PersistentClient(path=sys.argv[1],settings=Settings(anonymized_telemetry=False)); "
            "print(c.get_collection(sys.argv[2],embedding_function=None).count())"
        )
        proceso = subprocess.run(
            [sys.executable, "-c", codigo, str(self.ruta), primera["coleccion"]],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONPYCACHEPREFIX": "/tmp/tp3-pycache"},
        )
        self.assertEqual(proceso.stdout.strip(), "3")
        self.patch_indice.reset_mock()
        segunda = indice.indexar(self.documento, self.ruta, solo_local=True)
        self.assertEqual(segunda["chunks_nuevos"], 0)
        self.assertEqual(segunda["chunks_reutilizados"], 3)
        self.patch_indice.assert_not_called()

    def test_recupera_orden_distancia_y_trazabilidad(self):
        indice.indexar(self.documento, self.ruta)
        resultado = retrieval.recuperar("Estudiar coseno.", self.documento, self.ruta, k=10)
        self.assertEqual(resultado["k_efectivo"], 3)
        primero = resultado["referencias"][0]
        original = self.documento["chunks"][0]
        self.assertEqual(primero["chunk_id"], original["chunk_id"])
        self.assertEqual(primero["texto"], original["texto"])
        self.assertEqual(primero["metadata"], {k: v for k, v in original.items()
                                             if k not in ("chunk_id", "texto")})
        self.assertAlmostEqual(primero["distancia_coseno"], 0, places=5)
        self.assertAlmostEqual(primero["similitud_coseno"], 1, places=5)
        distancias = [r["distancia_coseno"] for r in resultado["referencias"]]
        self.assertEqual(distancias, sorted(distancias))

    def test_fallo_parcial_bloquea_consulta_y_se_reanuda(self):
        llamadas = 0

        def interrumpir(textos, **opciones):
            nonlocal llamadas
            llamadas += 1
            if llamadas == 2:
                raise RuntimeError("Interrupción simulada")
            return self.vectores(textos)

        self.modelo.encode.side_effect = interrumpir
        with self.assertRaisesRegex(RuntimeError, "Interrupción"):
            indice.indexar(self.documento, self.ruta, tamano_lote=1)
        with self.assertRaisesRegex(ValueError, "incompleta"):
            retrieval.recuperar("Estudiar coseno.", self.documento, self.ruta)
        self.modelo.encode.side_effect = self.vectores
        resumen = indice.indexar(self.documento, self.ruta)
        self.assertEqual(resumen["chunks_reutilizados"], 1)
        self.assertEqual(resumen["chunks_nuevos"], 2)
        self.assertEqual(len(retrieval.recuperar("Coseno", self.documento, self.ruta)["referencias"]), 3)

    def test_separa_configuraciones_y_rechaza_nombre_incompatible(self):
        resumen = indice.indexar(self.documento, self.ruta, nombre="prueba_tp3")
        otro = self.preparar(tokens=39)
        self.assertNotEqual(indice.nombre_coleccion(indice.crear_manifiesto(self.documento)),
                            indice.nombre_coleccion(indice.crear_manifiesto(otro)))
        with self.assertRaisesRegex(ValueError, "otro corpus"):
            indice.indexar(otro, self.ruta, nombre=resumen["coleccion"])
        with self.assertRaisesRegex(ValueError, "otro corpus"):
            retrieval.recuperar("Coseno", otro, self.ruta, nombre=resumen["coleccion"])

    def test_rechaza_coleccion_con_metrica_incorrecta(self):
        manifiesto = indice.crear_manifiesto(self.documento)
        c = indice.abrir_cliente(self.ruta).create_collection(
            "metrica_incorrecta", embedding_function=None,
            configuration={"hnsw": {"space": "l2"}},
            metadata={"manifiesto": json.dumps(manifiesto, sort_keys=True, ensure_ascii=False)},
        )
        with self.assertRaisesRegex(ValueError, "distancia coseno"):
            indice.validar_coleccion(c, manifiesto, exigir_completa=False)

    def test_rechaza_entradas_invalidas_y_truncamiento(self):
        for cambio in ("vacio", "duplicado", "configuracion", "texto"):
            d = copy.deepcopy(self.documento)
            if cambio == "vacio":
                d["chunks"] = []
            elif cambio == "duplicado":
                d["chunks"].append(d["chunks"][0])
            elif cambio == "configuracion":
                d["configuracion"]["tokens"] = 1
            else:
                d["chunks"][0]["texto"] = ""
            with self.subTest(cambio=cambio), self.assertRaises(ValueError):
                indice.indexar(d, self.ruta)
        for objetivo, k in [(" ", 4), ("Coseno", 0)]:
            with self.assertRaises(ValueError):
                retrieval.recuperar(objetivo, self.documento, self.ruta, k=k)
        with self.assertRaisesRegex(ValueError, "excede el encoder"):
            indice.codificar(self.modelo, ["coseno " * 300])
        self.modelo.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
