"""Configuración del baseline local; los valores iniciales no son requisitos docentes."""

from pathlib import Path


DATA = Path(__file__).resolve().parent / "data"
ARCHIVO_CHUNKS = DATA / "chunks/embeddings_2.chunks.json"
RUTA_CHROMA = DATA / "chroma"
CACHE_MODELOS = DATA / "modelos"
MODELO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DIMENSION = 384
K_INICIAL = 8
