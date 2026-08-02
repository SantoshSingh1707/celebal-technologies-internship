from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DIR = os.path.join(DATA_DIR, "vector_store")

INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_DIR, "chunks.json")
CONFIG_PATH = os.path.join(VECTOR_DIR, "config.json")

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384

CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))

TOP_K = int(os.environ.get("RAG_TOP_K", "4"))

LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "phi3:mini")
LLM_TEMPERATURE = float(os.environ.get("RAG_TEMPERATURE", "0.1"))
LLM_CONTEXT_WINDOW = 2048
