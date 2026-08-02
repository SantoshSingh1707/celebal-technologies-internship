from __future__ import annotations

import json
import os
from typing import List

from .config import (CHUNKS_PATH, CONFIG_PATH, EMBED_DIM, EMBED_MODEL,
                     INDEX_PATH, VECTOR_DIR)

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        print(f"[embed] Loading embedding model: {EMBED_MODEL} "
              f"(first load downloads ~90 MB, then fully offline)...")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed_texts(texts: List[str]) -> "list[list[float]]":
    return get_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def build_index(chunks: List[str], source: str):
    import numpy as np
    import faiss

    os.makedirs(VECTOR_DIR, exist_ok=True)

    vectors = embed_texts(chunks)
    matrix = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(matrix)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": os.path.basename(source),
                "embed_model": EMBED_MODEL,
                "chunk_count": len(chunks),
                "index_type": "IndexFlatIP (cosine, normalized)",
            },
            f,
            indent=2,
        )

    return index


def load_index():
    import faiss

    if not (os.path.isfile(INDEX_PATH) and os.path.isfile(CHUNKS_PATH)):
        return None, None

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks: List[str] = json.load(f)
    return index, chunks
