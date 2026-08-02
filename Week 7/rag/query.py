from __future__ import annotations

import sys
from typing import List, Tuple

from . import embed as embed_mod
from .config import LLM_CONTEXT_WINDOW, LLM_MODEL, LLM_TEMPERATURE, TOP_K

_PROMPT_TEMPLATE = """You are a precise assistant answering questions using ONLY the context below, which was retrieved from a user's private document.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- If the context does not contain the answer, say "I couldn't find that in the document." and stop.
- Quote the relevant part where it helps, and keep the answer concise.

--- Context ---
{context}

--- Question ---
{question}

--- Answer ---"""


def search(index, chunks: List[str], question: str,
           top_k: int = TOP_K) -> Tuple[List[str], List[float]]:
    import numpy as np

    vector = embed_mod.embed_texts([question])
    query_vec = np.array(vector, dtype="float32")

    scores, ids = index.search(query_vec, min(top_k, len(chunks)))
    results, result_scores = [], []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            break
        results.append(chunks[idx])
        result_scores.append(round(float(score), 4))
    return results, result_scores


def _ensure_model(model: str) -> None:
    import ollama

    client = ollama.Client()
    try:
        names = [m.get("name", "") for m in client.list().get("models", [])]
    except Exception as exc:
        print(
            "\n[error] Could not reach the Ollama daemon. "
            "Start the Ollama app, then retry.\n",
            file=sys.stderr,
        )
        raise SystemExit(f"Ollama unreachable: {exc}")

    if any(model == n or model in n for n in names):
        return

    print(f"[ollama] Model '{model}' not found — pulling it now "
          f"(one-time download, may take a while)...")
    for status in client.pull(model):
        print(f"  {status.get('status', '')} {status.get('completed', '')}")

    print(f"[ollama] Model '{model}' ready.\n")


def _build_context(chunks: List[str], max_chars: int = LLM_CONTEXT_WINDOW) -> str:
    parts, used = [], 0
    for i, chunk in enumerate(chunks, 1):
        remaining = max_chars - used
        if remaining <= 0:
            break
        snippet = chunk if len(chunk) <= remaining else chunk[:remaining] + "…"
        parts.append(f"[Excerpt {i}]\n{snippet}")
        used += len(snippet)
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: List[str],
                    model: str = LLM_MODEL) -> str:
    _ensure_model(model)

    import ollama

    context = _build_context(chunks)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": LLM_TEMPERATURE},
    )
    return response["message"]["content"].strip()


def ask(question: str, top_k: int = TOP_K,
        model: str = LLM_MODEL) -> Tuple[str, List[str], List[float]]:
    index, chunks = embed_mod.load_index()
    if index is None or chunks is None:
        raise RuntimeError(
            "No vector store found. Ingest a document first:\n"
            "    python main.py ingest <path/to/document.pdf>\n"
            "or move your PDF/text file into the data/ folder first."
        )

    retrieved, scores = search(index, chunks, question, top_k=top_k)
    answer = generate_answer(question, retrieved, model=model)
    return answer, retrieved, scores
