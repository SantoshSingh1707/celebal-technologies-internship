from __future__ import annotations

import os
import re
from typing import List

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def load_document(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Document not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return _load_pdf(path)
    if ext in {".txt", ".md", ".csv", ".log"}:
        return _load_text(path)

    raise ValueError(
        f"Unsupported file type '{ext}'. Please use a .pdf or .txt document."
    )


def _load_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")
    if not pages:
        raise ValueError(
            "No text could be extracted from the PDF "
            "(it may be a scanned/image-only document)."
        )
    return "\n\n".join(pages)


def _load_text(path: str) -> str:
    encodings = ["utf-8", "utf-16", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} with any common encoding.")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    atoms = _split_atoms(text)

    chunks: List[str] = []
    current = ""

    for atom in atoms:
        while len(atom) > chunk_size:
            if current:
                chunks.append(current)
                current = _tail(current, overlap)
            chunks.append(atom[:chunk_size])
            atom = atom[chunk_size:]

        if len(current) + len(atom) + 1 <= chunk_size:
            current = f"{current} {atom}".strip()
        else:
            if current:
                chunks.append(current)
            current = _tail(current, overlap) + atom

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def _split_atoms(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def _tail(chunk: str, n: int) -> str:
    return chunk[-n:] if len(chunk) >= n else chunk


def ingest(path: str, chunk_size: int = CHUNK_SIZE,
           chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    raw = load_document(path)
    chunks = chunk_text(raw, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Document produced no chunks (it may be empty).")
    return chunks
