# Week 7 — Fully Offline Document Question Answering System (RAG)

A **Retrieval-Augmented Generation (RAG)** system that answers questions about your own
private documents — resumes, notes, research papers, books. Everything runs **fully
offline**: embeddings via `sentence-transformers`, retrieval via a local FAISS index,
and answer generation via **Ollama** (a local LLM). No external API calls, no data
leaves your machine.

Built from the assignment spec in [`Week7_Project.txt`](Week7_Project.txt).

---

## How it works

```
                  ┌────────────────────────────────────────────────────┐
                  │                   STAGES (offline)                 │
 your PDF/.txt ──▶│ 1. Ingest   → raw text                              │
   (data/)        │ 2. Chunk    → overlapping text chunks              │
                  │ 3. Embed    → chunks → vectors (MiniLM, 384-d)     │
                  │ 4. VectorDB → FAISS index on disk                  │
                  │                                                   │
      question ──▶│ 5. Query    → question → vector                    │
                  │ 6. Retrieve → top-k similar chunks (cosine)        │
                  │ 7. Generate → Ollama LLM answers from context      │
                  └────────────────────────────────────────────────────┘
```

| Stage (per spec)      | Where                                                             |
|-----------------------|-------------------------------------------------------------------|
| 1. Document ingestion | `rag/ingest.py` — `load_document()` (pypdf / text)               |
| 2. Text chunking      | `rag/ingest.py` — `chunk_text()` (paragraph/sentence-aware + overlap) |
| 3. Embedding creation | `rag/embed.py` — `sentence-transformers` `all-MiniLM-L6-v2`      |
| 4. Vector database    | `rag/embed.py` — FAISS `IndexFlatIP` (cosine) saved to `data/vector_store/` |
| 5. Query processing   | `rag/query.py` — `search()` embeds the question                   |
| 6. Context retrieval  | `rag/query.py` — top-`k` chunks + similarity scores               |
| 7. Answer generation  | `rag/query.py` — `generate_answer()` via Ollama                   |

---

## Requirements

- **Python 3.9+**
- **Ollama** installed and running — <https://ollama.com> (the app auto-starts as a
  background service; first query auto-pulls the model, ~2.4 GB for `phi3:mini`).
- Packages:

```bash
pip install -r requirements.txt
```

`ollama` python package included in requirements (not yet installed on this machine).

---

## Quick start

1. **Put your document** in the `data/` folder (any `.pdf` or `.txt` — resume, notes,
   research paper, book). A `sample_document.txt` is already there to try it out.

2. **Build the vector store:**

```bash
python main.py ingest                       # uses the doc found in data/
python main.py ingest data/my_resume.pdf    # or point at a specific file
```

3. **Ask questions:**

```bash
python main.py ask "How many leave days do employees get?"
python main.py ask "What is the main idea of the document?"
```

4. **Interactive chat session:**

```bash
python main.py chat
```

> ⚠️ First run downloads the embedding model (~90 MB) and the Ollama LLM (~2.4 GB).
> After that, everything works fully offline.

---

## Project layout

```
Week 7/
├── Week7_Project.txt       # the assignment spec
├── README.md               # this file
├── requirements.txt
├── main.py                 # CLI: ingest / ask / chat
├── rag/
│   ├── __init__.py
│   ├── config.py           # paths + tunable parameters (env-overridable)
│   ├── ingest.py           # stage 1-2: load documents, chunk text
│   ├── embed.py            # stage 3-4: embeddings + FAISS persistence
│   └── query.py            # stage 5-7: retrieval + Ollama generation
└── data/
    ├── sample_document.txt # demo knowledge base (swap for your own)
    └── vector_store/       # generated FAISS index + chunk metadata
```

## Configuration

Tunable via environment variables (or edit `rag/config.py`):

| Env var             | Default               | Meaning                        |
|---------------------|-----------------------|--------------------------------|
| `RAG_EMBED_MODEL`   | `all-MiniLM-L6-v2`    | embedding model                |
| `RAG_CHUNK_SIZE`    | `1000`                | chars per chunk                |
| `RAG_CHUNK_OVERLAP` | `200`                 | overlap between chunks         |
| `RAG_TOP_K`         | `4`                   | chunks retrieved per question  |
| `RAG_LLM_MODEL`     | `phi3:mini`           | Ollama generation model        |
| `RAG_TEMPERATURE`   | `0.1`                 | lower = more factual           |

Other small models you can try: `llama3.2:1b` (~1.3 GB, fastest),
`qwen2.5:1.5b` (~1 GB), `llama3.2:3b` (better quality).

---

## Experiments to try (from the spec's "Improvements & Experiments")

- **Better chunking** — change `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` and re-ingest.
- **Different embedding model** — set `RAG_EMBED_MODEL` (e.g. `BAAI/bge-small-en-v1.5`).
- **Hybrid search** — add keyword (BM25) scores to the cosine retrieval.
- **Re-ranking** — re-rank the top-k candidates with a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) before generation.
- **Different LLM** — swap `RAG_LLM_MODEL` to compare answer quality.

## Key learnings (per spec)

- RAG combines **retrieval** (grounding) with **generation** (fluency).
- Retrieval quality drives answer accuracy — chunking, embeddings, and top-k matter.
- Embeddings + vector similarity enable search over *unstructured, private* text.
- The whole stack can run fully on-device, which matters for privacy-sensitive data.
