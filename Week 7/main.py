from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

from rag import config
from rag.ingest import ingest as do_ingest
from rag.query import ask


def _args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Fully offline RAG question-answering over your own documents.",
    )
    p.add_argument("--model", default=config.LLM_MODEL,
                   help="Ollama generation model (default: %(default)s)")
    p.add_argument("--top-k", type=int, default=config.TOP_K,
                   help="chunks retrieved per question (default: %(default)s)")
    sub = p.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="build the vector store from a document")
    ingest_p.add_argument("path", nargs="?", help="path to a .pdf or .txt file")
    sub.add_parser("chat", help="interactive Q&A session")

    ask_p = sub.add_parser("ask", help="answer a single question")
    ask_p.add_argument("question", nargs="?", help="the question to ask")
    return p


def cmd_ingest(path: Optional[str]) -> None:
    if not path:
        candidates = (
            sorted(glob.glob(os.path.join(config.DATA_DIR, "*.pdf")))
            + sorted(glob.glob(os.path.join(config.DATA_DIR, "*.txt")))
            + sorted(glob.glob(os.path.join(config.DATA_DIR, "*.md")))
        )
        if not candidates:
            raise SystemExit(
                "No document found in data/. Put a .pdf or .txt file in\n"
                f"    {config.DATA_DIR}\nthen re-run this command."
            )
        path = candidates[0]
        print(f"[ingest] No path given — using {path}")

    chunks = do_ingest(path)
    from rag.embed import build_index

    build_index(chunks, source=path)
    print(f"\n[ingest] Indexed {len(chunks)} chunks from {os.path.basename(path)} "
          f"into {config.VECTOR_DIR}")
    print("[ingest] Done. Try:  python main.py ask \"your question here\"")


def cmd_ask(question: str, model: str, top_k: int) -> None:
    if not question:
        question = input("Question: ").strip()
        if not question:
            raise SystemExit("No question given.")
    try:
        answer, retrieved, scores = ask(question, top_k=top_k, model=model)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("=" * 70)
    print(f"A: {answer}")
    print("=" * 70)
    if retrieved:
        print("\n[retrieved context — used to ground the answer]")
        for i, (chunk, score) in enumerate(zip(retrieved, scores), 1):
            print(f"\n  ({i}) similarity={score} | {chunk[:160].replace(chr(10), ' ')}…")


def cmd_chat(model: str, top_k: int) -> None:
    print("Interactive RAG session (Ctrl+C or 'quit' to exit)\n")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        try:
            answer, *_ = ask(question, top_k=top_k, model=model)
            print(f"\nRAG: {answer}")
        except RuntimeError as exc:
            print(f"[error] {exc}")


def main() -> None:
    args = _args().parse_args()
    if args.command == "ingest":
        cmd_ingest(getattr(args, "path", None))
    elif args.command == "ask":
        cmd_ask(args.question, args.model, args.top_k)
    elif args.command == "chat":
        cmd_chat(args.model, args.top_k)


if __name__ == "__main__":
    main()
