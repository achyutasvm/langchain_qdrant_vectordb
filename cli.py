#!/usr/bin/env python3
"""CLI to ingest PDFs into Qdrant (via LangChain) and test retrieval.

Pipeline: PyPDFLoader (page load) -> split into smaller chunks ->
OpenAI text-embedding-3-small (dense) + BM25 (sparse) -> Qdrant hybrid search.

Shares its ingestion/retrieval logic with the FastAPI service in app/server.py.
"""

import argparse
from pathlib import Path

from app import config, ingest, rag


def cmd_ingest(args: argparse.Namespace) -> None:
    config.require_env("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            raise SystemExit(f"PDF not found: {pdf_path}")
        print(f"Loading '{pdf_path}' (page load)...")
        pages = ingest.load_pdf(pdf_path)
    else:
        docs_dir = Path(config.DOCS_DIR)
        print(f"Loading PDF(s) from '{docs_dir}' (page load)...")
        pages = ingest.load_pages(docs_dir)
    print(f"Loaded {len(pages)} page(s).")

    print(f"Embedding with '{config.EMBEDDING_MODEL}' (dense) + '{config.SPARSE_MODEL}' (sparse) "
          f"and indexing into Qdrant collection '{config.QDRANT_COLLECTION_NAME}'...")
    chunk_count = ingest.ingest_documents(pages, force_recreate=args.recreate)
    print(f"Done. Indexed {chunk_count} chunk(s) into '{config.QDRANT_COLLECTION_NAME}'.")


def cmd_search(args: argparse.Namespace) -> None:
    config.require_env("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")

    retriever = rag.build_retriever()
    retriever.search_kwargs["k"] = args.top_k
    docs = retriever.invoke(args.query)

    if not docs:
        print("No results.")
        return

    for i, doc in enumerate(docs, start=1):
        page = rag.page_label(doc.metadata.get("page"))
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        print(f"\n[{i}] page={page}")
        print(snippet)


def cmd_ask(args: argparse.Namespace) -> None:
    config.require_env("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")

    answer, docs = rag.ask(args.query)
    print(answer)
    print("\nSources:")
    for doc in docs:
        print(f" - page {rag.page_label(doc.metadata.get('page'))} ({doc.metadata.get('source')})")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain + Qdrant PDF RAG test CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Parse PDF(s) and index them into Qdrant")
    p_ingest.add_argument("--pdf", help=f"Path to a single PDF file (default: all PDFs in {config.DOCS_DIR}/)")
    p_ingest.add_argument("--recreate", action="store_true",
                           help="Drop and recreate the Qdrant collection before indexing")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="Run a similarity search against the index")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--top-k", type=int, default=config.TOP_K,
                           help=f"Number of results (default: {config.TOP_K})")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="Ask a question answered via retrieved context (RAG)")
    p_ask.add_argument("query", help="Question to ask")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
