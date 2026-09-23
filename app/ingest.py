import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)


def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=config.SPARSE_MODEL)


def get_qdrant_client_options() -> dict:
    return {"url": config.QDRANT_URL, "api_key": config.QDRANT_API_KEY}


def load_pdf(pdf_path: Path) -> list[Document]:
    """Load a single PDF into one Document per page."""
    pages = PyPDFLoader(str(pdf_path), mode="page").load()
    for page in pages:
        page.metadata["source"] = pdf_path.name
    return pages


def load_pages(docs_dir: Path) -> list[Document]:
    """Load every PDF in docs_dir into one Document per page."""
    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {docs_dir}")

    pages: list[Document] = []
    for pdf_path in pdf_paths:
        pages.extend(load_pdf(pdf_path))
    return pages


def chunk_by_page(pages: list[Document]) -> list[Document]:
    """Recursively split within each page, so chunks never cross a page boundary."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    counts: dict[tuple[str, object], int] = {}
    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "doc")).stem
        page = chunk.metadata.get("page", 0)
        key = (source, page)
        counts[key] = counts.get(key, 0) + 1
        chunk.metadata["chunk_id"] = f"{source}-p{page}-{counts[key]}"
    return chunks


def ingest_documents(pages: list[Document], force_recreate: bool = False) -> int:
    """Chunk pages, embed (dense + sparse hybrid), and upsert into Qdrant.

    Point ids are deterministic (derived from source+page+index), so
    re-ingesting the same file overwrites its previous chunks instead of
    duplicating them.
    """
    chunks = chunk_by_page(pages)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, c.metadata["chunk_id"])) for c in chunks]

    QdrantVectorStore.from_documents(
        chunks,
        ids=ids,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name=config.QDRANT_COLLECTION_NAME,
        force_recreate=force_recreate,
        **get_qdrant_client_options(),
    )
    return len(chunks)


def run() -> None:
    pages = load_pages(Path(config.DOCS_DIR))
    print(f"Loaded {len(pages)} pages from {config.DOCS_DIR}")

    chunk_count = ingest_documents(pages)
    print(f"Upserted {chunk_count} chunks into collection '{config.QDRANT_COLLECTION_NAME}'.")


if __name__ == "__main__":
    config.require_env("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")
    run()
