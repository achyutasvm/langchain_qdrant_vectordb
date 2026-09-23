import os
import sys

import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()


def require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"Missing required environment variable(s): {', '.join(missing)}", file=sys.stderr)
        print("Copy .env.example to .env and fill in the values.", file=sys.stderr)
        sys.exit(1)


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "10k_filing")

# Dense (OpenAI) + sparse (BM25, via fastembed) hybrid retrieval.
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
SPARSE_MODEL = os.environ.get("SPARSE_MODEL", "Qdrant/bm25")

# OpenAI model used for the generation step of the RAG chain.
GENERATION_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

DOCS_DIR = os.environ.get("DOCS_DIR", "docs")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
TOP_K = int(os.environ.get("RAG_TOP_K", "8"))

# Comma-separated origins allowed to call the FastAPI service (e.g. the
# Next.js chat app's dev/prod URLs).
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
