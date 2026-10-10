# PDF RAG with Qdrant + LangChain

Retrieval-augmented generation over the PDF(s) in `docs/`.

- **Chunking**: each PDF page is loaded as its own `Document` (`PyPDFLoader`), then
  `RecursiveCharacterTextSplitter` splits within each page — so chunks respect
  page boundaries and every chunk keeps its `page` metadata.
- **Embeddings**: OpenAI `text-embedding-3-small` (dense) + BM25 via `fastembed`
  (sparse), combined through `langchain-qdrant`'s hybrid retrieval mode.
- **Vector store**: Qdrant Cloud, via `QdrantVectorStore`.
- **Reranking** (on by default): retrieves 20 candidate chunks, scores each
  against the question with a local cross-encoder (`fastembed`,
  `Xenova/ms-marco-MiniLM-L-12-v2`), and keeps the best top-k. See
  [Reranking](#reranking).
- **Generation**: OpenAI, via `langchain-openai`'s `ChatOpenAI` (default `gpt-4o-mini`).

## Architecture

```mermaid
flowchart TB
    PDF["docs/*.pdf"]

    subgraph ingest["Ingestion — python cli.py ingest"]
        direction TB
        Loader["PyPDFLoader\n(one Document per page)"]
        Splitter["RecursiveCharacterTextSplitter\n(splits within each page)"]
        EmbedDocs["OpenAI text-embedding-3-small (dense)\n+ BM25 via fastembed (sparse)"]
        Loader --> Splitter --> EmbedDocs
    end

    Index[("Qdrant Cloud collection\n(QdrantVectorStore, hybrid)")]

    subgraph query["Query — python cli.py ask \"...\""]
        direction TB
        Question["Question"]
        EmbedQuery["Dense + sparse query embedding"]
        Retrieve["Hybrid similarity search\n(20 candidates)"]
        Rerank["Local cross-encoder (fastembed)\nms-marco-MiniLM-L-12-v2\n(keep top-k)"]
        Prompt["ChatPromptTemplate\n(context + question)"]
        LLM["ChatOpenAI\ngpt-4o-mini"]
        Answer["Answer + cited page numbers"]
        Question --> EmbedQuery --> Retrieve --> Rerank --> Prompt --> LLM --> Answer
        Retrieve -. "rerank off: top-k" .-> Prompt
    end

    PDF --> Loader
    EmbedDocs -- "upsert chunks (deterministic ids)" --> Index
    Index -- "candidate chunks" --> Retrieve
```

Both the CLI (`cli.py`) and the API (`app/server.py`) share the same `app/ingest.py` +
`app/rag.py` chunking + retrieval + generation logic.

```mermaid
flowchart LR
    Browser["Chat app browser client"]
    NextRoute["Next.js route\n/api/rag-chat"]
    FastAPI["FastAPI\nPOST /chat"]
    RAGChain["app/rag.py\nhybrid retriever + reranker + prompt + ChatOpenAI"]
    QdrantIdx[("Qdrant collection")]

    Browser --> NextRoute --> FastAPI --> RAGChain
    RAGChain <--> QdrantIdx
    RAGChain -- "answer + sources" --> FastAPI --> NextRoute --> Browser
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

`QDRANT_URL`/`QDRANT_API_KEY` come from a Qdrant Cloud cluster (cloud.qdrant.io).

## Usage

```bash
# Chunk docs/*.pdf and upsert into Qdrant (all PDFs in docs/ by default)
python cli.py ingest

# --recreate drops and rebuilds the collection first
python cli.py ingest --recreate

# Similarity search (raw retrieved chunks, no generation)
python cli.py search "revenue recognition policy"

# Ask a question (RAG: retrieve + generate)
python cli.py ask "What are the main risk factors disclosed in this filing?"
```

## Run as an API

The same retrieval + generation chain is also exposed over HTTP via FastAPI, so
other apps (e.g. a chat frontend) can call it:

```bash
uvicorn app.server:app --reload --port 8001
```

- `GET /health` — liveness check.
- `POST /chat` — body `{"messages": [{"role": "user", "content": "..."}], "rerank": true}`,
  returns `{"text": "...", "sources": [{"source": "...", "page": 3, "rerank_score": 5.72}, ...]}`.
  Only the last `user` message is used as the retrieval query. `rerank` is
  optional and overrides `RERANK_ENABLED` for that request. `sources` are in
  rank order; `rerank_score` is `null` when reranking didn't run.
- `POST /ingest` — multipart upload, field name `file` (PDF only, 25MB max).
  Saves the file into `docs/`, then chunks/embeds/upserts it, same as
  `python cli.py ingest`. Returns `{"filename", "pages", "chunks"}`. Re-uploading
  the same file overwrites its existing chunks (deterministic point ids) rather
  than duplicating them.

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What fiscal year does this filing cover?"}]}'

curl -X POST http://localhost:8001/ingest -F "file=@docs/some-report.pdf"
```

Set `CORS_ORIGINS` (comma-separated, default `http://localhost:3000`) to allow
the calling frontend's origin.

### Integration with my_test_chat_app

[my_test_chat_app](https://github.com/achyutasvm/my_test_chat_app) is a Next.js
chat app with a "Document Q&A" mode that can call **two** RAG backends —
this Qdrant service and the sibling
[vector_embeddings](https://github.com/achyutasvm/vector_embeddings) (Pinecone)
service — chosen per-upload via a selector in the UI. Run whichever backend(s)
you want available and set both `RAG_API_URL_QDRANT` (pointing here, default
port 8001) and `RAG_API_URL_PINECONE` (pointing at vector_embeddings, default
port 8000) in that app's `.env.local`.

```bash
# Terminal 1 — this service
cd path/to/langchain_qdrant_vectordb
source .venv/bin/activate
uvicorn app.server:app --reload --port 8001

# Terminal 2 — the Pinecone sibling service (optional, if you want both options)
cd path/to/vector_embeddings
source .venv/bin/activate
uvicorn app.server:app --reload --port 8000

# Terminal 3 — the chat app
cd path/to/my_test_chat_app
npm run dev
```

## Reranking

Hybrid search finds chunks that *sound* like the question, which isn't always
the chunk that *answers* it. With reranking on, `app/rag.py`:

1. Retrieves `RERANK_CANDIDATES` (default 20) chunks via hybrid search instead of `RAG_TOP_K`.
2. Scores each (question, chunk) pair with a cross-encoder running locally
   through `fastembed` (`app/rerank.py`, default `Xenova/ms-marco-MiniLM-L-12-v2`,
   ~130 MB). No API key needed; `fastembed` is already a dependency for BM25.
3. Keeps the `RAG_TOP_K` highest-scoring chunks, in score order, for generation.

Notes:
- **First use downloads the model** from Hugging Face (about 20s), then it
  loads from the local cache in well under a second. The model is loaded once
  per process and shared across requests.
- **Scores are raw logits**, not probabilities: higher is more relevant, and
  they can be negative (e.g. `5.72` vs `-2.26`). Compare them only within one
  question.
- **`HF_HUB_DISABLE_XET=1`** is set in `.env.example` because `hf-xet` 1.6.0
  can fail the model download with "Byte range not sequential". If the
  download fails anyway, the reranker logs the error and falls back to the
  hybrid-search order rather than failing the question.
- `python cli.py search` shows raw hybrid-search results and doesn't rerank;
  `python cli.py ask` and `POST /chat` do.
- Other `fastembed` rerankers (`BAAI/bge-reranker-base`,
  `jinaai/jina-reranker-v2-base-multilingual`, ...) can be swapped in via
  `RERANK_MODEL`.

Settings (`.env`): `RERANK_ENABLED` (default `true`), `RERANK_MODEL`,
`RERANK_CANDIDATES` (default `20`).

## Config

All settings are environment variables (see `.env.example`): OpenAI/Qdrant
credentials, collection name, embedding/generation models, chunk size/overlap,
retrieval top-k, and reranking (`RERANK_ENABLED`, `RERANK_MODEL`,
`RERANK_CANDIDATES`).

## Project structure

- `app/config.py` — environment-variable configuration.
- `app/ingest.py` — PDF loading, chunking, embedding, and upsert into Qdrant.
- `app/rag.py` — hybrid retriever + reranking + prompt + generation chain,
  shared by the CLI and the API.
- `app/rerank.py` — local cross-encoder reranker (`fastembed`).
- `app/server.py` — FastAPI app (`/health`, `/chat`, `/ingest`).
- `cli.py` — CLI (`ingest`, `search`, `ask`) built on the same `app/` modules.
- `docs/` — source PDFs (also where uploaded files land).
