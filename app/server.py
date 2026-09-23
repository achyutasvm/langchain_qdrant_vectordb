from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import config, ingest, rag

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

config.require_env("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")

app = FastAPI(title="Qdrant RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in config.CORS_ORIGINS.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class Source(BaseModel):
    source: str
    page: Optional[int] = None


class ChatResponse(BaseModel):
    text: str
    sources: list[Source]


class IngestResponse(BaseModel):
    filename: str
    pages: int
    chunks: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    docs_dir = Path(config.DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / filename

    size = 0
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large (max 25MB)")
            out.write(chunk)

    try:
        pages = ingest.load_pdf(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    chunk_count = ingest.ingest_documents(pages)
    return IngestResponse(filename=filename, pages=len(pages), chunks=chunk_count)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    question = user_messages[-1].content
    answer, docs = rag.ask(question)

    sources = [
        Source(
            source=doc.metadata.get("source", ""),
            page=rag.page_label(doc.metadata.get("page")),
        )
        for doc in docs
    ]
    return ChatResponse(text=answer, sources=sources)
