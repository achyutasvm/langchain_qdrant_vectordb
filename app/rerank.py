import logging
import threading
from typing import Optional

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.documents import Document

from app import config

logger = logging.getLogger(__name__)

_model: Optional[TextCrossEncoder] = None
_model_lock = threading.Lock()


def _get_model() -> TextCrossEncoder:
    # Loading is slow (and downloads the model the first time), so do it once
    # per process. FastAPI runs sync endpoints in a thread pool, hence the lock.
    global _model
    with _model_lock:
        if _model is None:
            _model = TextCrossEncoder(model_name=config.RERANK_MODEL)
    return _model


def rerank(question: str, docs: list[Document], top_n: int) -> list[Document]:
    """Reorder docs by relevance to the question and keep the best top_n.

    Each returned doc gets its cross-encoder score in metadata["rerank_score"]
    (a raw logit: higher is more relevant, not a probability). If reranking
    fails, falls back to the hybrid search order.
    """
    if not docs:
        return docs
    try:
        scores = list(_get_model().rerank(question, [d.page_content for d in docs]))
    except Exception:
        logger.exception("Rerank failed; falling back to hybrid search order")
        return docs[:top_n]

    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
    reranked = []
    for score, doc in ranked[:top_n]:
        doc.metadata["rerank_score"] = float(score)
        reranked.append(doc)
    return reranked
