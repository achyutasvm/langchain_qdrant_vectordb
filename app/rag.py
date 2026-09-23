from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from app import config, ingest

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions using only the "
            "provided context. If the answer isn't in the context, say you "
            "don't know. Cite the page number(s) you relied on.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def page_label(page: object) -> Optional[int]:
    """Convert a 0-indexed page number to a human-readable 1-indexed one."""
    if isinstance(page, (int, float)):
        return int(page) + 1
    return None


def format_docs(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        page = page_label(doc.metadata.get("page"))
        source = doc.metadata.get("source", "")
        parts.append(f"[source={source} page={page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_retriever():
    client = QdrantClient(**ingest.get_qdrant_client_options())
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embedding=ingest.get_embeddings(),
        sparse_embedding=ingest.get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
    )
    return vector_store.as_retriever(search_kwargs={"k": config.TOP_K})


def build_chain():
    llm = ChatOpenAI(
        model=config.GENERATION_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    )
    return PROMPT | llm | StrOutputParser()


def ask(question: str) -> tuple[str, list[Document]]:
    retriever = build_retriever()
    docs = retriever.invoke(question)

    chain = build_chain()
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    return answer, docs


if __name__ == "__main__":
    import sys

    config.require_env("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")

    query = " ".join(sys.argv[1:]) or "What is this document about?"
    result, sources = ask(query)

    print("Answer:\n" + result)
    print("\nSources:")
    for source_doc in sources:
        print(f" - page {page_label(source_doc.metadata.get('page'))} ({source_doc.metadata.get('source')})")
