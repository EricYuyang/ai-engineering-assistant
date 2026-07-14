"""
Chroma collection management: one collection per ingested repo
(DECISIONS.md #4), with a global-top-k merge query across a project's
repos for multi-repo retrieval (DECISIONS.md #4 extension, #6).
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app.config import settings


def get_embeddings() -> Embeddings:
    return OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)


def get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents(collection_name: str, documents: list[Document]) -> None:
    if not documents:
        return
    get_vectorstore(collection_name).add_documents(documents)


def delete_by_source_file(collection_name: str, source_file: str) -> None:
    """Used by incremental re-ingestion (Decision 5) to drop a changed or
    deleted file's stale chunks before re-adding the current version."""
    get_vectorstore(collection_name).delete(where={"source_file": source_file})


def delete_collection(collection_name: str) -> None:
    get_vectorstore(collection_name).delete_collection()


def query_collection(collection_name: str, query: str, k: int) -> list[tuple[Document, float]]:
    return get_vectorstore(collection_name).similarity_search_with_relevance_scores(query, k=k)


def query_project(collection_names: list[str], query: str, k: int) -> list[Document]:
    """Global top-k across every repo collection in a project (Decision 6):
    query each collection, merge by relevance score, keep the k highest
    overall rather than reserving slots per repo."""
    scored: list[tuple[float, Document]] = []
    for name in collection_names:
        for doc, score in query_collection(name, query, k=k):
            doc.metadata["collection"] = name
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:k]]
