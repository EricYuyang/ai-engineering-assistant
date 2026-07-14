"""
Phase 1 (plain streaming chat) plus Phase 2 (RAG over an ingested repo).

Phase 1's endpoint shape is unchanged: session_id + message in, streamed
tokens out. `project_name` is an *optional* addition -- omit it and /chat
behaves exactly as it did in Phase 1 (no retrieval), so nothing that
already depended on this endpoint breaks.
"""
from __future__ import annotations

import uuid

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.config import settings
from app.llm import get_chat_model
from app.manifest import get_project_collections, load_projects
from app.pipeline import IngestResult, ingest_repo
from app.vectorstore import query_project

app = FastAPI(title="AI Engineering Assistant", version="0.1.0")

llm = get_chat_model()

SYSTEM_PROMPT = (
    "You are an AI Engineering Assistant that helps developers understand "
    "software projects. When retrieved context from a codebase is provided, "
    "ground your answer in it and cite file paths; if the context doesn't "
    "contain the answer, say so rather than guessing."
)

# Phase 1 has no persistent memory system (that arrives in Phase 3 via
# LangGraph checkpointing). This process-local dict is just enough state to
# hold a conversation while the server is running, keyed by session_id.
_sessions: dict[str, list[BaseMessage]] = {}

# Ingestion jobs run in the background (DECISIONS.md #7's implication: a
# repo can take minutes to summarize, so /ingest can't block on it) and are
# tracked here so a client can poll for completion.
_ingest_jobs: dict[str, dict] = {}


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    # Scopes retrieval to a project (a named group of one or more ingested
    # repos, DECISIONS.md #4 extension). Omit for plain Phase 1 chat.
    project_name: str | None = None


class IngestRequest(BaseModel):
    source: str
    project_name: str | None = None
    force: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _build_context_block(docs: list[Document]) -> str:
    parts = [f"--- {doc.metadata.get('source_file', 'unknown')} ---\n{doc.page_content}" for doc in docs]
    return "\n\n".join(parts)


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    history = _sessions.setdefault(req.session_id, [SystemMessage(content=SYSTEM_PROMPT)])
    system_message, conversation = history[0], history[1:]

    # Stopgap history cap (DECISIONS.md #6) ahead of Phase 3's proper
    # checkpointer-based memory -- prevents unbounded history from silently
    # overflowing the 4096-token budget once retrieved chunks share it.
    if len(conversation) > settings.max_history_messages:
        conversation = conversation[-settings.max_history_messages :]

    retrieved_docs: list[Document] = []
    if req.project_name:
        collections = get_project_collections(req.project_name)
        if collections:
            retrieved_docs = query_project(collections, req.message, k=settings.retrieval_top_k)

    messages: list[BaseMessage] = [system_message]
    if retrieved_docs:
        messages.append(
            SystemMessage(
                content="Retrieved context from the codebase:\n\n" + _build_context_block(retrieved_docs)
            )
        )
    messages.extend(conversation)
    messages.append(HumanMessage(content=req.message))

    def token_stream():
        chunks: list[str] = []
        for chunk in llm.stream(messages):
            text = chunk.content or ""
            chunks.append(text)
            yield text

        history.append(HumanMessage(content=req.message))
        history.append(AIMessage(content="".join(chunks)))

        if retrieved_docs:
            sources: list[str] = []
            for doc in retrieved_docs:
                source = doc.metadata.get("source_file", "unknown")
                if source not in sources:
                    sources.append(source)
            yield "\n\nSources: " + ", ".join(sources)

    return StreamingResponse(token_stream(), media_type="text/plain")


def _run_ingest_job(job_id: str, source: str, project_name: str | None, force: bool) -> None:
    try:
        result: IngestResult = ingest_repo(source, project_name=project_name, force=force)
        _ingest_jobs[job_id] = {"status": "done", "result": result.__dict__}
    except Exception as exc:  # surfaced to the poller rather than crashing the background task silently
        _ingest_jobs[job_id] = {"status": "error", "error": str(exc)}


@app.post("/ingest")
def ingest(req: IngestRequest, background_tasks: BackgroundTasks) -> dict:
    job_id = str(uuid.uuid4())
    _ingest_jobs[job_id] = {"status": "running"}
    background_tasks.add_task(_run_ingest_job, job_id, req.source, req.project_name, req.force)
    return {"job_id": job_id, "status": "running"}


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str) -> dict:
    return _ingest_jobs.get(job_id, {"status": "not_found"})


@app.get("/projects")
def list_projects() -> dict:
    return load_projects()
