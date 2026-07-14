"""
Phase 1 (plain streaming chat), Phase 2 (RAG), and Phase 3 (LangGraph
orchestration + memory).

Phase 3 replaces the hand-rolled message assembly and in-memory session
dict with a LangGraph StateGraph backed by a SqliteSaver checkpointer.
The /chat endpoint is async; token-level streaming is handled by
iterating the generate node's LLM output within the graph, then
yielding the full response plus metadata.
"""
from __future__ import annotations

import uuid

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from app.config import settings
from app.graph import build_graph
from app.manifest import load_projects
from app.pipeline import IngestResult, ingest_repo

app = FastAPI(title="AI Engineering Assistant", version="0.2.0")

_graph_builder = build_graph()

_ingest_jobs: dict[str, dict] = {}


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    project_name: str | None = None


class IngestRequest(BaseModel):
    source: str
    project_name: str | None = None
    force: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def token_stream():
        async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as checkpointer:
            graph = _graph_builder.compile(checkpointer=checkpointer)

            config = {"configurable": {"thread_id": req.session_id}}
            input_state = {
                "messages": [HumanMessage(content=req.message)],
                "project_name": req.project_name,
                "retrieved_docs": [],
                "best_score": 0.0,
                "response": "",
                "sources": [],
                "node_trace": [],
            }

            async for event in graph.astream_events(input_state, config=config, version="v2"):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    if event.get("metadata", {}).get("langgraph_node") == "generate":
                        token = event["data"]["chunk"].content or ""
                        if token:
                            yield token

                if kind == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    sources = output.get("sources", [])
                    node_trace = output.get("node_trace", [])

                    if sources:
                        yield "\n\nSources: " + ", ".join(sources)
                    if node_trace:
                        yield f"\n[nodes: {' → '.join(node_trace)}]"

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post("/chat/sync")
async def chat_sync(req: ChatRequest) -> dict:
    """Non-streaming fallback — useful for testing and debugging."""
    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as checkpointer:
        graph = _graph_builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": req.session_id}}
        input_state = {
            "messages": [HumanMessage(content=req.message)],
            "project_name": req.project_name,
            "retrieved_docs": [],
            "best_score": 0.0,
            "response": "",
            "sources": [],
            "node_trace": [],
        }

        result = await graph.ainvoke(input_state, config=config)
        return {
            "response": result["response"],
            "sources": result["sources"],
            "node_trace": result["node_trace"],
            "best_score": result["best_score"],
        }


def _run_ingest_job(job_id: str, source: str, project_name: str | None, force: bool) -> None:
    try:
        result: IngestResult = ingest_repo(source, project_name=project_name, force=force)
        _ingest_jobs[job_id] = {"status": "done", "result": result.__dict__}
    except Exception as exc:
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
