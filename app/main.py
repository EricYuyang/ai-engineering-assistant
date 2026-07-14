"""
Phase 1 (plain streaming chat), Phase 2 (RAG), Phase 3 (LangGraph
orchestration + memory), and Phase 4 (Guardrails: output validation).

The /chat endpoint streams tokens from the generate node, then shows
a "Validating..." indicator while the validate node runs. If validation
replaces the response (structured re-ask succeeded), the new response
is appended. The final output includes validation status in the node
trace so the user can see what the graph did.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from app.config import settings
from app.graph import build_graph
from app.manifest import load_projects
from app.pipeline import IngestResult, ingest_repo

app = FastAPI(title="AI Engineering Assistant", version="0.4.0")

_static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

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


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _make_input_state(req: ChatRequest) -> dict:
    return {
        "messages": [HumanMessage(content=req.message)],
        "project_name": req.project_name,
        "retrieved_docs": [],
        "best_score": 0.0,
        "response": "",
        "sources": [],
        "node_trace": [],
        "validation_status": "",
    }


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def token_stream():
        async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as checkpointer:
            graph = _graph_builder.compile(checkpointer=checkpointer)

            config = {"configurable": {"thread_id": req.session_id}}
            input_state = _make_input_state(req)

            validate_started = False
            original_response = ""

            async for event in graph.astream_events(input_state, config=config, version="v2"):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")

                if kind == "on_chat_model_stream" and node == "generate":
                    token = event["data"]["chunk"].content or ""
                    if token:
                        original_response += token
                        yield token

                if kind == "on_chain_start" and event.get("name") == "validate":
                    if req.project_name:
                        validate_started = True
                        yield "\n\n⏳ Validating..."

                if kind == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    sources = output.get("sources", [])
                    node_trace = output.get("node_trace", [])
                    validation_status = output.get("validation_status", "")
                    final_response = output.get("response", "")

                    if validate_started:
                        if validation_status == "structured-reask":
                            yield f" ✓ Re-validated"
                            yield f"\n\n{final_response}"
                        elif validation_status in ("structured", "plaintext-valid"):
                            yield " ✓ Validated"
                        elif validation_status == "unverified":
                            yield " ⚠ Could not verify citations"
                        elif validation_status == "skip":
                            pass

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
        input_state = _make_input_state(req)

        result = await graph.ainvoke(input_state, config=config)
        return {
            "response": result["response"],
            "sources": result["sources"],
            "node_trace": result["node_trace"],
            "best_score": result["best_score"],
            "validation_status": result.get("validation_status", ""),
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
