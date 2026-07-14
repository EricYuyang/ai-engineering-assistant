"""
LangGraph StateGraph for the chat pipeline.

Nodes:
  retrieve       – query the vector store (skipped when no project_name)
  route          – conditional edge: check retrieval confidence
  generate       – assemble messages and call the LLM
  validate       – post-generation: structured JSON validation + re-ask,
                   with plain-text fallback (Phase 4, Guardrails pattern)
  no_context     – hard gate: return "not enough context" message
  extract_facts  – scan the response for long-term facts to persist

Plain chat (no project_name):    retrieve(skip) → generate → validate(skip) → extract_facts
RAG with good context:           retrieve → generate → validate → extract_facts
RAG with poor context:           retrieve → no_context
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.facts import load_facts, save_facts
from app.llm import get_chat_model
from app.manifest import get_project_collections
from app.validate import try_parse_structured, validate_plaintext, validate_structured

SYSTEM_PROMPT = (
    "You are an AI Engineering Assistant that helps developers understand "
    "software projects. When retrieved context from a codebase is provided, "
    "ground your answer in it and cite file paths; if the context doesn't "
    "contain the answer, say so rather than guessing."
)

FACT_EXTRACTION_PROMPT = (
    "Review the conversation above. Extract any factual statements about "
    "the project's architecture, technology choices, conventions, or "
    "ownership that were explicitly stated (not inferred). Return a JSON "
    'array of short fact strings. If there are none, return [].\n'
    "Example: [\"The auth service uses JWT tokens\", \"Redis handles caching\"]"
)


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    project_name: str | None
    retrieved_docs: list[Document]
    best_score: float
    response: str
    sources: list[str]
    node_trace: Annotated[list[str], operator.add]
    validation_status: str


def _build_context_block(docs: list[Document]) -> str:
    parts = [
        f"--- {doc.metadata.get('source_file', 'unknown')} ---\n{doc.page_content}"
        for doc in docs
    ]
    return "\n\n".join(parts)


def _build_generation_messages(state: GraphState) -> list[BaseMessage]:
    msgs: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    project_name = state.get("project_name")
    if project_name:
        facts = load_facts(project_name)
        if facts:
            facts_block = "\n".join(f"- {f}" for f in facts)
            msgs.append(
                SystemMessage(content=f"Known facts about this project:\n{facts_block}")
            )

    retrieved_docs = state.get("retrieved_docs", [])
    if retrieved_docs:
        msgs.append(
            SystemMessage(
                content="Retrieved context from the codebase:\n\n"
                + _build_context_block(retrieved_docs)
            )
        )

    history = state.get("messages", [])
    if len(history) > settings.max_history_messages:
        history = history[-settings.max_history_messages :]
    msgs.extend(history)

    return msgs


def retrieve_node(state: GraphState) -> dict:
    project_name = state.get("project_name")
    if not project_name:
        return {
            "retrieved_docs": [],
            "best_score": -1.0,
            "node_trace": ["retrieve(skip)"],
        }

    collections = get_project_collections(project_name)
    if not collections:
        return {
            "retrieved_docs": [],
            "best_score": -1.0,
            "node_trace": ["retrieve(no-collections)"],
        }

    from app.vectorstore import query_collection

    scored: list[tuple[float, Document]] = []
    for name in collections:
        for doc, score in query_collection(name, state["messages"][-1].content, k=settings.retrieval_top_k):
            doc.metadata["collection"] = name
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: settings.retrieval_top_k]
    best = top[0][0] if top else 0.0
    docs = [doc for _, doc in top]

    return {
        "retrieved_docs": docs,
        "best_score": best,
        "node_trace": [f"retrieve(k={len(docs)}, best={best:.3f})"],
    }


def route_edge(state: GraphState) -> str:
    if state.get("best_score", -1.0) < 0:
        return "generate"
    if state["best_score"] >= settings.confidence_threshold:
        return "generate"
    return "no_context"


def generate_node(state: GraphState) -> dict:
    llm = get_chat_model()
    msgs = _build_generation_messages(state)

    full_response: list[str] = []
    for chunk in llm.stream(msgs):
        text = chunk.content or ""
        full_response.append(text)

    response_text = "".join(full_response)

    sources: list[str] = []
    for doc in state.get("retrieved_docs", []):
        src = doc.metadata.get("source_file", "unknown")
        if src not in sources:
            sources.append(src)

    return {
        "messages": [AIMessage(content=response_text)],
        "response": response_text,
        "sources": sources,
        "node_trace": ["generate"],
    }


def validate_node(state: GraphState) -> dict:
    retrieved_docs = state.get("retrieved_docs", [])
    if not retrieved_docs:
        return {
            "validation_status": "skip",
            "node_trace": ["validate(skip)"],
        }

    response_text = state.get("response", "")
    expected_sources = state.get("sources", [])
    trace: list[str] = []

    parsed, parse_err = try_parse_structured(response_text)
    if parsed:
        trace.append("existing-json:pass")
        return {
            "messages": [AIMessage(content=parsed.answer)],
            "response": parsed.answer,
            "sources": parsed.sources,
            "validation_status": "structured",
            "node_trace": [f"validate({','.join(trace)})"],
        }

    is_valid, reason = validate_plaintext(response_text, expected_sources)
    trace.append(f"plaintext:{reason}")

    if is_valid:
        return {
            "validation_status": "plaintext-valid",
            "node_trace": [f"validate({','.join(trace)})"],
        }

    llm = get_chat_model()
    msgs = _build_generation_messages(state)
    parsed, structured_trace = validate_structured(
        llm, msgs, max_retries=settings.guardrails_max_retries,
    )
    trace.extend(structured_trace)

    if parsed:
        return {
            "messages": [AIMessage(content=parsed.answer)],
            "response": parsed.answer,
            "sources": parsed.sources,
            "validation_status": "structured-reask",
            "node_trace": [f"validate({','.join(trace)})"],
        }

    return {
        "validation_status": "unverified",
        "node_trace": [f"validate({','.join(trace)})"],
    }


def no_context_node(state: GraphState) -> dict:
    msg = (
        "I don't have enough relevant context from the codebase to answer "
        "this confidently. Try rephrasing your question or check that the "
        "right project is ingested."
    )
    return {
        "messages": [AIMessage(content=msg)],
        "response": msg,
        "sources": [],
        "node_trace": ["no_context"],
    }


def extract_facts_node(state: GraphState) -> dict:
    project_name = state.get("project_name")
    if not project_name:
        return {"node_trace": ["extract_facts(skip)"]}

    llm = get_chat_model()
    recent = state.get("messages", [])[-4:]
    extraction_msgs: list[BaseMessage] = list(recent) + [
        HumanMessage(content=FACT_EXTRACTION_PROMPT)
    ]

    result = llm.invoke(extraction_msgs)
    text = (result.content or "").strip()

    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        facts = json.loads(text[start:end])
        if isinstance(facts, list):
            facts = [f for f in facts if isinstance(f, str) and len(f) > 5]
            save_facts(project_name, facts)
    except (ValueError, json.JSONDecodeError):
        facts = []

    return {"node_trace": [f"extract_facts({len(facts)})"]}


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("no_context", no_context_node)
    graph.add_node("extract_facts", extract_facts_node)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", route_edge, {"generate": "generate", "no_context": "no_context"})
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "extract_facts")
    graph.add_edge("extract_facts", END)
    graph.add_edge("no_context", END)

    return graph
