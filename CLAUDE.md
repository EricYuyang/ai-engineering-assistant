# AI Engineering Assistant — project memory

Read `README.md` first for the full architecture, setup instructions, and
"talking points" (this is portfolio/interview prep — the reasoning behind
each decision matters as much as the code).

## Purpose

Demonstrate, in one working system, how local LLMs, RAG, LangGraph
orchestration, agent memory, and Guardrails AI fit together. Built
incrementally, one concept at a time, so each phase runs on its own before
the next is layered on.

## Stack

Python, FastAPI, LangChain (`ChatOllama`), Ollama (local model), and in
later phases: Chroma (embedded vector store), LangGraph (orchestration +
checkpointer memory), Guardrails AI (structured output validation).

## Phase status

| Phase | What it adds | Status |
|---|---|---|
| 1 | Local LLM chat (FastAPI + LangChain `ChatOllama`, streaming) | **Done** |
| 2 | RAG: ingest docs, chunk, embed locally, retrieve, cite sources | **Done** |
| 3 | LangGraph orchestration + memory (checkpointer + long-term facts) | Next up |
| 4 | Guardrails AI: structured output validation, auto re-ask | Not started |
| Stretch | Multiple selectable knowledge bases | Not started |

Update this table (and the matching one in README.md) whenever a phase is
completed — that's the persistent record of progress across sessions.

## Conventions established so far

- `app/llm.py` is the only file that touches Ollama directly; everything
  else talks to a LangChain `BaseChatModel`, so the model can be swapped
  later without touching callers.
- Conversation history in Phase 1 is a plain in-memory dict — intentionally
  *not* the real memory system. Phase 3 replaces it with LangGraph's
  checkpointer (short-term) plus a small persisted store for facts that
  should survive across sessions (long-term). Keep these conceptually
  separate; don't quietly merge them.
- Two guardrail patterns are kept distinct on purpose: a routing guardrail
  in the graph (pre-generation, based on retrieval confidence) and
  Guardrails AI (post-generation, output schema validation). Phase 3 adds
  the first, Phase 4 adds the second.
- Phase 2 module responsibilities are separated: `app/ingest.py` owns
  source resolution + file collection + chunking; `app/vectorstore.py` owns
  Chroma; `app/pipeline.py` orchestrates the full flow. Don't merge them.
- Analysis doc generation (`app/summarize.py`) is deliberately serial (one
  local LLM, no parallelism) — the background job pattern in `main.py`
  absorbs the latency.
- Project grouping is an application-layer manifest (`data/projects.json`),
  not a Chroma feature — multi-repo querying merges results at query time.

## When starting a new phase

1. Confirm current phase status against the table above.
2. Implement the phase without breaking the previous phases' endpoints.
3. Update README.md's phase table and talking-points section.
4. Update this file's phase table to match.
5. Commit with a message naming the phase (e.g. `Phase 2: RAG retrieval`).
