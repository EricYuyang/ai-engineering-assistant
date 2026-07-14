# AI Engineering Assistant

A small assistant that helps a developer understand a codebase: ask a
question, it retrieves relevant chunks from the project's docs/source,
remembers facts established earlier in the conversation, and validates its
own output before returning it. Built to demonstrate, end to end, how
**local LLMs, RAG, LangGraph orchestration, agent memory, and Guardrails**
fit together in one real system -- not as a code sample, but as an
architecture to be able to explain.

## Target architecture

```
                    User question (+ session_id)
                              |
                        FastAPI endpoint
                              |
                     LangGraph StateGraph
                              |
              +---------------+---------------+
              |                               |
          retrieve                      (thread-level memory:
      (Chroma vector store,               LangGraph checkpointer,
       local embeddings)                  loaded automatically per
              |                            session_id)
      assess confidence
      (routing guardrail:
       low similarity -> "not enough
       context" branch, skip generate)
              |
           generate
      (ChatOllama + retrieved context
       + conversation history
       + long-term facts)
              |
      validate output
      (Guardrails AI: Pydantic schema
       for answer + sources, auto
       re-ask on failure)
              |
        update long-term memory
      (extract stated facts/decisions,
       persist keyed by session)
              |
        answer + sources + which
        nodes fired (for demoing)
```

Two distinct guardrail patterns are deliberately kept separate, because
they answer different questions:
- **Routing guardrail** (in the graph, pre-generation): "do we have enough
  retrieved context to even attempt an answer?"
- **Guardrails AI** (post-generation): "is the model's output well-formed
  and does it actually cite what it retrieved?"

## Build phases

| Phase | What it adds | Status |
|---|---|---|
| 1 | Local LLM chat (FastAPI + LangChain `ChatOllama`, streaming) | Done |
| 2 | RAG: ingest docs, chunk, embed locally, retrieve, cite sources | **Done** |
| 3 | LangGraph orchestration + memory (checkpointer + long-term facts) | **Done** |
| 4 | Guardrails AI: structured output validation, auto re-ask | **Done** |
| Stretch | Multiple selectable knowledge bases | Not started |

## Setup

1. Install [Ollama](https://ollama.com) and pull the chat + embedding models:
   ```
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
2. Create a virtualenv and install deps:
   ```
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` (defaults already match `ollama pull llama3.2:3b`).
4. Run the API:
   ```
   uvicorn app.main:app --reload
   ```
5. Chat with it via the CLI (streams tokens as they arrive):
   ```
   python scripts/chat_cli.py
   ```
   or hit `POST /chat` directly with `{"session_id": "x", "message": "..."}`.
6. Ingest a repo (local folder or GitHub URL) to enable RAG:
   ```
   curl -X POST http://127.0.0.1:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"source": "/path/to/repo"}'
   ```
   Poll `GET /ingest/status/{job_id}` until status is `"done"`, then chat
   with `{"session_id": "x", "message": "...", "project_name": "<collection>"}`.

## Phase 1 notes

- `app/llm.py` is the *only* place that knows about Ollama. Everything else
  talks to a LangChain `BaseChatModel`. That's the seam where a hosted
  model (or a different local one) gets swapped in later without touching
  the endpoint, the prompt, or (eventually) the graph.
- `app/main.py` holds conversation history in a plain in-memory dict keyed
  by `session_id`. This is intentionally *not* the memory system --
  Phase 3 replaces it with LangGraph's checkpointer (thread-level/short-term)
  plus a small persisted store for facts that should survive across
  sessions (long-term). Calling this out explicitly matters: it's easy to
  conflate "the app remembers the last few messages" with "the app has
  memory" -- they're different concerns, and the project is structured to
  keep them separate on purpose.
- No RAG, no orchestration graph, no guardrails yet -- by design, so each
  concept gets introduced with a working system underneath it rather than
  landing all at once.

## Talking points for Phase 1

- Why LangChain here even before LangGraph: `ChatOllama` and the message
  types (`SystemMessage`/`HumanMessage`/`AIMessage`) are LangChain
  primitives that LangGraph nodes will consume directly in Phase 3 -- no
  rewrite needed, just wrapping this same call in a graph node.
  This is also where "why Python" mattered for this project: LangChain and
  LangGraph are Python (and JS) native, and doing this in .NET would have
  meant reimplementing their concepts by hand rather than using the actual
  tools.
- Why streaming from the start: it's the difference between "the API
  works" and "the API is usable" -- and it forces the response shape
  (`StreamingResponse` over token chunks) to be decided early, before RAG
  and guardrails have to fit inside it too.
- Why a local model at all: no API cost/latency while iterating, and it's
  the more interesting architecture problem -- smaller local models make
  retrieval quality and guardrails *matter more*, since the model alone is
  less reliable than a frontier hosted model would be.

## Phase 2 notes

Phase 2 adds RAG: ingest a codebase (local folder or GitHub URL), chunk and
embed it locally, then retrieve relevant code when answering questions.

New modules and what they own:
- `app/ingest.py` — source resolution (local path vs. `git clone`), file
  collection (`.gitignore` + extension allowlist + size cap), language-aware
  chunking via `RecursiveCharacterTextSplitter.from_language()`.
- `app/vectorstore.py` — Chroma collection management: one collection per
  repo, add/delete/query, plus a multi-collection merge query for
  project-scoped retrieval.
- `app/manifest.py` — per-repo file-hash manifests (for incremental
  re-ingestion) and a project grouping manifest (`project_name →
  [collection_names]`).
- `app/summarize.py` — map-reduce analysis doc: per-file summaries →
  hierarchical reduce via folder structure → saved to
  `analysis/<collection>.md`, then chunked and embedded back into the
  collection so the doc itself is retrievable.
- `app/pipeline.py` — orchestrates the full flow: resolve source → diff
  manifest → chunk+embed changed files → update manifest → generate
  analysis doc → register project.

New endpoints:
- `POST /ingest` — kicks off a background ingestion job, returns a
  `job_id` for polling.
- `GET /ingest/status/{job_id}` — returns `running`, `done` (with stats),
  or `error`.
- `GET /projects` — lists all ingested projects and their collections.
- `POST /chat` gains an optional `project_name` field — when set, retrieves
  top-k chunks from the project's collections and injects them as context.
  Sources are appended to the streamed response.

Key constraints:
- `llama3.2:3b` has a 4096-token context window. At ~800-char chunks
  (~200 tokens each), top-k=6 costs ~1200 tokens, leaving room for system
  prompt, question, history, and answer.
- Conversation history is capped at 12 messages as a stopgap — Phase 3's
  LangGraph checkpointer replaces this with proper memory management.

## Talking points for Phase 2

- Why language-aware chunking over naive character splitting: a naive
  splitter will cut a function in half at an arbitrary character boundary.
  `RecursiveCharacterTextSplitter.from_language()` understands code
  structure (function/class boundaries) for each language and prefers those
  as split points, producing chunks that are semantically self-contained.
  This matters because an embedding of half a function signature is
  meaningless — the embedding model needs a coherent unit to produce a
  useful vector. (DECISIONS.md #2)
- Why per-repo Chroma collections instead of one shared collection with
  metadata filters: a shared collection has a silent-failure risk — every
  query must remember to filter by `repo_name`, and missing it once leaks
  cross-repo results. Separate collections make that bug class impossible
  by construction. The trade-off (querying across repos requires merging
  results from multiple collections) is the right cost — and the
  project-grouping layer handles it cleanly at the application level.
  (DECISIONS.md #4)
- Why incremental re-ingestion via content hashing: always-append creates
  stale duplicates (edit a function, both old and new chunks exist,
  retrieval might cite the old one). Always-rebuild is correct but wastes
  time re-embedding unchanged files. File-level SHA-256 hashing against a
  stored manifest gives us the diff cheaply — only changed/new files get
  re-chunked and re-embedded. (DECISIONS.md #5)
- Why map-reduce via folder structure for the analysis doc: flat
  "combine all summaries in one pass" breaks the token budget for any
  non-trivial repo. Instead of inventing a batching scheme, the reduce
  step reuses the repo's own directory tree — which already encodes the
  authors' organizational intent. Per-file summaries roll up to
  per-directory, then to the whole repo. (DECISIONS.md #7)
- The dependency resolution snag on Windows: `chromadb` 0.5.x requires
  compiling `chroma-hnswlib` from C++ source (needs Microsoft C++ Build
  Tools — a large, invasive system install). Solved by moving to
  `chromadb` 1.x which ships precompiled Windows wheels, then finding the
  specific `langchain-chroma` version (0.2.4) that bridges old
  `langchain-core` 0.3.x with new `chromadb` 1.x — resolved by reading
  PyPI dependency metadata rather than trial-and-error. (DECISIONS.md #9)

## Phase 3 notes

Phase 3 wraps the chat pipeline in a LangGraph `StateGraph`, adds
persistent session memory via a SqliteSaver checkpointer, and introduces
a routing guardrail that prevents hallucination when retrieval context
is poor.

New/changed modules:
- `app/graph.py` — the StateGraph definition with four nodes: `retrieve`,
  `generate`, `no_context`, and `extract_facts`. A conditional edge after
  `retrieve` routes based on retrieval confidence.
- `app/facts.py` — long-term fact storage in a separate `facts.sqlite3`.
  Extracted facts are keyed by `project_name` and injected as system
  messages in future sessions.
- `app/main.py` — rewritten to use the graph. The `/chat` endpoint is now
  async and streams tokens via `astream_events()`. A `/chat/sync` endpoint
  is available for debugging (returns full JSON result).

Three paths through the graph:
1. **Plain chat** (no `project_name`): `retrieve(skip) → generate` — same
   as Phase 1 behavior, no retrieval.
2. **RAG with good context**: `retrieve → generate → extract_facts` —
   retrieval score above threshold, model generates with context.
3. **RAG with poor context**: `retrieve → no_context` — score below
   threshold, hard gate returns "not enough context" instead of
   hallucinating.

Every response includes a `[nodes: ...]` trace showing which nodes fired
and the retrieval confidence score — making the graph's behavior
observable for debugging and demo walkthroughs.

## Talking points for Phase 3

- Why one graph handles both plain chat and RAG: instead of separate code
  paths, the retrieve node returns empty docs when no `project_name` is
  set, and the route edge skips straight to generate. This means the
  graph is the single source of truth for all chat behavior — no risk of
  the two paths diverging. (DECISIONS.md #10)
- Why SqliteSaver over MemorySaver or PostgresSaver: MemorySaver is what
  we already had (lost on restart, no improvement). PostgresSaver needs a
  running database server (overkill for a demo). SqliteSaver gives
  persistence with zero infrastructure — and the checkpointer is a
  one-line swap, so moving to Postgres later requires zero code changes
  to the graph. (DECISIONS.md #11)
- Why long-term facts are separate from the checkpointer: the
  checkpointer owns per-session history (short-term, scoped to a
  thread_id). Facts own cross-session knowledge (long-term, scoped to a
  project_name). Different lifecycles, different access patterns, clean
  separation. This distinction matters: "the app remembers the last few
  messages" is not the same as "the app has memory." (DECISIONS.md #12)
- Why a hard gate instead of a soft gate for the routing guardrail: a
  soft gate (generate with a disclaimer) hopes the model will
  self-regulate with bad context — exactly what small models are worst
  at. A hard gate is decisive and predictable: below threshold, no
  generation, clear message. The threshold is configurable and the score
  is logged on every request for tuning. (DECISIONS.md #13)
- Why `astream_events()` over `graph.stream()`: Phase 1 established
  streaming as the response shape. `graph.stream()` only returns full
  node outputs — the user would wait for the entire response. We chose
  the streaming approach that preserved the UX rather than the simpler
  one that would have regressed it. (DECISIONS.md #14)

## Phase 4 notes

Phase 4 adds post-generation output validation — the Guardrails AI pattern
(validate → re-ask with error context) implemented manually because the
`guardrails-ai` package is incompatible with our dependency stack.

New/changed modules:
- `app/validate.py` — Pydantic `ChatResponse` schema (answer + sources),
  structured JSON parsing with `try_parse_structured()`, plain-text
  citation checking with `check_source_citations()`, and the two-tier
  validation flow: `validate_structured()` (Tier 1: JSON re-ask) and
  `validate_plaintext()` (Tier 2: citation check).
- `app/graph.py` — new `validate` node wired between `generate` and
  `extract_facts`. Runs a cost-aware check order: parse existing response
  as JSON (free) → check citations (free) → structured re-ask (expensive,
  only if both free checks fail).
- `app/main.py` — streaming endpoint now shows `⏳ Validating...` after
  tokens finish, followed by `✓ Validated` / `✓ Re-validated` /
  `⚠ Could not verify citations`. Sync endpoint returns `validation_status`.

Updated graph paths:
1. **Plain chat**: `retrieve(skip) → generate → validate(skip) → extract_facts`
2. **RAG, good context**: `retrieve → generate → validate → extract_facts`
3. **RAG, poor context**: `retrieve → no_context`

No new dependencies — Phase 4 uses only `pydantic` (already installed).

## Talking points for Phase 4

- Why manual implementation instead of the guardrails-ai package: two hard
  incompatibilities — `guardrails-ai` requires `langchain-core>=1.0.0`
  (would cascade-break Phases 1–3) and pulls in `litellm` which needs
  Rust/Cargo. The pattern itself is simple: validate → re-ask with error
  context → retry. The value is in demonstrating the pattern, not in
  having the library do it. (DECISIONS.md #16)
- Why tiered validation with a cost-aware check order: small models are
  inconsistent at structured JSON. Most responses already cite sources in
  natural language, so the cheap plain-text check usually passes. The
  expensive structured re-ask only fires when the response has no
  citations at all — avoiding unnecessary LLM calls for formatting
  failures. (DECISIONS.md #17)
- Why two distinct guardrail patterns are kept separate: the routing
  guardrail (pre-generation, Phase 3) asks "do we have enough context?"
  The output validation (post-generation, Phase 4) asks "did the model
  cite what it retrieved?" Different questions, different places in the
  graph, deliberate separation. (DECISIONS.md #17)
- Why the "thinking mode" streaming UX: tokens stream in real time (low
  perceived latency), then a validation indicator appears — the user
  gets both immediate responsiveness and a confidence signal. The
  alternative (wait for validation before showing anything) would
  regress the streaming UX established in Phase 1. (DECISIONS.md #19)
