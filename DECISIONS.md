# Design decisions log

Running record of the non-obvious choices made while building this project,
and *why* — kept separate from README.md (architecture/setup) and
CLAUDE.md (phase status) because this doc is about reasoning, not state.
Updated as we go, organized by phase.

---

## Phase 2: RAG over an ingested repo (local folder or GitHub URL)

### 1. Which files to include when ingesting a repo
**Decision:** Respect the repo's own `.gitignore` first, then apply an
explicit extension allowlist on top (`.py .js .jsx .ts .tsx .go .rs .java
.c .cpp .rb .php .cs .md .yaml .yml .toml`, etc.), then cap individual
file size (skip anything over ~1MB), then explicit directory excludes as a
backstop (`.git`, `node_modules`, `venv`/`.venv`, `__pycache__`, `dist`,
`build`, `target`).

**Why:**
- `.gitignore` already encodes the repo authors' own signal-vs-noise
  judgment (build output, deps, secrets, caches) — re-deriving that from
  scratch is wasted effort and error-prone.
- Not everything noisy is gitignored (e.g. vendored deps some ecosystems
  commit directly), so the extension allowlist is a second filter, applied
  on top rather than instead of `.gitignore`.
- Guiding question for every inclusion: "would this help explain the
  *architecture*, or is it noise for that goal?" (e.g. `package-lock.json`
  is source-controlled text but explains nothing about design → excluded.)
- File size cap catches large generated/checked-in files that dodge the
  extension filter (e.g. a big generated `.py`, a checked-in minified `.js`
  bundle).
- Directory excludes are belt-and-suspenders for repos with no
  `.gitignore` at all, or one that doesn't cover everything.

### 2. Chunk size, overlap, and splitting strategy
**Decision:** ~800 characters per chunk, ~150 characters overlap, using a
language-aware splitter that tries to break at function/class boundaries
first and only falls back to a hard character cut if a single logical unit
exceeds the limit.

**Why:**
- Too small a chunk (e.g. 200 chars) → semantically incomplete on its own
  (a function signature with no body, an `if` with no idea which function
  it's in). Precise match, but useless once retrieved.
- Too large a chunk (e.g. 3000 chars) → blends multiple unrelated
  functions into one embedding, diluting what that embedding actually
  represents and making similarity search less precise.
- `llama3.2:3b` has only a **4096-token context window** (confirmed via
  `ollama ps`). At ~800 chars (~200 tokens) per chunk, retrieving the top 4
  chunks costs ~800 tokens, leaving room for conversation history, system
  prompt, question, and generated answer. Bigger chunks would eat that
  budget fast.
- Overlap (~15-20% of chunk size) stitches back together things a hard
  cut would otherwise split (e.g. a docstring separated from its function
  body). Too much overlap just duplicates content across chunks for little
  benefit.
- Naive character-splitting doesn't understand code structure and will
  slice a function in half at an arbitrary point; a language-aware
  splitter avoids that by preferring logical boundaries.

### 3. Embedding model choice
**Decision:** `nomic-embed-text`, pulled via Ollama (same tool as the chat
model, no new runtime dependency).

**Why:**
- Purpose-built for embeddings — a chat model like `llama3.2:3b` could
  technically produce embeddings, but wasn't optimized for "similar
  meaning → similar vector," so a dedicated embedding model gives stronger
  similarity signal.
- Staying inside Ollama avoids a second local ML runtime (e.g.
  `sentence-transformers` would pull in PyTorch, a much heavier
  dependency). Keeps one toolchain, one seam for "how does this app talk
  to models" — consistent with `app/llm.py` being the only Ollama
  touchpoint so far.
- Small footprint (~274MB) and 8192-token context — far more than the
  ~200-token chunks we're producing, so no truncation risk, including for
  the longer analysis-doc chunks later.
- Considered `mxbai-embed-large` (~669MB, higher quality, but only a
  512-token context window) and briefly pulled it, but reverted to
  `nomic-embed-text` — the quality step-up wasn't worth it at this scale
  (single-repo, personal-scale ingestion).
- Honest limitation: none of Ollama's standard embedding models are
  code-specific (trained mostly on natural-language text). Works
  reasonably well for source because identifiers/docstrings/comments
  carry semantic signal, but pure algorithmic similarity between
  differently-named-but-equivalent code won't be captured as well as a
  code-specialized embedding model would.
- **Design implication:** chunking and embedding are kept as separate
  steps, with raw chunk text+metadata persisted independently from the
  embedding vectors. This means switching embedding models later only
  requires re-embedding already-chunked text, not re-walking/re-cloning/
  re-splitting the source repo.

### 4. Vector store collection strategy (per-repo isolation vs. shared)
**Decision:** one Chroma collection per ingested repo, named by a
sanitized identifier (slugified repo/folder name + short hash for
uniqueness), rather than one shared collection filtered by metadata.

**Why:**
- A shared collection + metadata filter has a silent-failure risk: every
  query has to remember to apply the `repo_name` filter. Miss it once in
  one code path and Repo A's question can retrieve and cite Repo B's
  chunks. Separate collections make that bug class impossible — there's
  nothing to search except the repo currently in scope.
- Lifecycle operations become a single primitive: deleting/re-ingesting a
  repo is "drop this collection," not a metadata-filtered delete that's
  easy to get subtly wrong.
- Matches actual usage: a chat session is about understanding *a*
  codebase, not searching across unrelated repos simultaneously.
- Directly sets up the README's existing "Stretch: multiple selectable
  knowledge bases" goal — per-repo collections make that "add a dropdown"
  rather than a re-architecture.
- Trade-off accepted: searching across *all* ingested repos at once needs
  querying multiple collections and merging results — more code than one
  shared query, but the right cost for this project's actual goal. (See
  the project-grouping extension below, which needed exactly this.)

**Extension (in scope for Phase 2): project grouping for multi-repo setups
(e.g. microservices).** A single "project" a user wants to understand can
span several repos (auth-service, billing-service, etc.) that only make
full sense together.

- Chroma collections stay at repo granularity (no change above) — Chroma
  has no native "collection of collections," and forcing that hierarchy
  into the vector store (e.g. one shared collection tagged by
  `service_name`) would reintroduce the cross-contamination risk this
  decision was written to avoid.
- Instead, add a lightweight **project manifest** at the application
  layer — just a mapping of `project_name -> [repo_collection_1,
  repo_collection_2, ...]`. A "project" is nothing more than a named list
  of repos.
- A chat session is scoped to a *project*, not a single repo. Retrieval
  queries every collection listed under that project and merges results
  before generation.
- Isolation is preserved: a bug can't leak data into an unrelated *other*
  project's repos, since grouping is just a manifest, not merged storage.
- Open wrinkle carried into Decision 6 (context budget/top-k): once
  merging results from N collections, need to decide top-k *per repo* vs.
  a single global ranking across all repos in the project.

### 5. Re-ingestion behavior (duplicate vs. skip vs. replace on re-run)
**Decision:** file-level incremental re-ingestion. Track each file's
content hash (e.g. SHA-256) in a manifest alongside chunk metadata. On
re-ingestion: unchanged files are skipped and their existing chunks
reused; changed files have their old chunks deleted (via Chroma's
metadata-filtered delete on `source_file`) and re-chunked/re-embedded;
deleted files have their orphaned chunks removed; new files are
chunked/embedded and added. Keep a "force full rebuild" escape hatch for
when the *ingestion logic itself* changes (e.g. chunk size), not just the
repo's content.

**Why:**
- Always-append (no dedup) is wrong: editing a function and re-ingesting
  would leave both the old and new chunk in the store, and a query could
  retrieve and cite the stale version. Gets worse with every re-run.
- Always-full-rebuild is correct but wasteful: re-embeds every file even
  if only 2 out of 200 changed. Embedding is the most compute-heavy step,
  and paying that cost for unchanged files wastes time on a laptop with
  no dedicated GPU headroom to spare.
- File-level incremental isn't much extra engineering: the project's own
  "cite sources" requirement already means every chunk needs `source_file`
  metadata. Content-hash tracking piggybacks on that — one more field in
  a manifest already being built, not a new subsystem.
- The "force full rebuild" escape hatch exists because incremental
  diffing only helps when the *repo* changed; it can't detect that we
  changed how we chunk/split, which needs a clean rebuild regardless.

### 6. Retrieval context budget (top-k chunks vs. the 4096-token ceiling)
**Decision:** top-k = 6 retrieved chunks per question (~1200 tokens at
~200 tokens/chunk). For multi-repo project queries, use global top-k
across all repos' merged results (rank by relevance regardless of source
repo) rather than forcing per-repo representation. Cap Phase 1's in-memory
conversation history to the last few exchanges as a stopgap, ahead of
Phase 3's proper memory rework. Fallback: if total tokens exceed budget
anyway, truncate conversation history from the oldest turn first.

**Why:**
- 4096-token budget (llama3.2:3b) has to cover: system prompt (~300),
  current question (~150), answer headroom (~800-1000 — otherwise
  responses get cut off mid-sentence), leaving ~1800-2000 tokens for
  retrieved chunks + history combined.
- top-k=6 gives real topical coverage (a function plus related context,
  not just one isolated match) while leaving room for history and the
  answer. First knob to revisit if answers feel under-informed in
  practice — not over-tuned blindly upfront.
- Global top-k (vs. per-repo guaranteed slots) for multi-repo projects:
  per-repo guarantees would waste budget forcing in irrelevant chunks from
  repos unrelated to the actual question. Global ranking lets relevance
  decide. Deliberately *not* building defensive per-repo minimums now —
  that's solving a problem (one repo's chunks systematically starving out
  another) that hasn't been confirmed to exist yet. Revisit only if
  observed in practice.
- Conversation history cap is a stopgap surfaced by this decision, not
  solved by it: `app/main.py`'s in-memory history has no cap today, and
  once retrieved chunks share the same token budget, uncapped history
  could silently overflow 4096 tokens before Phase 3 replaces history
  with LangGraph's checkpointer. Capping now avoids a silent breakage
  window between Phase 2 and Phase 3.
- Truncation fallback prioritizes the current question and retrieved
  chunks over older history, since those are what's most directly needed
  to answer the question being asked right now.

### 7. Map-reduce strategy for the auto-generated analysis doc
**Decision:** map at per-file granularity (one summary per file, falling
back to chunk-level summarize-then-combine only for outlier oversized
files), then reduce hierarchically using the repo's own folder structure
(per-file summaries → per-directory summary → final whole-repo doc,
recursing another level for deep nesting). Skip near-empty files in the
map step. The resulting doc is saved to `analysis/<repo-name>.md`, then
chunked and embedded into the same collection as the raw code.

**Why:**
- Map at the file level, not the chunk level: a chunk might be half a
  class, so summarizing it in isolation loses whole-file context. A file
  is a meaningful conceptual unit (usually one responsibility), and most
  files comfortably fit in 4096 tokens alongside a summarization prompt —
  so most files need exactly one LLM call, not several.
- Reduce via folder structure, not an arbitrary batching scheme: a flat
  "combine all file summaries in one pass" breaks down fast (50 files ×
  ~100 tokens/summary already exceeds budget on its own). Rather than
  inventing a generic batch-and-combine scheme from scratch, reuse the
  directory tree the repo's authors already created — it encodes real
  organizational intent (e.g. a `services/auth/` folder groups related
  files for a reason).
- Skipping near-empty files (e.g. blank `__init__.py`) avoids spending an
  LLM call summarizing "this file is empty."
- **Practical implication, not itself a decision:** map-reduce over even a
  medium repo means dozens of sequential local LLM calls (one local model
  instance, no parallel throughput) — realistically minutes, not seconds.
  This means `/ingest` can't be a simple synchronous request that blocks
  for minutes; it needs to kick off background work and let the caller
  poll status instead.

### 8. Where cloned GitHub repos live, and cleanup policy
**Decision:** clone into a dedicated folder inside the project itself
(`data/repos/<sanitized-name>-<short-hash>/`, same naming convention as
the collection names from Decision 4), added to `.gitignore`, and kept
persistently rather than deleted after ingestion.

**Why:**
- Inside the project, not system temp or a user-home cache dir: preserves
  the expectation established early on that deleting the
  `ai-engineering-assistant` folder removes everything except Ollama
  itself. A cache dir living outside the project would silently break
  that guarantee.
- Persistent, not deleted after ingestion — connects directly to Decision
  5: incremental re-ingestion needs a current copy of the repo's files to
  hash and diff against. Keeping the clone means re-ingestion is just
  `git pull` (cheap) + the hash-diff, instead of a full re-clone from
  scratch every time. It also makes GitHub-sourced repos behave exactly
  like local folders after the first clone, consistent with how the two
  source types were originally scoped to behave identically.
- Trade-off accepted: this folder grows unboundedly with no automatic
  pruning if many large repos get ingested over time. Acceptable for now
  since it's contained and visible; a "clean up old clones" utility would
  be a cheap addition later if disk usage actually becomes a problem —
  not worth building defensively before it's needed.

### 9. Dependency resolution snag: chromadb on Windows
**Decision:** use `chromadb==1.5.9` + `langchain-chroma==0.2.4`, and bump
`langchain` to `0.3.30` / `langchain-ollama` to `0.3.9` / `langchain-core`
to `0.3.86` (all still within their pre-1.0 `0.3.x`/`0.2.x` lines, not the
newer `langchain` 1.x rewrite).

**Why:**
- The obvious pick, `chromadb==0.5.23` (matching the version the README
  originally anticipated), failed to install on Windows: its pinned
  `chroma-hnswlib==0.7.6` has no prebuilt Windows wheel and requires
  Microsoft C++ Build Tools to compile from source — a large, invasive
  system install, exactly what we've been avoiding.
- `chromadb` 1.x ships its own precompiled Windows wheel and no longer
  requires building `chroma-hnswlib` for the default embedded client —
  confirmed by checking PyPI metadata directly rather than guessing.
- `langchain-chroma==1.1.0` (latest) pulled in `langchain-core` 1.x, which
  conflicts with the `langchain`/`langchain-ollama` versions Phase 1 was
  pinned to (`<0.4.0`). `langchain-chroma==0.2.4` is the version that
  bridges old `langchain-core` 0.3.x with new `chromadb` 1.x.
- That in turn needed `langchain-core>=0.3.76` (per `langchain-ollama`
  0.3.9's constraint), which needed a newer `langsmith` range, which
  needed `langchain` bumped to 0.3.30. All resolved versions stay within
  the same pre-1.0 minor line as originally pinned — verified by checking
  PyPI dependency metadata for each package before picking a version,
  rather than trial-and-error installing.
- Verified Phase 1's `/chat` endpoint still works after the bump, per the
  project rule of not breaking previous phases when starting a new one.

---

## Phase 3: LangGraph orchestration + memory

### 10. Graph structure — nodes, edges, and state

**Decision:** wrap the chat pipeline in a LangGraph `StateGraph` with four
nodes: `retrieve`, `generate`, `no_context`, and `extract_facts`. A
conditional edge after `retrieve` routes to either `generate` (confidence
above threshold) or `no_context` (below threshold). Plain chat (no
`project_name`) flows through the same graph — `retrieve` returns empty
docs and `route` skips straight to `generate`, so one graph handles both
RAG and non-RAG modes with no separate code paths.

**Why:**
- Single graph for all modes avoids code duplication and makes the flow
  visible in one place. The routing guardrail is a conditional edge, not
  an `if` buried in a helper function — it's declarative and shows up in
  graph visualization.
- Node trace (`retrieve(k=6, best=0.553) → generate → extract_facts(2)`)
  is returned with every response, making the graph's behavior observable
  for debugging and for demo/interview walkthroughs.
- The `no_context` node is a hard gate — it returns a canned message
  instead of letting the model generate with irrelevant context. This is
  the routing guardrail's whole value: prevent hallucination rather than
  hoping the model self-corrects.

### 11. Checkpointer choice — SqliteSaver for session persistence

**Decision:** use `SqliteSaver` (from `langgraph-checkpoint-sqlite`) backed
by a local `checkpoints.sqlite3` file. This replaces the in-memory
`_sessions` dict and the `max_history_messages` stopgap from Phase 2.

**Why:**
- MemorySaver (in-memory) would be identical to what we already had —
  lost on restart, no improvement.
- SqliteSaver gives persistence with zero infrastructure: sessions survive
  server restarts, which is the one meaningful upgrade for a single-user
  demo. The file sits next to `chroma_db/` in the project folder.
- PostgresSaver would require running a Postgres server — overkill for
  this project's scale.
- The checkpointer is a one-line swap (the graph doesn't know which
  backend is behind it), so moving to Postgres later is trivial. This is
  a clean talking point for interviews: same graph, different durability
  guarantees, zero code changes.

### 12. Long-term memory — separate fact store

**Decision:** extract factual statements from conversations and persist them
in a separate `facts.sqlite3` database, keyed by `project_name`. Facts are
injected as a system message during generation in subsequent sessions.

**Why:**
- The checkpointer handles short-term memory (conversation history within
  a session, per `thread_id`). Long-term memory (facts that should survive
  across sessions) is a different concern with a different lifecycle — a
  fact about "the auth service uses JWT" should be available in every
  future session about that project, not just the one where it was stated.
- Storing facts in the same SQLite as the checkpointer would conflate two
  access patterns: the checkpointer needs fast read/write of full state
  per thread, while facts need a simple query by project name across all
  threads.
- The extraction is done by the LLM itself: the `extract_facts` node asks
  the model to identify factual statements about the project from the
  recent conversation and return them as a JSON array. This is deliberately
  best-effort — a small local model won't catch everything, but even
  partial extraction is better than none.

### 13. Routing guardrail — confidence threshold and hard gate

**Decision:** use the best retrieval similarity score as a gate. If the
highest score among all retrieved chunks is below a configurable threshold
(`CONFIDENCE_THRESHOLD`, default 0.4), route to `no_context` (hard gate)
instead of `generate`.

**Why:**
- The score comes from Chroma's `similarity_search_with_relevance_scores`,
  which we were already calling but not using the score value. No new
  computation needed.
- Hard gate (skip generation entirely) rather than soft gate (generate
  with a disclaimer): the whole point is to prevent hallucination from a
  small model. A soft gate hopes the model will self-regulate with bad
  context, which is exactly what small models are worst at.
- Configurable threshold (env var + config.py) because the "right" value
  depends on the embedding model, the data, and the query distribution.
  0.4 is a conservative starting point; real tuning requires observing
  score distributions on actual queries. The node trace logs the score on
  every request to make this observable.
- Plain chat (no `project_name`) bypasses the check entirely — no
  retrieval was attempted, so there's no score to evaluate, and no
  grounding claim to protect.

### 14. Streaming from inside the graph

**Decision:** use `graph.astream_events()` (async, version="v2") to stream
individual LLM tokens from the `generate` node, filtered by
`event["metadata"]["langgraph_node"] == "generate"`. Also provide a
`/chat/sync` endpoint using `graph.ainvoke()` for testing and debugging.

**Why:**
- Phase 1 established streaming as the response shape (talking point: "the
  difference between 'the API works' and 'the API is usable'"). Losing
  token-level streaming when wrapping the LLM call in a graph node would
  be a UX regression.
- `graph.stream()` only returns full node outputs — the user would wait
  for the entire response before seeing anything. `astream_events()`
  exposes inner events including individual LLM token chunks.
- The `langgraph_node` metadata filter ensures we only yield tokens from
  `generate`, not from `extract_facts` (which also calls the LLM but
  whose output is internal).
- `/chat/sync` exists for debugging and testing — it returns the full
  result as JSON including response, sources, node trace, and confidence
  score in one object.

### 15. Dependency version — langgraph on langchain-core 0.3.x

**Decision:** use `langgraph==0.6.11` and `langgraph-checkpoint-sqlite==3.0.3`.
This is the last `langgraph` line that accepts `langchain-core>=0.1` without
requiring `>=1.4`. `langgraph` 1.x requires `langchain-core>=1.4.7`, which
would break `langchain-chroma==0.2.4` and the entire Phase 1+2 dependency
chain.

**Why:**
- Same constraint pattern as Decision 9: the `langchain-core` version is
  the fragile link. Everything else (langchain, langchain-chroma,
  langchain-ollama) is pinned to `0.3.x` of langchain-core, and upgrading
  to `1.x` would cascade-break them all.
- Verified via `pip install --dry-run` before committing to the version —
  same approach that saved time in Phase 2's dependency resolution.
- `langgraph 0.6.11` has the full API we need: `StateGraph`, conditional
  edges, `astream_events(version="v2")`, and the checkpoint interface.

---

## Phase 4: Guardrails AI — structured output validation

### 16. Why manual implementation instead of the guardrails-ai package

**Decision:** implement the Guardrails AI pattern (Pydantic schema validation
+ auto re-ask with error context) manually using plain Pydantic and custom
retry logic, rather than installing the `guardrails-ai` package.

**Why:**
- Two hard incompatibilities with the package:
  1. `guardrails-ai` requires `langchain-core>=1.0.0`. Our entire stack
     (Phases 1–3) is pinned to `langchain-core 0.3.x` — upgrading would
     cascade-break `langchain-chroma`, `langchain-ollama`, and `langgraph`.
     Same fragile link identified in Decisions 9 and 15.
  2. `guardrails-ai` pulls in `litellm`, which requires Rust/Cargo to
     compile native extensions. A toolchain install that's invasive,
     platform-specific, and unrelated to what we're actually building.
- The pattern itself is simple enough to reimplement: validate output →
  on failure, re-prompt with the error → retry up to N times. The value
  is in *demonstrating the pattern*, not in having the library do it.
- Interview talking point: "I evaluated the framework, hit two blocking
  incompatibilities, decided the pattern was simple enough to implement
  directly, and got the same behavior with zero new dependencies."

### 17. Tiered validation strategy (structured → plain-text fallback)

**Decision:** two-tier validation with a cost-aware check order:
1. Try to parse the existing response as structured JSON (free, no LLM call)
2. If not JSON, check plain-text for source citations (free, no LLM call)
3. Only if both fail, re-ask the LLM with a structured JSON prompt (expensive,
   up to `GUARDRAILS_MAX_RETRIES` attempts)

**Why:**
- Small models (3B parameters) are inconsistent at producing valid JSON.
  Forcing structured output every time would burn re-ask retries on
  formatting failures rather than content quality issues.
- The cost-aware order avoids unnecessary LLM calls: most responses from
  the generate node already cite source files in natural language, so the
  cheap plain-text check usually passes without any re-ask.
- If the response has no citations at all, the structured re-ask gives
  the model explicit instructions and error feedback, which is the
  Guardrails AI re-ask pattern in action.
- Keeps two distinct guardrail patterns visible (as the README calls out):
  the routing guardrail (pre-generation, Decision 13) and this output
  validation (post-generation). Different questions, different places.

### 18. Validate node placement in the graph

**Decision:** add a `validate` node between `generate` and `extract_facts`.
The graph edge changes from `generate → extract_facts` to
`generate → validate → extract_facts`.

**Why:**
- Validation must happen after generation (it needs the response to check)
  and before fact extraction (if validation replaces the response with a
  structured re-ask result, fact extraction should operate on the final
  validated response, not the original).
- Validation is skipped for plain chat (no RAG context = nothing to
  validate against), keeping the non-RAG path fast.
- The node updates `validation_status` in state, which is visible in the
  node trace — e.g., `validate(plaintext:citations-found)` or
  `validate(structured:fail(attempt=1),structured:fail(attempt=2),fallback:citations-missing)`.

### 19. Streaming UX — "thinking mode" validation indicator

**Decision:** stream tokens in real time, then show a "Validating..."
indicator while the validate node runs, followed by the result status.

Flow:
1. Tokens from `generate` stream to the client as they're produced
2. After generation completes: `⏳ Validating...`
3. Validation runs (cheap checks, potentially expensive re-ask)
4. Result: `✓ Validated` / `✓ Re-validated` / `⚠ Could not verify citations`

If validation re-asks successfully, the new structured response is appended.

**Why:**
- User's idea (not mine): "make it like thinking mode — stream the
  response, then add a Validating... after that." Smart UX instinct: the
  user sees the answer forming immediately (low perceived latency), then
  gets a confidence indicator as a visual cue.
- The alternative (wait for validation before showing anything) would
  regress the streaming UX that Phase 1 established and Phase 3 preserved.
- In the sync endpoint, `validation_status` is returned in the JSON
  response for programmatic consumption.
