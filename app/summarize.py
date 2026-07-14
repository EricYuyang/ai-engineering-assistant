"""
Auto-generated architecture doc via map-reduce summarization
(DECISIONS.md #7): map at per-file granularity, reduce hierarchically
following the repo's own folder structure, falling back to a generic
batch-and-combine when a directory's combined summaries are too large for
one pass.

Note: the doc is regenerated in full from every currently-tracked file on
each run, not patched incrementally the way raw code chunks are (Decision
5). Caching per-file summaries to make this incremental too would need a
second manifest -- more machinery than this project's scale (single small
repos) currently justifies.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.ingest import chunk_file, is_near_empty
from app.llm import get_chat_model
from app.vectorstore import add_documents

# Rough chars-per-token approximation, used only to keep combine prompts
# under budget -- doesn't need to be exact, just conservative.
_CHARS_PER_TOKEN = 4
_MAX_COMBINE_CHARS = 2000 * _CHARS_PER_TOKEN


def _summarize_file(llm, root: Path, path: Path) -> str | None:
    if is_near_empty(path):
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel_path = path.relative_to(root).as_posix()
    prompt = (
        "Summarize what this file does in 2-4 sentences, for a developer "
        "trying to understand the overall architecture of the project. "
        "Focus on its responsibility and how it likely fits with the rest "
        f"of the codebase, not line-by-line detail.\n\nFile: {rel_path}\n\n{text}"
    )
    response = llm.invoke(
        [
            SystemMessage(content="You write terse, architecture-focused summaries of source files."),
            HumanMessage(content=prompt),
        ]
    )
    return response.content


def _combine_call(llm, label: str, text: str) -> str:
    prompt = (
        f"Combine these summaries into one coherent summary of '{label}', "
        "for a developer trying to understand the project's architecture. "
        "Describe the overall responsibility of this part of the codebase "
        f"and how its pieces relate, in a short paragraph.\n\n{text}"
    )
    response = llm.invoke(
        [
            SystemMessage(content="You write terse, architecture-focused summaries."),
            HumanMessage(content=prompt),
        ]
    )
    return response.content


def _combine(llm, label: str, sections: list[str]) -> str:
    """Combine already-summarized sections into one summary for `label`
    (a directory or the repo root). Batches when the combined text would
    exceed the model's context budget (Decision 7's generic fallback)."""
    combined_text = "\n\n".join(sections)
    if len(combined_text) <= _MAX_COMBINE_CHARS:
        return _combine_call(llm, label, combined_text)

    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for section in sections:
        if current and current_len + len(section) > _MAX_COMBINE_CHARS:
            batches.append(current)
            current, current_len = [], 0
        current.append(section)
        current_len += len(section)
    if current:
        batches.append(current)

    batch_summaries = [
        _combine_call(llm, f"{label} (part {i + 1})", "\n\n".join(batch)) for i, batch in enumerate(batches)
    ]
    return _combine(llm, label, batch_summaries)


def _build_tree(rel_paths: list[str]) -> dict:
    tree: dict = {}
    for rel in rel_paths:
        parts = Path(rel).parts
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault("__files__", []).append(rel)
    return tree


def _summarize_tree(llm, tree: dict, file_summaries: dict[str, str], label: str) -> str:
    sections: list[str] = []
    for name, subtree in sorted(tree.items()):
        if name == "__files__":
            continue
        sections.append(f"[{name}/]\n{_summarize_tree(llm, subtree, file_summaries, name)}")

    for rel in sorted(tree.get("__files__", [])):
        summary = file_summaries.get(rel)
        if summary:
            sections.append(f"[{rel}]\n{summary}")

    if not sections:
        return "(no summarizable content)"

    return _combine(llm, label, sections)


def generate_analysis_doc(collection_name: str, root: Path, files: list[Path]) -> Path:
    """Map (per-file) then reduce (hierarchical, via folder structure) into
    one analysis doc, saved to analysis/<collection_name>.md, then chunked
    and embedded back into the same collection as the raw code."""
    llm = get_chat_model()

    file_summaries: dict[str, str] = {}
    for path in files:
        summary = _summarize_file(llm, root, path)
        if summary:
            file_summaries[path.relative_to(root).as_posix()] = summary

    tree = _build_tree(list(file_summaries))
    doc_body = _summarize_tree(llm, tree, file_summaries, root.name)
    doc_text = f"# Architecture overview: {root.name}\n\n{doc_body}\n"

    analysis_dir = Path(settings.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    doc_path = analysis_dir / f"{collection_name}.md"
    doc_path.write_text(doc_text, encoding="utf-8")

    chunks = chunk_file(analysis_dir, doc_path)
    for chunk in chunks:
        chunk.metadata["kind"] = "analysis_doc"
    add_documents(collection_name, chunks)

    return doc_path
