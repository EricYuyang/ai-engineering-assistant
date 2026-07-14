"""
End-to-end ingestion pipeline: resolve source -> diff against the last
manifest -> chunk+embed only what changed -> update the manifest ->
regenerate the analysis doc. Implements DECISIONS.md #5 (incremental
re-ingestion) end to end, with the escape hatch it calls for: `force=True`
drops the collection and re-ingests everything, for when the ingestion
logic itself changes rather than the repo's content.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.ingest import chunk_files, collect_source_files, file_content_hash, repo_identifier, resolve_source
from app.manifest import add_repo_to_project, diff_manifest, load_file_manifest, save_file_manifest
from app.summarize import generate_analysis_doc
from app.vectorstore import add_documents, delete_by_source_file, delete_collection


@dataclass
class IngestResult:
    collection_name: str
    repo_root: str
    new_files: list[str]
    changed_files: list[str]
    deleted_files: list[str]
    unchanged_files: int
    chunks_added: int
    analysis_doc_path: str


def ingest_repo(source: str, project_name: str | None = None, force: bool = False) -> IngestResult:
    root = resolve_source(source)
    collection_name = repo_identifier(source)
    # Every repo is at least its own single-repo "project" by default, so a
    # chat session can always target `project_name=collection_name` even
    # when the caller never explicitly grouped it with anything else.
    project_name = project_name or collection_name

    if force:
        delete_collection(collection_name)
        save_file_manifest(collection_name, {})

    all_files = collect_source_files(root)
    current_manifest = {f.relative_to(root).as_posix(): file_content_hash(f) for f in all_files}
    old_manifest = load_file_manifest(collection_name)

    new_files, changed_files, deleted_files = diff_manifest(old_manifest, current_manifest)

    for rel_path in changed_files + deleted_files:
        delete_by_source_file(collection_name, rel_path)

    to_chunk_rel = set(new_files) | set(changed_files)
    to_chunk_paths = [root / rel for rel in to_chunk_rel]
    documents = chunk_files(root, to_chunk_paths)
    add_documents(collection_name, documents)

    save_file_manifest(collection_name, current_manifest)

    doc_path = generate_analysis_doc(collection_name, root, all_files)

    add_repo_to_project(project_name, collection_name)

    return IngestResult(
        collection_name=collection_name,
        repo_root=str(root),
        new_files=new_files,
        changed_files=changed_files,
        deleted_files=deleted_files,
        unchanged_files=len(current_manifest) - len(new_files) - len(changed_files),
        chunks_added=len(documents),
        analysis_doc_path=str(doc_path),
    )
