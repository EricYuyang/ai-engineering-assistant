"""
Repo ingestion: resolve a source (local folder or GitHub URL) into a list
of chunked Documents ready for embedding.

See DECISIONS.md #1 (file filtering), #2 (chunking), #8 (clone storage)
for the reasoning behind the choices made here.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pathspec
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.config import settings

_EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".java": Language.JAVA,
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".hpp": Language.CPP,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".cs": Language.CSHARP,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".md": Language.MARKDOWN,
}


def is_github_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "git@"))


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug or "repo"


def repo_identifier(source: str) -> str:
    """Sanitized name + short hash of the source string. Used for both the
    Chroma collection name (Decision 4) and the clone folder name
    (Decision 8), so the two stay in lockstep for a given source."""
    if is_github_url(source):
        name = source.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[: -len(".git")]
    else:
        name = Path(source).name
    digest = hashlib.sha256(source.encode()).hexdigest()[:8]
    return f"{_slugify(name)}-{digest}"


def resolve_source(source: str) -> Path:
    """Return a local directory for the repo, cloning it first if `source`
    is a GitHub URL. Clones are persistent (Decision 8) so re-ingestion can
    `git pull` instead of re-cloning from scratch."""
    if not is_github_url(source):
        path = Path(source).resolve()
        if not path.is_dir():
            raise ValueError(f"Local path does not exist or is not a directory: {source}")
        return path

    dest = Path(settings.repos_dir).resolve() / repo_identifier(source)
    if dest.is_dir():
        subprocess.run(
            ["git", "-C", str(dest), "pull", "--ff-only"], check=True, capture_output=True, text=True
        )
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", source, str(dest)], check=True, capture_output=True, text=True
        )
    return dest


def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return None
    lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def collect_source_files(root: Path) -> list[Path]:
    """Walk the repo, applying Decision 1's filters in order: .gitignore,
    extension allowlist, file size cap, explicit directory excludes."""
    spec = _load_gitignore(root)
    files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)

        if any(part in settings.excluded_dir_names for part in rel.parts[:-1]):
            continue
        if spec is not None and spec.match_file(rel.as_posix()):
            continue
        if path.suffix.lower() not in settings.included_extensions:
            continue
        try:
            if path.stat().st_size > settings.max_file_size_bytes:
                continue
        except OSError:
            continue

        files.append(path)

    return files


def file_content_hash(path: Path) -> str:
    """Used by the re-ingestion manifest (Decision 5) to detect changed files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_near_empty(path: Path, min_chars: int = 20) -> bool:
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").strip()) < min_chars
    except OSError:
        return True


def chunk_file(root: Path, path: Path) -> list[Document]:
    """Split one file into Documents tagged with source_file metadata,
    using Decision 2's chunk size/overlap and language-aware splitting
    where the file extension has a known language."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel_path = path.relative_to(root).as_posix()

    language = _EXTENSION_TO_LANGUAGE.get(path.suffix.lower())
    if language is not None:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    return [
        Document(
            page_content=chunk,
            metadata={"source_file": rel_path, "chunk_index": i, "kind": "code"},
        )
        for i, chunk in enumerate(splitter.split_text(text))
    ]


def chunk_files(root: Path, paths: list[Path]) -> list[Document]:
    """Chunk a batch of files, skipping near-empty ones (Decision 7's map
    step reuses this same skip so trivial files never cost an LLM call)."""
    documents: list[Document] = []
    for path in paths:
        if is_near_empty(path):
            continue
        documents.extend(chunk_file(root, path))
    return documents
