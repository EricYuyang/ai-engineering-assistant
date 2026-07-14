"""
Persisted state for incremental re-ingestion (DECISIONS.md #5) and
multi-repo project grouping (DECISIONS.md #4 extension).

Both are plain JSON files rather than a database -- proportionate to what
this needs to track (a handful of repos, a handful of projects), and easy
to inspect by hand during a walkthrough.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import settings


def _manifest_path(collection_name: str) -> Path:
    return Path(settings.manifests_dir) / f"{collection_name}.json"


def load_file_manifest(collection_name: str) -> dict[str, str]:
    """relative file path -> content hash, as of the last ingestion."""
    path = _manifest_path(collection_name)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_file_manifest(collection_name: str, manifest: dict[str, str]) -> None:
    path = _manifest_path(collection_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def diff_manifest(
    old: dict[str, str], current: dict[str, str]
) -> tuple[list[str], list[str], list[str]]:
    """Returns (new_files, changed_files, deleted_files), all relative paths."""
    old_paths = set(old)
    current_paths = set(current)

    new_files = sorted(current_paths - old_paths)
    deleted_files = sorted(old_paths - current_paths)
    changed_files = sorted(
        path for path in (current_paths & old_paths) if current[path] != old[path]
    )
    return new_files, changed_files, deleted_files


def load_projects() -> dict[str, list[str]]:
    """project_name -> [collection_name, ...]"""
    path = Path(settings.projects_manifest_path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_projects(projects: dict[str, list[str]]) -> None:
    path = Path(settings.projects_manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(projects, indent=2, sort_keys=True), encoding="utf-8")


def add_repo_to_project(project_name: str, collection_name: str) -> None:
    projects = load_projects()
    repos = projects.setdefault(project_name, [])
    if collection_name not in repos:
        repos.append(collection_name)
    save_projects(projects)


def get_project_collections(project_name: str) -> list[str]:
    return load_projects().get(project_name, [])
