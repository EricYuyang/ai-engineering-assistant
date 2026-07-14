"""
Centralised settings, loaded from environment variables / .env.

Kept deliberately dumb (no pydantic-settings) so it's easy to read at a
glance during an interview walkthrough: every setting the app depends on
lives in exactly one place.
"""
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

    # --- Phase 2: RAG over an ingested repo ---
    # See DECISIONS.md for why each of these values was picked.
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # Where per-repo Chroma collections are persisted (Decision 4).
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    # Where cloned GitHub repos and local-folder manifests live (Decision 8).
    repos_dir: str = os.getenv("REPOS_DIR", "data/repos")
    # Per-repo file-hash manifests and the project-grouping manifest (Decision 5, 4-ext).
    manifests_dir: str = os.getenv("MANIFESTS_DIR", "data/manifests")
    projects_manifest_path: str = os.getenv("PROJECTS_MANIFEST_PATH", "data/projects.json")
    # Where generated analysis docs are saved (Decision 7).
    analysis_dir: str = os.getenv("ANALYSIS_DIR", "analysis")

    # Chunking (Decision 2).
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # File filtering (Decision 1).
    max_file_size_bytes: int = int(os.getenv("MAX_FILE_SIZE_BYTES", str(1_000_000)))
    included_extensions: tuple[str, ...] = (
        ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
        ".c", ".h", ".cpp", ".hpp", ".rb", ".php", ".cs", ".swift",
        ".kt", ".md", ".yaml", ".yml", ".toml", ".json",
    )
    excluded_dir_names: tuple[str, ...] = (
        ".git", "node_modules", "venv", ".venv", "__pycache__",
        "dist", "build", "target", ".next", "vendor", ".idea", ".vscode",
    )

    # Retrieval (Decision 6).
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
    # Stopgap history cap ahead of Phase 3's proper memory system (Decision 6).
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))


settings = Settings()
