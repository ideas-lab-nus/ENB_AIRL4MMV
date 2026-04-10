"""Repository-local path helpers for the standalone AIRL4MMV release."""

from __future__ import annotations

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """Return the root of the standalone AIRL4MMV repository."""
    return PROJECT_ROOT


def resolve_repo_path(path: PathLike) -> Path:
    """Resolve a potentially relative path against the repository root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def ensure_repo_dir(path: PathLike) -> Path:
    """Create a repository-relative directory if needed and return its path."""
    resolved = resolve_repo_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
