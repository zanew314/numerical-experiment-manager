"""Collect Python source files for LLM analysis.

Text/hash based only: no AST and no dynamic probing. ``main.py`` and config
files are surfaced first so the agent can build an initial summary quickly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".nems",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "site-packages",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    ".demo_workspace",
    ".eggs",
    "third_party",
}

PRIORITY_NAMES = [
    "main.py",
    "config.py",
    "configs.py",
    "configuration.py",
    "settings.py",
    "train.py",
    "run.py",
    "model.py",
    "models.py",
    "utils.py",
]


@dataclass
class CodeFile:
    """A single collected Python source file."""

    path: str
    rel_path: str
    content: str
    size: int
    sha256: str


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class CodeReader:
    """Recursively read ``.py`` files from a project root."""

    def __init__(self, root, exclude_dirs: Optional[Iterable[str]] = None):
        self.root = Path(root).resolve()
        self.exclude_dirs = set(exclude_dirs or EXCLUDE_DIRS)

    def _priority_key(self, rel_path: str):
        parts = Path(rel_path).parts
        name = parts[-1].lower()
        if name in PRIORITY_NAMES:
            return (0, PRIORITY_NAMES.index(name), len(parts), rel_path)
        return (1, 0, len(parts), rel_path)

    def _iter_paths(self) -> List[Path]:
        found: List[Path] = []
        for path in self.root.rglob("*.py"):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(self.root).parts
            if any(part in self.exclude_dirs for part in rel_parts[:-1]):
                continue
            if rel_parts and rel_parts[-1] in self.exclude_dirs:
                continue
            if any(part.startswith(".") and part not in (".",) for part in rel_parts[:-1]):
                if rel_parts[-2] in (".nems",):
                    continue
            found.append(path)
        return found

    def scan(self, max_files: Optional[int] = None) -> List[CodeFile]:
        """Return collected files, priority-ordered (main/config first)."""
        paths = self._iter_paths()
        rels = [str(p.relative_to(self.root)).replace("\\", "/") for p in paths]
        order = sorted(range(len(paths)), key=lambda i: self._priority_key(rels[i]))
        files: List[CodeFile] = []
        for idx in order:
            path = paths[idx]
            rel = rels[idx]
            try:
                content = path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            files.append(
                CodeFile(
                    path=str(path),
                    rel_path=rel,
                    content=content,
                    size=len(content.encode("utf-8", errors="replace")),
                    sha256=hash_text(content),
                )
            )
        if max_files is not None:
            files = files[:max_files]
        return files

    def read_map(self, max_files: Optional[int] = None) -> Dict[str, str]:
        """Return a mapping ``rel_path -> content`` in priority order."""
        return {f.rel_path: f.content for f in self.scan(max_files=max_files)}

    def hashes(self) -> Dict[str, str]:
        """Return a mapping ``rel_path -> sha256``."""
        return {f.rel_path: f.sha256 for f in self.scan()}

    def read_matching(self, rel_paths: Iterable[str]) -> Dict[str, str]:
        """Read only the requested relative paths that exist."""
        out: Dict[str, str] = {}
        for rel in rel_paths:
            path = (self.root / rel).resolve()
            if not path.is_file():
                continue
            try:
                out[rel] = path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
        return out
