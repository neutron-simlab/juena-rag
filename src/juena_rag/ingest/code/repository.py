"""Getting a repository onto disk, and deciding which of its files to index.

Everything here is git and the filesystem -- no chunking, no embedding, no
database. Carried over from the chatbot's ``repo_manager.py`` and ``repo_config.py``
with the manifest and diff machinery removed, since this pipeline always re-indexes
a whole repository rather than the files that changed.

The ``repositories.yaml`` belongs to whoever runs the ingest -- juena-lab keeps its
own -- and it decides how the files are split. juena-rag reads it and holds no
chunk size of its own.
"""

import fnmatch
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the install, not the code
    raise ImportError(
        "reading repositories.yaml needs `pyyaml`, which is not installed. "
        "`pip install 'juena-rag[ingest]'`."
    ) from None

# Sensible for a Python project, and overridden per repository in the YAML.
DEFAULT_INCLUDE = ["**/*.py", "**/*.md", "**/*.rst", "**/*.txt"]
DEFAULT_EXCLUDE = [
    "**/node_modules/**",
    "**/.git/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/dist/**",
    "**/build/**",
    "**/*.egg-info/**",
]

#: 512 kB. Bigger files are generated, vendored or data -- not something anybody
#: asks a question about, and one of them can be a quarter of a repository.
DEFAULT_MAX_FILE_BYTES = 524_288


@dataclass
class RepoSource:
    """Where a repository is cloned from."""

    url: str
    branch: str = "main"


@dataclass
class RepoConfig:
    """One repository, and the rules for which of its files get indexed.

    The same field names the chatbot's ``config/repositories.yaml`` uses, so an
    entry can be copied between the two files unchanged.

    ``chunk_size`` and ``chunk_overlap`` have no default on purpose. They decide
    every chunk boundary, so the caller's config states them; a default here would
    quietly win whenever someone forgot, and nobody would notice until search
    results changed.
    """

    id: str
    name: str
    chunk_size: int
    chunk_overlap: int
    description: str = ""
    source: RepoSource = field(default_factory=lambda: RepoSource(url=""))
    include_globs: list[str] = field(default_factory=lambda: list(DEFAULT_INCLUDE))
    exclude_globs: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    docs_paths: list[str] = field(default_factory=list)


def load_repositories(path: Path) -> dict[str, RepoConfig]:
    """Read ``repositories.yaml`` into one :class:`RepoConfig` per entry, by id."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    configs: dict[str, RepoConfig] = {}
    for entry in raw.get("repositories", []):
        source = entry.get("source", {})
        if not source.get("url"):
            raise ValueError(f"repository {entry.get('id', '?')!r} names no source.url")
        for setting in ("chunk_size", "chunk_overlap"):
            if setting not in entry:
                raise ValueError(
                    f"repository {entry.get('id', '?')!r} sets no {setting}. The "
                    "config that runs the ingest decides how files are split; "
                    "juena-rag has no default to fall back on."
                )

        config = RepoConfig(
            id=entry["id"],
            name=entry.get("name", entry["id"]),
            chunk_size=entry["chunk_size"],
            chunk_overlap=entry["chunk_overlap"],
            description=entry.get("description", ""),
            source=RepoSource(url=source["url"], branch=source.get("branch", "main")),
            include_globs=entry.get("include_globs", list(DEFAULT_INCLUDE)),
            exclude_globs=entry.get("exclude_globs", list(DEFAULT_EXCLUDE)),
            max_file_bytes=entry.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES),
            docs_paths=entry.get("docs_paths", []),
        )
        if config.id in configs:
            raise ValueError(f"repository id {config.id!r} appears twice")
        configs[config.id] = config

    return configs


# --------------------------------------------------------------------------- #
# Git
# --------------------------------------------------------------------------- #


def _git(*arguments: str) -> str:
    """Run one git command and return its output, raising with stderr if it fails.

    Raising rather than returning a code, because every caller here would have to
    turn a non-zero exit into an exception anyway.
    """
    # `check=False` on purpose: the raise below carries git's own stderr, which
    # `CalledProcessError` does not.
    result = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`git {' '.join(arguments)}` failed with exit {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def sync(config: RepoConfig, cache_dir: Path) -> Path:
    """Clone the repository, or bring an existing clone up to date. Returns its root.

    Shallow (``--depth 1``) because nothing here reads history -- only the files at
    the tip and the sha naming them.
    """
    root = cache_dir / config.id

    if (root / ".git").is_dir():
        _git("-C", str(root), "remote", "set-url", "origin", config.source.url)
        _git("-C", str(root), "fetch", "--depth", "1", "origin", config.source.branch)
        _git("-C", str(root), "reset", "--hard", f"origin/{config.source.branch}")
        return root

    if root.exists():
        # A directory that is not a clone -- a half-finished download, or an
        # extracted tarball. Cloning into it would fail, so it goes.
        shutil.rmtree(root) if root.is_dir() else root.unlink()

    root.parent.mkdir(parents=True, exist_ok=True)
    _git(
        "clone",
        "--depth",
        "1",
        "--branch",
        config.source.branch,
        config.source.url,
        str(root),
    )
    return root


def current_revision(root: Path) -> str:
    """The sha of whatever is checked out. This is a source's ``revision``."""
    return _git("-C", str(root), "rev-parse", "HEAD").strip()


# --------------------------------------------------------------------------- #
# Which files, and which of them are documentation
# --------------------------------------------------------------------------- #


def _matches(rel_path: str, pattern: str) -> bool:
    """Match a path against one glob, treating ``**/`` as *zero* or more directories.

    ``fnmatch`` reads ``**/*.py`` as requiring at least one directory, so a
    top-level ``setup.py`` would not match. Retrying without the prefix is what
    makes the pattern mean what everybody expects it to.
    """
    if fnmatch.fnmatch(rel_path, pattern):
        return True
    if pattern.startswith("**/"):
        return _matches(rel_path, pattern[3:])
    return False


def wanted(config: RepoConfig, rel_path: str) -> bool:
    """Does this path pass the include/exclude rules? Exclusions win."""
    if any(_matches(rel_path, pattern) for pattern in config.exclude_globs):
        return False
    return any(_matches(rel_path, pattern) for pattern in config.include_globs)


def is_doc(config: RepoConfig, rel_path: str) -> bool:
    """Is this file documentation rather than source?

    An entry in ``docs_paths`` is either an exact file or a directory prefix, so
    ``doc/`` covers everything beneath it and ``README.rst`` covers only itself.
    """
    for entry in config.docs_paths:
        if rel_path == entry or rel_path.startswith(entry.rstrip("/") + "/"):
            return True
    return False


def list_files(config: RepoConfig, root: Path) -> list[str]:
    """Every repo-relative path worth indexing, sorted.

    Sorted so two runs over the same revision produce the same ``chunk_index`` for
    the same chunk -- ``os.walk`` returns whatever order the filesystem gives it.
    """
    found: list[str] = []
    for directory, _subdirectories, filenames in os.walk(root):
        for filename in filenames:
            path = Path(directory) / filename
            rel_path = str(path.relative_to(root))

            if not wanted(config, rel_path):
                continue
            try:
                if path.stat().st_size > config.max_file_bytes:
                    continue
            except OSError:
                # A broken symlink, or a file that vanished mid-walk.
                continue
            found.append(rel_path)

    found.sort()
    return found


def read_file(root: Path, rel_path: str) -> str:
    """Read one file as text, refusing to read outside the repository.

    ``errors="replace"`` rather than strict: a stray non-UTF-8 byte in one file of
    two hundred should cost that file's characters, not the whole ingest.
    """
    path = (root / rel_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{rel_path!r} resolves outside the repository")
    return path.read_text(errors="replace")
