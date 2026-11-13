"""Getting a repository onto disk, and deciding which of its files to index.

Everything here is git and the filesystem -- no chunking, no embedding, no
database. There is no change tracking: the pipeline always re-indexes the whole
repository, not only the files that changed.

The ``repositories.yaml`` belongs to whoever runs the ingest, and it decides how
the files are split. juena-rag reads it and holds no chunk size of its own.
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
    """Where a repository is cloned from.

    Attributes:
        url: The address to clone from.
        branch: The branch to index.
    """

    url: str
    branch: str = "main"


@dataclass
class RepoConfig:
    """One repository, and the rules for which of its files get indexed.

    The field names match juena-chatbot's ``config/repositories.yaml``, so an
    entry can be copied between the two files unchanged.

    ``chunk_size`` and ``chunk_overlap`` have no default: they decide every chunk
    boundary, and a default would quietly apply whenever a config forgot them.

    Attributes:
        id: The repository's id; becomes the source's ``source_id``.
        name: A readable name; becomes the source's title.
        chunk_size: The largest chunk, in characters.
        chunk_overlap: How many characters consecutive chunks may share.
        description: Free text about the repository.
        source: Where the repository is cloned from.
        include_globs: Which files to index.
        exclude_globs: Which files to skip; these win over ``include_globs``.
        max_file_bytes: Files bigger than this are skipped.
        docs_paths: Files or directories that hold documentation; see
            :func:`is_doc`.
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
    """Read a ``repositories.yaml``.

    Args:
        path: The file to read.

    Returns:
        One :class:`RepoConfig` per entry, keyed by id.

    Raises:
        ValueError: If an entry has no ``source.url``, ``chunk_size`` or
            ``chunk_overlap``, or an id appears twice.
    """
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
    """Run one git command and return its output, raising with stderr if it fails."""
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
    """Clone the repository, or bring an existing clone up to date.

    Shallow (``--depth 1``) because nothing here reads history -- only the files at
    the tip and the sha naming them.

    Args:
        config: The repository's settings.
        cache_dir: The directory that holds every clone.

    Returns:
        The clone's root directory.

    Raises:
        RuntimeError: If a git command fails.
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
    """Read the sha of whatever is checked out.

    Args:
        root: The clone's root directory.

    Returns:
        The sha, which becomes the source's ``revision``.

    Raises:
        RuntimeError: If git fails.
    """
    return _git("-C", str(root), "rev-parse", "HEAD").strip()


# --------------------------------------------------------------------------- #
# Which files, and which of them are documentation
# --------------------------------------------------------------------------- #


def _matches(rel_path: str, pattern: str) -> bool:
    """Match a path against one glob, treating ``**/`` as *zero* or more directories.

    ``fnmatch`` reads ``**/*.py`` as requiring at least one directory, so a
    top-level ``setup.py`` would not match; hence the retry without the prefix.
    """
    if fnmatch.fnmatch(rel_path, pattern):
        return True
    if pattern.startswith("**/"):
        return _matches(rel_path, pattern[3:])
    return False


def wanted(config: RepoConfig, rel_path: str) -> bool:
    """Check a path against the include and exclude rules. Exclusions win.

    Args:
        config: The repository's settings.
        rel_path: The file's path inside the repository.

    Returns:
        True if the file should be indexed.
    """
    if any(_matches(rel_path, pattern) for pattern in config.exclude_globs):
        return False
    return any(_matches(rel_path, pattern) for pattern in config.include_globs)


def is_doc(config: RepoConfig, rel_path: str) -> bool:
    """Tell whether a file is documentation rather than source.

    An entry in ``docs_paths`` is either an exact file or a directory prefix, so
    ``doc/`` covers everything beneath it and ``README.rst`` covers only itself.

    Args:
        config: The repository's settings.
        rel_path: The file's path inside the repository.

    Returns:
        True if an entry in ``docs_paths`` covers the file.
    """
    for entry in config.docs_paths:
        if rel_path == entry or rel_path.startswith(entry.rstrip("/") + "/"):
            return True
    return False


def list_files(config: RepoConfig, root: Path) -> list[str]:
    """List the files worth indexing.

    Sorted so two runs over the same revision produce the same ``chunk_index`` for
    the same chunk -- ``os.walk`` returns whatever order the filesystem gives it.

    Args:
        config: The repository's settings.
        root: The clone's root directory.

    Returns:
        Every repository-relative path that passes the rules and the size
        limit, sorted.
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

    Args:
        root: The clone's root directory.
        rel_path: The file's path inside the repository.

    Returns:
        The file's text.

    Raises:
        ValueError: If the path resolves outside the repository.
    """
    path = (root / rel_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{rel_path!r} resolves outside the repository")
    return path.read_text(errors="replace")
