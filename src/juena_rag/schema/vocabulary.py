"""The words the schema uses, and the pure functions that derive values.

Imports nothing else in this package, so it is the safest place to look first.
"""

import hashlib
from typing import Literal

#: Bumped when the meaning of a stored row changes. Written onto every chunk.
SCHEMA_VERSION = 1

#: Who may see a source. Closed on purpose, unlike ``category``: a typo in a free
#: string would match no entitlement check, hiding the chunk from everyone.
Visibility = Literal["public", "internal", "restricted"]

#: What the stored original is. The retrieval agent branches on it.
ArtifactFormat = Literal["docling-json", "markdown"]

#: The shape of a chunk's content -- a separate axis from ``category``, since a
#: table appears in papers, manuals and code READMEs alike.
#:
#: ``"table"`` covers a whole table and a window of its rows alike. How many rows
#: go in a chunk is a chunking decision, and chunking is the caller's job.
ChunkKind = Literal["text", "table", "equation", "code_block"]

#: The chunk fields with a real database column. Everything else goes to ``extra``.
#:
#: Listed explicitly rather than derived from ``Chunk.model_fields``: ``table``,
#: ``equation``, ``heading_path`` and the line range are declared on ``Chunk`` and
#: have no column, so deriving it would drop them silently.
CHUNK_COLUMNS = frozenset({"text", "source_uri", "language", "chunk_kind"})


def derive_chunk_id(source_id: str, source_revision: str, chunk_index: int) -> str:
    """A stable id for one chunk of one revision of one source.

    Deterministic, so re-ingesting the same input updates rows rather than
    duplicating them. The revision is part of it, which is what lets a new
    revision be written *beside* the old one.
    """
    # NUL-separated so ("ab", "c", 1) cannot collide with ("a", "bc", 1).
    joined = "\x00".join((source_id, source_revision, str(chunk_index)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def compute_content_hash(text: str) -> str:
    """A hash of a chunk's text, for detecting that content changed."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_range(
    start: int | None, end: int | None, start_name: str, end_name: str
) -> None:
    """Validation rule 3. Raise unless a start/end pair describes a range.

    A start alone is fine; an end alone is not, and neither is an end before its
    start. Used for lines on any chunk and character offsets on a code chunk.
    """
    if end is not None and start is None:
        raise ValueError(f"`{end_name}` is set but `{start_name}` is not")
    if start is not None and end is not None and start > end:
        raise ValueError(
            f"`{start_name}` ({start}) is after `{end_name}` ({end}) -- "
            "the range is empty"
        )
