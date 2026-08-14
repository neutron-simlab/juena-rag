"""The chunk schema.

A **chunk** is one piece of text that gets embedded, stored and searched as a
unit; a **source** is the document it came from. An ingestion pipeline builds a
``ChunkBatch`` -- one source and its chunks -- and hands it over; the store's
`sources` and `chunks` tables hold each one back as a ``SourceRow`` or a
``ChunkRow``.

Four modules, each importing only the ones above it:

===================== =========================================================
:mod:`vocabulary`     the closed vocabularies and the pure functions
:mod:`models`         ``SourceMetadata``, ``SourceRow``, ``Chunk``, ``ChunkRow``
:mod:`registry`       how a category is claimed, plus seven worked examples
:mod:`boundary`       ``ChunkBatch``
===================== =========================================================

To describe your own kind of document, subclass ``Chunk`` and claim a category::

    @register_chunk_class
    class SeminarChunk(Chunk):
        category: Literal["seminar"] = "seminar"
        speaker: str | None = None

This package validates; the caller transforms. Nothing here parses a PDF, splits
a table or decides where a chunk should end.
"""

from juena_rag.schema.boundary import ChunkBatch
from juena_rag.schema.models import (
    Chunk,
    ChunkRow,
    Equation,
    PageProvenance,
    SourceMetadata,
    SourceRow,
    Table,
)
from juena_rag.schema.registry import (
    CodeChunk,
    DatasetChunk,
    LayoutChunk,
    ManualChunk,
    PaperChunk,
    PresentationChunk,
    ReportChunk,
    WikiChunk,
    chunk_class,
    parse_chunk,
    register_chunk_class,
    registered_categories,
)
from juena_rag.schema.vocabulary import (
    CHUNK_COLUMNS,
    SCHEMA_VERSION,
    ArtifactFormat,
    ChunkKind,
    Visibility,
    compute_content_hash,
    derive_chunk_id,
)

__all__ = [
    "CHUNK_COLUMNS",
    "SCHEMA_VERSION",
    "ArtifactFormat",
    "Chunk",
    "ChunkBatch",
    "ChunkKind",
    "ChunkRow",
    "CodeChunk",
    "DatasetChunk",
    "Equation",
    "LayoutChunk",
    "ManualChunk",
    "PageProvenance",
    "PaperChunk",
    "PresentationChunk",
    "ReportChunk",
    "SourceMetadata",
    "SourceRow",
    "Table",
    "Visibility",
    "WikiChunk",
    "chunk_class",
    "compute_content_hash",
    "derive_chunk_id",
    "parse_chunk",
    "register_chunk_class",
    "registered_categories",
]
