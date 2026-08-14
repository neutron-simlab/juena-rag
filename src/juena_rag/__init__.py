"""juena-rag -- chunk schema, storage and retrieval for the JCNS knowledge base.

The package owns one thing: *a chunk, once it has been cleaned, gets stored,
versioned, searched and cited correctly.* Parsing and chunking stay with whoever
knows the document format.

The base install is pydantic alone; storage arrives with the ``[store]`` extra
and the web layer with ``[serve]``.
"""

from juena_rag.schema import (
    CHUNK_COLUMNS,
    SCHEMA_VERSION,
    ArtifactFormat,
    Chunk,
    ChunkBatch,
    ChunkKind,
    ChunkRow,
    CodeChunk,
    DatasetChunk,
    Equation,
    LayoutChunk,
    ManualChunk,
    PageProvenance,
    PaperChunk,
    PresentationChunk,
    ReportChunk,
    SourceMetadata,
    SourceRow,
    Table,
    Visibility,
    WikiChunk,
    chunk_class,
    compute_content_hash,
    derive_chunk_id,
    parse_chunk,
    register_chunk_class,
    registered_categories,
)

__version__ = "0.0.1"

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
    "__version__",
    "chunk_class",
    "compute_content_hash",
    "derive_chunk_id",
    "parse_chunk",
    "register_chunk_class",
    "registered_categories",
]
