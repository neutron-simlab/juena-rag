"""Turning a batch into rows -- the derivation, with no database in sight.

The ``chunks`` table has seventeen columns and :class:`~juena_rag.schema.models.Chunk`
has nine fields, none of them the embedding. This module is the twelve values in
between, written as pure functions so a test can check every one of them without
starting PostgreSQL.

Of those seventeen, exactly one costs anything: the embedding, which is a network
call and happens in :mod:`juena_rag.store.pgvector`. Everything here is a copy, a
constant, or a hash.
"""

from typing import Any

from juena_rag.schema.boundary import ChunkBatch
from juena_rag.schema.models import Chunk, SourceMetadata
from juena_rag.schema.vocabulary import (
    CHUNK_COLUMNS,
    SCHEMA_VERSION,
    compute_content_hash,
    derive_chunk_id,
)

#: The column order every insert uses. One tuple per chunk, built to match.
CHUNK_INSERT_COLUMNS = (
    "chunk_id",
    "source_id",
    "source_revision",
    "chunk_index",
    "category",
    "visibility",
    "language",
    "chunk_kind",
    "source_uri",
    "text",
    "content_hash",
    "extra",
    "embedding",
    "embedding_model",
    "schema_version",
)


def chunk_extra(chunk: Chunk) -> dict[str, Any]:
    """Everything on a chunk that has no column of its own.

    ``table``, ``equation``, ``heading_path`` and the line range are declared on
    :class:`~juena_rag.schema.models.Chunk` and none of them has a column, so the
    ones that *do* are listed in ``CHUNK_COLUMNS`` rather than derived -- deriving
    would drop those silently.
    """
    return chunk.model_dump(
        mode="json",  # jsonb wants JSON types, not Python ones
        exclude=set(CHUNK_COLUMNS),
        exclude_none=True,
    )


def chunk_row(
    source: SourceMetadata,
    chunk: Chunk,
    chunk_index: int,
    embedding: list[float],
    embedding_model: str,
) -> dict[str, Any]:
    """One chunk's row, ready to insert. ``ingested_at`` is the database's.

    ``category`` and ``visibility`` come from the *source*, not the chunk: a chunk
    has no say in either, so a document cannot hold a paragraph that is readable
    by a different audience than the rest of it.
    """
    return {
        "chunk_id": derive_chunk_id(source.source_id, source.revision, chunk_index),
        "source_id": source.source_id,
        "source_revision": source.revision,
        "chunk_index": chunk_index,
        "category": source.category,
        "visibility": source.visibility,
        "language": chunk.language,
        "chunk_kind": chunk.chunk_kind,
        "source_uri": chunk.source_uri,
        "text": chunk.text,
        "content_hash": compute_content_hash(chunk.text),
        "extra": chunk_extra(chunk),
        "embedding": embedding,
        "embedding_model": embedding_model,
        "schema_version": SCHEMA_VERSION,
    }


def chunk_rows(
    batch: ChunkBatch,
    embeddings: list[list[float]],
    embedding_model: str,
) -> list[dict[str, Any]]:
    """Every row of a batch, with ``chunk_index`` taken from list position.

    The caller never supplies ``chunk_index``; it is where the chunk sits in
    ``batch.chunks``. That is also why the vectors are matched positionally, and
    why the length check below is worth its two lines.
    """
    if len(embeddings) != len(batch.chunks):
        raise ValueError(
            f"{len(embeddings)} vectors for {len(batch.chunks)} chunks; they are "
            "matched by position, so an uneven pair would attach vectors to the "
            "wrong text without ever raising"
        )
    return [
        chunk_row(batch.source, chunk, index, embedding, embedding_model)
        for index, (chunk, embedding) in enumerate(
            zip(batch.chunks, embeddings, strict=True)
        )
    ]


def source_row(source: SourceMetadata) -> dict[str, Any]:
    """The `sources` row a batch declares. ``active_revision`` is moved separately.

    Registering a source and publishing a revision are two different moments --
    the second one is the end of ``replace_source``, once the chunks are in.
    """
    return {
        "source_id": source.source_id,
        "category": source.category,
        "title": source.title,
        "visibility": source.visibility,
        "artifact_uri": source.artifact_uri,
        "artifact_format": source.artifact_format,
        "artifact_hash": source.artifact_hash,
        "parser": source.parser,
    }
