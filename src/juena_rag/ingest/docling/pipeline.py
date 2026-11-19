"""Turn one rag-lab HybridChunker export into a book source in the store.

    juena-rag ingest-docling \\
        --raw-chunks data/output/5a_Crystallography_MZ/5a_Crystallography_MZ_raw_chunks.json \\
        --docling-json data/output/5a_Crystallography_MZ/5a_Crystallography_MZ.json \\
        --title "Neutron Scattering" --ref-chapter 5a

    juena-rag ingest-docling-csv use-cases/lecture-notes/books.csv

The second form ingests every export a CSV lists, one row each -- see
:mod:`juena_rag.ingest.docling.manifest` for its columns.

One export is one source, described by two files. The ``*_raw_chunks.json``
gives the chunks and the source's id -- the ``source`` every record carries --
and its content hash is the ``revision``, so re-running this on an unchanged
export is a no-op. The docling JSON it was chunked from gives the rest: the
chapter's title and authors as rag-lab recorded them, the PDF it was parsed
from, and the DoclingDocument version. The command line lives in
:mod:`juena_rag.service.cli`; this module is the work it calls.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from juena_rag.ingest.docling.chunking import adapt_chunks, load_raw_chunks
from juena_rag.ingest.docling.manifest import BOOK_INPUTS, load_manifest
from juena_rag.schema.boundary import ChunkBatch
from juena_rag.schema.models import SourceMetadata

#: Every chunk's ``language``: the books this pipeline reads are in English.
LANGUAGE = "en"


def _content_hash(path: Path) -> str:
    """A stable revision: the export's own bytes, not a timestamp.

    Re-running the chunker without changing the source document reproduces
    the same bytes, so this is what lets ``replace_source`` be a no-op then.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _source_id(records: list[dict[str, Any]], document: dict[str, Any]) -> str:
    """The export's name: the one ``source`` every record carries.

    Checked against the docling JSON's ``name`` as well, so raw chunks paired
    with another chapter's docling JSON are refused rather than stored under
    that chapter's authors and PDF.
    """
    sources = sorted({record["source"] for record in records})
    if len(sources) != 1:
        raise ValueError(
            f"expected every record to name the same one source, found {sources}; "
            "one *_raw_chunks.json is one source"
        )
    source_id = sources[0]
    if document.get("name") != source_id:
        raise ValueError(
            f"the raw chunks come from {source_id!r} but the docling JSON is "
            f"{document.get('name')!r}; pass the docling JSON they were chunked from"
        )
    return source_id


def build_batch(
    raw_chunks_path: Path,
    docling_json_path: Path,
    *,
    title: str | None = None,
    book: dict[str, Any] | None = None,
    contextualize: bool = True,
) -> ChunkBatch:
    """Load one export and build the batch to store it as.

    Args:
        raw_chunks_path: The HybridChunker's export.
        docling_json_path: The docling JSON it was chunked from, the
            ``<name>.json`` beside it.
        title: The book's title -- always the book's. The chapter's own title
            comes from the docling JSON, as every chunk's ``chapter_title``.
        book: What only the caller knows about the book -- any of
            :data:`BOOK_INPUTS`, copied onto every chunk.
        contextualize: See
            :func:`juena_rag.ingest.docling.chunking.adapt_record`.

    Returns:
        The batch, ready for ``Store.replace_source``.

    Raises:
        ValueError: If ``book`` sets a field that is not in
            :data:`BOOK_INPUTS`, or the two files are not one export.
    """
    book = book or {}
    unknown = sorted(book.keys() - set(BOOK_INPUTS))
    if unknown:
        raise ValueError(
            f"`book` takes only {list(BOOK_INPUTS)}, not {unknown}; every other "
            "field is read from the raw chunks or the docling JSON"
        )

    records = load_raw_chunks(raw_chunks_path)
    document = json.loads(docling_json_path.read_text(encoding="utf-8"))
    origin = document.get("origin") or {}
    binary_hash = origin.get("binary_hash")

    chunks = adapt_chunks(
        records,
        {
            "source_uri": origin.get("filename"),
            "language": LANGUAGE,
            "chapter_title": document.get("title"),
            **book,
        },
        contextualize=contextualize,
    )
    source = SourceMetadata(
        source_id=_source_id(records, document),
        category="book",
        revision=_content_hash(raw_chunks_path),
        title=title,
        authors=document.get("authors") or [],
        # The file's name only, not where this machine happens to keep it.
        artifact_uri=docling_json_path.name,
        artifact_format="docling-json",
        artifact_hash=None if binary_hash is None else str(binary_hash),
        parser=f"docling/{document['version']}",
    )
    return ChunkBatch(source=source, chunks=chunks, mode="replace")


def ingest_docling_export(
    raw_chunks_path: Path,
    docling_json_path: Path,
    *,
    title: str | None = None,
    book: dict[str, Any] | None = None,
    contextualize: bool = True,
    force: bool = False,
) -> int:
    """Store one HybridChunker export as a book source, embedding every chunk.

    Args:
        raw_chunks_path: The HybridChunker's export.
        docling_json_path: See :func:`build_batch`.
        title: See :func:`build_batch`.
        book: See :func:`build_batch`.
        contextualize: See :func:`build_batch`.
        force: Re-ingest even when the stored revision already matches.

    Returns:
        The process's exit code: 0 when done or already up to date, 1 when
        no database is configured.
    """
    dsn = _dsn()
    if not dsn:
        return 1

    batch = build_batch(
        raw_chunks_path,
        docling_json_path,
        title=title,
        book=book,
        contextualize=contextualize,
    )
    return _store(dsn, [batch], force=force)


def ingest_docling_manifest(
    manifest_path: Path, *, contextualize: bool = True, force: bool = False
) -> int:
    """Store every export a CSV lists, each as its own book source.

    Every row is built before anything is embedded, so a missing file or a bad
    cell in the last row stops the run before the first row costs an
    embedding call. Each source is then replaced on its own: a failure partway
    leaves the earlier ones stored, and re-running skips them.

    Args:
        manifest_path: The CSV; see :mod:`juena_rag.ingest.docling.manifest`.
        contextualize: See :func:`build_batch`.
        force: Re-ingest even the sources whose stored revision matches.

    Returns:
        The process's exit code: 0 when done or already up to date, 1 when
        no database is configured, 2 when any row cannot be built.
    """
    dsn = _dsn()
    if not dsn:
        return 1

    exports = load_manifest(manifest_path)
    batches: list[ChunkBatch] = []
    failures: list[str] = []
    for export in exports:
        try:
            batches.append(
                build_batch(
                    export.raw_chunks,
                    export.docling_json,
                    title=export.title,
                    book=export.book,
                    contextualize=contextualize,
                )
            )
        except (OSError, ValueError) as error:
            failures.append(f"  {export.raw_chunks.name}: {error}")

    if failures:
        print(
            f"{len(failures)} of {len(exports)} row(s) cannot be ingested, so "
            "nothing was stored:",
            *failures,
            sep="\n",
            file=sys.stderr,
        )
        return 2
    return _store(dsn, batches, force=force)


def _dsn() -> str | None:
    """The database to write to, or None after saying how to configure one."""
    # `JUENA_RAG_TEST_DSN` is only a fallback -- see ingest.code.pipeline for why.
    dsn = os.environ.get("JUENA_RAG_DSN") or os.environ.get("JUENA_RAG_TEST_DSN")
    if not dsn:
        print(
            "set JUENA_RAG_DSN to a database with the schema applied "
            "(see .env.example, and `docker compose up -d`)",
            file=sys.stderr,
        )
    return dsn


def _store(dsn: str, batches: list[ChunkBatch], *, force: bool) -> int:
    """Replace each batch's source, skipping those already at that revision."""
    from juena_rag.embedding.blablador import BlabladorEmbedder
    from juena_rag.store.pgvector import Store

    embedder = BlabladorEmbedder.from_env()

    with Store(dsn, embedder=embedder) as store:
        for batch in batches:
            source = batch.source
            print(f"  {source.source_id} at {source.revision}")
            stored = store.get_structure(source.source_id)
            if (
                stored is not None
                and stored.active_revision == source.revision
                and not force
            ):
                print("    already stored; nothing to do (--force to re-ingest)")
                continue

            print(
                f"    embedding {len(batch.chunks)} chunks with "
                f"{embedder.metadata.model} ..."
            )
            store.replace_source(batch)

        status = store.status()
        print(
            f"  store now holds {status.chunks} chunks across {status.sources} "
            f"source(s), embedded by {', '.join(status.embedding_models)}"
        )

    return 0
