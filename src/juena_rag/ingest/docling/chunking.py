"""Adapt rag-lab's HybridChunker records into :class:`BookChunk` rows.

rag-lab's ``src/hybrid_chunker.py`` (built on docling-core) writes one JSON
object per chunk to ``data/output/<name>/<name>_raw_chunks.json``::

    chunk_id, source, headings, pages, doc_item_refs, doc_item_labels,
    n_tokens, text, contextualized_text

None of those names is a :class:`~juena_rag.schema.registry.BookChunk` field.
``headings`` is ``heading_path``; ``pages`` and ``doc_item_refs`` already
exist, by that name, on the :class:`~juena_rag.schema.registry.LayoutChunk`
that ``BookChunk`` inherits. ``chunk_id``, ``source``, ``n_tokens`` and
``doc_item_labels`` have no equivalent -- the store derives its own chunk id
from ``(source_id, revision, chunk_index)``, never counts tokens, and has no
field for docling's raw item labels. ``provenance`` stays an empty list: a
record says which pages a chunk is on, but not where on them.

The other way round, nothing in a record says what every chunk of the export
has in common -- the PDF it came from (``source_uri``), its ``language``, and
``BookChunk``'s own fields such as ``chapter_title`` and ``isbn``. The caller
states those once as ``shared`` and each chunk gets a copy. This module is the
one place that translation happens.
"""

import json
from pathlib import Path
from typing import Any

from juena_rag.schema.registry import BookChunk

#: The ``BookChunk`` fields a record fills. ``shared`` may not set them too.
_FROM_RECORD = frozenset({"text", "heading_path", "pages", "doc_item_refs"})


def load_raw_chunks(path: Path) -> list[dict[str, Any]]:
    """Read one ``*_raw_chunks.json`` written by rag-lab's HybridChunker.

    Args:
        path: The file, e.g. ``data/output/<name>/<name>_raw_chunks.json``.

    Returns:
        The chunk records, in the chunker's own order.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def adapt_record(
    record: dict[str, Any],
    shared: dict[str, Any] | None = None,
    *,
    contextualize: bool = True,
) -> BookChunk:
    """Turn one HybridChunker record into a :class:`BookChunk`.

    Args:
        record: One element of a ``*_raw_chunks.json`` file.
        shared: The fields every chunk of this export has in common, e.g.
            ``{"source_uri": "5a_Crystallography_MZ.pdf", "language": "en",
            "chapter_title": "Crystallography", "ref_chapter": "5a"}``.
            Validated by ``BookChunk`` itself, so a misspelt key is refused
            rather than dropped.
        contextualize: Store ``contextualized_text`` (the heading path
            prepended to the text) rather than the bare ``text``.
            ``hybrid_chunker.py`` computes it for exactly this purpose: a
            paragraph on its own and the same paragraph under "5a.4
            Symmetry operations" should not embed identically, and
            ``Store.replace_source`` embeds ``chunk.text`` exactly as given,
            with no contextualizing step of its own.

    Returns:
        The chunk.

    Raises:
        ValueError: If ``shared`` sets a field the record already fills.
    """
    shared = shared or {}
    clash = sorted(_FROM_RECORD & shared.keys())
    if clash:
        raise ValueError(
            f"`shared` sets {clash}, which every record fills on its own; "
            "`shared` is only for what all chunks of the export have in common"
        )
    return BookChunk.model_validate(
        {
            **shared,
            "text": record["contextualized_text"] if contextualize else record["text"],
            "heading_path": record.get("headings") or [],
            "pages": record.get("pages") or [],
            "doc_item_refs": record.get("doc_item_refs") or [],
        }
    )


def adapt_chunks(
    records: list[dict[str, Any]],
    shared: dict[str, Any] | None = None,
    *,
    contextualize: bool = True,
) -> list[BookChunk]:
    """Adapt every record of one ``*_raw_chunks.json`` file, in order.

    The order is kept because the store numbers chunks by their position in
    the batch: record ``<name>-0002`` becomes ``chunk_index`` 2.

    Args:
        records: The chunker's records, from :func:`load_raw_chunks`.
        shared: See :func:`adapt_record`.
        contextualize: See :func:`adapt_record`.

    Returns:
        One chunk per record, in the same order.
    """
    return [
        adapt_record(record, shared, contextualize=contextualize) for record in records
    ]
