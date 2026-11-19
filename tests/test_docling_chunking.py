"""Adapting rag-lab's HybridChunker records -- `juena_rag.ingest.docling.chunking`.

Nothing here touches a file: `adapt_record`/`adapt_chunks` take records already
parsed from JSON, so a test writes the shape `hybrid_chunker.py` produces by hand.
"""

import pytest
from pydantic import ValidationError

from juena_rag import BookChunk
from juena_rag.ingest.docling.chunking import adapt_chunks, adapt_record

RECORD = {
    "chunk_id": "5a_Crystallography_MZ-0002",
    "source": "5a_Crystallography_MZ",
    "headings": ["5a.1 Introduction"],
    "pages": [2],
    "doc_item_refs": ["#/texts/7", "#/texts/8"],
    "doc_item_labels": ["picture", "text"],
    "n_tokens": 42,
    "text": "Quartz is a crystal.",
    "contextualized_text": "5a.1 Introduction\nQuartz is a crystal.",
}


def test_a_record_becomes_a_book_chunk_keeping_pages_and_doc_item_refs():
    chunk = adapt_record(RECORD)
    assert isinstance(chunk, BookChunk)
    assert chunk.category == "book"
    assert chunk.pages == [2]
    assert chunk.doc_item_refs == ["#/texts/7", "#/texts/8"]


def test_headings_become_heading_path():
    chunk = adapt_record(RECORD)
    assert chunk.heading_path == ["5a.1 Introduction"]


def test_contextualized_text_is_embedded_by_default():
    """`Store.replace_source` embeds `chunk.text` as given -- no contextualizing
    step of its own -- so the heading path has to already be baked in here."""
    chunk = adapt_record(RECORD)
    assert chunk.text == "5a.1 Introduction\nQuartz is a crystal."


def test_contextualize_false_keeps_the_bare_text():
    chunk = adapt_record(RECORD, contextualize=False)
    assert chunk.text == "Quartz is a crystal."


def test_a_chunk_with_no_headings_gets_an_empty_heading_path():
    record = {**RECORD, "headings": []}
    chunk = adapt_record(record)
    assert chunk.heading_path == []


def test_without_book_fields_they_stay_unset():
    chunk = adapt_record(RECORD)
    assert (chunk.isbn, chunk.chapter_title, chunk.ref_chapter) == (None, None, None)
    assert chunk.editors == []


def test_book_fields_are_copied_onto_every_chunk():
    """No record carries the ISBN or the chapter -- the caller states them once."""
    book = {
        "isbn": "9780198567691",
        "chapter_title": "Crystallography",
        "ref_chapter": "5a",
        "editors": ["T. Brückel"],
        "year": 2019,
        "publisher": "Forschungszentrum Jülich",
    }
    records = [RECORD, {**RECORD, "chunk_id": "5a_Crystallography_MZ-0003"}]
    chunks = adapt_chunks(records, book)
    for chunk in chunks:
        assert chunk.isbn == "9780198567691"
        assert (chunk.chapter_title, chunk.ref_chapter) == ("Crystallography", "5a")
        assert chunk.editors == ["T. Brückel"]
        assert (chunk.year, chunk.publisher) == (2019, "Forschungszentrum Jülich")


def test_a_misspelt_book_field_is_refused_not_dropped():
    with pytest.raises(ValidationError, match="chapter"):
        adapt_record(RECORD, {"chapter": "Crystallography"})


def test_a_book_field_the_record_already_fills_is_refused():
    """Otherwise one `pages` meant for a single chunk would land on all of them."""
    with pytest.raises(ValueError, match="pages"):
        adapt_record(RECORD, {"pages": [1]})


def test_adapt_chunks_preserves_record_order():
    records = [
        {
            **RECORD,
            "chunk_id": "a-0000",
            "text": "first",
            "contextualized_text": "first",
        },
        {
            **RECORD,
            "chunk_id": "a-0001",
            "text": "second",
            "contextualized_text": "second",
        },
    ]
    chunks = adapt_chunks(records)
    assert [chunk.text for chunk in chunks] == ["first", "second"]
