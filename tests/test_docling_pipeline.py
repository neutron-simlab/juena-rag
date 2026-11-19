"""Building the batch for one export -- `juena_rag.ingest.docling.pipeline`.

No database: `build_batch` stops at the `ChunkBatch`, before anything is
embedded. The two files are written by hand in the shapes rag-lab produces.
"""

import json

import pytest

from juena_rag import BookChunk
from juena_rag.ingest.docling.pipeline import build_batch

NAME = "5a_Crystallography_MZ"

RECORD = {
    "chunk_id": f"{NAME}-0000",
    "source": NAME,
    "headings": ["5a.1 Introduction"],
    "pages": [2],
    "doc_item_refs": ["#/texts/7"],
    "doc_item_labels": ["text"],
    "n_tokens": 5,
    "text": "Quartz is a crystal.",
    "contextualized_text": "5a.1 Introduction\nQuartz is a crystal.",
}

#: Only the keys the pipeline reads; a real one also has `texts`, `pages`, ….
DOCUMENT = {
    "schema_name": "DoclingDocument",
    "version": "1.10.0",
    "name": NAME,
    "origin": {
        "mimetype": "application/pdf",
        "binary_hash": 18321577791418801589,
        "filename": f"{NAME}.pdf",
    },
    "title": "Crystallography",
    "authors": ["M. Zobel"],
}


@pytest.fixture
def export(tmp_path):
    """Write one export's two files; return a function taking their contents."""

    def write(records=(RECORD,), document=DOCUMENT):
        folder = tmp_path / NAME
        folder.mkdir(exist_ok=True)
        raw_chunks = folder / f"{NAME}_raw_chunks.json"
        docling_json = folder / f"{NAME}.json"
        raw_chunks.write_text(json.dumps(list(records)), encoding="utf-8")
        docling_json.write_text(json.dumps(document), encoding="utf-8")
        return raw_chunks, docling_json

    return write


def test_the_source_comes_from_the_records_and_the_docling_json(export):
    batch = build_batch(*export(), title="Neutron Scattering")
    source = batch.source

    assert source.source_id == NAME
    assert source.category == "book"
    assert len(source.revision) == 16
    assert source.title == "Neutron Scattering"
    assert source.authors == ["M. Zobel"]
    assert source.artifact_uri == f"{NAME}.json"
    assert source.artifact_format == "docling-json"
    assert source.artifact_hash == "18321577791418801589"
    assert source.parser == "docling/1.10.0"


def test_every_chunk_gets_the_pdf_language_and_chapter_title(export):
    batch = build_batch(*export())
    chunk = batch.chunks[0]

    assert isinstance(chunk, BookChunk)
    assert chunk.source_uri == f"{NAME}.pdf"
    assert chunk.language == "en"
    assert chunk.chapter_title == "Crystallography"
    assert chunk.chunk_kind == "text"


def test_the_callers_book_fields_reach_every_chunk(export):
    batch = build_batch(*export(), book={"ref_chapter": "5a", "year": 2019})
    assert (batch.chunks[0].ref_chapter, batch.chunks[0].year) == ("5a", 2019)


def test_the_extras_are_the_records_the_docling_title_and_the_callers_input(export):
    batch = build_batch(*export(), book={"isbn": "9783958067448", "editors": ["T. B."]})
    chunk = batch.chunks[0]

    assert chunk.heading_path == ["5a.1 Introduction"]
    assert chunk.doc_item_refs == ["#/texts/7"]
    assert chunk.pages == [2]
    assert chunk.provenance == []
    assert chunk.chapter_title == "Crystallography"
    assert (chunk.isbn, chunk.editors) == ("9783958067448", ["T. B."])


def test_book_takes_only_the_manual_fields(export):
    """`chapter_title` is the docling JSON's; input must not quietly replace it."""
    with pytest.raises(ValueError, match="chapter_title"):
        build_batch(*export(), book={"chapter_title": "Something else"})


def test_records_naming_two_sources_are_refused(export):
    other = {**RECORD, "source": "5b_Nandi_Structural_Analysis"}
    with pytest.raises(ValueError, match="same one source"):
        build_batch(*export(records=[RECORD, other]))


def test_a_docling_json_from_another_chapter_is_refused(export):
    document = {**DOCUMENT, "name": "5b_Nandi_Structural_Analysis"}
    with pytest.raises(ValueError, match="chunked from"):
        build_batch(*export(document=document))


def test_a_docling_json_without_rag_labs_title_and_authors_still_builds(export):
    """`title` and `authors` are rag-lab's additions; plain docling has neither."""
    document = {
        key: value for key, value in DOCUMENT.items() if key not in ("title", "authors")
    }
    batch = build_batch(*export(document=document))

    assert batch.source.authors == []
    assert batch.chunks[0].chapter_title is None
