"""What crosses into the store -- `juena_rag.schema.boundary`.

Validation rules 4 to 7, and the `SerializeAsAny` round trip -- the failure that
is invisible to every type checker and every test that does not read the dump.
"""

import pytest
from builders import a_docling_source, a_source
from pydantic import ValidationError

from juena_rag import (
    Chunk,
    ChunkBatch,
    CodeChunk,
    ManualChunk,
    ReportChunk,
    SourceMetadata,
)


def test_the_end_to_end_example_from_the_plan_validates():
    my_texts = ["The first paragraph.", "The second paragraph."]

    batch = ChunkBatch(
        source=SourceMetadata(
            source_id="my-notes", category="report", revision="2026-08"
        ),
        chunks=[Chunk(text=t) for t in my_texts],
    )

    assert len(batch.chunks) == 2
    assert batch.mode == "replace"


def test_subclass_fields_survive_serialisation():
    """Without SerializeAsAny these vanish silently on the way to the database."""
    batch = ChunkBatch(
        source=SourceMetadata(source_id="juena", category="code", revision="a1b2c3d"),
        chunks=[CodeChunk(text="def build_index(): ...", symbol="build_index")],
    )

    dumped = batch.model_dump()

    assert dumped["chunks"][0]["symbol"] == "build_index"
    assert dumped["chunks"][0]["category"] == "code"


def test_layout_anchors_survive_serialisation():
    batch = ChunkBatch(
        source=a_docling_source(),
        chunks=[ManualChunk(text="Before each run …", doc_item_refs=["#/texts/104"])],
    )

    dumped = batch.model_dump()

    assert dumped["chunks"][0]["doc_item_refs"] == ["#/texts/104"]


def test_a_batch_round_trips_through_json():
    original = ChunkBatch(
        source=a_docling_source(),
        chunks=[ManualChunk(text="Before each run …", pages=[4], instrument="KWS-1")],
    )

    restored = ChunkBatch.model_validate_json(original.model_dump_json())

    assert isinstance(restored.chunks[0], ManualChunk)
    assert restored.chunks[0].instrument == "KWS-1"
    assert restored.chunks[0].pages == [4]


def test_a_bare_chunk_is_upgraded_to_the_category_class():
    batch = ChunkBatch(
        source=a_source(),
        chunks=[Chunk(text="Before each run …", heading_path=["3", "3.2"])],
    )

    chunk = batch.chunks[0]
    assert isinstance(chunk, ManualChunk)
    assert chunk.heading_path == ["3", "3.2"]  # nothing was lost


def test_the_wrong_subclass_is_rejected():
    """A CodeChunk in a manual batch would filter as a manual and render as code."""
    with pytest.raises(ValidationError) as exc:
        ChunkBatch(source=a_source(), chunks=[CodeChunk(text="def f(): ...")])

    message = str(exc.value)
    assert "chunk 0 is a CodeChunk" in message
    assert "category='manual'" in message
    assert "ManualChunk" in message


def test_dictionaries_are_built_as_the_category_class():
    batch = ChunkBatch(
        source=a_source(),
        chunks=[{"text": "Before each run …", "instrument": "KWS-1"}],
    )

    assert isinstance(batch.chunks[0], ManualChunk)
    assert batch.chunks[0].instrument == "KWS-1"


def test_a_category_specific_field_on_the_wrong_category_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ChunkBatch(
            source=SourceMetadata(
                source_id="juena", category="code", revision="a1b2c3d"
            ),
            chunks=[{"text": "def f(): ...", "instrument": "KWS-1"}],
        )

    assert "instrument" in str(exc.value)


def test_a_source_with_no_artifact_is_fine_when_nothing_points_into_one():
    """Pasted notes, a record from an API: the text *is* the original."""
    batch = ChunkBatch(
        source=SourceMetadata(
            source_id="my-notes", category="report", revision="2026-08"
        ),
        chunks=[ReportChunk(text="Some notes I typed.")],
    )

    assert batch.source.artifact_uri is None


def test_pages_without_an_artifact_are_rejected():
    with pytest.raises(ValidationError) as exc:
        ChunkBatch(source=a_source(), chunks=[ManualChunk(text="…", pages=[4])])

    message = str(exc.value)
    assert "chunk 0 carries pages" in message
    assert "no artifact_uri" in message


def test_doc_item_refs_without_an_artifact_are_rejected():
    with pytest.raises(ValidationError) as exc:
        ChunkBatch(
            source=a_source(),
            chunks=[ManualChunk(text="…", doc_item_refs=["#/texts/1"])],
        )

    assert "chunk 0 carries doc_item_refs" in str(exc.value)


def test_pages_with_an_artifact_are_accepted():
    batch = ChunkBatch(
        source=a_docling_source(),
        chunks=[ManualChunk(text="Before each run …", pages=[4])],
    )

    assert batch.chunks[0].pages == [4]


def test_node_addresses_against_a_markdown_artifact_are_rejected():
    """The realistic mistake: parse with docling, then store the smaller markdown."""
    with pytest.raises(ValidationError) as exc:
        ChunkBatch(
            source=a_source(
                artifact_uri="s3://juena/manuals/kws1.md",
                artifact_format="markdown",
            ),
            chunks=[
                ManualChunk(text="Before each run …", doc_item_refs=["#/texts/104"])
            ],
        )

    message = str(exc.value)
    assert "chunk 0 sets doc_item_refs" in message
    assert "'markdown'" in message
    assert "start_line/end_line" in message


def test_node_addresses_against_a_docling_artifact_are_accepted():
    batch = ChunkBatch(
        source=a_docling_source(),
        chunks=[ManualChunk(text="Before each run …", doc_item_refs=["#/texts/104"])],
    )

    assert batch.chunks[0].doc_item_refs == ["#/texts/104"]


def test_a_markdown_artifact_carries_line_numbers_instead():
    batch = ChunkBatch(
        source=a_source(
            artifact_uri="s3://juena/manuals/kws1.md", artifact_format="markdown"
        ),
        chunks=[ManualChunk(text="Before each run …", start_line=12, end_line=14)],
    )

    assert batch.chunks[0].start_line == 12


def test_an_empty_replace_batch_is_rejected():
    """The atomic replace would delete every chunk of the source and succeed."""
    with pytest.raises(ValidationError) as exc:
        ChunkBatch(source=a_source(), chunks=[])

    message = str(exc.value)
    assert "would delete every chunk" in message
    assert "delete_source()" in message


def test_an_empty_upsert_batch_is_a_harmless_no_op():
    batch = ChunkBatch(source=a_source(), chunks=[], mode="upsert")

    assert batch.chunks == []
