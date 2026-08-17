"""Claiming a category -- `juena_rag.schema.registry`."""

from typing import Literal

import pytest

from juena_rag import (
    Chunk,
    ChunkBatch,
    CodeChunk,
    LayoutChunk,
    ManualChunk,
    PaperChunk,
    SourceMetadata,
    WikiChunk,
    chunk_class,
    parse_chunk,
    register_chunk_class,
    registered_categories,
)


def test_the_seven_category_classes_are_registered():
    assert set(registered_categories()) >= {
        "code",
        "dataset",
        "manual",
        "paper",
        "presentation",
        "report",
        "wiki",
    }


def test_layout_chunk_does_not_register():
    """It pins no category, and nothing registers it, so it cannot be named."""
    assert LayoutChunk not in registered_categories().values()


def test_registering_a_category_twice_raises(clean_registry):
    """Otherwise which class wins depends on import order."""
    with pytest.raises(TypeError) as exc:

        @register_chunk_class
        class RivalCodeChunk(Chunk):
            category: Literal["code"] = "code"

    assert "'code' is already registered" in str(exc.value)
    assert "CodeChunk" in str(exc.value)


def test_registering_a_class_that_pins_no_category_raises(clean_registry):
    with pytest.raises(TypeError) as exc:

        @register_chunk_class
        class NamelessChunk(Chunk):
            speaker: str | None = None

    assert "pins no category" in str(exc.value)


def test_registering_the_same_class_twice_is_harmless():
    """Re-running the decorator, e.g. after a module reload, is not a collision."""
    assert register_chunk_class(PaperChunk) is PaperChunk
    assert registered_categories()["paper"] is PaperChunk


def test_specialising_a_category_class_without_repinning_is_allowed(clean_registry):
    """No decorator, no registration -- so it cannot fight PaperChunk for 'paper'."""

    class AnnotatedPaperChunk(PaperChunk):
        annotator: str | None = None

    assert registered_categories()["paper"] is PaperChunk


def test_a_subclass_defined_outside_the_package_validates(clean_registry):
    @register_chunk_class
    class SeminarChunk(Chunk):
        category: Literal["seminar"] = "seminar"
        speaker: str | None = None

    batch = ChunkBatch(
        source=SourceMetadata(
            source_id="seminar-2026",
            category="seminar",
            revision="2026-08-14",
            visibility="internal",
        ),
        chunks=[{"text": "Welcome everyone.", "speaker": "A. Ihsan"}],
    )

    assert isinstance(batch.chunks[0], SeminarChunk)
    assert batch.chunks[0].speaker == "A. Ihsan"


def test_an_unregistered_category_falls_back_to_plain_chunk():
    """A caller with no category-specific fields need not define a class."""
    assert chunk_class("no-such-category") is Chunk

    batch = ChunkBatch(
        source=SourceMetadata(
            source_id="misc",
            category="no-such-category",
            revision="1",
            visibility="internal",
        ),
        chunks=[{"text": "Just some text."}],
    )

    assert type(batch.chunks[0]) is Chunk


def test_parse_chunk_names_the_unknown_category_and_lists_the_known_ones():
    with pytest.raises(ValueError) as exc:
        parse_chunk("seminar", {"text": "Welcome everyone.", "speaker": "A. Ihsan"})

    message = str(exc.value)
    assert "'seminar' is not registered" in message
    assert "'manual'" in message  # the registry is listed
    assert not isinstance(exc.value, KeyError)


def test_parse_chunk_builds_the_registered_class():
    chunk = parse_chunk("paper", {"text": "We measured …", "section": "Methods"})

    assert isinstance(chunk, PaperChunk)
    assert chunk.section == "Methods"


def test_a_code_chunk_is_a_chunk():
    assert isinstance(CodeChunk(text="def f(): ..."), Chunk)


def test_code_and_wiki_chunks_carry_no_layout_anchors():
    """A source file has lines; a wiki page is markdown. Neither has pages."""
    assert not issubclass(CodeChunk, LayoutChunk)
    assert not issubclass(WikiChunk, LayoutChunk)


def test_laid_out_categories_inherit_the_anchors():
    assert issubclass(PaperChunk, LayoutChunk)
    assert issubclass(ManualChunk, LayoutChunk)
