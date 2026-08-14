"""The closed vocabularies and the pure functions -- `juena_rag.schema.vocabulary`."""

import pytest

from juena_rag import (
    CHUNK_COLUMNS,
    Chunk,
    ManualChunk,
    compute_content_hash,
    derive_chunk_id,
)


def test_chunk_ids_are_deterministic():
    """Ingesting twice must update rows rather than duplicate them."""
    first = derive_chunk_id("juena", "a1b2c3d", 7)
    second = derive_chunk_id("juena", "a1b2c3d", 7)

    assert first == second


@pytest.mark.parametrize(
    "args",
    [
        ("juena", "a1b2c3d", 8),
        ("juena", "e4f5g6h", 7),
        ("other", "a1b2c3d", 7),
    ],
)
def test_changing_any_part_changes_the_chunk_id(args):
    assert derive_chunk_id(*args) != derive_chunk_id("juena", "a1b2c3d", 7)


def test_chunk_ids_do_not_collide_across_field_boundaries():
    """("ab", "c") must not hash the same as ("a", "bc")."""
    assert derive_chunk_id("ab", "c", 0) != derive_chunk_id("a", "bc", 0)


def test_a_new_revision_gets_new_ids_so_it_can_sit_beside_the_old_one():
    old = derive_chunk_id("kws1-manual", "v2", 0)
    new = derive_chunk_id("kws1-manual", "v3", 0)

    assert old != new


def test_content_hash_is_deterministic_and_content_sensitive():
    assert compute_content_hash("hello") == compute_content_hash("hello")
    assert compute_content_hash("hello") != compute_content_hash("Hello")


def test_the_base_chunk_fields_without_columns_would_land_in_extra():
    """The tempting shortcut -- exclude Chunk's own fields -- drops all of these."""
    without_columns = set(Chunk.model_fields) - CHUNK_COLUMNS

    assert without_columns == {
        "table",
        "equation",
        "heading_path",
        "start_line",
        "end_line",
    }


def test_extra_keeps_everything_that_has_no_column():
    chunk = ManualChunk(
        text="Before each run …",
        language="en",
        heading_path=["3", "3.2"],
        instrument="KWS-1",
    )

    extra = chunk.model_dump(exclude=CHUNK_COLUMNS)

    assert extra["heading_path"] == ["3", "3.2"]
    assert extra["instrument"] == "KWS-1"
    assert "text" not in extra
    assert "language" not in extra
