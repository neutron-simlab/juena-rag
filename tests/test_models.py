"""The models -- `juena_rag.schema.models`.

Validation rules 1, 2 and 3 live here, because each is answerable by one chunk
on its own. Rules 4 to 7 need the source too, so they are in test_boundary.py.
"""

from datetime import UTC, datetime

import pytest
from builders import a_source
from pydantic import ValidationError

from juena_rag import (
    SCHEMA_VERSION,
    Chunk,
    ChunkBatch,
    ChunkRow,
    CodeChunk,
    Equation,
    SourceMetadata,
    SourceRow,
    Table,
    compute_content_hash,
    derive_chunk_id,
)


def test_artifact_uri_without_format_is_rejected():
    with pytest.raises(ValidationError) as exc:
        a_source(artifact_uri="s3://juena/manuals/kws1.pdf")

    assert "artifact_format is missing" in str(exc.value)


def test_code_sources_leave_every_artifact_field_unset():
    source = SourceMetadata(source_id="juena", category="code", revision="a1b2c3d")

    assert source.artifact_uri is None
    assert source.artifact_format is None
    assert source.artifact_hash is None
    assert source.parser is None


def test_a_misspelled_source_field_is_rejected_rather_than_ignored():
    with pytest.raises(ValidationError) as exc:
        a_source(artifact_url="s3://juena/manuals/kws1.pdf")

    assert "artifact_url" in str(exc.value)


def a_source_row(**overrides) -> SourceRow:
    defaults = {
        "source_id": "kws1-manual",
        "category": "manual",
        "active_revision": "v3",
        "updated_at": datetime.now(UTC),
    }
    return SourceRow(**{**defaults, **overrides})


def test_a_source_row_is_not_a_source_metadata():
    """Inheriting would make a database row valid input to register_source()."""
    assert not issubclass(SourceRow, SourceMetadata)
    assert not isinstance(a_source_row(), SourceMetadata)


def test_a_source_row_names_active_revision_not_revision():
    """`revision` is what a caller is submitting now; `active_revision` is what
    is currently live -- the atomic replace can briefly make them differ."""
    row = a_source_row(active_revision="v3")

    assert row.active_revision == "v3"
    assert "revision" not in type(row).model_fields


def test_a_source_row_may_have_no_active_revision_yet():
    """register_source() declares a source before anything is ingested into it."""
    row = a_source_row(active_revision=None)

    assert row.active_revision is None


def test_a_source_row_reads_an_artifact_format_this_version_does_not_know():
    row = a_source_row(
        artifact_uri="s3://juena/manuals/kws1.pdf", artifact_format="pdf"
    )

    assert row.artifact_format == "pdf"


def test_a_source_row_does_not_replay_the_artifact_format_validator():
    """That rule guards what a caller may submit, not what a read may see."""
    row = a_source_row(artifact_uri="s3://juena/manuals/kws1.pdf", artifact_format=None)

    assert row.artifact_uri is not None
    assert row.artifact_format is None


def test_a_source_row_carries_updated_at():
    row = a_source_row()

    assert isinstance(row.updated_at, datetime)


def test_a_plain_text_chunk_carries_no_detail():
    chunk = Chunk(text="Ordinary prose.")

    assert chunk.chunk_kind == "text"
    assert chunk.table is None
    assert chunk.equation is None


@pytest.mark.parametrize("kind", ["text", "code_block"])
def test_a_plain_kind_must_not_carry_a_table(kind):
    with pytest.raises(ValidationError) as exc:
        Chunk(text="…", chunk_kind=kind, table=Table(table_ref="t1"))

    assert f"chunk_kind={kind!r} must not set `table`" in str(exc.value)


def test_a_whole_table_is_accepted():
    chunk = Chunk(
        text="Table 3: Calibration settings",
        chunk_kind="table",
        table=Table(
            table_ref="#/tables/7",
            table_title="Calibration settings",
            column_descriptors=["Tmin", "Tmax", "Step"],
            rows=[["4", "600", "0.5"], ["10", "300", "1.0"]],
            row_offset=0,
            n_rows=2,
        ),
    )

    assert len(chunk.table.rows) == chunk.table.n_rows


def test_one_row_of_a_large_table_is_the_same_kind():
    """No separate 'table_row': how many rows is the caller's chunking decision."""
    chunk = Chunk(
        text="minimum temperature 4 K, maximum 600 K, step 0.5",
        chunk_kind="table",
        table=Table(
            column_descriptors=["Tmin", "Tmax", "Step"],
            rows=[["4", "600", "0.5"]],
            row_offset=42,
            n_rows=200,
        ),
    )

    assert dict(zip(chunk.table.column_descriptors, chunk.table.rows[0])) == {
        "Tmin": "4",
        "Tmax": "600",
        "Step": "0.5",
    }


def test_a_window_of_rows_is_expressible():
    """The case the old table/table_row pair could not say at all."""
    chunk = Chunk(
        text="rows 40 to 59 of the calibration table",
        chunk_kind="table",
        table=Table(
            column_descriptors=["Tmin", "Tmax", "Step"],
            rows=[[str(i), "600", "0.5"] for i in range(40, 60)],
            row_offset=40,
            n_rows=200,
        ),
    )

    assert len(chunk.table.rows) == 20
    assert chunk.table.row_offset == 40


def test_a_row_that_is_not_as_wide_as_the_header_is_rejected():
    with pytest.raises(ValidationError) as exc:
        Table(
            column_descriptors=["Tmin", "Tmax", "Step"],
            rows=[["4", "600", "0.5"], ["10", "300"]],
        )

    message = str(exc.value)
    assert "row 1 holds 2 values" in message
    assert "names 3 columns" in message


def test_rows_without_descriptors_are_rejected():
    """Positional values with nothing to line up against carry no meaning."""
    with pytest.raises(ValidationError) as exc:
        Table(rows=[["4", "600", "0.5"]])

    assert "`column_descriptors` must name the columns" in str(exc.value)


def test_a_table_chunk_without_a_table_is_rejected():
    with pytest.raises(ValidationError) as exc:
        Chunk(text="Table 3", chunk_kind="table")

    assert "chunk_kind='table' requires `table`" in str(exc.value)


def test_an_equation_chunk_is_accepted():
    chunk = Chunk(
        text="The differential scattering cross section is given by",
        chunk_kind="equation",
        equation=Equation(latex=r"\frac{d\sigma}{d\Omega}", label="Eq. 3"),
    )

    assert chunk.equation.label == "Eq. 3"


def test_an_equation_chunk_without_an_equation_is_rejected():
    with pytest.raises(ValidationError) as exc:
        Chunk(text="…", chunk_kind="equation")

    assert "chunk_kind='equation' requires `equation`" in str(exc.value)


def test_a_placeholder_marks_a_formula_that_could_not_be_translated_to_words():
    """No number in the placeholder: one chunk carries at most one equation."""
    chunk = Chunk(
        text="The differential cross section is given by $EQUATION$.",
        chunk_kind="equation",
        equation=Equation(latex=r"\frac{d\sigma}{d\Omega} = b^2", label="Eq. 4"),
    )

    assert "$EQUATION$" in chunk.text
    assert chunk.equation.latex == r"\frac{d\sigma}{d\Omega} = b^2"


def test_two_formulas_in_one_passage_split_into_two_chunks():
    """Structuring is opt-in per formula; a chunk never carries more than one."""
    scattering_length = Chunk(text="The scattering length is denoted b.")
    cross_section = Chunk(
        text="The differential cross section is given by $EQUATION$.",
        chunk_kind="equation",
        equation=Equation(latex=r"\frac{d\sigma}{d\Omega} = b^2", label="Eq. 4"),
    )

    assert scattering_length.equation is None
    assert cross_section.equation is not None


def test_an_equation_chunk_must_not_carry_a_table():
    with pytest.raises(ValidationError) as exc:
        Chunk(
            text="…",
            chunk_kind="equation",
            equation=Equation(latex="x"),
            table=Table(table_ref="t1"),
        )

    assert "chunk_kind='equation' must not set `table`" in str(exc.value)


def test_an_unknown_chunk_kind_is_rejected_on_the_way_in():
    with pytest.raises(ValidationError):
        Chunk(text="…", chunk_kind="figure")


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_a_blank_chunk_is_rejected(text):
    with pytest.raises(ValidationError) as exc:
        Chunk(text=text)

    assert "empty or only whitespace" in str(exc.value)


def test_a_start_without_an_end_is_fine():
    assert CodeChunk(text="def f(): ...", start_line=231).end_line is None


def test_an_end_without_a_start_is_rejected():
    with pytest.raises(ValidationError) as exc:
        Chunk(text="…", end_line=260)

    assert "`end_line` is set but `start_line` is not" in str(exc.value)


def test_a_backwards_line_range_is_rejected():
    with pytest.raises(ValidationError) as exc:
        Chunk(text="…", start_line=300, end_line=12)

    assert "is after `end_line`" in str(exc.value)


def test_a_backwards_character_range_is_rejected():
    with pytest.raises(ValidationError) as exc:
        CodeChunk(text="def f(): ...", start_index=900, end_index=100)

    assert "is after `end_index`" in str(exc.value)


def test_a_misspelled_chunk_field_is_rejected_rather_than_silently_stored():
    with pytest.raises(ValidationError) as exc:
        Chunk(text="…", headingpath=["3.2"])

    assert "headingpath" in str(exc.value)


def test_line_numbers_are_on_the_base_so_every_category_can_record_them():
    """A README in a code repository has headings and line numbers too."""
    assert {"start_line", "end_line", "heading_path"} <= set(Chunk.model_fields)


def a_chunk_row(**overrides) -> ChunkRow:
    defaults = {
        "chunk_id": derive_chunk_id("kws1-manual", "v3", 0),
        "source_id": "kws1-manual",
        "source_revision": "v3",
        "chunk_index": 0,
        "category": "manual",
        "chunk_kind": "text",
        "text": "Before each run …",
        "content_hash": compute_content_hash("Before each run …"),
        "embedding_model": "alias-qwen3-8b-embeddings",
        "schema_version": SCHEMA_VERSION,
    }
    return ChunkRow(**{**defaults, **overrides})


def test_a_chunk_row_is_not_a_chunk():
    """Inheriting would make a database row valid batch input."""
    assert not issubclass(ChunkRow, Chunk)
    assert not isinstance(a_chunk_row(), Chunk)


def test_a_chunk_row_is_not_accepted_as_batch_input():
    with pytest.raises(ValidationError):
        ChunkBatch(source=a_source(), chunks=[a_chunk_row()])


def test_a_chunk_row_reads_a_kind_this_version_does_not_know():
    """A row written by a newer package version must still be readable."""
    assert a_chunk_row(chunk_kind="figure").chunk_kind == "figure"


def test_a_chunk_row_without_a_kind_raises_rather_than_defaulting_to_text():
    with pytest.raises(ValidationError):
        ChunkRow(
            chunk_id="x",
            source_id="kws1-manual",
            source_revision="v3",
            chunk_index=0,
            category="manual",
            text="…",
            content_hash="y",
            embedding_model="m",
            schema_version=SCHEMA_VERSION,
        )
