"""The models: a source, a chunk, and what a chunk carries.

Read top to bottom, this file is the whole data model -- the document, the detail
a chunk carries when it is more than prose, the chunk itself, and the two shapes
that come back from the database: :class:`SourceRow` and :class:`ChunkRow`.

Each has a declaration counterpart -- :class:`SourceMetadata`, :class:`Chunk` --
and deliberately does not inherit from it: a row is not a valid declaration, and
letting one be accepted as the other is exactly the mistake this split prevents.

Extending :class:`Chunk` for your own kind of document happens in
:mod:`juena_rag.schema.registry`, which has seven worked examples.
"""

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from juena_rag.schema.vocabulary import (
    ArtifactFormat,
    ChunkKind,
    check_range,
)

# --------------------------------------------------------------------------- #
# The source
# --------------------------------------------------------------------------- #


class SourceMetadata(BaseModel):
    """One document, one revision, and the file it was parsed from."""

    model_config = {"extra": "forbid"}

    source_id: str = Field(min_length=1)
    category: str = Field(min_length=1)

    #: The caller's name for this version -- a commit sha, a date, a label. Never
    #: parsed or ordered, only compared for equality. One rule governs it:
    #: *change the bytes, change the revision.*
    revision: str = Field(min_length=1)

    title: str | None = None

    # The original file, for documents that have one. All four are unset for code
    # sources, and that is correct rather than incomplete.
    artifact_uri: str | None = None
    artifact_format: ArtifactFormat | None = None
    artifact_hash: str | None = None
    parser: str | None = None  # e.g. "docling/2.55.1"

    @model_validator(mode="after")
    def _format_accompanies_uri(self) -> Self:
        if self.artifact_uri and not self.artifact_format:
            raise ValueError(
                "artifact_uri is set but artifact_format is missing; the retrieval "
                "agent branches on the format and cannot read an artifact without it"
            )
        return self


# --------------------------------------------------------------------------- #
# The detail models -- what a chunk carries beyond its text
# --------------------------------------------------------------------------- #


class Table(BaseModel):
    """The table this chunk came from -- all of it, or a window of its rows.

    There is no separate kind for a single row. How many rows go in a chunk is a
    chunking decision, and chunking belongs to the caller; this only records what
    they decided::

        whole 3-row table   rows=[[…],[…],[…]]  row_offset=0   n_rows=3
        one row of 200      rows=[["4","600"]]  row_offset=42  n_rows=200
        a window of 20      rows=[…20 lists…]   row_offset=40  n_rows=200
    """

    model_config = {"extra": "forbid"}

    table_ref: str | None = None
    table_title: str | None = None

    #: What each column means -- the header row, verbatim and in order. Every
    #: chunk of a split table repeats it: alone, a row reads ``4 | 600 | 0.5``.
    column_descriptors: list[str] = Field(default_factory=list)

    #: The rows carried by *this* chunk, each positionally aligned with
    #: ``column_descriptors``. Positional rather than dicts keyed by column, which
    #: would store the header again in every row and could not be checked.
    rows: list[list[str]] = Field(default_factory=list)

    #: Where ``rows`` begins in the full table, and how long the full table is.
    #: With ``len(rows)`` they say exactly what the chunker did.
    row_offset: int | None = None
    n_rows: int | None = None

    # Validator to check whether the data is aligned with the column descriptors
    @model_validator(mode="after")
    def _rows_align_with_descriptors(self) -> Self:
        if not self.rows:
            return self
        if not self.column_descriptors:
            raise ValueError(
                "`rows` is positional, so `column_descriptors` must name the columns "
                "it lines up against; without them the values carry no meaning"
            )
        width = len(self.column_descriptors)
        for offset, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(
                    f"row {offset} holds {len(row)} values but `column_descriptors` "
                    f"names {width} columns; rows are positionally aligned, so they "
                    "must be the same length"
                )
        return self


class Equation(BaseModel):
    """The formula itself. The chunk's ``text`` carries the prose around it.

    Nobody searches for the LaTeX; they search for *differential scattering cross
    section*, which is in the sentence introducing the formula, not the formula.
    A caller who cannot translate a formula to words may instead leave the
    sentence intact and mark the formula's position with a placeholder, e.g.
    ``$EQUATION$`` -- one chunk carries at most one equation, so the marker never
    needs a number.

    Not every formula needs this structure -- a formula can sit as ordinary
    characters inside a plain ``chunk_kind="text"`` chunk with no ``Equation`` at
    all. Reach for this only when a formula is worth citing on its own. If a
    passage has more than one such formula, split into one chunk per formula,
    each repeating enough prose to stand alone -- the same pattern a table row
    uses to repeat its header.
    """

    model_config = {"extra": "forbid"}

    latex: str | None = None  # exact source, so it renders faithfully
    label: str | None = None  # "Eq. 3", so a citation can name it


class PageProvenance(BaseModel):
    """Where on a page a chunk physically sat."""

    model_config = {"extra": "forbid"}

    page_no: int
    bbox: tuple[float, float, float, float] | None = None
    charspan: tuple[int, int] | None = None


# --------------------------------------------------------------------------- #
# The chunk
# --------------------------------------------------------------------------- #


class Chunk(BaseModel):
    """One piece of text. Subclass it to add category-specific fields."""

    # Typos are rejected, not silently stored.
    model_config = {"extra": "forbid"}

    text: str
    source_uri: str | None = None
    language: str | None = None

    chunk_kind: ChunkKind = "text"

    table: Table | None = None  # iff chunk_kind is "table"
    equation: Equation | None = None  # iff chunk_kind is "equation"

    heading_path: list[str] = Field(default_factory=list)

    # Line-based artifacts: code, Markdown.
    start_line: int | None = None
    end_line: int | None = None

    @model_validator(mode="after")
    def _text_is_not_blank(self) -> Self:
        if not self.text.strip():
            raise ValueError(
                "`text` is empty or only whitespace; embedding it costs a model call "
                "and returns a vector that matches everything weakly and nothing well"
            )
        return self

    @model_validator(mode="after")
    def _detail_matches_kind(self) -> Self:
        kind = self.chunk_kind
        wants_table = kind == "table"
        wants_equation = kind == "equation"

        if wants_table and self.table is None:
            raise ValueError(f"chunk_kind={kind!r} requires `table`")
        if not wants_table and self.table is not None:
            raise ValueError(
                f"chunk_kind={kind!r} must not set `table`; a chunk carrying a table "
                "should say so, with chunk_kind='table'"
            )

        if wants_equation and self.equation is None:
            raise ValueError(f"chunk_kind={kind!r} requires `equation`")
        if not wants_equation and self.equation is not None:
            raise ValueError(
                f"chunk_kind={kind!r} must not set `equation`; a chunk carrying a "
                "formula should say so, with chunk_kind='equation'"
            )

        return self

    @model_validator(mode="after")
    def _line_range_is_coherent(self) -> Self:
        """Validation rule 3."""
        check_range(self.start_line, self.end_line, "start_line", "end_line")
        return self


# --------------------------------------------------------------------------- #
# What comes back out -- one row of each table
# --------------------------------------------------------------------------- #


class SourceRow(BaseModel):
    """One row of the `sources` table -- a database row, not a declaration.

    Deliberately not a subclass of :class:`SourceMetadata`, for the same reason
    :class:`ChunkRow` is not a subclass of :class:`Chunk`: the two mean different
    things at different moments, even though most fields overlap.

    Two differences beyond a rename:

    - ``active_revision`` replaces ``revision``. ``SourceMetadata.revision`` is
      the version a caller is submitting *right now*; ``active_revision`` is
      whichever revision is currently live. The atomic replace can briefly make
      the two differ -- the new revision is written before the pointer moves.
    - ``updated_at`` has no equivalent on the input side; nobody submits it, the
      store sets it.

    Read leniently on purpose -- ``artifact_format`` widened to a plain string --
    so a row written by a newer version of this package stays readable by an
    older one. Does not repeat ``SourceMetadata``'s
    artifact-format validator either: that rule guards what a caller may submit,
    not what a routine read is allowed to see.
    """

    source_id: str
    category: str
    active_revision: str | None = None  # unset before a source's first ingest

    title: str | None = None

    artifact_uri: str | None = None
    artifact_format: str | None = None  # widened; see the docstring
    artifact_hash: str | None = None
    parser: str | None = None

    #: How many chunks this source currently has stored. Counted per read rather
    #: than kept as a column, so it cannot drift from the table it describes;
    #: `source_id` is the leading column of `chunks_revision_idx`, so it is an
    #: index-only aggregate. Zero means registered but never ingested -- which is
    #: what a caller asking "is this searchable?" needs to tell apart from absent.
    chunk_count: int = 0

    updated_at: datetime


class ChunkRow(BaseModel):
    """One row of the `chunks` table -- a database row, not a chunk.

    Deliberately not a subclass of :class:`Chunk`, though the category classes
    are: inheriting would make a stored row valid batch input. Declares its own
    ``text``/``source_uri``/``language`` rather than sharing a base with
    :class:`Chunk` -- what identifies *which* chunk this is is the trio below,
    not those three, so there was nothing that needed to be shared.

    Read leniently on purpose -- no ``extra="forbid"``, and ``chunk_kind``
    widened to a plain string -- so a row written by a newer version of this
    package stays readable by an older one.

    **No vector field, deliberately.** The row has an ``embedding`` column, but
    nothing in Python reads it: search runs in SQL and only ever returns text.
    Materialising 4096 floats costs roughly 130 kB per row as Python objects --
    eight times what it occupies in the database -- so carrying it here would be
    paid on every read for a value no caller uses. ``Chunk`` omits it for a
    different reason: the package embeds, so a caller who could supply a vector
    could supply one from the wrong model.

    ``extra`` keeps its name from the database column. It holds every chunk field
    without a column of its own -- the table, the equation, the heading path, the
    line range, and whatever the category class added.
    """

    # Together, the key that tells which chunk this is: derive_chunk_id() hashes
    # exactly these three into chunk_id.
    source_id: str
    source_revision: str
    chunk_index: int  # position within the source

    chunk_id: str

    text: str
    source_uri: str | None = None
    language: str | None = None

    category: str
    chunk_kind: str  # widened; see the docstring

    content_hash: str
    extra: dict[str, Any] = Field(default_factory=dict)

    embedding_model: str
    schema_version: int


# --------------------------------------------------------------------------- #
# What a search returns
# --------------------------------------------------------------------------- #


class SearchHit(ChunkRow):
    """A stored chunk, and how far the query landed from it.

    A subclass rather than a narrower copy, because a hit *is* a row -- the same
    fields, read the same leniently, reconstructed into a typed chunk by the same
    ``parse_chunk(hit.category, {…, **hit.extra})``. Only the distance is new.
    """

    #: Cosine distance, so **smaller is closer** -- 0 is identical and 2 is
    #: opposite. Deliberately not called ``score``: it is the raw number the
    #: database computed, and inverting it into a similarity would claim the
    #: vectors are unit-length, which is the embedding model's business and not
    #: something this package can check. A reranker adds its own field beside
    #: this one rather than overwriting it.
    distance: float

    #: The reranking model's relevance score, so **larger is more relevant**, on
    #: that model's own scale. ``None`` when the search was not reranked -- see
    #: ``SearchResult.reranked_by``. Kept apart from ``distance`` because the two
    #: numbers mean different things and must never be compared with each other.
    rerank_score: float | None = None


class SearchResult(BaseModel):
    """The hits, and the documents they came from.

    Two lists rather than one, mirroring :class:`~juena_rag.schema.boundary.ChunkBatch`
    on the write side: a handful of documents usually carry many hits, and the
    navigator in plan 03 needs ``artifact_format`` and ``title``, which live on the
    source. Join them with :meth:`source_of`.
    """

    sources: list[SourceRow] = Field(default_factory=list)
    hits: list[SearchHit] = Field(default_factory=list)

    #: Which model put ``hits`` in their order. ``None`` means the order is by
    #: vector distance -- either no reranking was asked for, or it failed and
    #: ``rerank_error`` says why. Stated on every result, because the order means
    #: something different in each case.
    reranked_by: str | None = None

    #: Why a requested rerank did not happen, when it did not. The search still
    #: answers, in vector-distance order.
    rerank_error: str | None = None

    def source_of(self, hit: SearchHit) -> SourceRow:
        """The document one hit came from.

        Linear over ``sources``, which holds one entry per *document* rather than
        per hit -- single digits in every real call.
        """
        for source in self.sources:
            if source.source_id == hit.source_id:
                return source
        raise KeyError(
            f"no source {hit.source_id!r} in this result; a hit and its source "
            "always travel together, so this means the two lists were built apart"
        )
