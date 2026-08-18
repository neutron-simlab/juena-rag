"""The `Store`: what a batch becomes, and what comes back out.

Split the way the module is. The derivation tests need no database at all -- they
check the twelve values a row gets that a chunk never had. The rest need one, and
skip when it is absent.
"""

import threading
import time

import pytest
from builders import a_docling_source, a_source
from fakes import FakeEmbedder

from juena_rag import (
    SCHEMA_VERSION,
    Chunk,
    ChunkBatch,
    CodeChunk,
    ManualChunk,
    SearchHit,
    SearchResult,
    SourceRow,
    compute_content_hash,
    derive_chunk_id,
    parse_chunk,
)
from juena_rag.schema.vocabulary import CHUNK_COLUMNS
from juena_rag.store import chunk_extra, chunk_rows, source_row

# --------------------------------------------------------------------------- #
# The derivation -- no database
# --------------------------------------------------------------------------- #


def a_batch(*texts, **source_overrides) -> ChunkBatch:
    return ChunkBatch(
        source=a_source(**source_overrides),
        chunks=[Chunk(text=text) for text in texts],
    )


def test_every_column_a_chunk_does_not_have_is_derived():
    """The twelve values between a Chunk and a row, checked one by one."""
    batch = a_batch("The detector moves on rails.")
    rows = chunk_rows(batch, [[0.5] * 8], "fake-embedder")

    assert len(rows) == 1
    row = rows[0]
    assert row["source_id"] == "kws1-manual"
    assert row["source_revision"] == "v3"
    assert row["chunk_index"] == 0
    assert row["chunk_id"] == derive_chunk_id("kws1-manual", "v3", 0)
    assert row["content_hash"] == compute_content_hash("The detector moves on rails.")
    assert row["embedding"] == [0.5] * 8
    assert row["embedding_model"] == "fake-embedder"
    assert row["schema_version"] == SCHEMA_VERSION


def test_category_and_visibility_come_from_the_source_not_the_chunk():
    """A chunk cannot claim a category or an audience of its own."""
    batch = ChunkBatch(
        source=a_source(category="code", visibility="public"),
        chunks=[CodeChunk(text="def f(): ...", symbol="f")],
    )
    row = chunk_rows(batch, [[0.0] * 8], "fake-embedder")[0]

    assert row["category"] == "code"
    assert row["visibility"] == "public"
    # And there is no way for the chunk to have said otherwise:
    assert "visibility" not in CodeChunk.model_fields


def test_chunk_index_is_list_position_and_the_caller_never_supplies_it():
    batch = a_batch("first", "second", "third")
    rows = chunk_rows(batch, [[0.0] * 8] * 3, "fake-embedder")

    assert [row["chunk_index"] for row in rows] == [0, 1, 2]
    # Distinct ids follow from distinct indexes, which is what the UNIQUE index
    # on (source_id, source_revision, chunk_index) also says.
    assert len({row["chunk_id"] for row in rows}) == 3


def test_extra_holds_exactly_the_fields_with_no_column():
    """`table`, `equation`, the heading path and the line range have no column."""
    chunk = CodeChunk(
        text="def build_index(): ...",
        symbol="build_index",
        start_line=231,
        end_line=240,
        heading_path=["indexing", "build"],
    )
    extra = chunk_extra(chunk)

    assert extra["symbol"] == "build_index"
    assert extra["start_line"] == 231
    assert extra["heading_path"] == ["indexing", "build"]
    # Read from CHUNK_COLUMNS rather than repeated here: spelling the names out
    # once let `category` grow a column without this test noticing.
    for column in CHUNK_COLUMNS:
        assert column not in extra


def test_a_typed_chunk_survives_the_round_trip_through_extra():
    """extra is only worth having if what goes in comes back out typed."""
    original = CodeChunk(
        text="def f(x): return x",
        symbol="f",
        start_line=1,
        end_line=1,
        language="python",
    )
    row = chunk_rows(
        ChunkBatch(source=a_source(category="code"), chunks=[original]),
        [[0.0] * 8],
        "fake-embedder",
    )[0]

    rebuilt = parse_chunk(
        row["category"],
        {
            "text": row["text"],
            "source_uri": row["source_uri"],
            "language": row["language"],
            "chunk_kind": row["chunk_kind"],
            **row["extra"],
        },
    )
    assert rebuilt == original


def test_a_mismatched_number_of_vectors_is_refused():
    """Vectors are matched by position, so an uneven pair must not be silent."""
    batch = a_batch("one", "two")
    with pytest.raises(ValueError, match="matched by position"):
        chunk_rows(batch, [[0.0] * 8], "fake-embedder")


def test_the_source_row_leaves_active_revision_alone():
    """Registering a source and publishing a revision are two moments."""
    row = source_row(a_docling_source())

    assert row["artifact_format"] == "docling-json"
    assert "active_revision" not in row


# --------------------------------------------------------------------------- #
# Against a real database
# --------------------------------------------------------------------------- #


def test_a_batch_becomes_rows_that_come_back_as_hits(store):
    store.replace_source(a_batch("The detector moves on rails.", "Neutrons scatter."))

    result = store.search("detector", visibility=["internal"])

    assert isinstance(result, SearchResult)
    assert len(result.hits) == 2
    assert {hit.text for hit in result.hits} == {
        "The detector moves on rails.",
        "Neutrons scatter.",
    }
    assert all(isinstance(hit, SearchHit) for hit in result.hits)


def test_the_query_finds_the_chunk_whose_text_it_is(store):
    """The FakeEmbedder hashes text, so identical text is distance zero."""
    store.replace_source(a_batch("alpha", "beta", "gamma"))

    result = store.search("beta", visibility=["internal"], n=1)

    assert len(result.hits) == 1
    assert result.hits[0].text == "beta"
    assert result.hits[0].distance == pytest.approx(0.0, abs=1e-6)


def test_hits_arrive_ordered_by_distance(store):
    store.replace_source(a_batch("alpha", "beta", "gamma", "delta"))

    distances = [
        hit.distance for hit in store.search("beta", visibility=["internal"]).hits
    ]

    assert distances == sorted(distances)


def test_the_documents_travel_with_the_hits(store):
    """The whole reason search returns two lists rather than one."""
    store.replace_source(
        ChunkBatch(
            source=a_docling_source(title="KWS-1 User Manual"),
            chunks=[ManualChunk(text="The detector moves on rails.")],
        )
    )

    result = store.search("detector", visibility=["internal"])
    document = result.source_of(result.hits[0])

    assert isinstance(document, SourceRow)
    assert document.title == "KWS-1 User Manual"
    assert document.artifact_format == "docling-json"


def test_one_document_appears_once_however_many_hits_it_produced(store):
    store.replace_source(a_batch("alpha", "beta", "gamma"))

    result = store.search("beta", visibility=["internal"])

    assert len(result.hits) == 3
    assert len(result.sources) == 1


def test_visibility_filters_inside_the_query(store):
    store.replace_source(a_batch("public thing", source_id="open", visibility="public"))
    store.replace_source(
        a_batch("restricted thing", source_id="closed", visibility="restricted")
    )

    visible = store.search("thing", visibility=["public"])

    assert [hit.text for hit in visible.hits] == ["public thing"]
    assert [source.source_id for source in visible.sources] == ["open"]


def test_visibility_is_required_and_an_empty_list_is_refused(store):
    """An empty list can only ever match nothing, which is never what was meant."""
    with pytest.raises(TypeError):
        store.search("anything")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="can only ever match nothing"):
        store.search("anything", visibility=[])


def test_asking_for_more_than_was_retrieved_is_refused(store):
    with pytest.raises(ValueError, match="exceeds k"):
        store.search("anything", visibility=["internal"], k=5, n=10)


def test_sources_and_categories_narrow_the_search(store):
    store.replace_source(a_batch("alpha", source_id="one", category="manual"))
    store.replace_source(a_batch("alpha", source_id="two", category="code"))

    by_source = store.search("alpha", visibility=["internal"], sources=["one"])
    by_category = store.search("alpha", visibility=["internal"], categories=["code"])

    assert [hit.source_id for hit in by_source.hits] == ["one"]
    assert [hit.source_id for hit in by_category.hits] == ["two"]


def test_replacing_a_source_leaves_exactly_the_new_revision(store):
    store.replace_source(a_batch("old one", "old two", revision="v1"))
    store.replace_source(a_batch("new one", revision="v2"))

    result = store.search("one", visibility=["internal"])

    assert [hit.text for hit in result.hits] == ["new one"]
    assert result.hits[0].source_revision == "v2"
    assert store.status().chunks == 1


def test_replacing_a_source_at_the_same_revision_works(store):
    """Found by the first real ingest, which is later than it should have been.

    `chunk_id` hashes the revision, so re-running at an unchanged one rebuilds the
    ids already stored -- and the old delete spared exactly those rows, leaving the
    insert to collide on the primary key.
    """
    store.replace_source(a_batch("first go", revision="v1"))
    store.replace_source(a_batch("chunked differently", revision="v1"))

    result = store.search("chunked", visibility=["internal"])

    assert [hit.text for hit in result.hits] == ["chunked differently"]
    assert store.status().chunks == 1


def test_replacing_a_source_with_fewer_chunks_leaves_none_behind(store):
    """The other half of the same bug: a source that gets shorter.

    Overwriting on conflict would have fixed the collision above and still left
    rows 1 and 2 here, since nothing in the new batch has their ids.
    """
    store.replace_source(a_batch("one", "two", "three", revision="v1"))
    store.replace_source(a_batch("only this", revision="v2"))

    assert store.status().chunks == 1


def test_a_reader_never_sees_a_half_replaced_source(store):
    """The guarantee the whole pgvector migration was chosen for.

    One thread replaces a source in a loop while this one searches; every single
    result set must be wholly one revision or wholly the other, never a mixture
    and never empty.
    """
    store.replace_source(a_batch(*[f"chunk {i}" for i in range(20)], revision="v1"))

    stop = threading.Event()
    failures: list[str] = []

    def rewrite() -> None:
        for number in range(2, 8):
            revision = f"v{number}"
            store.replace_source(
                a_batch(
                    *[f"chunk {i} of {revision}" for i in range(20)], revision=revision
                )
            )
            time.sleep(0.01)
        stop.set()

    writer = threading.Thread(target=rewrite)
    writer.start()
    try:
        while not stop.is_set():
            hits = store.search("chunk 3", visibility=["internal"], k=20, n=20).hits
            if not hits:
                failures.append("a search saw no chunks at all")
                break
            revisions = {hit.source_revision for hit in hits}
            if len(revisions) != 1:
                failures.append(f"a search saw two revisions at once: {revisions}")
                break
    finally:
        writer.join()

    assert not failures, failures[0]


def test_upsert_changes_some_chunks_and_leaves_the_rest(store):
    store.replace_source(a_batch("alpha", "beta", "gamma", revision="v1"))

    changed = ChunkBatch(
        source=a_source(revision="v1"),
        chunks=[Chunk(text="alpha"), Chunk(text="BETA")],
        mode="upsert",
    )
    written = store.upsert_chunks(changed)

    texts = {hit.text for hit in store.search("alpha", visibility=["internal"]).hits}
    assert written == 2
    assert texts == {"alpha", "BETA", "gamma"}


def test_upsert_into_a_source_that_does_not_exist_says_so(store):
    batch = ChunkBatch(
        source=a_source(source_id="never-registered"),
        chunks=[Chunk(text="orphan")],
        mode="upsert",
    )
    from juena_rag.store.pgvector import SourceNotFound

    with pytest.raises(SourceNotFound, match="replace_source"):
        store.upsert_chunks(batch)


def test_deleting_one_files_chunks_leaves_the_source(store):
    store.replace_source(
        ChunkBatch(
            source=a_source(category="code"),
            chunks=[
                CodeChunk(text="def a(): ...", source_uri="a.py"),
                CodeChunk(text="def b(): ...", source_uri="b.py"),
            ],
        )
    )

    removed = store.delete_chunks("kws1-manual", uri="a.py")

    assert removed == 1
    assert store.status().chunks == 1
    assert store.status().sources == 1


def test_deleting_a_source_takes_its_chunks_with_it(store):
    store.replace_source(a_batch("alpha", "beta"))

    assert store.delete_source("kws1-manual") is True
    assert store.status().chunks == 0
    assert store.delete_source("kws1-manual") is False


def test_register_source_is_not_a_precondition_for_replace(store):
    """Two calls where one would do is a call somebody eventually forgets."""
    store.replace_source(a_batch("alpha"))

    assert [source.source_id for source in store.list_sources()] == ["kws1-manual"]


def test_register_source_declares_one_with_no_content_yet(store):
    store.register_source(a_source())

    catalogue = store.list_sources()

    assert len(catalogue) == 1
    assert catalogue[0].active_revision is None


def test_get_structure_resolves_only_a_visible_current_revision(store):
    store.replace_source(a_batch("alpha", revision="v3", visibility="internal"))

    assert store.get_structure("kws1-manual", visibility=["internal"]) is not None
    assert store.get_structure("kws1-manual", visibility=["public"]) is None
    assert (
        store.get_structure("kws1-manual", visibility=["internal"], revision="v3")
        is not None
    )
    assert (
        store.get_structure("kws1-manual", visibility=["internal"], revision="v1")
        is None
    )


def test_status_reports_one_embedding_model_for_a_consistent_corpus(store):
    store.replace_source(a_batch("alpha", "beta"))

    status = store.status()

    assert status.dimension == 8
    assert status.sources == 1
    assert status.chunks == 2
    assert status.embedding_models == ["fake-embedder"]


def test_the_search_query_reaches_the_vector_index(store):
    """The one failure with no symptom: a plan that works but scans everything.

    Postgres matches an index expression after parsing, so dropping the
    ``::bit(N)`` cast -- or letting the statistics go stale after an ingest --
    silently swaps the index for a sort over every matching row. Nothing about
    the results changes, only how long they take.
    """
    from pgvector import Vector

    from juena_rag.store.pgvector import _search_sql

    store.replace_source(
        ChunkBatch(
            source=a_source(),
            chunks=[Chunk(text=f"sentence number {i}") for i in range(1000)],
        )
    )

    parameters = {
        "query": Vector(FakeEmbedder(dimension=8).embed_query("sentence number 42")),
        "visibility": ["internal"],
        "sources": None,
        "categories": None,
        "chunk_kinds": None,
        "k": 100,
        "n": 5,
    }
    with store._pool.connection() as connection:
        rows = connection.execute(
            "EXPLAIN (COSTS OFF) " + _search_sql(8), parameters
        ).fetchall()
    plan = "\n".join(row[0] for row in rows)

    assert "chunks_embedding_bits_idx" in plan, plan
    assert "Seq Scan" not in plan, plan


def test_an_embedder_of_the_wrong_width_is_refused_at_construction(store, database):
    """The check that turns a failed insert six minutes in into a startup error."""
    from juena_rag.store.pgvector import DimensionMismatch, Store

    with pytest.raises(DimensionMismatch, match="vector\\(8\\)"):
        Store(
            store._pool.conninfo,
            embedder=FakeEmbedder(dimension=16, model="wider-embedder"),
        )
