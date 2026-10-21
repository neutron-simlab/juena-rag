"""The read half, asserted against both stores at once.

`Store` reads the database and `HttpStore` reads the service that reads the
database, and plan 04 swaps one for the other by changing a constructor line. That
only works if they stay identical, so this file checks the two things that could
drift: the *shape* -- method names, argument names, return annotations, which needs
no database -- and the *behaviour*, which runs every read twice.
"""

import inspect

import pytest
from builders import a_source
from fakes import FailingReranker, FakeReranker
from fastapi.testclient import TestClient

from juena_rag import Chunk, ChunkBatch, SearchResult, SourceRow
from juena_rag.service.app import create_app
from juena_rag.store.base import StoreError, StoreStatus
from juena_rag.store.http import HttpStore
from juena_rag.store.pgvector import Store

#: The contract. `HttpStore` has exactly these and nothing else of `Store`'s.
READ_METHODS = ("search", "list_sources", "get_structure", "status", "ping")

#: Removed from the network by plan 02's S2, so their absence is the design and
#: not an omission -- ingestion opens the database directly, as a job.
WRITE_METHODS = (
    "register_source",
    "replace_source",
    "upsert_chunks",
    "delete_chunks",
    "delete_source",
)


def a_batch(*texts, **source_overrides) -> ChunkBatch:
    return ChunkBatch(
        source=a_source(**source_overrides),
        chunks=[Chunk(text=text) for text in texts],
    )


@pytest.fixture(params=["library", "http"])
def any_store(request, store):
    """The same corpus, reached both ways -- every test below runs twice.

    ``TestClient`` is an ``httpx.Client``, so the HTTP half talks to the real app
    in this process: no port, no server, and the same code path a network caller
    takes.
    """
    if request.param == "library":
        yield store
        return

    with TestClient(create_app(store)) as client:
        yield HttpStore("http://testserver", client=client)


# --------------------------------------------------------------------------- #
# The shape -- no database, so this half runs anywhere
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", READ_METHODS)
def test_the_client_has_every_read_method(name):
    assert callable(getattr(HttpStore, name, None)), f"HttpStore is missing {name}()"


@pytest.mark.parametrize("name", WRITE_METHODS)
def test_the_client_has_none_of_the_write_methods(name):
    """Missing on purpose. A read method missing would be a bug; these are not."""
    assert not hasattr(HttpStore, name), (
        f"HttpStore grew {name}(). There is no write side over HTTP -- see plan 02's S2."
    )


@pytest.mark.parametrize("name", READ_METHODS)
def test_the_read_methods_take_the_same_arguments(name):
    """Names *and* kinds: a keyword-only argument turned positional still breaks."""
    library = inspect.signature(getattr(Store, name)).parameters
    client = inspect.signature(getattr(HttpStore, name)).parameters

    assert list(library) == list(client)
    assert [p.kind for p in library.values()] == [p.kind for p in client.values()]
    assert [p.default for p in library.values()] == [p.default for p in client.values()]


@pytest.mark.parametrize(
    ("name", "returns"),
    [
        ("search", SearchResult),
        ("list_sources", list[SourceRow]),
        ("get_structure", SourceRow | None),
        ("status", StoreStatus),
        ("ping", None),
    ],
)
def test_the_read_methods_return_the_same_shape(name, returns):
    for implementation in (Store, HttpStore):
        annotation = inspect.signature(getattr(implementation, name)).return_annotation
        assert annotation == returns, f"{implementation.__name__}.{name}"


def test_a_service_that_cannot_be_reached_says_which_one():
    """A connection failure is a `StoreError`, not an httpx exception two layers up."""
    unreachable = HttpStore("http://127.0.0.1:1", timeout=1.0)

    with pytest.raises(StoreError, match="cannot reach the juena-rag service"):
        unreachable.ping()


def test_the_client_refuses_a_structure_with_no_revision():
    """The one place the two differ, and it fails before the request is sent."""
    with pytest.raises(ValueError, match="`revision` is required"):
        HttpStore("http://testserver").get_structure("kws1-manual")


# --------------------------------------------------------------------------- #
# The behaviour -- every one of these runs twice
# --------------------------------------------------------------------------- #


def test_a_search_finds_the_chunk_whose_text_it_is(any_store, store):
    store.replace_source(a_batch("alpha", "beta", "gamma"))

    result = any_store.search("beta", n=1)

    assert result.hits[0].text == "beta"
    assert result.hits[0].distance == pytest.approx(0.0, abs=1e-6)


def test_the_documents_travel_with_the_hits(any_store, store):
    store.replace_source(a_batch("alpha", "beta"))

    result = any_store.search("alpha")

    assert [source.source_id for source in result.sources] == ["kws1-manual"]
    assert result.source_of(result.hits[0]).title is None


def test_the_filters_narrow_the_search(any_store, store):
    store.replace_source(a_batch("alpha", source_id="one", category="manual"))
    store.replace_source(a_batch("alpha", source_id="two", category="code"))

    by_source = any_store.search("alpha", sources=["two"])
    by_category = any_store.search("alpha", categories=["manual"])

    assert [hit.source_id for hit in by_source.hits] == ["two"]
    assert [hit.source_id for hit in by_category.hits] == ["one"]


def test_asking_for_more_than_was_retrieved_is_refused(any_store):
    """A `ValueError` from both, so a caller catches one thing."""
    with pytest.raises(ValueError, match="exceeds"):
        any_store.search("alpha", k=2, n=10)


def test_the_catalogue_lists_every_source(any_store, store):
    store.replace_source(a_batch("alpha", source_id="one"))
    store.replace_source(a_batch("beta", source_id="two"))

    assert {source.source_id for source in any_store.list_sources()} == {"one", "two"}


def test_the_catalogue_reports_how_many_chunks_each_source_has(any_store, store):
    """The question a caller asks to decide whether a source is worth searching."""
    store.replace_source(a_batch("alpha", "beta", "gamma", source_id="one"))
    store.replace_source(a_batch("delta", source_id="two"))

    counts = {
        source.source_id: source.chunk_count for source in any_store.list_sources()
    }

    assert counts == {"one": 3, "two": 1}


def test_a_registered_source_with_no_chunks_reports_zero(any_store, store):
    """Zero and absent are different answers, and a caller acts on the difference."""
    store.register_source(a_source(source_id="declared-only"))

    listed = {
        source.source_id: source.chunk_count for source in any_store.list_sources()
    }

    assert listed["declared-only"] == 0


def test_replacing_a_source_recounts_rather_than_accumulates(any_store, store):
    """A stored count would drift here; a counted one cannot."""
    store.replace_source(a_batch("alpha", "beta", "gamma", revision="v3"))
    store.replace_source(a_batch("only one now", revision="v4"))

    listed = {
        source.source_id: source.chunk_count for source in any_store.list_sources()
    }

    assert listed["kws1-manual"] == 1


def test_a_structure_at_the_active_revision_is_the_document(any_store, store):
    store.replace_source(a_batch("alpha", revision="v3"))

    found = any_store.get_structure("kws1-manual", revision="v3")

    assert found is not None
    assert found.active_revision == "v3"
    assert found.chunk_count == 1


def test_a_structure_at_a_superseded_revision_is_nothing(any_store, store):
    store.replace_source(a_batch("alpha", revision="v3"))
    store.replace_source(a_batch("alpha again", revision="v4"))

    assert any_store.get_structure("kws1-manual", revision="v3") is None


def test_a_structure_of_a_source_that_does_not_exist_is_nothing(any_store):
    assert any_store.get_structure("nothing-here", revision="v1") is None


def test_status_counts_what_is_stored(any_store, store):
    store.replace_source(a_batch("alpha", "beta"))

    reported = any_store.status()

    assert reported.dimension == 8
    assert (reported.sources, reported.chunks) == (1, 2)
    assert reported.embedding_models == ["fake-embedder"]


def test_ping_returns_nothing_and_raises_nothing(any_store):
    assert any_store.ping() is None


# --------------------------------------------------------------------------- #
# Reranking -- the second stage, asked for one search at a time
# --------------------------------------------------------------------------- #


def test_a_reranked_search_puts_the_rerankers_choice_first(any_store, store):
    store.replace_source(a_batch("alpha", "beta", "gamma"))
    store.reranker = FakeReranker(favourite="gamma")

    plain = any_store.search("beta", n=2)
    reranked = any_store.search("beta", n=2, rerank=True)

    assert plain.hits[0].text == "beta" and plain.reranked_by is None
    assert reranked.hits[0].text == "gamma"
    assert reranked.reranked_by == "fake-reranker" and reranked.rerank_error is None
    assert [hit.rerank_score for hit in reranked.hits] == [1.0, 0.0]
    # The distance is still the database's own number, beside the reranker's.
    assert reranked.hits[0].distance > 0


def test_the_reranker_reads_only_the_closest_candidates(any_store, store):
    store.replace_source(a_batch("alpha", "beta", "gamma", "delta", "epsilon"))
    store.reranker = FakeReranker(favourite="gamma")
    store.rerank_candidates = 3

    any_store.search("beta", n=1, rerank=True)

    [candidates] = store.reranker.calls
    assert len(candidates) == 3 and candidates[0] == "beta"


def test_a_reranker_that_is_down_leaves_the_distance_order_and_says_why(
    any_store, store
):
    """The search still answers. Only the order is the first stage's."""
    store.replace_source(a_batch("alpha", "beta", "gamma"))
    expected = [hit.text for hit in store.search("beta", n=2).hits]
    store.reranker = FailingReranker()

    result = any_store.search("beta", n=2, rerank=True)

    assert [hit.text for hit in result.hits] == expected
    assert result.reranked_by is None
    assert "temporarily unavailable" in result.rerank_error
    assert all(hit.rerank_score is None for hit in result.hits)


def test_reranking_without_a_reranker_is_refused(any_store):
    """A setup mistake rather than a model being down, so it is loud.

    The fixture's store has no reranker. The library raises `ValueError`; the
    service answers 400, which the client reads back as the same `ValueError`.
    """
    with pytest.raises(ValueError, match="without a reranker"):
        any_store.search("alpha", rerank=True)


# --------------------------------------------------------------------------- #
# The two of them, side by side
# --------------------------------------------------------------------------- #


def test_the_two_stores_answer_the_same_question_identically(store):
    """Not "both look right" -- the same objects, field for field."""
    store.replace_source(a_batch("alpha", "beta", "gamma", revision="v9"))

    with TestClient(create_app(store)) as client:
        over_http = HttpStore("http://testserver", client=client)

        assert over_http.search("beta", n=2) == store.search("beta", n=2)
        assert over_http.list_sources() == store.list_sources()
        assert over_http.status() == store.status()
        assert over_http.get_structure(
            "kws1-manual", revision="v9"
        ) == store.get_structure("kws1-manual", revision="v9")
