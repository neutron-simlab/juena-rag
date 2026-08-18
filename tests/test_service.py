"""The HTTP service: what each endpoint answers, and what has no endpoint at all.

Runs the real app against the real store, in-process, on the same throwaway
schema the store tests use -- so a handler that stops matching its `Store` method
fails here rather than at cutover. Skips with the rest when there is no database.
"""

import anyio
import pytest
from builders import a_source
from fastapi.testclient import TestClient

from juena_rag import Chunk, ChunkBatch
from juena_rag.service.app import API_PREFIX, create_app


def a_batch(*texts, **source_overrides) -> ChunkBatch:
    return ChunkBatch(
        source=a_source(**source_overrides),
        chunks=[Chunk(text=text) for text in texts],
    )


@pytest.fixture
def client(store):
    """The service in this process, sharing the `store` fixture's schema.

    A context manager because that is what runs the lifespan; without it the
    startup block never executes and the test would prove less than it looks.
    """
    with TestClient(create_app(store)) as opened:
        yield opened


# --------------------------------------------------------------------------- #
# POST /v1/search
# --------------------------------------------------------------------------- #


def test_a_search_returns_hits_and_the_documents_they_came_from(client, store):
    store.replace_source(a_batch("The detector moves on rails.", "Neutrons scatter."))

    response = client.post(f"{API_PREFIX}/search", json={"query": "detector"})

    assert response.status_code == 200
    body = response.json()
    assert [source["source_id"] for source in body["sources"]] == ["kws1-manual"]
    assert len(body["hits"]) == 2
    hit = body["hits"][0]
    assert hit["source_id"] == "kws1-manual"
    assert hit["source_revision"] == "v3"
    assert "distance" in hit and "text" in hit


def test_the_closest_hit_is_the_chunk_whose_text_the_question_is(client, store):
    store.replace_source(a_batch("alpha", "beta", "gamma"))

    response = client.post(f"{API_PREFIX}/search", json={"query": "beta", "n": 1})

    hits = response.json()["hits"]
    assert hits[0]["text"] == "beta"
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-6)


def test_the_filters_reach_the_store(client, store):
    store.replace_source(a_batch("alpha", source_id="one", category="manual"))
    store.replace_source(a_batch("alpha", source_id="two", category="code"))

    response = client.post(
        f"{API_PREFIX}/search", json={"query": "alpha", "sources": ["two"]}
    )

    assert [hit["source_id"] for hit in response.json()["hits"]] == ["two"]


def test_asking_for_more_than_was_retrieved_is_the_callers_mistake(client):
    """`n > k` is a 400, not a 500: the request is wrong, the service is not."""
    response = client.post(
        f"{API_PREFIX}/search", json={"query": "alpha", "k": 2, "n": 10}
    )

    assert response.status_code == 400
    assert "n=10" in response.json()["detail"]


def test_a_misspelled_filter_is_refused_rather_than_ignored(client):
    """`source` for `sources` would otherwise search the whole corpus, quietly."""
    response = client.post(
        f"{API_PREFIX}/search", json={"query": "alpha", "source": ["one"]}
    )

    assert response.status_code == 422


def test_a_search_with_no_question_is_refused(client):
    assert client.post(f"{API_PREFIX}/search", json={"query": ""}).status_code == 422


# --------------------------------------------------------------------------- #
# GET /v1/sources, and one source's structure
# --------------------------------------------------------------------------- #


def test_the_catalogue_lists_every_source(client, store):
    store.replace_source(a_batch("alpha", source_id="one"))
    store.replace_source(a_batch("beta", source_id="two"))

    response = client.get(f"{API_PREFIX}/sources")

    assert response.status_code == 200
    assert {source["source_id"] for source in response.json()} == {"one", "two"}


def test_a_structure_with_no_revision_is_refused_rather_than_guessed(client, store):
    store.replace_source(a_batch("alpha"))

    response = client.get(f"{API_PREFIX}/sources/kws1-manual/structure")

    assert response.status_code == 400
    assert "revision" in response.json()["detail"]


def test_a_structure_at_the_active_revision_is_the_document(client, store):
    store.replace_source(a_batch("alpha", revision="v3"))

    response = client.get(
        f"{API_PREFIX}/sources/kws1-manual/structure", params={"revision": "v3"}
    )

    assert response.status_code == 200
    assert response.json()["active_revision"] == "v3"


def test_a_structure_at_a_superseded_revision_is_not_found(client, store):
    store.replace_source(a_batch("alpha", revision="v3"))
    store.replace_source(a_batch("alpha again", revision="v4"))

    response = client.get(
        f"{API_PREFIX}/sources/kws1-manual/structure", params={"revision": "v3"}
    )

    assert response.status_code == 404


def test_a_structure_of_a_source_that_does_not_exist_is_not_found(client):
    response = client.get(
        f"{API_PREFIX}/sources/nothing-here/structure", params={"revision": "v1"}
    )

    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# The two that report on the service itself
# --------------------------------------------------------------------------- #


def test_status_counts_what_is_stored(client, store):
    store.replace_source(a_batch("alpha", "beta"))

    body = client.get(f"{API_PREFIX}/status").json()

    assert body == {
        "dimension": 8,
        "sources": 1,
        "chunks": 2,
        "embedding_models": ["fake-embedder"],
    }


def test_health_says_the_database_is_reachable(client):
    response = client.get(f"{API_PREFIX}/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# The shape of the whole thing
# --------------------------------------------------------------------------- #


def test_every_endpoint_is_under_the_version_prefix(store):
    """Read off the OpenAPI schema, which is the same list a client generator uses."""
    paths = create_app(store).openapi()["paths"]

    assert len(paths) == 5, paths
    assert all(path.startswith(API_PREFIX) for path in paths), paths


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/sources"),  # register_source
        ("PUT", "/v1/sources/kws1-manual"),  # replace_source
        ("POST", "/v1/sources/kws1-manual/chunks"),  # upsert_chunks
        ("DELETE", "/v1/sources/kws1-manual/chunks"),  # delete_chunks
        ("DELETE", "/v1/sources/kws1-manual"),  # delete_source
    ],
)
def test_there_is_no_way_to_write_over_http(client, method, path):
    """Plan 02's S2: the five write methods have no endpoint, deliberately."""
    response = client.request(method, path, json={})

    # 405 where the path exists for reading, 404 where it does not. Both are
    # "there is nothing here"; what matters is that neither is a handler.
    assert response.status_code in (404, 405), response.text


def test_the_threadpool_is_sized_by_configuration(store):
    """S6's second number, which is set inside the event loop or not at all."""
    app = create_app(store, max_concurrency=7)

    async def start_it_and_look() -> float:
        async with app.router.lifespan_context(app):
            return anyio.to_thread.current_default_thread_limiter().total_tokens

    assert anyio.run(start_it_and_look) == 7
