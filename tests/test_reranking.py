"""The Blablador reranker, against a pretend endpoint. No network.

The client is the real ``openai`` one, wired to an in-process transport, so these
go through the SDK's own handling of a 503 and of a plain-text body rather than
through a stand-in for it. Searching with a reranker is tested in
``test_store_read_contract.py``, both through the library and over HTTP.
"""

import json

import pytest

pytest.importorskip("openai", reason="needs the `[embed]` extra")

import httpx2  # noqa: E402 -- the HTTP library the installed `openai` is built on
from openai import OpenAI  # noqa: E402

from juena_rag.reranking import RerankFailed  # noqa: E402
from juena_rag.reranking.blablador import DEFAULT_MODEL, BlabladorReranker  # noqa: E402


def a_reranker(answer, **settings):
    """A reranker whose every request is answered by ``answer(request)``.

    Returns the reranker and the list every request body lands in.
    """
    bodies = []

    def handle(request):
        bodies.append({"path": request.url.path, **json.loads(request.content)})
        return answer(request)

    client = OpenAI(
        api_key="test-key",
        base_url="https://blablador.test/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
    )
    return BlabladorReranker(api_key="unused", client=client, **settings), bodies


def scored_by(score_of):
    """An endpoint that answers each document with ``score_of(its text)``."""

    def answer(request):
        documents = json.loads(request.content)["documents"]
        results = [
            {"index": index, "relevance_score": score_of(text)}
            for index, text in enumerate(documents)
        ]
        return httpx2.Response(200, json={"results": results})

    return answer


# --------------------------------------------------------------------------- #
# What is sent
# --------------------------------------------------------------------------- #


def test_candidates_go_a_few_at_a_time_and_cut_to_size():
    """The endpoint was seen to take about ten short documents, so that is the shape."""
    reranker, bodies = a_reranker(
        scored_by(lambda text: 0.5), max_documents_per_request=10, max_document_chars=6
    )

    reranker.rerank("q", [f"document {i}" for i in range(23)], n=5)

    assert [len(body["documents"]) for body in bodies] == [10, 10, 3]
    assert all(len(text) <= 6 for body in bodies for text in body["documents"])
    assert {body["path"] for body in bodies} == {"/v1/rerank"}
    assert {body["model"] for body in bodies} == {DEFAULT_MODEL}
    assert {body["query"] for body in bodies} == {"q"}


def test_scores_from_separate_requests_are_ranked_together():
    """Each pair is scored on its own, so the best can sit in any request."""
    reranker, _ = a_reranker(
        scored_by(lambda text: 1.0 if text == "doc 12" else 0.1),
        max_documents_per_request=10,
    )

    best = reranker.rerank("q", [f"doc {i}" for i in range(15)], n=3)

    assert best[0] == (12, 1.0)
    assert len(best) == 3


# --------------------------------------------------------------------------- #
# What counts as failing -- every case here is one a search falls back from
# --------------------------------------------------------------------------- #


def test_a_model_that_is_temporarily_unavailable_is_a_rerank_failure():
    """What both Blablador reranking aliases answered on 2026-10-20."""
    reranker, _ = a_reranker(
        lambda request: httpx2.Response(
            503,
            json={"error": f"Model '{DEFAULT_MODEL}' is temporarily unavailable."},
        )
    )

    with pytest.raises(RerankFailed, match="temporarily unavailable"):
        reranker.rerank("q", ["a", "b"], n=1)


def test_a_plain_text_answer_is_a_rerank_failure():
    """Blablador answers a bad key with 200 and a sentence, not with an error."""
    reranker, _ = a_reranker(
        lambda request: httpx2.Response(
            200, text="You must provide a valid API key.", headers={"content-type": "text/plain"}
        )
    )

    with pytest.raises(RerankFailed, match="valid API key"):
        reranker.rerank("q", ["a", "b"], n=1)


def test_an_answer_that_skips_a_document_is_a_rerank_failure():
    """A dropped candidate would vanish from the ranking without a word."""
    reranker, _ = a_reranker(
        lambda request: httpx2.Response(
            200, json={"results": [{"index": 0, "relevance_score": 0.9}]}
        )
    )

    with pytest.raises(RerankFailed, match="scored positions"):
        reranker.rerank("q", ["a", "b"], n=2)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_the_environment_picks_the_model(monkeypatch):
    monkeypatch.setenv("BLABLADOR_API_KEY", "not-a-real-key")
    monkeypatch.setenv("BLABLADOR_RERANK_MODEL", "alias-qwen3-8b-reranking")

    assert BlabladorReranker.from_env(client=object()).name == "alias-qwen3-8b-reranking"


def test_no_api_key_is_refused_with_a_sentence(monkeypatch):
    monkeypatch.delenv("BLABLADOR_API_KEY", raising=False)

    with pytest.raises(ValueError, match="BLABLADOR_API_KEY"):
        BlabladorReranker.from_env()
