"""The one real reranker: Qwen3-Reranker on Blablador, through ``/v1/rerank``.

Blablador lists two reranking aliases, ``alias-qwen3-4b-reranking`` and
``alias-qwen3-8b-reranking``, both marked *experimental* -- they "come and go
without notice". So a failure here never fails a search: it raises
:class:`~juena_rag.reranking.base.RerankFailed`, and the store falls back to the
vector-distance order and says so.

The ``openai`` client has no rerank method, but it can call an endpoint it does
not wrap (``client.post``) and still applies its own API key, timeout and retries.
That is the same client the embedder uses, so this needs the ``[embed]`` extra and
nothing more.

**Two limits, observed rather than documented.** Another team calling the same
endpoint (eve-esa/backend#283, October 2026) found it accepts about ten short
documents per request, or a single document under 8,000 characters. A chunk can be
3,000 characters, so each candidate is cut to its first ``max_document_chars`` and
candidates are sent ``max_documents_per_request`` at a time. A cross-encoder scores
each (question, candidate) pair on its own, so scores from separate requests are
directly comparable.
"""

import os
from typing import Any

from juena_rag.reranking.base import RerankFailed

try:
    import openai
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on the install, not the code
    raise ImportError(
        "the Blablador reranker needs the `openai` package, which is not "
        "installed. `pip install 'juena-rag[embed]'`."
    ) from None

# The embedder's module owns the one copy of Blablador's address.
from juena_rag.embedding.blablador import DEFAULT_BASE_URL

#: The smaller of the two. It runs on every reranked search, where the person is
#: waiting, so the faster model is the default. ``BLABLADOR_RERANK_MODEL`` switches.
DEFAULT_MODEL = "alias-qwen3-4b-reranking"

#: See the module docstring: about ten documents per request is what the endpoint
#: was seen to accept, and 800 characters keeps ten of them near 8,000 in total.
DEFAULT_MAX_DOCUMENTS_PER_REQUEST = 10
DEFAULT_MAX_DOCUMENT_CHARS = 800


class BlabladorReranker:
    """Re-order candidates with a reranking model on Blablador.

    Every setting is a plain keyword argument with a default, like
    :class:`~juena_rag.embedding.blablador.BlabladorEmbedder`, and :meth:`from_env`
    fills them from the environment for a deployment.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_documents_per_request: int = DEFAULT_MAX_DOCUMENTS_PER_REQUEST,
        max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
        timeout: float = 10.0,
        max_retries: int = 0,
        client: Any | None = None,
    ) -> None:
        """
        :param timeout: seconds to wait for one request. Short on purpose: a
            person is waiting on the search, and a slow reranker costs less as a
            fallback to the vector order than as a hung tool call.
        :param max_retries: how many times the SDK retries a failed request.
            **Zero by default**, unlike the embedder's three. When the model is
            "temporarily unavailable", a retry only adds waiting before the same
            fallback.
        :param client: an already-built client, for tests. Anything with
            ``.post(path, body=…, cast_to=object)`` will do.
        """
        self.name = model
        self.max_documents_per_request = max_documents_per_request
        self.max_document_chars = max_document_chars
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> "BlabladorReranker":
        """Build from the environment the embedder already reads, plus the model.

        ``BLABLADOR_API_KEY`` and ``BLABLADOR_BASE_URL`` are shared with the
        embedder; ``BLABLADOR_RERANK_MODEL`` picks the reranking alias. Anything
        passed as a keyword wins.
        """
        api_key = os.getenv("BLABLADOR_API_KEY", "").strip()
        if not api_key and "api_key" not in overrides:
            raise ValueError(
                "BLABLADOR_API_KEY is not set. It is the only value with no "
                "default, because there is no sensible one."
            )

        settings: dict[str, Any] = {"api_key": api_key}
        base_url = os.getenv("BLABLADOR_BASE_URL", "").strip()
        if base_url:
            settings["base_url"] = base_url
        model = os.getenv("BLABLADOR_RERANK_MODEL", "").strip()
        if model:
            settings["model"] = model

        settings.update(overrides)
        return cls(**settings)

    def rerank(self, query: str, candidates: list[str], n: int) -> list[tuple[int, float]]:
        """Score every candidate, a few per request, and return the best ``n``.

        Stops at the first request that fails, rather than ranking half the
        candidates: a partial order would look complete and be wrong.
        """
        scored: list[tuple[int, float]] = []
        for start in range(0, len(candidates), self.max_documents_per_request):
            group = candidates[start : start + self.max_documents_per_request]
            for position, score in self._score(query, group):
                scored.append((start + position, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:n]

    def _score(self, query: str, group: list[str]) -> list[tuple[int, float]]:
        """One request: a score for each document in ``group``, by its position."""
        body = {
            "model": self.name,
            "query": query,
            "documents": [text[: self.max_document_chars] for text in group],
        }
        try:
            # `cast_to=object` hands back the decoded JSON as it arrived.
            response = self._client.post("/rerank", body=body, cast_to=object)
        except openai.APIError as error:
            # Every failure the SDK raises -- a 503, a 500, a timeout, a dropped
            # connection -- is one of these.
            raise RerankFailed(f"{self.name}: {error}") from error

        # Blablador answers a bad API key with 200 and a sentence of plain text,
        # not with an error, so the shape has to be checked before it is trusted.
        results = response.get("results") if isinstance(response, dict) else None
        if not isinstance(results, list):
            raise RerankFailed(
                f"{self.name} answered without a `results` list: {str(response)[:200]!r}"
            )

        scores: list[tuple[int, float]] = []
        for item in results:
            try:
                scores.append((int(item["index"]), float(item["relevance_score"])))
            except (KeyError, TypeError, ValueError):
                raise RerankFailed(
                    f"{self.name} returned a result without an index and a "
                    f"relevance_score: {str(item)[:200]!r}"
                ) from None

        # Every document scored exactly once. Fewer would silently drop a
        # candidate from the ranking; a repeat would rank one twice.
        if sorted(index for index, _ in scores) != list(range(len(group))):
            raise RerankFailed(
                f"{self.name} scored positions {sorted(i for i, _ in scores)} "
                f"for {len(group)} documents"
            )
        return scores
