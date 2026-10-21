"""What a reranker is, and the one way it is allowed to fail.

A search runs in two stages. The first compares the question's vector with every
chunk's vector, which the database can do quickly because the chunk vectors were
computed in advance -- but each side was embedded without seeing the other, so the
order is only roughly right. The second stage, the *reranker*, is a model that
reads the question and one candidate together and judges how well they match.
That is more accurate, and too slow for the whole corpus, so it only re-reads the
few dozen chunks the first stage found.

This module imports nothing beyond the standard library, so the store can accept
a reranker on the base install. The real one is
:class:`~juena_rag.reranking.blablador.BlabladorReranker`.
"""

from typing import Protocol


class RerankFailed(Exception):
    """The reranking model could not answer: down, overloaded, or refusing.

    The one failure :meth:`Store.search <juena_rag.store.pgvector.Store.search>`
    recovers from. It returns the vector-distance order instead, and says in
    ``SearchResult.rerank_error`` why. A bug in a reranker -- a ``TypeError``, say
    -- is not this, and still raises.
    """


class Reranker(Protocol):
    """Anything that re-orders search candidates by reading each with the question.

    It sees plain strings and returns positions, so it knows nothing about chunks,
    rows or this package's schema: a hosted model, a local one and a test fake all
    fit the same way.
    """

    #: Which model did the re-ordering. Every reranked result reports it, because
    #: its scores mean something only for that model.
    name: str

    def rerank(self, query: str, candidates: list[str], n: int) -> list[tuple[int, float]]:
        """Return ``(index into candidates, score)`` for the best ``n``, best first.

        A larger score means more relevant. Raises :class:`RerankFailed` when the
        model cannot answer.
        """
        ...
