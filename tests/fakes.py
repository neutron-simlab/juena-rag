"""Embedders and rerankers that never touch the network. Test scaffolding, not shipped.

They live here rather than in the package because none is something a user of
juena-rag should reach for: :class:`FakeEmbedder` invents numbers,
:class:`NullEmbedder` exists only to be the wrong answer, :class:`FakeReranker`
judges by one word, and :class:`FailingReranker` is a model that is down.
"""

import hashlib
import math

from juena_rag.embedding import EmbeddingMetadata, NoEmbedderConfigured
from juena_rag.reranking import RerankFailed


class FakeEmbedder:
    """Deterministic vectors of the right width, from a hash of the text.

    The same text always gives the same vector and different texts give different
    ones, which is all a test needs to assert that a vector reached the row it
    belongs to. The numbers mean nothing, so this proves wiring and never
    retrieval quality.
    """

    def __init__(
        self,
        dimension: int = 8,
        model: str = "fake-embedder",
        max_input_tokens: int = 8192,
    ) -> None:
        self.metadata = EmbeddingMetadata(
            model=model,
            dimension=dimension,
            max_input_tokens=max_input_tokens,
        )
        #: Every ``embed_documents`` call, in order -- so a test can check what
        #: was actually asked for.
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        # SHAKE-256 is a hash with a length you choose, so one call gives exactly
        # `dimension` bytes with no repeating pattern. Shifting and scaling puts
        # them in -1 to 1, and dividing by the length makes it a unit vector,
        # which is the shape a cosine distance expects.
        raw = hashlib.shake_256(text.encode("utf-8")).digest(self.metadata.dimension)
        values = [(byte - 127.5) / 127.5 for byte in raw]
        length = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / length for value in values]


class NullEmbedder:
    """An embedder that refuses, so "no embedder" is never a ``None`` surprise.

    Everything raises :class:`~juena_rag.embedding.NoEmbedderConfigured` with a
    sentence saying what to do, rather than an ``AttributeError`` on ``None``
    three frames away from the mistake. ``metadata`` raises too, because there is
    no honest answer and a placeholder width is exactly the plausible-looking
    wrong number that :func:`~juena_rag.embedding.verify_dimension` exists to
    catch.
    """

    _COMPLAINT = (
        "this was built without an embedder, so it can read but not search. "
        "Pass one -- BlabladorEmbedder.from_env() in production."
    )

    @property
    def metadata(self) -> EmbeddingMetadata:
        raise NoEmbedderConfigured(self._COMPLAINT)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NoEmbedderConfigured(self._COMPLAINT)

    def embed_query(self, text: str) -> list[float]:
        raise NoEmbedderConfigured(self._COMPLAINT)


class FakeReranker:
    """Scores a candidate 1.0 if it contains ``favourite``, otherwise 0.0.

    Ties keep the order they arrived in, which is the vector-distance order -- so a
    test can see exactly which candidate the reranker pulled up, and that nothing
    else moved. Proves wiring, never ranking quality.
    """

    def __init__(self, favourite: str) -> None:
        self.name = "fake-reranker"
        self.favourite = favourite
        #: Every candidate list it was handed, in order.
        self.calls: list[list[str]] = []

    def rerank(self, query: str, candidates: list[str], n: int) -> list[tuple[int, float]]:
        self.calls.append(list(candidates))
        scored = [
            (index, 1.0 if self.favourite in text else 0.0)
            for index, text in enumerate(candidates)
        ]
        # `sorted` keeps equal scores in their incoming order.
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:n]


class FailingReranker:
    """A reranking model that is down -- what Blablador answered on 2026-10-20."""

    name = "failing-reranker"

    def rerank(self, query: str, candidates: list[str], n: int) -> list[tuple[int, float]]:
        raise RerankFailed(
            f"{self.name}: Error code: 503 - Model is temporarily unavailable"
        )
