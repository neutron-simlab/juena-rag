"""What both stores share: the errors, the status object, and the read contract.

These live here rather than beside the pgvector client because a consumer holding
only :class:`~juena_rag.store.http.HttpStore` must not have to install a database
driver to catch an exception. Same split as :mod:`juena_rag.embedding.base`:
everything here is a description or a plain model, so it imports pydantic and
nothing else.
"""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel

from juena_rag.schema.models import SearchResult, SourceRow

#: How many candidates the index retrieves (k), and how many survive to the
#: caller (n). Over a million rows, k=100 gives 96.4% recall@10 (the share of
#: the true ten closest chunks that come back), k=200 gives 99.2%, and above 200
#: it barely improves. At ~10k chunks k=100 gave ~99.8%; the corpus is now ~38k
#: and not re-measured, so raise k as the corpus grows.
DEFAULT_K = 100
DEFAULT_N = 5

#: How many of the closest chunks a reranker re-reads when a search asks for it.
#: Two requests' worth at Blablador's ~10 documents each: enough for the reranker
#: to pull a good chunk up from 15th place, without making a person wait on a
#: dozen model calls. Never more than ``k``, which is all the index retrieved.
DEFAULT_RERANK_CANDIDATES = 20

#: The HTTP API's version, in every path. Defined here because the service and
#: its client both need it, and a client pinning a different one than the server
#: serves is a 404 nobody can explain.
API_PREFIX = "/v1"


# --------------------------------------------------------------------------- #
# What went wrong
# --------------------------------------------------------------------------- #


class StoreError(Exception):
    """Base class, so a caller can catch everything this layer raises."""


class DimensionMismatch(StoreError):
    """The table's vector column is not as wide as the embedder's vectors."""


class SourceNotFound(StoreError):
    """A write named a source that has to exist first."""


# --------------------------------------------------------------------------- #
# What a readiness check reports
# --------------------------------------------------------------------------- #


class StoreStatus(BaseModel):
    """What is stored, and the one inconsistency worth reporting."""

    dimension: int
    sources: int
    chunks: int

    #: Every distinct ``embedding_model`` in the table. More than one means two
    #: sources were embedded by different models, whose vectors are not comparable
    #: -- search would mix them and rank the result as if it meant something.
    embedding_models: list[str]


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


class ReadableStore(Protocol):
    """The five methods a caller needs to read the corpus.

    Both stores implement these, so an application switches between them by
    changing one constructor line; only ``Store`` also writes. Not
    ``runtime_checkable``, for the same reason as
    :class:`~juena_rag.embedding.base.Embedder`.
    """

    def search(
        self,
        query: str,
        *,
        sources: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
        chunk_kinds: Sequence[str] | None = None,
        k: int | None = None,
        n: int | None = None,
        rerank: bool = False,
    ) -> SearchResult:
        """Find the ``n`` chunks closest to ``query``, among ``k`` the index retrieved.

        Args:
            query: The question.
            sources: Only these source ids. None means all.
            categories: Only these categories. None means all.
            chunk_kinds: Only these chunk kinds. None means all.
            k: How many candidates the index retrieves. None means the
                store's default.
            n: How many hits to return. None means the store's default.
            rerank: Have a reranking model re-order the closest candidates first.

        Returns:
            The hits and the documents they came from.
        """

    def list_sources(self) -> list[SourceRow]:
        """List the catalogue.

        Returns:
            Every source, newest first.
        """

    def get_structure(
        self, source_id: str, *, revision: str | None = None
    ) -> SourceRow | None:
        """Look up one document, if it is still at the revision asked for.

        Args:
            source_id: The source's id.
            revision: The revision the caller holds.

        Returns:
            The source's row, or None if it is absent or has moved on.
        """

    def status(self) -> StoreStatus:
        """Report what is stored. Scans the chunks table, so not a readiness probe.

        Returns:
            The counts and the embedding models in use.
        """

    def ping(self) -> None:
        """Reach the store and come straight back. Raises if it cannot."""
