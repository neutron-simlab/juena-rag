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

#: How many candidates the index retrieves, and how many survive to the caller.
#: Measured over a million rows: k=100 gives 96.4% recall@10, k=200 gives 99.2%,
#: and above 200 the curve stops paying. Today's corpus is ~10k, where k=100 is
#: ~99.8% -- so this is a floor to raise as the corpus grows, not a ceiling.
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
    """The five methods a caller needs to *read* the corpus.

    Both stores implement exactly these, which is what lets an application swap
    one constructor line: ``Store`` also writes, ``HttpStore`` cannot and is not
    meant to. Deliberately not ``runtime_checkable``, for the same reason
    :class:`~juena_rag.embedding.base.Embedder` is not -- an ``isinstance`` check
    against a protocol only asks whether the attributes exist.
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
        """The ``n`` chunks closest to ``query``, among ``k`` the index retrieved.

        With ``rerank``, a reranking model re-orders the closest candidates first.
        """

    def list_sources(self) -> list[SourceRow]:
        """The catalogue, newest first."""

    def get_structure(
        self, source_id: str, *, revision: str | None = None
    ) -> SourceRow | None:
        """One document, if it is still at the revision asked for."""

    def status(self) -> StoreStatus:
        """What is stored. Scans the chunks table, so not a readiness probe."""

    def ping(self) -> None:
        """Reach the store and come straight back. Raises if it cannot."""
