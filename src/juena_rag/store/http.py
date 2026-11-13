"""The same read methods, over the network. Needs the ``[client]`` extra.

An application that should not open the database gets identical behaviour from
``HttpStore(url=…)`` as from ``Store(dsn=…)``: same method names, same argument
names, same return shapes. Moving between them is one constructor line.
"""

from collections.abc import Sequence
from typing import Any, Self
from urllib.parse import quote

from juena_rag.schema.boundary import SearchRequest
from juena_rag.schema.models import SearchResult, SourceRow
from juena_rag.store.base import API_PREFIX, SourceNotFound, StoreError, StoreStatus

try:
    import httpx
except ImportError:  # pragma: no cover - depends on the install, not the code
    raise ImportError(
        "the HTTP client needs `httpx`, which is not installed. "
        "`pip install 'juena-rag[client]'`."
    ) from None

#: How long to wait on the service. Well above a search -- 0.2 s measured against
#: the live corpus, most of it the embedding round trip -- and well under the
#: point where a person has given up.
DEFAULT_TIMEOUT = 30.0


class HttpStore:
    """The read half of :class:`~juena_rag.store.pgvector.Store`, over HTTP.

    Holds no embedder and no database driver: the service does both. It has no
    ``k``/``n`` defaults of its own either -- left unset, whatever the service is
    configured with applies.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: Any | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            url: Where the service is, such as
                ``https://juena-rag.jcns.internal``.
            timeout: Seconds to wait on the service.
            client: An already-built ``httpx.Client``, for tests. Every request
                goes to an absolute URL built from ``url``, so an injected client
                needs no ``base_url`` of its own. None means build one with
                ``timeout``.
        """
        self._url = url.rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    # ----------------------------------------------------------------- #
    # Lifecycle
    # ----------------------------------------------------------------- #

    def close(self) -> None:
        """Close the underlying connection. Safe to call twice."""
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ----------------------------------------------------------------- #
    # Reading -- the five methods `Store` also has
    # ----------------------------------------------------------------- #

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

        The filters are checked here before the request is sent, because
        :class:`~juena_rag.schema.boundary.SearchRequest` is the same object the
        service validates -- a typo fails in the caller's own traceback.

        Args:
            query: The question.
            sources: Only these source ids. None means all.
            categories: Only these categories. None means all.
            chunk_kinds: Only these chunk kinds. None means all.
            k: How many candidates the index retrieves. None means the
                service's default.
            n: How many hits to return. None means the service's default.
            rerank: Have the service's reranking model re-order the closest
                candidates first.

        Returns:
            The hits and the documents they came from.

        Raises:
            ValueError: If an argument is invalid, here or at the service.
            StoreError: If the service cannot be reached or answers with an
                error.
        """
        request = SearchRequest(
            query=query,
            sources=list(sources) if sources is not None else None,
            categories=list(categories) if categories is not None else None,
            chunk_kinds=list(chunk_kinds) if chunk_kinds is not None else None,
            k=k,
            n=n,
            rerank=rerank,
        )
        return SearchResult.model_validate(
            self._read("POST", "/search", json=request.model_dump())
        )

    def list_sources(self) -> list[SourceRow]:
        """List the catalogue.

        Returns:
            Every source, newest first.

        Raises:
            StoreError: If the service cannot be reached or answers with an
                error.
        """
        return [SourceRow.model_validate(row) for row in self._read("GET", "/sources")]

    def get_structure(
        self,
        source_id: str,
        *,
        revision: str | None = None,
    ) -> SourceRow | None:
        """Look up one document, if it is still at the revision asked for.

        The one place the two stores differ: ``revision`` is required here
        because the endpoint requires it, and refusing before the round trip
        says so more clearly than a `400` coming back would.

        Args:
            source_id: The source's id.
            revision: The revision the caller holds.

        Returns:
            The source's row, or None if it is absent or has moved on.

        Raises:
            ValueError: If ``revision`` is None.
            StoreError: If the service cannot be reached or answers with an
                error.
        """
        if revision is None:
            raise ValueError(
                "`revision` is required over HTTP: the endpoint resolves only a "
                "revision the caller already holds and does not guess at the "
                "current one. `Store` defaults it because a library caller holds "
                "the DSN and can already see everything."
            )

        response = self._send(
            "GET",
            f"/sources/{quote(source_id, safe='')}/structure",
            params={"revision": revision},
        )
        # A 404 covers both ways this fails -- absent, or moved on -- which is
        # exactly what the method answers with `None`.
        if response.status_code == 404:
            return None
        return SourceRow.model_validate(_checked(response))

    def status(self) -> StoreStatus:
        """Report what is stored. Scans the chunks table server-side, so not a probe.

        Returns:
            The counts and the embedding models in use.

        Raises:
            StoreError: If the service cannot be reached or answers with an
                error.
        """
        return StoreStatus.model_validate(self._read("GET", "/status"))

    def ping(self) -> None:
        """Reach the service, which reaches the database.

        Raises:
            StoreError: If the service cannot be reached or reports a failure.
        """
        self._read("GET", "/health")

    # ----------------------------------------------------------------- #
    # The two lines every method above goes through
    # ----------------------------------------------------------------- #

    def _send(self, method: str, path: str, **kwargs: Any) -> "httpx.Response":
        """One request, with an unreachable service turned into a `StoreError`."""
        try:
            return self._client.request(
                method, f"{self._url}{API_PREFIX}{path}", **kwargs
            )
        except httpx.RequestError as error:
            raise StoreError(
                f"cannot reach the juena-rag service at {self._url}: {error}"
            ) from error

    def _read(self, method: str, path: str, **kwargs: Any) -> Any:
        """The body of a successful response, or the exception it stands for."""
        return _checked(self._send(method, path, **kwargs))


def _checked(response: "httpx.Response") -> Any:
    """Turn a status code back into the exception the library would have raised.

    The same three the service maps on the way out, read in the other direction,
    so a caller catches one set of exceptions whichever store it holds.
    """
    if response.is_success:
        return response.json()

    detail = _detail(response)
    if response.status_code in (400, 422):
        raise ValueError(detail)
    if response.status_code == 404:
        raise SourceNotFound(detail)
    raise StoreError(f"the service answered {response.status_code}: {detail}")


def _detail(response: "httpx.Response") -> str:
    """The service's own sentence, or the raw body when it did not send one."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)
