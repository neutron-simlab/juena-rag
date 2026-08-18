"""The same read methods, over the network. Needs the ``[client]`` extra.

An application that should not open the database gets identical behaviour from
``HttpStore(url=…)`` as from ``Store(dsn=…)``: same method names, same argument
names, same return shapes. Moving between them is one constructor line, which is
what makes the chatbot's cutover cheap.
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

    Holds no embedder and no database driver: the service does both, which is most
    of the point of it. It has no ``k``/``n`` defaults of its own either -- left
    unset, whatever the service is configured with applies.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: Any | None = None,
    ) -> None:
        """
        :param url: where the service is -- ``https://juena-rag.jcns.internal``.
        :param client: an already-built ``httpx.Client``, for tests. Every request
            goes to an absolute URL built from ``url``, so an injected client needs
            no ``base_url`` of its own.
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
    ) -> SearchResult:
        """Find the ``n`` chunks closest to ``query``, among ``k`` the index retrieved.

        The filters are checked here before the request is sent, because
        :class:`~juena_rag.schema.boundary.SearchRequest` is the same object the
        service validates -- a typo fails in the caller's own traceback.
        """
        request = SearchRequest(
            query=query,
            sources=list(sources) if sources is not None else None,
            categories=list(categories) if categories is not None else None,
            chunk_kinds=list(chunk_kinds) if chunk_kinds is not None else None,
            k=k,
            n=n,
        )
        return SearchResult.model_validate(
            self._read("POST", "/search", json=request.model_dump())
        )

    def list_sources(self) -> list[SourceRow]:
        """The catalogue, newest first."""
        return [SourceRow.model_validate(row) for row in self._read("GET", "/sources")]

    def get_structure(
        self,
        source_id: str,
        *,
        revision: str | None = None,
    ) -> SourceRow | None:
        """One document, if it is still at the revision asked for.

        **The one place the two stores differ.** ``revision`` is required here
        because the endpoint requires it (plan 02's S3), and refusing before the
        round trip says so more clearly than a `400` coming back would.
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
        """What is stored. Scans the chunks table server-side, so not a probe."""
        return StoreStatus.model_validate(self._read("GET", "/status"))

    def ping(self) -> None:
        """Reach the service, which reaches the database. Raises if either cannot."""
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
