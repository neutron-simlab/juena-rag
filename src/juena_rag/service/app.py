"""The endpoints: five of them, each one call to a `Store` method.

Needs the ``[serve]`` extra. The rule this file exists to keep is plan 02's --
a handler deserialises, calls exactly one method and serialises -- so a question
asked over the network and the same question asked through the library can never
be answered two different ways.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio.to_thread

from juena_rag import __version__
from juena_rag.schema.boundary import SearchRequest
from juena_rag.schema.models import SearchResult, SourceRow
from juena_rag.store.base import API_PREFIX, SourceNotFound, StoreStatus
from juena_rag.store.pgvector import Store

try:
    from fastapi import APIRouter, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - depends on the install, not the code
    raise ImportError(
        "the HTTP service needs FastAPI, which is not installed. "
        "`pip install 'juena-rag[serve]'`."
    ) from None

#: Which store failure becomes which status code. Anything absent is a 500 --
#: FastAPI's own default, and the honest answer for a bug in this service.
#:
#: ``ValueError`` is the one `search` raises when ``n > k``, which is a caller's
#: mistake and not a broken service. A malformed request body never reaches here:
#: FastAPI registers pydantic's own failure separately and answers it with a 422.
STATUS_BY_ERROR: dict[type[Exception], int] = {
    ValueError: 400,
    SourceNotFound: 404,
}


def create_app(store: Store, *, max_concurrency: int | None = None) -> FastAPI:
    """Build the app around one already-open store.

    The store is a parameter rather than something built here: it holds a
    connection pool that has to outlive a request and be closed by whoever opened
    it, which is :mod:`juena_rag.service.cli` in production and a fixture in tests.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Every handler below is `def`, not `async def`, so each one runs in
        # anyio's threadpool -- which is what keeps a blocking psycopg call off
        # the event loop. How many may run at once is the only thing to set here,
        # and it has to happen inside the loop, which is why it is not in the CLI.
        if max_concurrency is not None:
            limiter = anyio.to_thread.current_default_thread_limiter()
            limiter.total_tokens = max_concurrency
        yield

    app = FastAPI(
        title="juena-rag",
        version=__version__,
        summary="Where to look in the JCNS corpus, never the thing being looked at.",
        lifespan=lifespan,
    )

    for error_class, status_code in STATUS_BY_ERROR.items():
        app.add_exception_handler(error_class, _refuse_with(status_code))

    router = APIRouter(prefix=API_PREFIX)

    @router.post("/search")
    def search(request: SearchRequest) -> SearchResult:
        """The chunks closest to a question, and the documents they came from.

        A ``POST`` despite changing nothing, because a set of filters does not fit
        comfortably or safely in a query string.
        """
        return store.search(
            request.query,
            sources=request.sources,
            categories=request.categories,
            chunk_kinds=request.chunk_kinds,
            k=request.k,
            n=request.n,
            rerank=request.rerank,
        )

    @router.get("/sources")
    def list_sources() -> list[SourceRow]:
        """The catalogue, newest first."""
        return store.list_sources()

    @router.get("/sources/{source_id}/structure")
    def get_structure(source_id: str, revision: str | None = None) -> SourceRow:
        """One document, if ``revision`` is still the one it is serving.

        Required here although the method defaults it, because a network caller is
        the one who pins a superseded revision by accident and gets a citation
        pointing at a line that no longer holds what was quoted.
        """
        if revision is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "`revision` is required. This resolves a revision you already "
                    "hold; it does not guess at the current one."
                ),
            )

        found = store.get_structure(source_id, revision=revision)
        if found is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no source {source_id!r} at revision {revision!r}. Absent and "
                    "moved on are one answer here, because a caller cannot act on "
                    "the difference."
                ),
            )
        return found

    @router.get("/status")
    def status() -> StoreStatus:
        """What the corpus holds, and whether one model embedded all of it.

        **This scans the chunks table** -- 25 ms over the 38,477 rows in the live
        corpus -- so it is for a person or a monitor. A readiness probe calls
        ``/v1/health`` instead.
        """
        return store.status()

    @router.get("/health")
    def health() -> dict[str, str]:
        """Can this pod reach the database? What the readiness probe calls."""
        store.ping()
        return {"status": "ok"}

    app.include_router(router)
    return app


def _refuse_with(status_code: int):
    """Answer one class of exception with a body shaped like FastAPI's own errors.

    Same ``{"detail": …}`` shape as :class:`HTTPException`, so a caller parses one
    thing whether the refusal came from the framework or from the store.
    """

    async def handler(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(error)})

    return handler
