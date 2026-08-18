"""``juena-rag serve`` -- one `Store` built from the environment, handed to uvicorn.

Nothing heavy is imported at the top, so ``juena-rag --help`` still answers on the
base install where neither FastAPI nor psycopg is present.
"""

import argparse
import sys

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main(argv: list[str] | None = None) -> int:
    """The console script. Returns the process's exit code."""
    parser = argparse.ArgumentParser(
        prog="juena-rag", description="The JCNS knowledge base, over HTTP."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve = subcommands.add_parser("serve", help="run the read-only HTTP service")
    serve.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="which address to listen on. Use 0.0.0.0 in a container, where "
        "nothing outside the pod can reach its loopback (default: %(default)s)",
    )
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
    )
    arguments = parser.parse_args(argv)

    return _serve(
        host=arguments.host, port=arguments.port, log_level=arguments.log_level
    )


def _serve(*, host: str, port: int, log_level: str) -> int:
    """Check the configuration is real, open the pool, then serve until stopped."""
    try:
        import uvicorn

        from juena_rag.embedding.base import verify_dimension
        from juena_rag.embedding.blablador import BlabladorEmbedder
        from juena_rag.service.app import create_app
        from juena_rag.service.settings import ServiceSettings
        from juena_rag.store.pgvector import Store
    except ImportError as error:
        print(
            f"the service is not installed: {error}.\n  pip install 'juena-rag[serve]'",
            file=sys.stderr,
        )
        return 1

    # Both of these raise with a sentence meant to be read, so print it rather
    # than a traceback -- a missing environment variable is not a crash.
    try:
        settings = ServiceSettings.from_env()
        embedder = BlabladorEmbedder.from_env()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    # Two checks, in this order, because they catch different mistakes and both
    # are far cheaper here than on the first real search: `verify_dimension`
    # embeds one short string and asks the *model* whether the configured width is
    # real, and `Store` then asks the *column* the same question. Neither can see
    # what the other does, and a pod that fails either cannot serve a search.
    verify_dimension(embedder)
    print(
        f"embedding with {embedder.metadata.model}, "
        f"{embedder.metadata.dimension} numbers a vector",
        flush=True,
    )

    with Store(
        settings.dsn,
        embedder=embedder,
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
    ) as store:
        print(
            f"{settings.pool_max_size} database connections, "
            f"{settings.max_concurrency} requests at a time",
            flush=True,
        )
        application = create_app(store, max_concurrency=settings.max_concurrency)
        uvicorn.run(application, host=host, port=port, log_level=log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover - the console script calls main()
    raise SystemExit(main())
