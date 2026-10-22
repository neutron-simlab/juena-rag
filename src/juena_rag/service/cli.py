"""The ``juena-rag`` command: ``serve`` the HTTP service, or ``ingest-repo``.

    juena-rag serve --host 0.0.0.0
    juena-rag ingest-repo sampledb --config repositories.yaml --dry-run

Nothing heavy is imported at the top, so ``juena-rag --help`` still answers on the
base install where neither FastAPI nor psycopg is present. Neither subcommand
reads a ``.env`` file -- the package never reads files for its settings. From a
clone, ``set -a; source .env; set +a`` exports it first; ``uv run --env-file``
would not expand the ``${POSTGRES_PORT:-5433}`` references that file uses.
"""

import argparse
import sys
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: Where clones live, relative to wherever the command runs. From a juena-rag
#: checkout this is the same `data/repos` the old use-case script used.
DEFAULT_CACHE_DIR = Path("data/repos")


def main(argv: list[str] | None = None) -> int:
    """The console script. Returns the process's exit code."""
    parser = argparse.ArgumentParser(
        prog="juena-rag", description="The JCNS knowledge base: serve it, or fill it."
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

    ingest = subcommands.add_parser(
        "ingest-repo",
        help="clone a git repository, chunk it, embed it and store it as one source",
    )
    ingest.add_argument("repository", help="an id from the --config file")
    ingest.add_argument(
        "--config",
        type=Path,
        required=True,
        help="your repositories.yaml: which files to index, and the chunk_size and "
        "chunk_overlap to split them with",
    )
    ingest.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="where clones live (default: %(default)s)",
    )
    ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="chunk and count only -- no embedding, no database",
    )
    ingest.add_argument(
        "--no-sync",
        action="store_true",
        help="use the clone as it stands, without fetching -- for working offline, "
        "or comparing against the exact revision somebody else indexed",
    )
    ingest.add_argument(
        "--force",
        action="store_true",
        help="re-ingest even when the stored revision already matches",
    )
    ingest.add_argument(
        "--limit",
        type=int,
        help="only the first N files, for a cheap smoke test against the real model",
    )

    arguments = parser.parse_args(argv)

    if arguments.command == "ingest-repo":
        return _ingest(arguments)
    return _serve(
        host=arguments.host, port=arguments.port, log_level=arguments.log_level
    )


def _ingest(arguments: argparse.Namespace) -> int:
    """Run one ingest, turning a missing extra or a bad config into a sentence."""
    try:
        from juena_rag.ingest.code.pipeline import ingest_repository
    except ImportError as error:
        print(
            f"ingestion is not installed: {error}\n  pip install 'juena-rag[ingest]'",
            file=sys.stderr,
        )
        return 1

    # A config without chunk_size, or a missing API key, raises with a sentence
    # meant to be read -- print it rather than a traceback.
    try:
        return ingest_repository(
            arguments.repository,
            config_path=arguments.config,
            cache_dir=arguments.cache_dir,
            dry_run=arguments.dry_run,
            no_sync=arguments.no_sync,
            force=arguments.force,
            limit=arguments.limit,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2


def _serve(*, host: str, port: int, log_level: str) -> int:
    """Check the configuration is real, open the pool, then serve until stopped."""
    try:
        import uvicorn

        from juena_rag.embedding.base import verify_dimension
        from juena_rag.embedding.blablador import BlabladorEmbedder
        from juena_rag.reranking.blablador import BlabladorReranker
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
        reranker = BlabladorReranker.from_env()
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
    # No check at startup for the reranker: its Blablador alias is experimental
    # and can be down for hours, and a search that asks for reranking falls back
    # to the vector order meanwhile. Refusing to start would lose search entirely.
    print(
        f"reranking with {reranker.name} when a search asks for it, "
        f"{settings.rerank_candidates} candidates",
        flush=True,
    )

    with Store(
        settings.dsn,
        embedder=embedder,
        reranker=reranker,
        rerank_candidates=settings.rerank_candidates,
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
