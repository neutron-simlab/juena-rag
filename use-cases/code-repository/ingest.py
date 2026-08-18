"""Clone a repository, chunk it, embed it, and store it as one source.

    uv run --group use-cases python use-cases/code-repository/ingest.py jscatter --dry-run
    uv run --group use-cases python use-cases/code-repository/ingest.py jscatter

One repository is one source: ``source_id`` is the repository's id, ``revision`` is
the git sha, and each file is a chunk's ``source_uri``. So a run replaces the whole
repository atomically -- a search during an ingest sees the old revision or the new
one, never a half-built index.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from juena_rag.schema.boundary import ChunkBatch
from juena_rag.schema.models import SourceMetadata

from chunking import chunk_file  # isort: skip -- this directory is a flat import root
from repository import (  # isort: skip
    RepoConfig,
    current_revision,
    is_doc,
    list_files,
    load_repositories,
    read_file,
    sync,
)

HERE = Path(__file__).parent
DEFAULT_CONFIG = HERE / "repositories.yaml"

#: Where clones live, relative to the juena-rag checkout. Point `--cache-dir` at
#: the chatbot's `data/repos` to reuse a clone it already has.
DEFAULT_CACHE_DIR = HERE.parent.parent / "data" / "repos"

#: Public open-source code. Every chunk of every source here inherits this, so it
#: is a per-run decision rather than something buried in the YAML.
VISIBILITY = "public"


def collect(config: RepoConfig, root: Path, limit: int | None) -> list:
    """Chunk every indexable file, numbered continuously across the repository.

    Continuity matters: ``chunk_index`` is a chunk's position in the batch, so
    restarting the count per file would give several chunks the same index.
    """
    paths = list_files(config, root)
    if limit is not None:
        paths = paths[:limit]

    chunks = []
    for rel_path in paths:
        try:
            source = read_file(root, rel_path)
        except OSError as error:
            print(f"  skipped {rel_path}: {error}", file=sys.stderr)
            continue

        chunks.extend(
            chunk_file(
                source,
                rel_path,
                is_doc=is_doc(config, rel_path),
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        )

    print(f"  {len(paths)} files -> {len(chunks)} chunks")
    return chunks


def report_tokens(chunks: list, ceiling: int) -> None:
    """Measure the chunks against the model's input limit, and say so.

    Worth the two seconds the vocabulary takes to load: past the ceiling the model
    truncates and returns a normal-looking vector, so nothing else would ever
    mention it.
    """
    from juena_rag.embedding.tokenizer import ModelTokenCounter

    counts = ModelTokenCounter().count_many([chunk.text for chunk in chunks])
    over = [n for n in counts if n > ceiling]
    print(
        f"  tokens: max {max(counts)}, mean {sum(counts) // len(counts)}, "
        f"ceiling {ceiling} -- {len(over)} over"
    )
    if over:
        print(
            "  WARNING: the model truncates anything over the ceiling and returns "
            "a vector that looks fine. Lower chunk_size.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repository", help="an id from repositories.yaml")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="chunk and count only -- no embedding, no database",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="use the clone as it stands, without fetching -- for working offline, "
        "or comparing against the exact revision somebody else indexed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-ingest even when the stored revision already matches",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="only the first N files, for a cheap smoke test against the real model",
    )
    arguments = parser.parse_args()

    load_dotenv()

    configs = load_repositories(arguments.config)
    config = configs.get(arguments.repository)
    if config is None:
        parser.error(
            f"no repository {arguments.repository!r} in {arguments.config}. "
            f"Configured: {', '.join(sorted(configs))}"
        )

    print(f"{config.id}: {config.source.url} ({config.source.branch})")
    if arguments.no_sync:
        root = arguments.cache_dir / config.id
        if not (root / ".git").is_dir():
            parser.error(f"--no-sync, but there is no clone at {root}")
    else:
        root = sync(config, arguments.cache_dir)
    revision = current_revision(root)
    print(f"  at {revision[:12]} in {root}")

    chunks = collect(config, root, arguments.limit)
    if not chunks:
        print("nothing to ingest", file=sys.stderr)
        return 1

    if arguments.dry_run:
        report_tokens(chunks, ceiling=8192)
        kinds = {chunk.chunk_kind for chunk in chunks}
        docs = sum(1 for chunk in chunks if chunk.is_doc)
        named = sum(1 for chunk in chunks if chunk.symbol)
        print(f"  kinds: {sorted(kinds)}; {docs} doc chunks; {named} with a symbol")
        print("dry run -- nothing was embedded or stored")
        return 0

    # `JUENA_RAG_TEST_DSN` is the fallback rather than the name, because the test
    # suite works in throwaway schemas it drops afterwards and this writes a real
    # corpus that stays. Pointing both at one database is fine and is what a
    # laptop does; they should still be separately settable.
    dsn = os.environ.get("JUENA_RAG_DSN") or os.environ.get("JUENA_RAG_TEST_DSN")
    if not dsn:
        print(
            "set JUENA_RAG_DSN to a database with the schema applied "
            "(see .env.example, and `docker compose up -d`)",
            file=sys.stderr,
        )
        return 1

    from juena_rag.embedding.blablador import BlabladorEmbedder
    from juena_rag.store.pgvector import Store

    embedder = BlabladorEmbedder.from_env()
    report_tokens(chunks, ceiling=embedder.metadata.max_input_tokens)

    with Store(dsn, embedder=embedder) as store:
        stored = store.get_structure(config.id, visibility=[VISIBILITY])
        if (
            stored is not None
            and stored.active_revision == revision
            and not arguments.force
        ):
            print(f"  already at {revision[:12]}; nothing to do (--force to re-ingest)")
            return 0

        batch = ChunkBatch(
            source=SourceMetadata(
                source_id=config.id,
                category="code",
                revision=revision,
                title=config.name,
                visibility=VISIBILITY,
                # The four artifact columns stay unset. A repository is its own
                # original -- there is no parsed file to point back at.
            ),
            chunks=chunks,
            mode="replace",
        )

        print(f"  embedding {len(chunks)} chunks with {embedder.metadata.model} ...")
        started = time.monotonic()
        store.replace_source(batch)
        print(f"  done in {time.monotonic() - started:.0f}s")

        status = store.status()
        print(
            f"  store now holds {status.chunks} chunks across {status.sources} "
            f"source(s), embedded by {', '.join(status.embedding_models)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
