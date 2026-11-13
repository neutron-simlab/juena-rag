"""Clone a repository, chunk it, embed it, and store it as one source.

    juena-rag ingest-repo jscatter --config repositories.yaml --dry-run
    juena-rag ingest-repo jscatter --config repositories.yaml

One repository is one source: ``source_id`` is the repository's id, ``revision`` is
the git sha, and each file is a chunk's ``source_uri``. So a run replaces the whole
repository atomically -- a search during an ingest sees the old revision or the new
one, never a half-built index.

Needs the ``[ingest]`` extra. The command line lives in
:mod:`juena_rag.service.cli`; this module is the work it calls.
"""

import os
import sys
import time
from pathlib import Path

from juena_rag.ingest.code.chunking import chunk_file
from juena_rag.ingest.code.repository import (
    RepoConfig,
    current_revision,
    is_doc,
    list_files,
    load_repositories,
    read_file,
    sync,
)
from juena_rag.schema.boundary import ChunkBatch
from juena_rag.schema.models import SourceMetadata


def collect(config: RepoConfig, root: Path, limit: int | None) -> list:
    """Chunk every indexable file, numbered continuously across the repository.

    Continuity matters: ``chunk_index`` is a chunk's position in the batch, so
    restarting the count per file would give several chunks the same index. A
    file that cannot be read is skipped with a message.

    Args:
        config: The repository's settings.
        root: The clone's root directory.
        limit: Only the first this-many files. None means all of them.

    Returns:
        The chunks of every file, in sorted path order.
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

    Past the ceiling the model truncates and returns a normal-looking vector, so
    nothing else would report it.

    Args:
        chunks: The chunks to measure.
        ceiling: The model's input limit, in tokens.
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


def ingest_repository(
    repository_id: str,
    *,
    config_path: Path,
    cache_dir: Path,
    dry_run: bool = False,
    no_sync: bool = False,
    force: bool = False,
    limit: int | None = None,
) -> int:
    """Run one ingest.

    Settings come from two places, each owned by the caller: how to split the
    files from ``config_path``, and where to store them and which model embeds
    them from the environment (``JUENA_RAG_DSN``, ``BLABLADOR_*``).

    Args:
        repository_id: The repository's id in the config file.
        config_path: The ``repositories.yaml`` to read.
        cache_dir: The directory that holds every clone.
        dry_run: Chunk and count tokens only, without embedding or storing.
        no_sync: Use the existing clone as it stands, without fetching.
        force: Re-ingest even when the stored revision already matches.
        limit: Only the first this-many files. None means all of them.

    Returns:
        The process's exit code: 0 when done or already up to date, 1 when
        there is nothing to ingest or no database is set, 2 for an unknown
        repository or a missing clone.

    Raises:
        ValueError: If the config is invalid, ``BLABLADOR_API_KEY`` is not set,
            or a numeric ``BLABLADOR_*`` setting is not a whole number.
    """
    configs = load_repositories(config_path)
    config = configs.get(repository_id)
    if config is None:
        print(
            f"no repository {repository_id!r} in {config_path}. "
            f"Configured: {', '.join(sorted(configs))}",
            file=sys.stderr,
        )
        return 2

    print(f"{config.id}: {config.source.url} ({config.source.branch})")
    print(f"  chunk_size {config.chunk_size}, overlap {config.chunk_overlap}")
    if no_sync:
        root = cache_dir / config.id
        if not (root / ".git").is_dir():
            print(f"--no-sync, but there is no clone at {root}", file=sys.stderr)
            return 2
    else:
        root = sync(config, cache_dir)
    revision = current_revision(root)
    print(f"  at {revision[:12]} in {root}")

    chunks = collect(config, root, limit)
    if not chunks:
        print("nothing to ingest", file=sys.stderr)
        return 1

    if dry_run:
        report_tokens(chunks, ceiling=8192)
        kinds = {chunk.chunk_kind for chunk in chunks}
        docs = sum(1 for chunk in chunks if chunk.is_doc)
        named = sum(1 for chunk in chunks if chunk.symbol)
        print(f"  kinds: {sorted(kinds)}; {docs} doc chunks; {named} with a symbol")
        print("dry run -- nothing was embedded or stored")
        return 0

    # `JUENA_RAG_TEST_DSN` is only a fallback: the tests work in throwaway schemas
    # they drop, while this writes a corpus that stays. Both may point at one
    # database, as on a laptop, but they stay separately settable.
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
        stored = store.get_structure(config.id)
        if stored is not None and stored.active_revision == revision and not force:
            print(f"  already at {revision[:12]}; nothing to do (--force to re-ingest)")
            return 0

        batch = ChunkBatch(
            source=SourceMetadata(
                source_id=config.id,
                category="code",
                revision=revision,
                title=config.name,
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
