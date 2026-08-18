"""Reading and writing the two tables -- the one class a caller actually holds.

Needs the ``[store]`` extra, which is why nothing in ``store/__init__.py`` imports
it: creating the table works on the base install, talking to it does not. The
table itself is created once by :func:`juena_rag.store.schema.apply_ddl`, never
here -- a service that creates its own tables at startup is a service that
quietly creates the wrong one.
"""

from collections.abc import Sequence
from typing import Any, Self

from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from juena_rag.embedding.base import Embedder
from juena_rag.schema.boundary import ChunkBatch
from juena_rag.schema.models import SearchHit, SearchResult, SourceMetadata, SourceRow
from juena_rag.store.rows import CHUNK_INSERT_COLUMNS, chunk_rows, source_row

#: How many candidates the index retrieves, and how many survive to the caller.
#: Measured over a million rows: k=100 gives 96.4% recall@10, k=200 gives 99.2%,
#: and above 200 the curve stops paying. Today's corpus is ~10k, where k=100 is
#: ~99.8% -- so this is a floor to raise as the corpus grows, not a ceiling.
DEFAULT_K = 100
DEFAULT_N = 5


class StoreError(Exception):
    """Base class, so a caller can catch everything this layer raises."""


class DimensionMismatch(StoreError):
    """The table's vector column is not as wide as the embedder's vectors."""


class SourceNotFound(StoreError):
    """A write named a source that has to exist first."""


class StoreStatus(BaseModel):
    """What a readiness check reports. Plan 02 serialises this out of ``/status``."""

    dimension: int
    sources: int
    chunks: int

    #: Every distinct ``embedding_model`` in the table. More than one means two
    #: sources were embedded by different models, whose vectors are not comparable
    #: -- search would mix them and rank the result as if it meant something.
    embedding_models: list[str]


class Store:
    """The pgvector-backed store: one pool, one embedder, nine methods.

    Opens its own connection pool from ``dsn`` and configures each connection once
    -- the pgvector type adapters, and the iterative-scan setting a filtered vector
    search silently needs. Close it, or use it as a context manager.
    """

    def __init__(
        self,
        dsn: str,
        *,
        embedder: Embedder,
        k: int = DEFAULT_K,
        n: int = DEFAULT_N,
        min_size: int = 1,
        max_size: int = 4,
        check_dimension: bool = True,
        analyze_after_write: bool = True,
    ) -> None:
        self._embedder = embedder
        self._dimension = embedder.metadata.dimension
        self.k = k
        self.n = n
        self._analyze_after_write = analyze_after_write

        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            configure=_configure_connection,
            open=False,
        )
        self._pool.open()

        # A constructor that raises leaves nobody holding the object, so nobody
        # can close the pool it already opened. Close it here or the connections
        # leak until the garbage collector happens to notice.
        try:
            if check_dimension:
                self._check_dimension()
        except Exception:
            self._pool.close()
            raise

    # ----------------------------------------------------------------- #
    # Lifecycle
    # ----------------------------------------------------------------- #

    def close(self) -> None:
        """Close every pooled connection. Safe to call twice."""
        self._pool.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _check_dimension(self) -> None:
        """Compare the column's width to the embedder's, once, at construction.

        Without it the mismatch surfaces as a failed ``INSERT`` -- correct, but
        six minutes into an ingest rather than at startup.
        """
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
            ).fetchone()
        if row is None:
            raise StoreError(
                "no `chunks` table here. Run `apply_ddl()` once against this "
                "database before opening a Store on it."
            )
        # For a pgvector column `atttypmod` is the dimension itself, with none of
        # the offset that `varchar` and friends apply.
        column_width = row[0]
        if column_width != self._dimension:
            raise DimensionMismatch(
                f"the `embedding` column is vector({column_width}) but "
                f"{self._embedder.metadata.model!r} makes vectors of "
                f"{self._dimension}. The column's width is fixed when the table is "
                "created, so this is either the wrong database or the wrong model."
            )

    # ----------------------------------------------------------------- #
    # Writing
    # ----------------------------------------------------------------- #

    def register_source(self, metadata: SourceMetadata) -> None:
        """Declare a source before it has any content.

        Never a precondition for :meth:`replace_source`, which registers the source
        itself -- two calls where one would do is a call somebody eventually forgets.
        """
        with self._pool.connection() as connection:
            _upsert_source(connection, metadata)

    def replace_source(self, batch: ChunkBatch) -> str:
        """Replace a source's entire content, atomically. Returns the new revision.

        Readers see the old revision or the new one and never a half-built index.
        Everything -- embedding included -- happens inside one transaction, so a
        failure anywhere leaves the previous revision serving.
        """
        vectors = self._embedder.embed_documents([chunk.text for chunk in batch.chunks])
        rows = chunk_rows(batch, vectors, self._embedder.metadata.model)
        revision = batch.source.revision

        with self._pool.connection() as connection, connection.transaction():
            _upsert_source(connection, batch.source)
            # Every chunk of the source goes, before the new ones arrive -- not
            # just the other revisions, and not after the insert.
            #
            # `chunk_id` hashes (source_id, revision, chunk_index), so re-running
            # this at an *unchanged* revision produces exactly the ids already
            # stored. Deleting `source_revision <> revision` afterwards spared
            # them, and the insert collided on the primary key: "replace" failed
            # on the one input it should have been most able to handle. Deleting
            # first, unconditionally, also drops rows past the end when a source
            # gets shorter -- which conflict-update alone would have left behind.
            #
            # Still atomic. The delete and the insert are one transaction, so a
            # concurrent reader sees the previous content until it commits.
            connection.execute(
                "DELETE FROM chunks WHERE source_id = %s", (batch.source.source_id,)
            )
            _insert_chunks(connection, rows)
            connection.execute(
                "UPDATE sources SET active_revision = %s, updated_at = now() "
                "WHERE source_id = %s",
                (revision, batch.source.source_id),
            )
        self._analyze()
        return revision

    def upsert_chunks(self, batch: ChunkBatch) -> int:
        """Add or overwrite individual chunks without touching the rest. Returns how many.

        For incremental ingestion, where one file of many changed. ``chunk_index``
        is still position within the batch, so a caller must number a source's
        chunks continuously rather than restarting at zero per file.
        """
        vectors = self._embedder.embed_documents([chunk.text for chunk in batch.chunks])
        rows = chunk_rows(batch, vectors, self._embedder.metadata.model)
        if not rows:
            return 0

        with self._pool.connection() as connection, connection.transaction():
            if not _source_exists(connection, batch.source.source_id):
                raise SourceNotFound(
                    f"no source {batch.source.source_id!r}. upsert_chunks() changes "
                    "part of an existing source; use replace_source() to create one."
                )
            _insert_chunks(connection, rows, on_conflict_update=True)
        self._analyze()
        return len(rows)

    def _analyze(self) -> None:
        """Refresh the planner's statistics after a bulk write. Measured, not assumed.

        Without it the planner still believes whatever it last sampled -- right
        after a first ingest, that the table is nearly empty -- and answers a
        search with a filter-and-sort over every matching row instead of using the
        HNSW index. Autoanalyze gets there on its own, a minute or so later; these
        few milliseconds close the window where searches are slow and nothing says
        why.
        """
        if not self._analyze_after_write:
            return
        with self._pool.connection() as connection:
            connection.execute("ANALYZE chunks")

    def delete_chunks(self, source_id: str, *, uri: str) -> int:
        """Remove one file's chunks from a source. Returns how many went."""
        with self._pool.connection() as connection:
            cursor = connection.execute(
                "DELETE FROM chunks WHERE source_id = %s AND source_uri = %s",
                (source_id, uri),
            )
            return cursor.rowcount

    def delete_source(self, source_id: str) -> bool:
        """Remove a source and every chunk of it. ``ON DELETE CASCADE`` does the chunks."""
        with self._pool.connection() as connection:
            cursor = connection.execute(
                "DELETE FROM sources WHERE source_id = %s", (source_id,)
            )
            return cursor.rowcount > 0

    # ----------------------------------------------------------------- #
    # Reading
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

        Every filter here is applied inside the SQL, never after it: filtering a
        result set in Python leaks through the count and the ordering.
        """
        k = self.k if k is None else k
        n = self.n if n is None else n
        if n > k:
            raise ValueError(
                f"n={n} exceeds k={k}: the database retrieves k candidates and only "
                "those can be returned, so asking for more yields fewer, quietly."
            )

        embedded = Vector(self._embedder.embed_query(query))
        parameters = {
            "query": embedded,
            "sources": list(sources) if sources is not None else None,
            "categories": list(categories) if categories is not None else None,
            "chunk_kinds": list(chunk_kinds) if chunk_kinds is not None else None,
            "k": k,
            "n": n,
        }

        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            # Transaction-local, so it cannot leak onto the next borrower of this
            # pooled connection. `set_config` rather than `SET` because only the
            # function form takes a parameter.
            cursor.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(k),))
            cursor.execute(_search_sql(self._dimension), parameters)
            hits = [SearchHit.model_validate(row) for row in cursor.fetchall()]

            # The same cursor: its first result set is already drained, and one
            # extra round trip for the documents beats repeating them per hit.
            found = sorted({hit.source_id for hit in hits})
            documents: list[SourceRow] = []
            if found:
                cursor.execute(
                    "SELECT * FROM sources WHERE source_id = ANY(%s)", (found,)
                )
                documents = [SourceRow.model_validate(row) for row in cursor.fetchall()]

        return SearchResult(sources=documents, hits=hits)

    def list_sources(self) -> list[SourceRow]:
        """The catalogue, newest first."""
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute("SELECT * FROM sources ORDER BY updated_at DESC, source_id")
            return [SourceRow.model_validate(row) for row in cursor.fetchall()]

    def get_structure(
        self,
        source_id: str,
        *,
        revision: str | None = None,
    ) -> SourceRow | None:
        """One document, if it is still at the revision asked for.

        ``None`` covers both ways this fails -- absent, or moved on -- because a
        caller that cannot act on the difference is better served by one answer
        than by two it has to branch on.
        """
        with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                "SELECT * FROM sources "
                "WHERE source_id = %(source_id)s "
                "  AND (%(revision)s::text IS NULL "
                "       OR active_revision = %(revision)s)",
                {"source_id": source_id, "revision": revision},
            )
            row = cursor.fetchone()
        return SourceRow.model_validate(row) if row else None

    def status(self) -> StoreStatus:
        """Readiness, and the one inconsistency worth reporting.

        ``embedding_models`` scans the chunks table, so this is a startup or
        health-check call rather than something to put on a request path.
        """
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT (SELECT count(*) FROM sources), "
                "       (SELECT count(*) FROM chunks), "
                "       (SELECT coalesce(array_agg(DISTINCT embedding_model), "
                "                        ARRAY[]::text[]) FROM chunks)"
            ).fetchone()
        assert row is not None  # three scalar subqueries always give one row
        return StoreStatus(
            dimension=self._dimension,
            sources=row[0],
            chunks=row[1],
            embedding_models=sorted(row[2]),
        )


# --------------------------------------------------------------------------- #
# The SQL, and the bits of it that are not one-liners
# --------------------------------------------------------------------------- #


def _configure_connection(connection: Connection) -> None:
    """Run once per pooled connection, before anybody borrows it.

    ``iterative_scan`` is the one that would otherwise be a silent bug: with a
    filter applied, HNSW returns fewer than ``k`` rows and says nothing.
    """
    register_vector(connection)
    connection.execute("SET hnsw.iterative_scan = strict_order")
    # The pool's docs are explicit: leave no transaction open here. `SET` without
    # `LOCAL` survives the commit and applies for the connection's whole life.
    connection.commit()


def _search_sql(dimension: int) -> str:
    """The two-stage query. ``dimension`` is interpolated for the same reason the DDL's is.

    A cast to ``bit(N)`` is a type, and a type cannot be a query parameter -- so
    the number goes into the text, exactly as it does in ``ddl.sql``.
    """
    return f"""
        WITH candidates AS (
            SELECT chunk_id, source_id, source_revision, chunk_index,
                   text, source_uri, language, category, chunk_kind,
                   content_hash, extra, embedding_model, schema_version,
                   embedding <=> %(query)s AS distance
            FROM chunks
            WHERE (%(sources)s::text[] IS NULL OR source_id = ANY(%(sources)s))
              AND (%(categories)s::text[] IS NULL OR category = ANY(%(categories)s))
              AND (%(chunk_kinds)s::text[] IS NULL OR chunk_kind = ANY(%(chunk_kinds)s))
            -- The indexed expression, applied to the same column with the same
            -- cast. Drop the `::bit(N)` and Postgres plans a sequential scan
            -- without saying so.
            ORDER BY binary_quantize(embedding)::bit({dimension})
                     <~> binary_quantize(%(query)s)::bit({dimension})
            LIMIT %(k)s
        )
        SELECT * FROM candidates ORDER BY distance LIMIT %(n)s
    """


_SOURCE_UPSERT = """
    INSERT INTO sources (source_id, category, title,
                         artifact_uri, artifact_format, artifact_hash, parser)
    VALUES (%(source_id)s, %(category)s, %(title)s,
            %(artifact_uri)s, %(artifact_format)s, %(artifact_hash)s, %(parser)s)
    ON CONFLICT (source_id) DO UPDATE SET
        category        = EXCLUDED.category,
        title           = EXCLUDED.title,
        artifact_uri    = EXCLUDED.artifact_uri,
        artifact_format = EXCLUDED.artifact_format,
        artifact_hash   = EXCLUDED.artifact_hash,
        parser          = EXCLUDED.parser,
        updated_at      = now()
"""


def _upsert_source(connection: Connection, metadata: SourceMetadata) -> None:
    """Write the `sources` row, leaving ``active_revision`` where it is."""
    connection.execute(_SOURCE_UPSERT, source_row(metadata))


def _source_exists(connection: Connection, source_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sources WHERE source_id = %s", (source_id,)
    ).fetchone()
    return row is not None


def _insert_chunks(
    connection: Connection,
    rows: list[dict[str, Any]],
    *,
    on_conflict_update: bool = False,
) -> None:
    """Insert prepared rows, wrapping the two values psycopg needs told about.

    ``executemany`` rather than ``COPY`` deliberately: the embedding call ahead of
    this takes minutes and the insert takes seconds, so the faster path would buy
    nothing that anybody could measure.
    """
    if not rows:
        return
    placeholders = ", ".join(f"%({column})s" for column in CHUNK_INSERT_COLUMNS)
    statement = f"INSERT INTO chunks ({', '.join(CHUNK_INSERT_COLUMNS)}) VALUES ({placeholders})"
    if on_conflict_update:
        assignments = ", ".join(
            f"{column} = EXCLUDED.{column}"
            for column in CHUNK_INSERT_COLUMNS
            if column != "chunk_id"
        )
        statement += f" ON CONFLICT (chunk_id) DO UPDATE SET {assignments}"

    adapted = [
        {**row, "extra": Json(row["extra"]), "embedding": Vector(row["embedding"])}
        for row in rows
    ]
    with connection.cursor() as cursor:
        cursor.executemany(statement, adapted)
