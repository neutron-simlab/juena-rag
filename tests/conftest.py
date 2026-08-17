"""Fixtures shared by the tests."""

import os
from uuid import uuid4

import pytest
from dotenv import load_dotenv

from juena_rag.schema.registry import _REGISTRY

# Copying `.env.example` to `.env` is all it takes to run the tests that need a key
# or a database. Real environment variables win over the file, so CI sets them
# normally and needs no file at all.
load_dotenv()


@pytest.fixture
def clean_registry():
    """Let a test define its own chunk classes without leaking them."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


@pytest.fixture
def database():
    """A connection into a throwaway namespace, skipping when there is no database.

    Every test gets its own schema on the search path and drops it afterwards, so
    the tests cannot touch anything else living in whatever `JUENA_RAG_TEST_DSN`
    points at -- and `uv run pytest` still works with no database at all.
    """
    dsn = os.environ.get("JUENA_RAG_TEST_DSN")
    if not dsn:
        pytest.skip("set JUENA_RAG_TEST_DSN to run the tests that need a database")

    import psycopg

    # Unreachable is as good a reason to skip as unset: a `.env` copied weeks ago
    # points at a container that is no longer running, and that is the case the
    # "works on a plane" promise is actually about.
    try:
        connection = psycopg.connect(dsn)
    except psycopg.OperationalError as error:
        pytest.skip(f"no database at JUENA_RAG_TEST_DSN: {error}")

    namespace = f"juena_rag_test_{uuid4().hex[:8]}"
    with connection:
        # A test database is somewhere you have rights; a cluster is not, which is
        # the difference `apply_ddl(create_extension=...)` exists to express.
        try:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.commit()
        except psycopg.Error as error:
            connection.rollback()
            pytest.skip(f"no `vector` extension and no rights to create one: {error}")

        # Committed before the test runs, so a test that aborts its own
        # transaction does not also lose the search path.
        connection.execute(f'CREATE SCHEMA "{namespace}"')
        connection.execute(f'SET search_path TO "{namespace}", public')
        connection.commit()
        try:
            yield connection
        finally:
            connection.rollback()
            connection.execute(f'DROP SCHEMA "{namespace}" CASCADE')
            connection.commit()


@pytest.fixture
def store(database):
    """A `Store` on the same throwaway schema, with the table already applied.

    The `Store` opens its own pool, so it cannot borrow the `database` fixture's
    connection -- it gets the same namespace through a `search_path` in its own
    connection string instead. Torn down before `database` drops the schema,
    because a pool still holding connections would block the ``DROP``.
    """
    from psycopg.conninfo import make_conninfo

    from juena_rag.store import apply_ddl
    from juena_rag.store.pgvector import Store

    from fakes import FakeEmbedder  # isort: skip -- tests/ is a flat import root

    namespace = database.execute("SELECT current_schema()").fetchone()[0]
    apply_ddl(database, dimension=8, create_extension=True)
    database.commit()

    dsn = make_conninfo(
        os.environ["JUENA_RAG_TEST_DSN"],
        options=f"-c search_path={namespace},public",
    )
    with Store(dsn, embedder=FakeEmbedder(dimension=8), k=20, n=5) as opened:
        yield opened
