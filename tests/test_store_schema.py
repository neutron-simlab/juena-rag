"""Creating the tables -- `juena_rag.store.schema`.

The tests above the divider render SQL and need no database. The ones below need
one, and skip themselves when `JUENA_RAG_TEST_DSN` is unset.
"""

import os

import psycopg
import pytest

from juena_rag.store import (
    MAX_DIMENSION,
    SchemaAlreadyApplied,
    VectorExtensionMissing,
    apply_ddl,
    read_ddl,
    render_ddl,
)
from juena_rag.store import schema as schema_module

# --------------------------------------------------------------------------- #
# Rendering the SQL
# --------------------------------------------------------------------------- #


def statements_only(sql: str) -> str:
    """The SQL with its comment lines dropped.

    Every assertion about what the DDL *does* goes through this, so a test cannot
    be satisfied by a comment that happens to mention the right words.
    """
    return "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )


def test_the_ddl_ships_inside_the_package():
    """Read as package data, so an installed wheel can create its own tables."""
    text = read_ddl()

    assert "CREATE TABLE sources" in text
    assert "CREATE TABLE chunks" in text


def test_the_width_reaches_both_the_column_and_the_index():
    """One number, two places -- and the index is useless if it disagrees."""
    sql = statements_only(render_ddl(1024))

    assert "vector(1024)" in sql
    assert "bit(1024)" in sql
    assert "<dim>" not in render_ddl(1024)


def test_the_index_expression_is_the_one_a_query_has_to_repeat():
    """Written differently in a query, Postgres plans a sequential scan and is quiet.

    So the exact string is asserted here, where a change to it is visible, rather
    than discovered as a search that got slower for no apparent reason.
    """
    assert "binary_quantize(embedding)::bit(4096)" in statements_only(render_ddl(4096))


def test_the_chunk_kind_index_stays_partial():
    """An index over a value most rows share earns nothing and still costs writes."""
    assert "WHERE chunk_kind <> 'text'" in statements_only(render_ddl(4096))


def test_nothing_is_created_conditionally():
    """`IF NOT EXISTS` would turn applying this twice into a silent no-op.

    Re-applying should be a refusal instead: the second caller believes it created
    the tables it is about to write into.
    """
    assert "IF NOT EXISTS" not in statements_only(render_ddl(4096))


@pytest.mark.parametrize("dimension", [0, -1, MAX_DIMENSION + 1])
def test_an_impossible_width_is_refused(dimension):
    with pytest.raises(ValueError, match="pgvector's own limit|outside"):
        render_ddl(dimension)


@pytest.mark.parametrize("dimension", ["4096", 4096.0, None, True])
def test_a_width_that_is_not_an_int_is_refused(dimension):
    """Including `True`, which is an int in Python and would render `vector(True)`."""
    with pytest.raises(TypeError, match="must be an int"):
        render_ddl(dimension)


# --------------------------------------------------------------------------- #
# Against a real database
# --------------------------------------------------------------------------- #


def test_the_ddl_applies_to_a_fresh_database(database):
    apply_ddl(database, dimension=8)

    assert schema_module._existing_tables(database) == ["sources", "chunks"]


def test_the_vector_index_is_built_on_the_bits(database):
    """Not on the vector: 4096 floats are over every index size cap."""
    apply_ddl(database, dimension=8)

    definition = database.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_bits_idx'"
    ).fetchone()

    assert definition is not None, "the vector index was not created"
    assert "hnsw" in definition[0]
    assert "binary_quantize" in definition[0]
    assert "bit_hamming_ops" in definition[0]


def test_applying_it_twice_is_a_refusal(database):
    apply_ddl(database, dimension=8)

    with pytest.raises(SchemaAlreadyApplied, match="already exist"):
        apply_ddl(database, dimension=8)


def test_a_half_built_schema_is_refused_rather_than_completed(database):
    """The dangerous case: one table there, one missing.

    Filling in the gap would leave a table whose width nobody chose, so the check
    is for *any* of the two rather than for both.
    """
    database.execute("CREATE TABLE sources (source_id text PRIMARY KEY)")

    with pytest.raises(SchemaAlreadyApplied, match="sources"):
        apply_ddl(database, dimension=8)

    assert schema_module._existing_tables(database) == ["sources"]


def test_a_failure_part_way_through_leaves_the_database_untouched(
    database, monkeypatch
):
    """This is the whole reason it goes in one transaction."""
    broken = render_ddl(8) + "\nCREATE INDEX ON no_such_table (nope);"
    monkeypatch.setattr(schema_module, "render_ddl", lambda dimension: broken)

    with pytest.raises(psycopg.Error):
        apply_ddl(database, dimension=8)

    assert schema_module._existing_tables(database) == []


def test_the_width_is_pinned_into_the_column(database):
    """A vector of the wrong width is refused by the database, not by Python.

    This is why the width is a required argument: it is fixed when the column is
    created, and changing it afterwards means re-embedding the whole corpus.
    """
    apply_ddl(database, dimension=8)
    database.execute("INSERT INTO sources (source_id, category) VALUES ('s', 'manual')")

    with pytest.raises(psycopg.Error, match="8 dimensions"):
        database.execute(
            """
            INSERT INTO chunks (
                chunk_id, source_id, source_revision, chunk_index,
                category, visibility, text, content_hash,
                embedding, embedding_model, schema_version
            ) VALUES (
                'c', 's', 'r1', 0,
                'manual', 'internal', 'some text', 'deadbeef',
                '[1,2,3]', 'alias-qwen3-8b-embeddings', 1
            )
            """
        )


def test_a_database_without_the_extension_says_so(database):
    """`CREATE EXTENSION vector` needs superuser, which a cluster account will not have.

    So the default is to check rather than to try, and the message names who has to
    run it -- a permission error from Postgres explains none of that.
    """
    # The fixture guarantees this is set, and unlike `connection.info.dsn` it still
    # carries the password -- psycopg strips that from what it reports back.
    dsn = os.environ["JUENA_RAG_TEST_DSN"]
    try:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute("DROP DATABASE IF EXISTS juena_rag_without_vector")
            admin.execute("CREATE DATABASE juena_rag_without_vector")
    except psycopg.Error as error:
        pytest.skip(f"cannot create a database to test against: {error}")

    try:
        with (
            psycopg.connect(dsn, dbname="juena_rag_without_vector") as plain,
            pytest.raises(VectorExtensionMissing, match="superuser"),
        ):
            apply_ddl(plain, dimension=8)
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute("DROP DATABASE IF EXISTS juena_rag_without_vector")
