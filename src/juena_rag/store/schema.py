"""Creating the tables, and pinning the embedding width into them.

The width has to be decided before a single row exists, because pgvector fixes it
when the column is created -- so it is a required argument here rather than a
default anyone could get wrong quietly. Reads ``ddl.sql``, needs no database to
render it, and imports nothing outside the standard library.
"""

from importlib.resources import files
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers only, never at runtime
    from psycopg import Connection

#: pgvector's own ceiling for the ``vector`` type. Nothing here can raise it.
MAX_DIMENSION = 16000

#: The placeholder in ``ddl.sql``. A plain word rather than ``{}`` so the SQL can
#: keep its own braces -- ``extra jsonb NOT NULL DEFAULT '{}'`` is one of them.
DIMENSION_PLACEHOLDER = "<dim>"


class SchemaError(Exception):
    """Base class, so a caller can catch everything this module raises."""


class SchemaAlreadyApplied(SchemaError):
    """The tables are already there, so nothing was changed."""


class VectorExtensionMissing(SchemaError):
    """The database has no ``vector`` extension, and creating it was not asked for."""


def read_ddl() -> str:
    """Return ``ddl.sql`` as it ships, placeholder and all.

    Read through :mod:`importlib.resources` rather than by path, so it also works
    from an installed wheel where the file is not next to a directory listing.
    """
    return files("juena_rag.store").joinpath("ddl.sql").read_text(encoding="utf-8")


def render_ddl(dimension: int) -> str:
    """Return the DDL with the embedding width filled in.

    The width is checked to be a plain positive integer before it is substituted,
    which is what makes writing it straight into the SQL safe -- pgvector cannot
    take it as a query parameter.
    """
    # `True` passes an `isinstance(..., int)` check in Python and would render
    # `vector(True)`, so booleans are excluded explicitly.
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError(f"`dimension` must be an int, not {type(dimension).__name__}")
    if not 1 <= dimension <= MAX_DIMENSION:
        raise ValueError(
            f"`dimension` is {dimension}, which is outside 1..{MAX_DIMENSION} -- "
            "pgvector's own limit for the `vector` type"
        )

    template = read_ddl()
    if DIMENSION_PLACEHOLDER not in template:
        raise SchemaError(
            f"ddl.sql carries no `{DIMENSION_PLACEHOLDER}` placeholder, so the "
            "embedding width would silently be whatever is written in the file"
        )
    return template.replace(DIMENSION_PLACEHOLDER, str(dimension))


def apply_ddl(
    connection: "Connection[Any]",
    *,
    dimension: int,
    create_extension: bool = False,
) -> None:
    """Create the tables and indexes in one transaction, or change nothing.

    ``create_extension`` defaults to off because ``CREATE EXTENSION vector`` needs
    superuser rights, which an application account on a managed cluster will not
    have -- there, an administrator runs it once and this only checks it happened.
    """
    with connection.transaction():
        if create_extension:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        elif not _has_vector_extension(connection):
            raise VectorExtensionMissing(
                "this database has no `vector` extension. A superuser runs "
                "`CREATE EXTENSION vector` once per database; pass "
                "create_extension=True only where this connection has the rights "
                "to do it itself, such as a local test container."
            )

        existing = _existing_tables(connection)
        if existing:
            raise SchemaAlreadyApplied(
                f"{', '.join(existing)} already exist(s), so nothing was changed. "
                "Applying this to a populated database is not a migration -- to "
                "start over, drop the tables first."
            )

        # One `execute` with the whole file: psycopg sends multiple statements in
        # one call as long as none of them carries a parameter, and none does.
        connection.execute(render_ddl(dimension))


def _has_vector_extension(connection: "Connection[Any]") -> bool:
    """Is the pgvector extension installed in this database?"""
    row = connection.execute(
        "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    return row is not None


def _existing_tables(connection: "Connection[Any]") -> list[str]:
    """Which of our two tables are already visible on the search path?"""
    row = connection.execute(
        "SELECT to_regclass('sources'), to_regclass('chunks')"
    ).fetchone()
    assert row is not None  # a two-column SELECT always returns exactly one row
    return [
        name for name, found in zip(("sources", "chunks"), row, strict=True) if found
    ]
