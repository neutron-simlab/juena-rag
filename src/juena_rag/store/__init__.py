"""Where chunks are kept: the table, and the client that reads and writes it.

Creating the table is separate from talking to it, so ``schema`` needs no database
driver at all -- it renders SQL and takes a connection somebody else opened::

    from juena_rag.store import apply_ddl, render_ddl  # always works

``pgvector.py`` is the exception and is deliberately **not** imported here: it
needs the ``[store]`` extra, and this module must keep working on the base
install. Reach for it by name::

    from juena_rag.store.pgvector import Store   # needs [store]
"""

from juena_rag.store.rows import (
    CHUNK_INSERT_COLUMNS,
    chunk_extra,
    chunk_row,
    chunk_rows,
    source_row,
)
from juena_rag.store.schema import (
    DIMENSION_PLACEHOLDER,
    MAX_DIMENSION,
    SchemaAlreadyApplied,
    SchemaError,
    VectorExtensionMissing,
    apply_ddl,
    read_ddl,
    render_ddl,
)

__all__ = [
    "CHUNK_INSERT_COLUMNS",
    "DIMENSION_PLACEHOLDER",
    "MAX_DIMENSION",
    "SchemaAlreadyApplied",
    "SchemaError",
    "VectorExtensionMissing",
    "apply_ddl",
    "chunk_extra",
    "chunk_row",
    "chunk_rows",
    "read_ddl",
    "render_ddl",
    "source_row",
]
