"""Where chunks are kept: the table, and the client that reads and writes it.

Creating the table is separate from talking to it, so ``schema`` needs no database
driver at all -- it renders SQL and takes a connection somebody else opened::

    from juena_rag.store import apply_ddl, render_ddl  # always works

The two client modules are the exception and are deliberately **not** imported
here: each needs an extra, and this module must keep working on the base install.
Reach for them by name::

    from juena_rag.store.pgvector import Store       # needs [store]
    from juena_rag.store.http import HttpStore       # needs [client]

What they share -- the errors, ``StoreStatus`` and the ``ReadableStore`` contract
-- is in ``base``, which needs neither.
"""

from juena_rag.store.base import (
    API_PREFIX,
    DEFAULT_K,
    DEFAULT_N,
    DimensionMismatch,
    ReadableStore,
    SourceNotFound,
    StoreError,
    StoreStatus,
)
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
    "API_PREFIX",
    "CHUNK_INSERT_COLUMNS",
    "DEFAULT_K",
    "DEFAULT_N",
    "DIMENSION_PLACEHOLDER",
    "MAX_DIMENSION",
    "DimensionMismatch",
    "ReadableStore",
    "SchemaAlreadyApplied",
    "SchemaError",
    "SourceNotFound",
    "StoreError",
    "StoreStatus",
    "VectorExtensionMissing",
    "apply_ddl",
    "chunk_extra",
    "chunk_row",
    "chunk_rows",
    "read_ddl",
    "render_ddl",
    "source_row",
]
