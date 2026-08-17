"""Where chunks are kept: the table, and the client that reads and writes it.

Creating the table is separate from talking to it, so ``schema`` needs no database
driver at all -- it renders SQL and takes a connection somebody else opened::

    from juena_rag.store import apply_ddl, render_ddl  # always works
"""

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
    "DIMENSION_PLACEHOLDER",
    "MAX_DIMENSION",
    "SchemaAlreadyApplied",
    "SchemaError",
    "VectorExtensionMissing",
    "apply_ddl",
    "read_ddl",
    "render_ddl",
]
