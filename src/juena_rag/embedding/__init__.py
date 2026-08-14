"""Turning text into vectors.

**This package embeds; the caller does not.** A caller sends text, never numbers.

That is not about saving anyone work. Two vectors can only be compared if the
same model made both, and nothing checks that at write time -- the widths match,
the writes succeed, the searches return rows. They are just the wrong rows, and
no query afterwards can tell you which half of the table came from which model.
The only way to find out is to embed the whole corpus again. Owning the model
here means there is only ever one of it.

**The caller chunks; this package does not.** Text arrives ready to embed and is
sent whole. Nothing here cuts, joins, pads or shortens anything, and there is no
chunker in this repository at all -- where a chunk should end depends on the
format it came from, and the format is exactly what this package does not know.

The model's input ceiling is published as
:attr:`EmbeddingMetadata.max_input_tokens` and measurable with
:mod:`juena_rag.embedding.tokenizer`, so a chunker can size its output against
the same vocabulary the server uses. Staying under it is the chunker's job.

Three modules:

===================== =========================================================
:mod:`base`           the contract, and the two shared checks
:mod:`tokenizer`      counting tokens the model's way -- **not imported here**
:mod:`blablador`      the one real client -- **not imported here**
===================== =========================================================

The last two lines are the point of the split. Importing this package gets you
pydantic and nothing else, so::

    from juena_rag.embedding import Embedder, EmbeddingMetadata  # always works
    from juena_rag.embedding.blablador import BlabladorEmbedder  # needs [embed]
    from juena_rag.embedding.tokenizer import ModelTokenCounter  # needs [embed]

To embed against something other than Blablador, write a class with a
``metadata`` attribute and the two methods on :class:`Embedder`. There is no base
class to inherit and nothing to register.
"""

from juena_rag.embedding.base import (
    DEFAULT_MAX_BATCH_TEXTS,
    DimensionMismatchError,
    Embedder,
    EmbeddingError,
    EmbeddingMetadata,
    MisalignedResponseError,
    NoEmbedderConfigured,
    check_width,
    take_in_order,
    verify_dimension,
)

__all__ = [
    "DEFAULT_MAX_BATCH_TEXTS",
    "DimensionMismatchError",
    "Embedder",
    "EmbeddingError",
    "EmbeddingMetadata",
    "MisalignedResponseError",
    "NoEmbedderConfigured",
    "check_width",
    "take_in_order",
    "verify_dimension",
]
