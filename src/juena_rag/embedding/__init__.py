"""Turning text into vectors.

**This package embeds; the caller does not** -- two vectors are comparable only if
the same model made both, nothing catches a mismatch at write time, and the only
cure afterwards is to embed the whole corpus again. **The caller chunks; this
package does not** -- text arrives ready to embed and is sent whole, because where
a chunk should end depends on a format this package deliberately does not know::

    from juena_rag.embedding import Embedder, EmbeddingMetadata  # always works
    from juena_rag.embedding.blablador import BlabladorEmbedder  # needs [embed]
    from juena_rag.embedding.tokenizer import ModelTokenCounter  # needs [embed]
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
