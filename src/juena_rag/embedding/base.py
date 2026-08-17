"""What an embedder is, and the checks every embedder owes its caller.

An **embedding** is a list of numbers standing in for a piece of text, arranged so
that two texts about the same thing land close together. Everything here is a
description, a contract or a pure function, so this module imports pydantic and
nothing else -- the real client lives in :mod:`juena_rag.embedding.blablador`,
behind the ``[embed]`` extra.
"""

from typing import Protocol

from pydantic import BaseModel, Field

#: How many texts go in one request. What the chatbot already runs against this
#: endpoint, so it is a measured limit rather than a guess.
DEFAULT_MAX_BATCH_TEXTS = 64


# --------------------------------------------------------------------------- #
# What went wrong
# --------------------------------------------------------------------------- #


class EmbeddingError(Exception):
    """Base class, so a caller can catch everything this layer raises."""


class DimensionMismatchError(EmbeddingError):
    """The model returned a width other than the configured ``dimension``."""


class MisalignedResponseError(EmbeddingError):
    """A response did not carry exactly one vector per input, once each."""


class NoEmbedderConfigured(EmbeddingError):
    """Something needed an embedder and was built without one."""


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


class EmbeddingMetadata(BaseModel):
    """Which model made a vector, and the shape of what it makes.

    All three values are configuration, not discovery: nothing is looked up from
    the server, so a service changing underneath us is loud rather than quiet. A
    pydantic model rather than a dataclass because plan 02 serialises it out of
    ``/status``.
    """

    model_config = {"extra": "forbid", "frozen": True}

    #: The configured name, verbatim -- ``alias-qwen3-8b-embeddings``. This string
    #: goes into the ``embedding_model`` column unchanged.
    model: str = Field(min_length=1)

    #: How many numbers one vector has. 4096 for this model.
    dimension: int = Field(gt=0)

    #: The longest input the model accepts, in tokens. Published here, enforced by
    #: the caller -- 8192 for this model.
    max_input_tokens: int = Field(gt=0)


class Embedder(Protocol):
    """Anything that can turn text into vectors.

    Two methods rather than one because writing and searching are the same model
    but not the same call: an ingest embeds thousands and cares only about
    throughput, a search embeds exactly one and is what a person is waiting for.
    Deliberately not ``runtime_checkable`` -- an ``isinstance`` check against a
    protocol only asks whether the attributes exist, and it reports ``False`` for
    any implementation whose ``metadata`` raises.
    """

    metadata: EmbeddingMetadata

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks on the way in: one vector per text, in the order given."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one question. One round trip, on every search."""


# --------------------------------------------------------------------------- #
# The two shared checks
# --------------------------------------------------------------------------- #


def take_in_order(
    indexed_vectors: list[tuple[int, list[float]]], expected: int
) -> list[list[float]]:
    """Put a response back into input order, refusing anything that does not fit.

    Each item carries its own ``index``, so vectors are placed by that rather than
    by list position, and the response is refused unless every position was filled
    exactly once. Zipped blindly instead, a wrong count or a repeated index still
    yields real vectors of the right width -- so the write succeeds and the wrong
    rows come back at search time.
    """
    if len(indexed_vectors) != expected:
        raise MisalignedResponseError(
            f"asked for {expected} vectors and got {len(indexed_vectors)}; "
            "refusing rather than guessing which text lost its vector"
        )

    ordered: list[list[float] | None] = [None] * expected
    for index, vector in indexed_vectors:
        if not 0 <= index < expected:
            raise MisalignedResponseError(
                f"the response claims index {index}, which is outside the "
                f"{expected} texts that were sent"
            )
        if ordered[index] is not None:
            raise MisalignedResponseError(
                f"the response carries index {index} twice, so at least one "
                "text has no vector of its own"
            )
        ordered[index] = vector

    return [vector for vector in ordered if vector is not None]


def check_width(vectors: list[list[float]], dimension: int) -> None:
    """Raise unless every vector is the configured width.

    One ``len()`` per vector, and the only thing standing between a model changing
    underneath a running ingest and a table full of rows nobody can compare.
    """
    for position, vector in enumerate(vectors):
        if len(vector) != dimension:
            raise DimensionMismatchError(
                f"vector #{position} has {len(vector)} numbers but the "
                f"configured dimension is {dimension}"
            )


def verify_dimension(embedder: Embedder, probe: str = "dimension check") -> None:
    """Embed one short string at startup and confirm the width. Refuse if not.

    pgvector fixes the column width at ``CREATE TABLE``, so a disagreement found
    at write time is found halfway through filling a table that can no longer hold
    what is being written. The model *name* is deliberately not checked: several
    aliases share one model, so a response routinely names a different one and
    comparing them would fail constantly while detecting nothing.
    """
    vector = embedder.embed_query(probe)
    if len(vector) != embedder.metadata.dimension:
        raise DimensionMismatchError(
            f"`{embedder.metadata.model}` returned {len(vector)} numbers but "
            f"`dimension` is configured as {embedder.metadata.dimension}. Fix "
            "the configuration before starting: a table created at the wrong "
            "width cannot be widened, only dropped and refilled."
        )
