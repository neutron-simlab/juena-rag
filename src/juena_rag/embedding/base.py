"""What an embedder is, and the checks every embedder owes its caller.

An **embedding** is a list of numbers standing in for a piece of text, arranged
so that two texts about the same thing land close together. Storing them is how
a search can find a passage that never uses the words the question used.

Everything in this module is either a description (:class:`EmbeddingMetadata`),
a contract (:class:`Embedder`) or a pure function. Nothing here makes a network
call, so it imports pydantic and nothing else -- the real client lives in
:mod:`juena_rag.embedding.blablador`, behind the ``[embed]`` extra.

**Text arrives ready to embed.** A caller hands over whole texts and gets whole
vectors back; nothing here cuts, joins, pads or shortens anything. In particular
nothing checks a text against the model's ceiling -- see
:attr:`EmbeddingMetadata.max_input_tokens` for whose job that is and why it is
not this one's.

Two rules are shared by every implementation, and each exists because breaking
it fails *silently*:

1. **Never let a vector attach to the wrong text.** See :func:`take_in_order`.
2. **Check the width before writing anything.** See :func:`verify_dimension`.
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

    All three values are *configuration*, not discovery. Nothing is looked up
    from the server: the name is the one in the environment file, and the two
    numbers are pinned so that a service changing underneath us is loud rather
    than quiet.

    A pydantic model rather than a dataclass, even though it lives inside the
    package, because plan 02 serialises it out of ``/status``.
    """

    model_config = {"extra": "forbid", "frozen": True}

    #: The configured name, verbatim -- ``alias-qwen3-8b-embeddings``. This
    #: string goes into the ``embedding_model`` column unchanged. There is no
    #: second name for the same model and no resolution step.
    model: str = Field(min_length=1)

    #: How many numbers one vector has. 4096 for this model.
    #:
    #: The field that must not be skipped: pgvector fixes the width when the
    #: table is created (``vector(4096)``, never bare ``vector``), so a wrong
    #: value is only discovered halfway through filling a table that can no
    #: longer hold what is being written. :func:`verify_dimension` moves that
    #: discovery to startup, where the repair is a typo and a restart.
    dimension: int = Field(gt=0)

    #: The longest input the model accepts, in tokens. 8192 for this model,
    #: which is what the endpoint reports as ``max_model_len``.
    #:
    #: **Published here, enforced by the caller.** Past this the server truncates
    #: and returns a normal-looking vector, so a chunk sent over the ceiling is
    #: indexed by its opening paragraph and everything after the cut becomes
    #: unsearchable, with no error anywhere. Deciding how long a chunk is, is a
    #: chunking decision, and chunking belongs to whoever knows the format --
    #: this package validates rather than transforms, and cannot split what it is
    #: given. So the number lives here, where a caller can read it off the
    #: embedder and size its chunks against it, and the guarantee is theirs.
    max_input_tokens: int = Field(gt=0)


class Embedder(Protocol):
    """Anything that can turn text into vectors.

    Two methods rather than one, because writing and searching are the same
    model but not the same call. An ingest embeds thousands of texts and cares
    only about throughput; a search embeds exactly one and is what a person is
    waiting for.

    A protocol rather than a base class: an embedder is anything with these
    three members, so nothing has to import this to satisfy it. Deliberately not
    ``runtime_checkable`` either -- an ``isinstance`` check against a protocol
    only asks whether the attributes exist, which is a weaker claim than it
    looks, and it reports ``False`` for any implementation whose ``metadata``
    raises rather than letting that error through.
    """

    metadata: EmbeddingMetadata

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks on the way into the table.

        Returns one vector per text, **in the order the texts were given**.
        """

    def embed_query(self, text: str) -> list[float]:
        """Embed one question. One round trip, on every search."""


# --------------------------------------------------------------------------- #
# The two shared checks
# --------------------------------------------------------------------------- #


def take_in_order(
    indexed_vectors: list[tuple[int, list[float]]], expected: int
) -> list[list[float]]:
    """Put a response back into input order, refusing anything that does not fit.

    This is the one silent corruption in this checkpoint, so it gets its own
    function and its own test. An OpenAI-compatible ``/v1/embeddings`` response
    is a list of objects each carrying its own ``index``, and the order of that
    list is *not promised* to match the order of the input. Zip it against the
    input without looking and the vectors attach to the wrong texts: every text
    gets a real vector of the right width, every write succeeds, and the index is
    wrong in a way no test that counts rows will ever see.

    So the vectors are placed by their stated index, and the response is refused
    unless every position was filled exactly once.
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

    Cheap -- one ``len()`` per vector -- and it is the only thing standing
    between a model changing underneath a running ingest and a table full of
    rows nobody can compare.
    """
    for position, vector in enumerate(vectors):
        if len(vector) != dimension:
            raise DimensionMismatchError(
                f"vector #{position} has {len(vector)} numbers but the "
                f"configured dimension is {dimension}"
            )


def verify_dimension(embedder: Embedder, probe: str = "dimension check") -> None:
    """Embed one short string at startup and confirm the width. Refuse if not.

    Call this once when a service starts, before it opens for business. The
    alternative is finding out at write time, and pgvector fixes the column width
    at ``CREATE TABLE`` -- so a disagreement discovered then is discovered halfway
    through filling a table that can no longer hold what is being written, and the
    repair is to drop it and ingest again.

    The model *name* is deliberately not verified against the server. A response
    routinely names a different alias than the request did, because several
    aliases share one model, so comparing them would fail constantly while
    detecting nothing.
    """
    vector = embedder.embed_query(probe)
    if len(vector) != embedder.metadata.dimension:
        raise DimensionMismatchError(
            f"`{embedder.metadata.model}` returned {len(vector)} numbers but "
            f"`dimension` is configured as {embedder.metadata.dimension}. Fix "
            "the configuration before starting: a table created at the wrong "
            "width cannot be widened, only dropped and refilled."
        )
