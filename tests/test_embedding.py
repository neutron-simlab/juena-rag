"""The embedder -- `juena_rag.embedding`.

Every test here runs without a network. The one that needs Blablador is at the
bottom and skips itself when `BLABLADOR_API_KEY` is unset.
"""

import os
from dataclasses import dataclass, field

import pytest
from fakes import FakeEmbedder, NullEmbedder
from pydantic import ValidationError

from juena_rag.embedding import (
    DimensionMismatchError,
    EmbeddingMetadata,
    MisalignedResponseError,
    check_width,
    take_in_order,
    verify_dimension,
)
from juena_rag.embedding.blablador import BlabladorEmbedder
from juena_rag.embedding.tokenizer import ModelTokenCounter, TokenizerUnavailable

# --------------------------------------------------------------------------- #
# Standing in for the OpenAI client
# --------------------------------------------------------------------------- #


@dataclass
class _Item:
    """One entry of a `/v1/embeddings` response: a vector and where it belongs."""

    index: int
    embedding: list[float]


@dataclass
class _Response:
    data: list[_Item]
    #: What the *server* says it used. Routinely a different alias than the one
    #: asked for, which is why nothing compares it to the configured name.
    model: str = "alias-embeddings"


@dataclass
class StubEmbeddings:
    """`client.embeddings`, without a client.

    Each vector's first number is the first letter of the text that made it, so
    a vector attached to the wrong text is visible rather than inferred.
    """

    dimension: int = 4
    #: Return the entries back to front. The data is identical and every `index`
    #: is still correct -- only the order of the list changes, which is exactly
    #: what the API declines to promise.
    reversed_order: bool = False
    requests: list[list[str]] = field(default_factory=list)
    models: list[str] = field(default_factory=list)

    def create(self, *, input: list[str], model: str) -> _Response:
        self.requests.append(list(input))
        self.models.append(model)
        items = [
            _Item(index=position, embedding=self._vector(text))
            for position, text in enumerate(input)
        ]
        if self.reversed_order:
            items.reverse()
        return _Response(data=items)

    def _vector(self, text: str) -> list[float]:
        return [float(ord(text[0]))] + [0.0] * (self.dimension - 1)


@dataclass
class StubClient:
    embeddings: StubEmbeddings


def marker(vector: list[float]) -> str:
    """Which text made this vector, per `StubEmbeddings`."""
    return chr(int(vector[0]))


def build(**overrides) -> tuple[BlabladorEmbedder, StubEmbeddings]:
    """A Blablador client wired to a stub, at a width small enough to read."""
    stub = StubEmbeddings(dimension=overrides.pop("stub_dimension", 4))
    stub.reversed_order = overrides.pop("reversed_order", False)
    settings = {"api_key": "unused", "dimension": 4, "client": StubClient(stub)}
    settings.update(overrides)
    return BlabladorEmbedder(**settings), stub


class WrongWidthEmbedder:
    """Claims 4096 and returns 8, with nothing in it that would notice."""

    metadata = EmbeddingMetadata(model="wrong", dimension=4096, max_input_tokens=8192)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 8


# --------------------------------------------------------------------------- #
# The metadata
# --------------------------------------------------------------------------- #


def test_metadata_is_frozen_because_a_model_does_not_change_mid_run():
    metadata = EmbeddingMetadata(
        model="alias-qwen3-8b-embeddings", dimension=4096, max_input_tokens=8192
    )

    with pytest.raises(ValidationError):
        metadata.dimension = 2000


@pytest.mark.parametrize("bad", [{"dimension": 0}, {"max_input_tokens": 0}])
def test_metadata_rejects_a_width_or_ceiling_of_zero(bad):
    settings = {"model": "m", "dimension": 4096, "max_input_tokens": 8192} | bad

    with pytest.raises(ValueError):
        EmbeddingMetadata(**settings)


def test_the_configured_model_name_is_used_verbatim_whatever_the_server_says():
    """One name for one model, everywhere -- the request and the stored column.

    The response names a different alias, because several aliases share this
    model on Blablador. Nothing compares them, so nothing fails over it.
    """
    embedder, stub = build(model="alias-qwen3-8b-embeddings")

    embedder.embed_query("how do I align the detector")

    assert embedder.metadata.model == "alias-qwen3-8b-embeddings"
    assert stub.models == ["alias-qwen3-8b-embeddings"]


# --------------------------------------------------------------------------- #
# The ceiling belongs to the caller
# --------------------------------------------------------------------------- #


def test_the_ceiling_is_published_so_a_chunker_can_size_against_it():
    """The number a caller needs, on the object they already hold."""
    embedder, _ = build(max_input_tokens=8192)

    assert embedder.metadata.max_input_tokens == 8192


def test_text_over_the_ceiling_is_sent_as_it_is_rather_than_cut():
    """There is no chunker in this package, and this is what that means.

    Shortening the text would make the stored vector's recall a permanent
    ceiling, recoverable only by re-embedding; refusing would put a chunking
    decision in a layer that cannot see the format. So it goes as given, and
    staying under the limit is the job of whoever cut it.
    """
    embedder, stub = build(max_input_tokens=10)
    long_text = "a text far longer than ten tokens, whatever a token is here"

    embedder.embed_documents([long_text])

    assert stub.requests == [[long_text]], "sent whole, unchanged"


# --------------------------------------------------------------------------- #
# Batching -- grouping whole texts into requests, never cutting one
# --------------------------------------------------------------------------- #


def test_texts_are_grouped_into_requests_and_come_back_in_input_order():
    embedder, stub = build(max_batch_texts=2)

    vectors = embedder.embed_documents(["a", "b", "c", "d", "e"])

    assert stub.requests == [["a", "b"], ["c", "d"], ["e"]]
    assert [marker(vector) for vector in vectors] == ["a", "b", "c", "d", "e"]


def test_embedding_nothing_sends_nothing():
    embedder, stub = build()

    assert embedder.embed_documents([]) == []
    assert stub.requests == []


# --------------------------------------------------------------------------- #
# The one silent corruption
# --------------------------------------------------------------------------- #


def test_an_out_of_order_response_still_attaches_each_vector_to_its_own_text():
    """The response order is not promised to match the input order.

    Zipped blindly, every text still gets a real vector of the right width and
    every write still succeeds -- the index is just wrong, and no count of rows
    would ever show it.
    """
    embedder, _ = build(reversed_order=True)

    vectors = embedder.embed_documents(["a", "b", "c"])

    assert [marker(vector) for vector in vectors] == ["a", "b", "c"]


def test_a_short_response_is_refused_rather_than_guessed_at():
    with pytest.raises(MisalignedResponseError):
        take_in_order([(0, [1.0]), (1, [2.0])], expected=3)


def test_a_repeated_index_is_refused_because_some_text_lost_its_vector():
    with pytest.raises(MisalignedResponseError):
        take_in_order([(0, [1.0]), (0, [2.0])], expected=2)


def test_an_index_outside_the_request_is_refused():
    with pytest.raises(MisalignedResponseError):
        take_in_order([(0, [1.0]), (7, [2.0])], expected=2)


def test_a_response_in_any_order_lands_correctly():
    ordered = take_in_order([(2, [3.0]), (0, [1.0]), (1, [2.0])], expected=3)

    assert ordered == [[1.0], [2.0], [3.0]]


# --------------------------------------------------------------------------- #
# The width
# --------------------------------------------------------------------------- #


def test_a_wrong_width_is_caught_at_startup_not_at_write_time():
    """pgvector fixes the column width at CREATE TABLE. Found at startup this is
    a typo and a restart; found at write time it is a table to drop and refill.
    """
    with pytest.raises(DimensionMismatchError) as raised:
        verify_dimension(WrongWidthEmbedder())

    assert "4096" in str(raised.value)
    assert "8" in str(raised.value)


def test_a_matching_width_passes_quietly():
    assert verify_dimension(FakeEmbedder(dimension=8)) is None


def test_the_client_checks_every_vector_it_receives():
    embedder, _ = build(dimension=4096, stub_dimension=4)

    with pytest.raises(DimensionMismatchError):
        embedder.embed_query("anything")


def test_check_width_says_which_vector_was_wrong():
    with pytest.raises(DimensionMismatchError) as raised:
        check_width([[0.0, 0.0], [0.0]], dimension=2)

    assert "#1" in str(raised.value)


# --------------------------------------------------------------------------- #
# The query instruction -- still an open question, so it is off
# --------------------------------------------------------------------------- #


def test_queries_go_out_unwrapped_by_default():
    """What the chatbot does today. Changing it only affects queries, so the
    decision stays reversible after the corpus is embedded.
    """
    embedder, stub = build()

    embedder.embed_query("what is the beam stop diameter")

    assert stub.requests == [["what is the beam stop diameter"]]


def test_a_configured_instruction_wraps_the_query_and_only_the_query():
    embedder, stub = build(query_instruction="Retrieve documents for this question")

    embedder.embed_query("what is the beam stop diameter")
    embedder.embed_documents(["the beam stop is 20 mm"])

    wrapped = (
        "Instruct: Retrieve documents for this question\n"
        "Query: what is the beam stop diameter"
    )

    assert stub.requests[0] == [wrapped]
    assert stub.requests[1] == ["the beam stop is 20 mm"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_from_env_reads_the_same_variables_the_chatbot_does(monkeypatch):
    monkeypatch.setenv("BLABLADOR_API_KEY", "not-a-real-key")
    monkeypatch.setenv("BLABLADOR_EMBEDDING_MODEL", "alias-qwen3-8b-embeddings")
    monkeypatch.setenv("BLABLADOR_EMBEDDING_DIMENSION", "4096")

    embedder = BlabladorEmbedder.from_env()

    assert embedder.metadata.model == "alias-qwen3-8b-embeddings"
    assert embedder.metadata.dimension == 4096
    assert embedder.metadata.max_input_tokens == 8192


def test_from_env_refuses_without_a_key(monkeypatch):
    monkeypatch.delenv("BLABLADOR_API_KEY", raising=False)

    with pytest.raises(ValueError, match="BLABLADOR_API_KEY"):
        BlabladorEmbedder.from_env()


def test_from_env_refuses_a_dimension_that_is_not_a_number(monkeypatch):
    monkeypatch.setenv("BLABLADOR_API_KEY", "not-a-real-key")
    monkeypatch.setenv("BLABLADOR_EMBEDDING_DIMENSION", "four thousand")

    with pytest.raises(ValueError, match="whole number"):
        BlabladorEmbedder.from_env()


def test_an_explicit_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("BLABLADOR_API_KEY", "not-a-real-key")
    monkeypatch.setenv("BLABLADOR_EMBEDDING_MODEL", "from-the-environment")

    embedder = BlabladorEmbedder.from_env(model="from-the-argument")

    assert embedder.metadata.model == "from-the-argument"


# --------------------------------------------------------------------------- #
# The test doubles
# --------------------------------------------------------------------------- #


def test_the_fake_gives_the_same_vector_for_the_same_text_and_the_right_width():
    fake = FakeEmbedder(dimension=16)

    assert fake.embed_query("hello") == fake.embed_query("hello")
    assert fake.embed_query("hello") != fake.embed_query("Hello")
    assert len(fake.embed_query("hello")) == 16


def test_the_null_embedder_complains_on_every_member_including_metadata():
    """Including `metadata`: there is no honest width to report, and a
    plausible-looking placeholder is the exact failure `verify_dimension`
    exists to catch.
    """
    null = NullEmbedder()

    for call in (
        lambda: null.metadata,
        lambda: null.embed_query("x"),
        lambda: null.embed_documents(["x"]),
    ):
        with pytest.raises(Exception, match="without an embedder"):
            call()


# --------------------------------------------------------------------------- #
# The token counter -- a tool for whoever writes the chunks
# --------------------------------------------------------------------------- #


def test_a_missing_vocabulary_refuses_rather_than_falling_back_to_a_guess():
    """There is no safe estimate to fall back to -- that is the whole point."""
    with pytest.raises(TokenizerUnavailable, match="tokenizer.json"):
        ModelTokenCounter(tokenizer_path="/nowhere/tokenizer.json")


@pytest.fixture(scope="module")
def qwen_tokens():
    """The real vocabulary, downloaded once and cached by Hugging Face.

    Skipped rather than failed when there is no route out: these tests describe
    the model, and the rest of the file describes this package.
    """
    try:
        return ModelTokenCounter()
    except TokenizerUnavailable as exc:
        pytest.skip(f"the Qwen vocabulary is not reachable here: {exc}")


def test_counting_characters_would_be_wrong_in_both_directions(qwen_tokens):
    """Why this ships a vocabulary rather than a rule of thumb.

    English prose costs far fewer tokens than characters, so a character count
    over-states it -- wasteful but harmless. Multi-byte text costs *more* tokens
    than characters, because tokens are built out of bytes, and that is the
    harmful direction: a chunker trusting the "conservative" estimate walks past
    the ceiling and the server truncates without saying so.
    """
    english, japanese = "small-angle neutron scattering", "中性子小角散乱"
    counts = qwen_tokens.count_many([english, japanese])

    assert counts[0] < len(english), "prose is cheaper than one token per character"
    assert counts[1] > len(japanese), "and multi-byte text is more expensive"


def test_a_chunker_can_size_its_output_against_the_published_ceiling(qwen_tokens):
    """The whole intended use, in four lines: read the limit off the embedder,
    measure with the counter, and decide what to do -- all on the caller's side.
    """
    embedder, _ = build()
    chunks = ["short enough", "far too long " * 20]

    counts = qwen_tokens.count_many(chunks)
    over = [text for text, n in zip(chunks, counts) if n > 8]

    assert embedder.metadata.max_input_tokens == 8192, "the limit is readable"
    assert over == [chunks[1]], "and the caller can find what exceeds it"


# --------------------------------------------------------------------------- #
# The one test that needs the real thing
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not os.getenv("BLABLADOR_API_KEY"),
    reason="needs BLABLADOR_API_KEY; every other test here runs offline",
)
def test_the_real_endpoint_returns_the_width_this_package_is_configured_for():
    embedder = BlabladorEmbedder.from_env()

    verify_dimension(embedder)

    vectors = embedder.embed_documents(["small-angle scattering", "Streuung"])
    assert len(vectors) == 2
    assert all(len(vector) == 4096 for vector in vectors)
