"""The one real embedding client: Blablador, the Helmholtz-hosted inference API.

Blablador speaks the OpenAI REST shape, so this is the official ``openai`` client
pointed at a different address rather than hand-written HTTP -- the retry rules
this needs are already in the SDK and are easy to get subtly wrong by hand. Needs
the ``[embed]`` extra, and nothing else in the package imports it, so installing
the schema alone never brings an HTTP stack.
"""

import os
from typing import Any

from juena_rag.embedding.base import (
    DEFAULT_MAX_BATCH_TEXTS,
    EmbeddingMetadata,
    check_width,
    take_in_order,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on the install, not the code
    raise ImportError(
        "the Blablador client needs the `openai` package, which is not "
        "installed. Either `pip install 'juena-rag[embed]'`, or import only "
        "`juena_rag.embedding`, which has no network dependency."
    ) from None

DEFAULT_BASE_URL = "https://api.blablador.fz-juelich.de/v1"

#: The configured name, used verbatim in the request and in the
#: ``embedding_model`` column. Blablador's ``/v1/models`` also reports the
#: underlying weights (``Qwen/Qwen3-Embedding-8B``), deliberately unused here so
#: there is only ever one name for one model.
DEFAULT_MODEL = "alias-qwen3-8b-embeddings"

#: How many numbers this model returns. Not negotiable and not discovered: the
#: endpoint rejects a ``dimensions`` parameter outright.
DEFAULT_DIMENSION = 4096

#: What the endpoint reports as ``max_model_len``. Published for callers to size
#: their chunks against; nothing here enforces it.
DEFAULT_MAX_INPUT_TOKENS = 8192


class BlabladorEmbedder:
    """Embed text against Blablador, in batches, in order, at the right width.

    Every setting is a plain keyword argument with a default, so the whole
    configuration is visible in one place. :meth:`from_env` fills them from the
    environment for a deployment.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_batch_texts: int = DEFAULT_MAX_BATCH_TEXTS,
        query_instruction: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        """
        :param max_batch_texts: how many texts ride in one request. Purely about
            round trips -- each text is sent whole either way.
        :param query_instruction: what the *question* is for, if queries should be
            wrapped as ``Instruct: {task}\\nQuery: {text}``. **Off by default**,
            which is what the chatbot does today; only queries are affected, so
            the decision stays reversible after the corpus is embedded.
        :param max_retries: how many times a *retryable* failure is tried again,
            with backoff. The SDK retries connection errors, 408, 409, 429 and
            5xx, and never a 400.
        :param client: an already-built client, for tests. Anything with
            ``.embeddings.create(input=…, model=…)`` will do.
        """
        self.metadata = EmbeddingMetadata(
            model=model,
            dimension=dimension,
            max_input_tokens=max_input_tokens,
        )
        self.max_batch_texts = max_batch_texts
        self.query_instruction = query_instruction
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> "BlabladorEmbedder":
        """Build from the same environment variables the chatbot already reads.

        ``BLABLADOR_API_KEY`` is required; everything else falls back to the
        defaults at the top of this module. Anything passed as a keyword wins,
        which is how a test pins a value without touching the environment.
        """
        api_key = os.getenv("BLABLADOR_API_KEY", "").strip()
        if not api_key and "api_key" not in overrides:
            raise ValueError(
                "BLABLADOR_API_KEY is not set. It is the only value with no "
                "default, because there is no sensible one."
            )

        settings: dict[str, Any] = {"api_key": api_key}
        base_url = os.getenv("BLABLADOR_BASE_URL", "").strip()
        if base_url:
            settings["base_url"] = base_url
        model = os.getenv("BLABLADOR_EMBEDDING_MODEL", "").strip()
        if model:
            settings["model"] = model
        for name, variable in (
            ("dimension", "BLABLADOR_EMBEDDING_DIMENSION"),
            ("max_input_tokens", "BLABLADOR_EMBEDDING_MAX_INPUT_TOKENS"),
        ):
            raw = os.getenv(variable, "").strip()
            if not raw:
                continue
            if not raw.isdigit():
                raise ValueError(f"{variable} must be a whole number, not {raw!r}")
            settings[name] = int(raw)

        settings.update(overrides)
        return cls(**settings)

    # ----------------------------------------------------------------------- #
    # The two calls
    # ----------------------------------------------------------------------- #

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks on the way into the table, a batch at a time.

        A batch is a group of *whole* texts sharing one request -- 64 of them
        instead of 64 round trips, with no text ever cut to make one fit.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.max_batch_texts):
            vectors.extend(self._embed(texts[start : start + self.max_batch_texts]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one question, wrapped as an instruction if one is configured."""
        if self.query_instruction:
            text = f"Instruct: {self.query_instruction}\nQuery: {text}"
        return self._embed([text])[0]

    def _embed(self, block: list[str]) -> list[list[float]]:
        """One request, put back into input order and checked for width.

        ``dimensions`` is never passed: this endpoint rejects it outright, since
        the model behind the alias is not served with shortening enabled.
        """
        response = self._client.embeddings.create(
            input=block,
            model=self.metadata.model,
        )
        vectors = take_in_order(
            [(item.index, list(item.embedding)) for item in response.data],
            len(block),
        )
        check_width(vectors, self.metadata.dimension)
        return vectors
