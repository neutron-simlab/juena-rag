"""What `juena-rag serve` reads from the environment, in one object.

A dataclass with a ``from_env`` classmethod, like
:meth:`BlabladorEmbedder.from_env <juena_rag.embedding.blablador.BlabladorEmbedder.from_env>`:
every setting is a keyword argument, and the environment is one way of filling
them rather than the only way.
"""

import os
from dataclasses import dataclass

from juena_rag.store.base import DEFAULT_RERANK_CANDIDATES

#: How many database connections the pool holds. Four is `Store`'s own default.
DEFAULT_POOL_MIN_SIZE = 1
DEFAULT_POOL_MAX_SIZE = 4

#: How many requests may be inside a handler at once: anyio's own default, and
#: not meant to match the pool size.
#:
#: A search embeds the question before it borrows a connection, so the ~200 ms
#: round trip to Blablador holds a thread and no connection, while the query
#: holds one for about 4 ms. Capping threads at the pool's four would allow only
#: four concurrent searches while the pool sat idle.
DEFAULT_MAX_CONCURRENCY = 40


@dataclass(frozen=True)
class ServiceSettings:
    """Everything the service needs that is not the embedding model's business.

    The model's own configuration stays in ``BLABLADOR_*`` and is read by the
    embedder, so there is one place each value is named rather than two.

    Attributes:
        dsn: Where pgvector is. The only setting with no default.
        pool_min_size: The fewest database connections the pool keeps open.
        pool_max_size: The most database connections the pool opens.
        max_concurrency: How many requests may be inside a handler at once.
        rerank_candidates: How many of the closest chunks the reranker re-reads,
            for a search that asks for reranking. The reranking model itself is
            ``BLABLADOR_RERANK_MODEL``.
    """

    dsn: str

    pool_min_size: int = DEFAULT_POOL_MIN_SIZE
    pool_max_size: int = DEFAULT_POOL_MAX_SIZE
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY

    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES

    @classmethod
    def from_env(cls, **overrides: object) -> "ServiceSettings":
        """Build from the ``JUENA_RAG_*`` environment variables.

        Args:
            **overrides: Settings that win over the environment.

        Returns:
            The settings.

        Raises:
            ValueError: If no DSN is set, or a number is not a whole number.
        """
        settings: dict[str, object] = {}

        dsn = os.getenv("JUENA_RAG_DSN", "").strip()
        if dsn:
            settings["dsn"] = dsn

        for field, variable in (
            ("pool_min_size", "JUENA_RAG_POOL_MIN_SIZE"),
            ("pool_max_size", "JUENA_RAG_POOL_MAX_SIZE"),
            ("max_concurrency", "JUENA_RAG_MAX_CONCURRENCY"),
            ("rerank_candidates", "JUENA_RAG_RERANK_CANDIDATES"),
        ):
            raw = os.getenv(variable, "").strip()
            if not raw:
                continue
            try:
                settings[field] = int(raw)
            except ValueError:
                raise ValueError(
                    f"{variable} is {raw!r}, which is not a whole number"
                ) from None

        settings.update(overrides)
        if not settings.get("dsn"):
            raise ValueError(
                "JUENA_RAG_DSN is not set, so there is no database to serve. It is "
                "the only setting with no default; see .env.example."
            )
        return cls(**settings)  # type: ignore[arg-type]
