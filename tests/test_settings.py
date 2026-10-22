"""What `ServiceSettings.from_env` reads, and what it refuses.

No database and no network: every test sets the environment and reads it back.
"""

import pytest

from juena_rag.service.settings import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_POOL_MAX_SIZE,
    DEFAULT_POOL_MIN_SIZE,
    DEFAULT_RERANK_CANDIDATES,
    ServiceSettings,
)

VARIABLES = (
    "JUENA_RAG_DSN",
    "JUENA_RAG_POOL_MIN_SIZE",
    "JUENA_RAG_POOL_MAX_SIZE",
    "JUENA_RAG_MAX_CONCURRENCY",
    "JUENA_RAG_RERANK_CANDIDATES",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Start every test from none of the variables, whatever the shell exported."""
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_a_dsn_alone_gives_every_default(monkeypatch):
    monkeypatch.setenv("JUENA_RAG_DSN", "postgresql://example/db")

    settings = ServiceSettings.from_env()

    assert settings == ServiceSettings(
        dsn="postgresql://example/db",
        pool_min_size=DEFAULT_POOL_MIN_SIZE,
        pool_max_size=DEFAULT_POOL_MAX_SIZE,
        max_concurrency=DEFAULT_MAX_CONCURRENCY,
        rerank_candidates=DEFAULT_RERANK_CANDIDATES,
    )


def test_every_variable_is_read(monkeypatch):
    monkeypatch.setenv("JUENA_RAG_DSN", "postgresql://example/db")
    monkeypatch.setenv("JUENA_RAG_POOL_MIN_SIZE", "2")
    monkeypatch.setenv("JUENA_RAG_POOL_MAX_SIZE", "8")
    monkeypatch.setenv("JUENA_RAG_MAX_CONCURRENCY", "16")
    monkeypatch.setenv("JUENA_RAG_RERANK_CANDIDATES", "30")

    settings = ServiceSettings.from_env()

    assert (
        settings.pool_min_size,
        settings.pool_max_size,
        settings.max_concurrency,
        settings.rerank_candidates,
    ) == (2, 8, 16, 30)


def test_a_keyword_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("JUENA_RAG_DSN", "postgresql://from-env/db")
    monkeypatch.setenv("JUENA_RAG_RERANK_CANDIDATES", "30")

    settings = ServiceSettings.from_env(
        dsn="postgresql://given/db", rerank_candidates=5
    )

    assert settings.dsn == "postgresql://given/db"
    assert settings.rerank_candidates == 5


def test_a_blank_variable_counts_as_unset(monkeypatch):
    """An `.env` line like `JUENA_RAG_POOL_MAX_SIZE=` means the default, not zero."""
    monkeypatch.setenv("JUENA_RAG_DSN", "postgresql://example/db")
    monkeypatch.setenv("JUENA_RAG_POOL_MAX_SIZE", "  ")

    assert ServiceSettings.from_env().pool_max_size == DEFAULT_POOL_MAX_SIZE


def test_a_number_that_is_not_one_is_refused_with_its_name(monkeypatch):
    monkeypatch.setenv("JUENA_RAG_DSN", "postgresql://example/db")
    monkeypatch.setenv("JUENA_RAG_RERANK_CANDIDATES", "twenty")

    with pytest.raises(ValueError, match="JUENA_RAG_RERANK_CANDIDATES is 'twenty'"):
        ServiceSettings.from_env()


def test_no_dsn_is_refused_with_a_sentence():
    with pytest.raises(ValueError, match="JUENA_RAG_DSN is not set"):
        ServiceSettings.from_env()
