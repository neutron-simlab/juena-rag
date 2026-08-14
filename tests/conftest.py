"""Fixtures shared by the schema tests."""

import pytest

from juena_rag.schema.registry import _REGISTRY


@pytest.fixture
def clean_registry():
    """Let a test define its own chunk classes without leaking them."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)
