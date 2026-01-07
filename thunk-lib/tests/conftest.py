"""Pytest configuration for thunk tests."""

import pytest

from thunk import configure_cache


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the global cache before and after each test."""
    configure_cache(None)
    yield
    configure_cache(None)
