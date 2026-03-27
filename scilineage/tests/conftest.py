"""Pytest configuration for scilineage tests."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "canonical-hash" / "src"))

import pytest

from scilineage import _clear_backend


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the global cache before and after each test."""
    _clear_backend()
    yield
    _clear_backend()
