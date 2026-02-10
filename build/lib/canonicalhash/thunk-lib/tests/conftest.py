"""Pytest configuration for thunk tests."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "canonical-hash" / "src"))

import pytest

from thunk import Thunk


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the global cache before and after each test."""
    Thunk.query = None
    yield
    Thunk.query = None
