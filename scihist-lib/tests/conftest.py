"""Pytest configuration for scihist tests."""

import sys
from pathlib import Path

import pytest

# Add all relevant packages to path
_here = Path(__file__).parent.parent
_root = _here.parent
sys.path.insert(0, str(_here / "src"))
sys.path.insert(0, str(_root / "scidb" / "src"))
sys.path.insert(0, str(_root / "thunk-lib" / "src"))
sys.path.insert(0, str(_root / "scifor" / "src"))
sys.path.insert(0, str(_root / "canonical-hash" / "src"))
sys.path.insert(0, str(_root / "path-gen" / "src"))
sys.path.insert(0, str(_root / "pipelinedb-lib" / "src"))
sys.path.insert(0, str(_root / "sciduck" / "src"))


DEFAULT_TEST_SCHEMA_KEYS = ["subject", "trial"]


@pytest.fixture
def db(tmp_path):
    """Provide a fresh configured database with Thunk.query set."""
    from scihist import configure_database
    db_path = tmp_path / "test_db.duckdb"
    db = configure_database(db_path, DEFAULT_TEST_SCHEMA_KEYS)
    yield db
    db.close()
    from thunk import Thunk
    Thunk.query = None
    from scidb.database import _local
    if hasattr(_local, 'database'):
        delattr(_local, 'database')


@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear global state before and after each test."""
    from thunk import Thunk
    from scidb.database import _local
    Thunk.query = None
    if hasattr(_local, 'database'):
        delattr(_local, 'database')
    yield
    Thunk.query = None
    if hasattr(_local, 'database'):
        delattr(_local, 'database')
