"""Shared fixtures for scidb-net tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add all local packages to path
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "scidb-net" / "src"))
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "canonical-hash" / "src"))
sys.path.insert(0, str(_root / "thunk-lib" / "src"))
sys.path.insert(0, str(_root / "path-gen" / "src"))
sys.path.insert(0, str(_root / "pipelinedb-lib" / "src"))
sys.path.insert(0, str(_root / "scirun-lib" / "src"))
sys.path.insert(0, str(_root / "sciduck" / "src"))

from httpx import ASGITransport
from scidb.variable import BaseVariable
from scidb.database import _local
from scidb.thunk import Thunk
from scidbnet.server import create_app
from scidbnet.client import RemoteDatabaseManager


# Schema keys used across all tests
TEST_SCHEMA_KEYS = [
    "subject", "trial", "session", "channel", "experiment",
    "name", "sensor", "condition", "category", "key",
]


# -------------------------------------------------------------------------
# Test variable classes
# -------------------------------------------------------------------------

class ScalarVar(BaseVariable):
    """Simple scalar — native SciDuck storage."""
    schema_version = 1


class ArrayVar(BaseVariable):
    """1-D numpy array — native SciDuck storage."""
    schema_version = 1


class MatrixVar(BaseVariable):
    """2-D numpy array — native SciDuck storage."""
    schema_version = 1


class DFVar(BaseVariable):
    """DataFrame with custom serialization."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return self.data.copy()

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_global_state():
    """Reset global singletons between tests."""
    if hasattr(_local, "database"):
        delattr(_local, "database")
    Thunk.query = None
    yield
    if hasattr(_local, "database"):
        delattr(_local, "database")
    Thunk.query = None


@pytest.fixture
def server_app(tmp_path):
    """Create a FastAPI app backed by a temporary database."""
    db_path = tmp_path / "test.duckdb"
    app = create_app(
        dataset_db_path=str(db_path),
        dataset_schema_keys=TEST_SCHEMA_KEYS,
    )
    yield app
    app.state.db.close()


@pytest.fixture
def client(server_app):
    """RemoteDatabaseManager wired to the test app via ASGI transport."""
    transport = ASGITransport(app=server_app)
    import httpx
    http_client = httpx.Client(transport=transport, base_url="http://testserver")
    rdb = RemoteDatabaseManager.__new__(RemoteDatabaseManager)
    rdb.base_url = "http://testserver"
    rdb._client = http_client
    rdb._registered_types = {}
    yield rdb
    http_client.close()
