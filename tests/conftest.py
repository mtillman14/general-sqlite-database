"""Pytest configuration and shared fixtures for scidb tests."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add all local packages to path for imports
import sys
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "canonical-hash" / "src"))
sys.path.insert(0, str(_root / "thunk-lib" / "src"))
sys.path.insert(0, str(_root / "path-gen" / "src"))
sys.path.insert(0, str(_root / "pipelinedb-lib" / "src"))
sys.path.insert(0, str(_root / "scirun-lib" / "src"))
sys.path.insert(0, str(_root / "sciduck" / "src"))

from scidb import BaseVariable, configure_database
from scidb.database import _local
from scidb.thunk import Thunk


# Default schema keys for testing - defines the hierarchical dataset structure.
# Keys not in this list are treated as version parameters (e.g. stage, type, dtype).
DEFAULT_TEST_SCHEMA_KEYS = ["subject", "trial"]


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a temporary database path."""
    return tmp_path / "test_db.sqlite"


@pytest.fixture
def db(tmp_path):
    """Provide a fresh configured database instance."""
    db_path = tmp_path / "test_db.duckdb"
    pipeline_path = tmp_path / "test_pipeline.db"
    db = configure_database(db_path, DEFAULT_TEST_SCHEMA_KEYS, pipeline_path)
    yield db
    db.close()


@pytest.fixture
def configured_db(tmp_path):
    """Provide a configured global database."""
    db_path = tmp_path / "test_db.duckdb"
    pipeline_path = tmp_path / "test_pipeline.db"
    db = configure_database(db_path, DEFAULT_TEST_SCHEMA_KEYS, pipeline_path)
    yield db
    db.close()
    # Clear the global state
    if hasattr(_local, 'database'):
        delattr(_local, 'database')


@pytest.fixture(autouse=True)
def clear_global_db():
    """Clear global database state before each test."""
    if hasattr(_local, 'database'):
        delattr(_local, 'database')
    Thunk.query = None
    yield
    if hasattr(_local, 'database'):
        delattr(_local, 'database')
    Thunk.query = None


# --- Sample Variable Classes for Testing ---
# ScalarValue, ArrayValue, MatrixValue use native SciDuck storage (no to_db/from_db).
# DataFrameValue uses custom serialization to test that path too.

class ScalarValue(BaseVariable):
    """Simple scalar value — uses native SciDuck storage."""
    schema_version = 1


class ArrayValue(BaseVariable):
    """1D numpy array — uses native SciDuck storage."""
    schema_version = 1


class MatrixValue(BaseVariable):
    """2D numpy array — uses native SciDuck storage."""
    schema_version = 2  # Different schema version for testing


class DataFrameValue(BaseVariable):
    """Pandas DataFrame — uses custom serialization (to_db/from_db)."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return self.data.copy()

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


@pytest.fixture
def scalar_class():
    return ScalarValue


@pytest.fixture
def array_class():
    return ArrayValue


@pytest.fixture
def matrix_class():
    return MatrixValue


@pytest.fixture
def dataframe_class():
    return DataFrameValue
