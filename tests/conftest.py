"""Pytest configuration and shared fixtures for scidb tests."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scidb import DatabaseManager, BaseVariable, configure_database
from scidb.database import _local


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a temporary database path."""
    return tmp_path / "test_db.sqlite"


@pytest.fixture
def db(temp_db_path):
    """Provide a fresh DatabaseManager instance."""
    db = DatabaseManager(temp_db_path)
    yield db
    db.close()


@pytest.fixture
def configured_db(temp_db_path):
    """Provide a configured global database."""
    db = configure_database(temp_db_path)
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
    yield
    if hasattr(_local, 'database'):
        delattr(_local, 'database')


# --- Sample Variable Classes for Testing ---

class ScalarValue(BaseVariable):
    """Simple scalar value for testing."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": [self.data]})

    @classmethod
    def from_db(cls, df: pd.DataFrame):
        return df["value"].iloc[0]


class ArrayValue(BaseVariable):
    """1D numpy array for testing."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({
            "values": [self.data.tobytes()],
            "shape": [str(self.data.shape)],
            "dtype": [str(self.data.dtype)],
        })

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        row = df.iloc[0]
        shape = eval(row["shape"])
        dtype = np.dtype(row["dtype"])
        arr = np.frombuffer(row["values"], dtype=dtype)
        return arr.reshape(shape)


class MatrixValue(BaseVariable):
    """2D numpy array for testing."""
    schema_version = 2  # Different schema version for testing

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({
            "values": [self.data.tobytes()],
            "shape": [str(self.data.shape)],
            "dtype": [str(self.data.dtype)],
        })

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        row = df.iloc[0]
        shape = eval(row["shape"])
        dtype = np.dtype(row["dtype"])
        arr = np.frombuffer(row["values"], dtype=dtype)
        return arr.reshape(shape)


class DataFrameValue(BaseVariable):
    """Pandas DataFrame for testing."""
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
