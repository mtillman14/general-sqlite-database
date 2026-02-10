"""SciDuck — A thin DuckDB layer for managing versioned scientific data."""

from .sciduck import (
    SciDuck,
    _infer_duckdb_type,
    _numpy_dtype_to_duckdb,
    _python_to_storage,
    _storage_to_python,
)

__all__ = [
    "SciDuck",
    "_infer_duckdb_type",
    "_numpy_dtype_to_duckdb",
    "_python_to_storage",
    "_storage_to_python",
]
