"""SciDuck — A thin DuckDB layer for managing versioned scientific data."""

from .sciduckdb import (
    SciDuck,
    _infer_duckdb_type,
    _numpy_dtype_to_duckdb,
    _python_to_storage,
    _storage_to_python,
    _infer_data_columns,
    _value_to_storage_row,
    _dataframe_to_storage_rows,
    _flatten_dict,
    _unflatten_dict,
)

__all__ = [
    "SciDuck",
    "_infer_duckdb_type",
    "_numpy_dtype_to_duckdb",
    "_python_to_storage",
    "_storage_to_python",
    "_infer_data_columns",
    "_value_to_storage_row",
    "_dataframe_to_storage_rows",
    "_flatten_dict",
    "_unflatten_dict",
]
