"""Database connection and management using SciDuck backend."""

import json
import os
import random
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Type, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .lineage import LineageRecord

from .exceptions import (
    DatabaseNotConfiguredError,
    NotFoundError,
    NotRegisteredError,
)
from .hashing import generate_record_id, canonical_hash
from .variable import BaseVariable


def _schema_str(value):
    """Stringify a schema key value, converting whole-number floats to int.

    Schema keys are stored as VARCHAR in DuckDB.  str(1.0) → "1.0" but
    str(1) → "1".  MATLAB sends all numbers as float, so without this
    conversion, queries and cache lookups fail because "1.0" ≠ "1".
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _from_schema_str(value):
    """Convert a schema VARCHAR value back to a numeric type if possible.

    Schema keys are stored as VARCHAR, so loaded values are always strings.
    This restores the original type (int or float) so that user-facing
    metadata has the same type as what was originally saved.
    """
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return value

# Add sub-package paths
import sys
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root / "sciduck" / "src"))
sys.path.insert(0, str(_project_root / "pipelinedb-lib" / "src"))

from sciduck import SciDuck, _infer_duckdb_type, _python_to_storage
from pipelinedb import PipelineDB


# Global database instance (thread-local for safety)
_local = threading.local()


def _is_tabular_dict(data):
    """Return True if data is a dict where ALL values are 1D (or Nx1 column-vector) numpy arrays of equal length."""
    if not isinstance(data, dict) or len(data) == 0:
        print(f"    [_is_tabular_dict] FAIL: not dict or empty (type={type(data).__name__}, len={len(data) if isinstance(data, dict) else 'N/A'})")
        return False
    lengths = set()
    for k, v in data.items():
        if not isinstance(v, np.ndarray):
            print(f"    [_is_tabular_dict] FAIL: key={k!r} is {type(v).__name__}, not ndarray")
            return False
        # Accept 1D arrays, Nx1 column vectors, and 1xN row vectors (from MATLAB)
        if v.ndim == 1:
            lengths.add(v.shape[0])
        elif v.ndim == 2 and v.shape[0] == 1:
            lengths.add(v.shape[1])
        elif v.ndim == 2 and v.shape[1] == 1:
            lengths.add(v.shape[0])
        else:
            print(f"    [_is_tabular_dict] FAIL: key={k!r} has shape={v.shape}, ndim={v.ndim}")
            return False
    result = len(lengths) == 1
    if result:
        print(f"    [_is_tabular_dict] OK: {len(data)} keys, uniform length={lengths.pop()}")
    else:
        print(f"    [_is_tabular_dict] FAIL: unequal lengths={lengths}")
    return result


def _get_leaf_paths(d, prefix=()):
    """Recursively get all leaf paths in a nested dict.

    A leaf is any value that is NOT a dict.  Returns a list of tuples,
    each tuple being the sequence of keys from root to leaf.
    """
    paths = []
    for key, value in d.items():
        current = prefix + (key,)
        if isinstance(value, dict):
            paths.extend(_get_leaf_paths(value, current))
        else:
            paths.append(current)
    return paths


def _get_nested_value(d, path):
    """Get a value from a nested dict following *path* (tuple of keys)."""
    current = d
    for key in path:
        current = current[key]
    return current


def _set_nested_value(d, path, value):
    """Set a value in a nested dict by *path*, creating intermediate dicts."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def _flatten_struct_columns(df):
    """Flatten DataFrame columns that contain nested dicts into dot-separated columns.

    For each object-dtype column whose first non-null value is a ``dict``,
    recursively extract all leaf paths and create new columns named
    ``"original_col.key1.key2.leaf"``.

    **Leaf handling:**
    - Scalar leaves (int, float, str, bool, None) are stored directly.
    - Array leaves (numpy arrays, Python lists) are serialised to a JSON
      string so every cell in the resulting column is a simple scalar type
      that DuckDB can ingest.

    Returns
    -------
    (flattened_df, struct_columns_info)
        *struct_columns_info* maps each flattened original column name to
        metadata needed by ``_unflatten_struct_columns`` on load.
        Empty dict when no struct columns are found.
    """
    if len(df) == 0:
        return df, {}

    struct_info = {}
    cols_to_drop = []
    new_col_data = {}  # ordered: col_name -> list of values

    for col_idx, col in enumerate(df.columns):
        if df[col].dtype != object:
            continue

        # Find first non-null value
        first_val = None
        for v in df[col]:
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                first_val = v
                break

        if not isinstance(first_val, dict):
            continue

        # This column contains nested dicts — flatten it
        leaf_paths = _get_leaf_paths(first_val)
        if not leaf_paths:
            continue

        print(f"    [_flatten_struct_columns] column {col!r}: "
              f"{len(leaf_paths)} leaf paths detected")

        array_leaves = {}  # dot_path -> {"dtype": ..., "shape": ...}

        for path in leaf_paths:
            dot_path = ".".join(path)
            flat_col_name = f"{col}.{dot_path}"
            values = []
            for row_val in df[col]:
                if row_val is None or (isinstance(row_val, float) and np.isnan(row_val)):
                    values.append(None)
                    continue
                try:
                    leaf = _get_nested_value(row_val, path)
                except (KeyError, TypeError):
                    values.append(None)
                    continue

                if isinstance(leaf, np.ndarray):
                    # Track array metadata from first occurrence
                    if dot_path not in array_leaves:
                        array_leaves[dot_path] = {
                            "dtype": str(leaf.dtype),
                            "shape": list(leaf.shape),
                        }
                    values.append(json.dumps(leaf.tolist()))
                elif isinstance(leaf, list):
                    if dot_path not in array_leaves:
                        array_leaves[dot_path] = {"dtype": "list"}
                    values.append(json.dumps(leaf))
                else:
                    values.append(leaf)

            new_col_data[flat_col_name] = values
            print(f"      -> {flat_col_name} "
                  f"({'array' if dot_path in array_leaves else 'scalar'})")

        cols_to_drop.append(col)
        struct_info[col] = {
            "paths": [list(p) for p in leaf_paths],
            "array_leaves": array_leaves,
            "col_position": col_idx,
        }

    if not cols_to_drop:
        return df, {}

    result = df.drop(columns=cols_to_drop)
    for name, values in new_col_data.items():
        result[name] = values

    print(f"    [_flatten_struct_columns] flattened {len(cols_to_drop)} struct column(s) "
          f"into {len(new_col_data)} flat columns")
    return result, struct_info


def _unflatten_struct_columns(df, struct_info):
    """Reconstruct nested-dict columns from dot-separated flat columns.

    Inverse of ``_flatten_struct_columns``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with dot-separated columns produced by ``_flatten_struct_columns``.
    struct_info : dict
        The metadata dict that was stored alongside the data.

    Returns
    -------
    pd.DataFrame with the original nested-dict object columns restored.
    """
    if not struct_info:
        return df

    result = df.copy()

    # Process struct columns in reverse position order so inserts don't shift indices
    for col_name, info in sorted(
        ((k, v) for k, v in struct_info.items() if k != "__list_columns__"),
        key=lambda x: x[1]["col_position"],
        reverse=True,
    ):
        paths = [tuple(p) for p in info["paths"]]
        array_leaves = info.get("array_leaves", {})
        col_position = info["col_position"]

        # Collect all flat column names belonging to this struct
        flat_col_names = [f"{col_name}.{'.'.join(p)}" for p in paths]
        existing_flat = [c for c in flat_col_names if c in result.columns]

        if not existing_flat:
            print(f"    [_unflatten_struct_columns] WARNING: no flat columns found "
                  f"for {col_name!r}, skipping")
            continue

        print(f"    [_unflatten_struct_columns] reconstituting {col_name!r} "
              f"from {len(existing_flat)} flat columns")

        # Build nested dicts row by row
        nested_values = []
        n_rows = len(result)

        # DIAGNOSTIC: Log array_leaves metadata
        print(f"    [_unflatten DIAG] col {col_name!r}: "
              f"array_leaves keys={list(array_leaves.keys())}, "
              f"array_leaves={array_leaves}")

        for row_idx in range(n_rows):
            row_dict = {}
            for path, flat_col in zip(paths, flat_col_names):
                if flat_col not in result.columns:
                    if row_idx == 0:
                        print(f"    [_unflatten DIAG]   flat_col {flat_col!r} NOT in result columns!")
                    continue
                val = result[flat_col].iloc[row_idx]

                # DIAGNOSTIC: Log each leaf value on first row
                dot_path = ".".join(path)
                if row_idx == 0:
                    print(f"    [_unflatten DIAG]   path={dot_path!r}, "
                          f"raw val type={type(val).__name__}, "
                          f"in_array_leaves={dot_path in array_leaves}, "
                          f"val_preview={repr(val)[:120]}")

                # Restore arrays from JSON
                if dot_path in array_leaves and val is not None:
                    arr_meta = array_leaves[dot_path]
                    if isinstance(val, str):
                        parsed = json.loads(val)
                    else:
                        parsed = val
                    if arr_meta.get("dtype") == "list":
                        val = parsed
                    else:
                        val = np.array(parsed, dtype=np.dtype(arr_meta["dtype"]))
                        expected_shape = arr_meta.get("shape")
                        if (expected_shape and list(val.shape) != expected_shape
                                and val.size == np.prod(expected_shape)):
                            val = val.reshape(expected_shape)
                    if row_idx == 0:
                        print(f"    [_unflatten DIAG]   RESTORED: type={type(val).__name__}"
                              f"{f', dtype={val.dtype}, shape={val.shape}' if isinstance(val, np.ndarray) else ''}")
                elif row_idx == 0 and isinstance(val, str) and val.strip().startswith('['):
                    print(f"    [_unflatten DIAG]   WARNING: string starting with '[' but "
                          f"dot_path {dot_path!r} NOT in array_leaves! "
                          f"This value will remain a string.")

                _set_nested_value(row_dict, path, val)
            nested_values.append(row_dict)

        # Drop the flat columns
        result = result.drop(columns=existing_flat)

        # Insert the reconstituted column at its original position
        # (clamped to current column count since other columns may have shifted)
        insert_pos = min(col_position, len(result.columns))
        result.insert(insert_pos, col_name, nested_values)

    # Convert list-valued cells to numpy arrays for MATLAB interop.
    # DuckDB DOUBLE[] columns come back as Python lists; old VARCHAR saves
    # come back as string representations like "[1.0, 2.0, 3.0]".
    for col in result.columns:
        if result[col].dtype != object:
            continue
        first_val = next(
            (v for v in result[col]
             if v is not None and not (isinstance(v, float) and np.isnan(v))),
            None,
        )
        if first_val is None:
            continue

        # DIAGNOSTIC: Log what we find in post-unflatten object columns
        print(f"    [_unflatten DIAG postprocess] col {col!r}: "
              f"first_val type={type(first_val).__name__}"
              f"{f', is_dict={True}' if isinstance(first_val, dict) else ''}"
              f"{f', dtype={first_val.dtype}' if isinstance(first_val, np.ndarray) else ''}"
              f"{f', preview={repr(first_val)[:100]}' if isinstance(first_val, str) else ''}")

        if isinstance(first_val, (list, np.ndarray)):
            # DuckDB DOUBLE[] returns as lists or numpy arrays — ensure numpy
            # DIAGNOSTIC: Check if we're losing bool type
            if isinstance(first_val, list) and any(isinstance(x, bool) for x in first_val):
                print(f"    [_unflatten DIAG postprocess]   WARNING: list contains bools "
                      f"but converting to dtype=float, losing bool type!")
            result[col] = result[col].apply(
                lambda v: np.array(v, dtype=float) if isinstance(v, list) else v)
        elif isinstance(first_val, str) and first_val.strip().startswith('['):
            # Backwards compat: parse VARCHAR strings from old saves
            print(f"    [_unflatten DIAG postprocess]   Parsing string as JSON -> float array")
            def _parse_list_str(v):
                if not isinstance(v, str):
                    return v
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return np.array(parsed, dtype=float)
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
                return v
            result[col] = result[col].apply(_parse_list_str)

    return result


def get_user_id() -> str | None:
    """
    Get the current user ID from environment.

    The user ID is used for attribution in cross-user provenance tracking.
    Set the SCIDB_USER_ID environment variable to identify the current user.

    Returns:
        The user ID string, or None if not set.
    """
    return os.environ.get("SCIDB_USER_ID")


def configure_database(
    dataset_db_path: str | Path,
    dataset_schema_keys: list[str],
    pipeline_db_path: str | Path,
    lineage_mode: str = "strict",
) -> "DatabaseManager":
    """
    Configure the global database connection.

    Single-call setup that creates the database, auto-registers all known
    BaseVariable subclasses, and enables thunk caching.

    Args:
        dataset_db_path: Path to the DuckDB database file
        dataset_schema_keys: List of metadata keys that define the dataset schema
            (e.g., ["subject", "visit", "channel"]). These keys identify the
            logical location of data and are used for the folder hierarchy.
            Any metadata keys not in this list are treated as version parameters
            that distinguish different computational versions of the same data.
        pipeline_db_path: Path to the SQLite database for lineage storage.
        lineage_mode: How to handle intermediate variables in lineage tracking.
            - "strict" (default): All upstream BaseVariables must be saved
              before saving downstream results. Raises UnsavedIntermediateError
              if an unsaved intermediate is detected.
            - "ephemeral": Allows unsaved intermediates. Stores the computation
              graph (function, inputs) without storing actual data for unsaved
              variables. Enables full provenance tracking with smaller database
              size, but cache hits won't work for ephemeral intermediates.

    Returns:
        The DatabaseManager instance

    Raises:
        ValueError: If lineage_mode is not "strict" or "ephemeral"
    """
    db = DatabaseManager(
        dataset_db_path,
        dataset_schema_keys=dataset_schema_keys,
        pipeline_db_path=pipeline_db_path,
        lineage_mode=lineage_mode,
    )
    for cls in BaseVariable._all_subclasses.values():
        db.register(cls)
    db.set_current_db()
    return db


def get_database() -> "DatabaseManager":
    """
    Get the global database connection.

    Returns:
        The DatabaseManager instance

    Raises:
        DatabaseNotConfiguredError: If configure_database() hasn't been called
    """
    db = getattr(_local, "database", None)
    if db is None:
        raise DatabaseNotConfiguredError(
            "Database not configured. Call configure_database(path) first."
        )
    if getattr(db, "_closed", False):
        db.reopen()
    return db


class DatabaseManager:
    """
    Manages data storage (DuckDB via SciDuck) and lineage persistence (SQLite via PipelineDB).

    Example:
        db = configure_database("experiment.duckdb", ["subject", "session"], "pipeline.db")

        RawSignal.save(np.eye(3), subject=1, session=1)
        loaded = RawSignal.load(subject=1, session=1)
    """

    VALID_LINEAGE_MODES = ("strict", "ephemeral")

    def __init__(
        self,
        dataset_db_path: str | Path,
        dataset_schema_keys: list[str],
        pipeline_db_path: str | Path,
        lineage_mode: str = "strict",
    ):
        """
        Initialize database connection.

        Args:
            dataset_db_path: Path to DuckDB database file (created if doesn't exist)
            dataset_schema_keys: List of metadata keys that define the dataset schema
                (e.g., ["subject", "visit", "channel"]). These keys identify the
                logical location of data. Any other metadata keys are treated as
                version parameters.
            pipeline_db_path: Path to SQLite database for lineage storage.
            lineage_mode: How to handle intermediate variables ("strict" or "ephemeral")

        Raises:
            ValueError: If lineage_mode is not valid
        """
        if lineage_mode not in self.VALID_LINEAGE_MODES:
            raise ValueError(
                f"lineage_mode must be one of {self.VALID_LINEAGE_MODES}, "
                f"got '{lineage_mode}'"
            )

        self.dataset_db_path = Path(dataset_db_path)
        self.lineage_mode = lineage_mode

        if isinstance(dataset_schema_keys, (set, frozenset)):
            raise TypeError(
                "dataset_schema_keys must be an ordered sequence (list or tuple), "
                "not a set. Schema key order defines the dataset hierarchy."
            )
        self.dataset_schema_keys = list(dataset_schema_keys)
        self.pipeline_db_path = Path(pipeline_db_path)
        self._registered_types: dict[str, Type[BaseVariable]] = {}

        # Initialize SciDuck backend for data storage
        self._duck = SciDuck(self.dataset_db_path, dataset_schema=dataset_schema_keys)

        # Initialize PipelineDB for lineage storage (SQLite)
        self._pipeline_db = PipelineDB(pipeline_db_path)

        # Create metadata tables for type registration (in DuckDB)
        self._ensure_meta_tables()
        self._ensure_record_metadata_table()

        self._closed = False # Track connection open/closed state

    def _ensure_meta_tables(self):
        """Create internal metadata tables for type registration."""
        # Registered types table (remains in DuckDB for data type discovery)
        # Note: Only type_name is unique (PRIMARY KEY). table_name is not unique
        # to avoid DuckDB's ON CONFLICT ambiguity with multiple unique constraints.
        self._duck._execute("""
            CREATE TABLE IF NOT EXISTS _registered_types (
                type_name VARCHAR PRIMARY KEY,
                table_name VARCHAR NOT NULL,
                schema_version INTEGER NOT NULL,
                registered_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

    def _ensure_record_metadata_table(self):
        """Create the _record_metadata side table for record-level metadata."""
        self._duck._execute("""
            CREATE TABLE IF NOT EXISTS _record_metadata (
                record_id VARCHAR PRIMARY KEY,
                variable_name VARCHAR NOT NULL,
                parameter_id INTEGER NOT NULL,
                version_id INTEGER NOT NULL DEFAULT 1,
                schema_id INTEGER NOT NULL,
                content_hash VARCHAR,
                lineage_hash VARCHAR,
                schema_version INTEGER,
                user_id VARCHAR,
                created_at VARCHAR NOT NULL
            )
        """)

    def _create_variable_view(self, variable_class: Type[BaseVariable]):
        """Create a view joining a variable table with _schema and _variables."""
        table_name = variable_class.table_name()
        view_name = variable_class.view_name()
        schema_cols = ", ".join(f's."{col}"' for col in self.dataset_schema_keys)
        self._duck._execute(f"""
            CREATE OR REPLACE VIEW "{view_name}" AS
            SELECT
                t.*,
                s.schema_level, {schema_cols},
                v.version_keys, v.description
            FROM "{table_name}" t
            LEFT JOIN _schema s ON t.schema_id = s.schema_id
            LEFT JOIN _variables v
                ON v.variable_name = '{table_name}'
                AND t.parameter_id = v.parameter_id
        """)

    def _split_metadata(self, flat_metadata: dict) -> dict:
        """
        Split flat metadata into nested schema/version structure.

        Keys in schema_keys go to "schema", all other keys go to "version".
        """
        schema = {}
        version = {}
        for key, value in flat_metadata.items():
            if key in self.dataset_schema_keys:
                schema[key] = value
            else:
                version[key] = value
        return {"schema": schema, "version": version}

    def _infer_schema_level(self, schema_keys: dict) -> str | None:
        """
        Infer the schema level from provided keys.

        Walks dataset_schema_keys top-down. Returns the deepest provided key.
        Keys need not be contiguous — any subset of schema keys is valid.

        Returns None if no schema keys are provided.
        """
        if not schema_keys:
            return None

        level = None
        for key in self.dataset_schema_keys:
            if key in schema_keys:
                level = key
        return level

    def _save_record_metadata(
        self,
        record_id: str,
        variable_name: str,
        parameter_id: int,
        version_id: int,
        schema_id: int,
        content_hash: str,
        lineage_hash: str | None,
        schema_version: int,
        user_id: str | None,
        created_at: str,
    ) -> None:
        """Insert a record into _record_metadata. Idempotent via ON CONFLICT DO NOTHING."""
        self._duck._execute(
            """
            INSERT INTO _record_metadata (
                record_id, variable_name, parameter_id, version_id, schema_id,
                content_hash, lineage_hash, schema_version,
                user_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (record_id) DO NOTHING
            """,
            [
                record_id, variable_name, parameter_id, version_id, schema_id,
                content_hash, lineage_hash, schema_version,
                user_id, created_at,
            ],
        )

    def _resolve_parameter_slot(
        self,
        table_name: str,
        schema_id: int,
        version_keys: dict | None = None,
    ) -> tuple[int, int, bool]:
        """
        Determine (parameter_id, version_id, is_new_parameter) for a save operation.

        If a _variables row exists for (table_name, version_keys) JSON,
        reuse that parameter_id and increment version_id from _record_metadata.
        Otherwise, allocate a new parameter_id with version_id=1.

        Returns:
            (parameter_id, version_id, is_new_parameter) where is_new_parameter
            is True only when a brand-new parameter_id was allocated (i.e. no
            _variables row existed). Callers should only INSERT into _variables
            when is_new_parameter is True.
        """
        vk_json = json.dumps(version_keys or {}, sort_keys=True)

        # Check if _variables has a matching row for this variable + version_keys
        existing = self._duck._fetchall(
            "SELECT parameter_id FROM _variables "
            "WHERE variable_name = ? AND version_keys = ?",
            [table_name, vk_json],
        )

        if existing:
            parameter_id = existing[0][0]
            # Increment version_id for this parameter slot at this schema location
            version_rows = self._duck._fetchall(
                "SELECT COALESCE(MAX(version_id), 0) FROM _record_metadata "
                "WHERE variable_name = ? AND parameter_id = ? AND schema_id = ?",
                [table_name, parameter_id, schema_id],
            )
            version_id = version_rows[0][0] + 1
            return parameter_id, version_id, False
        else:
            # Allocate new parameter_id
            param_rows = self._duck._fetchall(
                "SELECT COALESCE(MAX(parameter_id), 0) FROM _variables "
                "WHERE variable_name = ?",
                [table_name],
            )
            parameter_id = param_rows[0][0] + 1
            version_id = 1
            return parameter_id, version_id, True

    def _save_columnar(
        self,
        table_name: str,
        variable_class: Type[BaseVariable],
        df: pd.DataFrame,
        schema_level: str | None,
        schema_keys: dict,
        content_hash: str,
        version_keys: dict | None = None,
        dict_of_arrays: bool = False,
        ndarray_keys: dict | None = None,
        struct_columns: dict | None = None,
    ) -> tuple[int, int, int]:
        """
        Save a DataFrame into SciDuck's table format (columnar storage).

        Used for custom-serialized data (to_db/from_db), native DataFrames,
        and dict-of-arrays data.

        Creates a table with schema_id, parameter_id, version_id, and data columns.
        Registers in _variables with appropriate dtype metadata.

        Args:
            dict_of_arrays: If True, the data originated from a dict of 1D numpy arrays.
            ndarray_keys: When dict_of_arrays=True, maps column names to
                {"dtype": str, "shape": list} for lossless round-tripping.

        Returns (parameter_id, schema_id, version_id).
        """
        _t0 = time.perf_counter()
        schema_id = (
            self._duck._get_or_create_schema_id(schema_level, schema_keys)
            if schema_level is not None and schema_keys
            else 0
        )

        # Resolve parameter_id and version_id
        parameter_id, version_id, is_new_parameter = self._resolve_parameter_slot(
            table_name, schema_id, version_keys
        )
        _t_resolve = time.perf_counter() - _t0
        print(f"      [_save_columnar] resolve schema/param slot: {_t_resolve:.4f}s")

        # Ensure table exists with SciDuck-compatible schema
        _t0 = time.perf_counter()
        if not self._duck._table_exists(table_name):
            # Infer column types from DataFrame
            col_defs = []
            for col in df.columns:
                # Map pandas dtypes to DuckDB types
                dtype = df[col].dtype
                if pd.api.types.is_integer_dtype(dtype):
                    ddb_type = "BIGINT"
                elif pd.api.types.is_float_dtype(dtype):
                    ddb_type = "DOUBLE"
                elif pd.api.types.is_bool_dtype(dtype):
                    ddb_type = "BOOLEAN"
                elif dtype == object:
                    # Check if this column contains numeric lists/arrays
                    first_val = next(
                        (v for v in df[col]
                         if v is not None and not (isinstance(v, float) and np.isnan(v))),
                        None,
                    )
                    # DIAGNOSTIC: Log what we find in object-dtype columns
                    print(f"      [_save_columnar DIAG] object col {col!r}: "
                          f"first_val type={type(first_val).__name__}"
                          f"{f', dtype={first_val.dtype}, shape={first_val.shape}' if isinstance(first_val, np.ndarray) else ''}"
                          f"{f', len={len(first_val)}, elem_types={set(type(x).__name__ for x in first_val[:5])}' if isinstance(first_val, list) and len(first_val) > 0 else ''}"
                          f"{f', preview={repr(first_val)[:100]}' if isinstance(first_val, str) else ''}")
                    if isinstance(first_val, np.ndarray) and np.issubdtype(first_val.dtype, np.number):
                        ddb_type = "DOUBLE[]"
                        print(f"      [_save_columnar DIAG]   -> DOUBLE[] (numeric ndarray)")
                    elif isinstance(first_val, np.ndarray) and first_val.dtype.kind == 'b':
                        # DIAGNOSTIC: Bool arrays fail np.issubdtype(bool, number)!
                        print(f"      [_save_columnar DIAG]   -> VARCHAR (PROBLEM: bool ndarray "
                              f"fails np.issubdtype check, should be BOOLEAN[])")
                        ddb_type = "VARCHAR"
                    elif (isinstance(first_val, list) and len(first_val) > 0
                          and all(isinstance(x, (int, float)) for x in first_val)):
                        # Note: isinstance(True, int) is True in Python, so bool lists pass this
                        has_bools = any(isinstance(x, bool) for x in first_val)
                        ddb_type = "DOUBLE[]"
                        print(f"      [_save_columnar DIAG]   -> DOUBLE[] (list of numbers"
                              f"{', CONTAINS BOOLS - will lose bool type!' if has_bools else ''})")
                    else:
                        ddb_type = "VARCHAR"
                        print(f"      [_save_columnar DIAG]   -> VARCHAR (fallback)")
                else:
                    ddb_type = "VARCHAR"
                col_defs.append(f'"{col}" {ddb_type}')

            data_cols_sql = ", ".join(col_defs)
            self._duck._execute(f"""
                CREATE TABLE "{table_name}" (
                    schema_id INTEGER NOT NULL,
                    parameter_id INTEGER NOT NULL,
                    version_id INTEGER NOT NULL,
                    {data_cols_sql}
                )
            """)
            self._create_variable_view(variable_class)
        _t_table = time.perf_counter() - _t0
        print(f"      [_save_columnar] ensure table exists: {_t_table:.4f}s")

        # Insert all DataFrame rows with the same (schema_id, parameter_id, version_id)
        _t0 = time.perf_counter()
        insert_df = df.copy()
        insert_df.insert(0, "version_id", version_id)
        insert_df.insert(0, "parameter_id", parameter_id)
        insert_df.insert(0, "schema_id", schema_id)
        _t_prep = time.perf_counter() - _t0
        print(f"      [_save_columnar] DataFrame prep (copy+insert cols): {_t_prep:.4f}s")
        print(f"      [_save_columnar] DataFrame shape: {insert_df.shape}, dtypes: {dict(insert_df.dtypes)}")
        print(f"      [_save_columnar] DataFrame memory: {insert_df.memory_usage(deep=True).sum() / 1024:.1f} KB")

        _t0 = time.perf_counter()
        col_str = ", ".join(f'"{c}"' for c in insert_df.columns)
        self._duck.con.execute(
            f'INSERT INTO "{table_name}" ({col_str}) SELECT * FROM insert_df'
        )
        _t_insert = time.perf_counter() - _t0
        print(f"      [_save_columnar] DuckDB INSERT: {_t_insert:.4f}s")

        # Register in _variables only for truly new parameter slots
        effective_level = schema_level or self.dataset_schema_keys[-1]
        if is_new_parameter:
            if dict_of_arrays:
                dtype_json = json.dumps({
                    "custom": True,
                    "dict_of_arrays": True,
                    "ndarray_keys": ndarray_keys or {},
                })
            elif struct_columns:
                dtype_json = json.dumps({
                    "custom": True,
                    "struct_columns": struct_columns,
                })
            else:
                dtype_json = json.dumps({"custom": True})
            self._duck._execute(
                "INSERT INTO _variables (variable_name, parameter_id, schema_level, "
                "dtype, version_keys, description) VALUES (?, ?, ?, ?, ?, ?)",
                [table_name, parameter_id, effective_level,
                 dtype_json,
                 json.dumps(version_keys or {}, sort_keys=True), ""],
            )

        return parameter_id, schema_id, version_id

    def _save_single_value(
        self,
        table_name: str,
        variable_class: Type[BaseVariable],
        data: Any,
        content_hash: str,
        schema_level: str | None = None,
        schema_keys: dict | None = None,
        version_keys: dict | None = None,
    ) -> tuple[int, int, int]:
        """
        Save native data as a single 'value' column.

        Handles both schema-aware and schema-less storage. When schema_level
        and schema_keys are provided, resolves a schema_id; otherwise uses 0.

        Returns (parameter_id, schema_id, version_id).
        """
        _t0 = time.perf_counter()
        if schema_level is not None and schema_keys:
            schema_id = self._duck._get_or_create_schema_id(
                schema_level, {k: _schema_str(v) for k, v in schema_keys.items()}
            )
        else:
            schema_id = 0

        parameter_id, version_id, is_new_parameter = self._resolve_parameter_slot(
            table_name, schema_id, version_keys
        )
        print(f"      [_save_single_value] resolve slot: {time.perf_counter() - _t0:.4f}s")

        _t0 = time.perf_counter()
        ddb_type, col_meta = _infer_duckdb_type(data)
        print(f"      [_save_single_value] _infer_duckdb_type: {time.perf_counter() - _t0:.4f}s  -> {ddb_type}, python_type={col_meta.get('python_type')}")

        _t0 = time.perf_counter()
        storage_value = _python_to_storage(data, col_meta)
        print(f"      [_save_single_value] _python_to_storage: {time.perf_counter() - _t0:.4f}s  -> value type={type(storage_value).__name__}, len={len(storage_value) if isinstance(storage_value, str) else 'N/A'}")
        dtype_meta = {"mode": "single_column", "columns": {"value": col_meta}}

        # Ensure table exists
        _t0 = time.perf_counter()
        if not self._duck._table_exists(table_name):
            self._duck._execute(f"""
                CREATE TABLE "{table_name}" (
                    schema_id INTEGER NOT NULL,
                    parameter_id INTEGER NOT NULL,
                    version_id INTEGER NOT NULL,
                    "value" {ddb_type}
                )
            """)
            self._create_variable_view(variable_class)
        print(f"      [_save_single_value] ensure table: {time.perf_counter() - _t0:.4f}s")

        # Insert data
        _t0 = time.perf_counter()
        self._duck._execute(
            f'INSERT INTO "{table_name}" ("schema_id", "parameter_id", "version_id", "value") VALUES (?, ?, ?, ?)',
            [schema_id, parameter_id, version_id, storage_value],
        )
        print(f"      [_save_single_value] DuckDB INSERT: {time.perf_counter() - _t0:.4f}s")

        # Register in _variables only for truly new parameter slots
        effective_level = schema_level or self.dataset_schema_keys[-1]
        if is_new_parameter:
            self._duck._execute(
                "INSERT INTO _variables (variable_name, parameter_id, schema_level, "
                "dtype, version_keys, description) VALUES (?, ?, ?, ?, ?, ?)",
                [table_name, parameter_id, effective_level,
                 json.dumps(dtype_meta),
                 json.dumps(version_keys or {}, sort_keys=True), ""],
            )

        return parameter_id, schema_id, version_id

    def save_batch(
        self,
        variable_class: Type[BaseVariable],
        data_items: list[tuple[Any, dict]],
        profile: bool = False,
    ) -> list[str]:
        """
        Bulk-save a list of (data_value, metadata_dict) pairs for a single variable type.

        Amortizes setup work (registration, table creation, parameter slot resolution)
        and batches SQL operations using executemany for ~100x speedup over per-row save().

        Args:
            variable_class: The BaseVariable subclass to save as
            data_items: List of (data_value, flat_metadata_dict) tuples
            profile: If True, print phase-by-phase timing summary

        Returns:
            List of record_ids for each saved item (in input order)
        """
        if not data_items:
            return []

        timings = {}
        t0 = time.perf_counter()

        table_name = self._ensure_registered(variable_class)
        type_name = variable_class.__name__
        schema_version = variable_class.schema_version
        user_id = get_user_id()

        # --- One-time setup from first item ---
        first_data, first_meta = data_items[0]
        nested_first = self._split_metadata(first_meta)
        version_keys = nested_first.get("version", {})

        # Detect whether data is dict-of-arrays for columnar storage
        is_dict_of_arrays = _is_tabular_dict(first_data)

        if is_dict_of_arrays:
            # Dict-of-arrays: multi-column table
            dict_columns = list(first_data.keys())
            ndarray_keys = {
                k: {"dtype": str(v.dtype), "shape": list(v.shape)}
                for k, v in first_data.items()
            }
            dtype_meta = {
                "custom": True,
                "dict_of_arrays": True,
                "ndarray_keys": ndarray_keys,
            }
            col_meta = None  # not used for dict-of-arrays path

            if not self._duck._table_exists(table_name):
                col_defs = []
                for col in dict_columns:
                    arr = first_data[col]
                    if np.issubdtype(arr.dtype, np.integer):
                        ddb_type = "BIGINT"
                    elif np.issubdtype(arr.dtype, np.floating):
                        ddb_type = "DOUBLE"
                    elif np.issubdtype(arr.dtype, np.bool_):
                        ddb_type = "BOOLEAN"
                    else:
                        ddb_type = "VARCHAR"
                    col_defs.append(f'"{col}" {ddb_type}')
                data_cols_sql = ", ".join(col_defs)
                self._duck._execute(f"""
                    CREATE TABLE "{table_name}" (
                        schema_id INTEGER NOT NULL,
                        parameter_id INTEGER NOT NULL,
                        version_id INTEGER NOT NULL,
                        {data_cols_sql}
                    )
                """)
                self._create_variable_view(variable_class)
        else:
            # Single-value path
            ddb_type, col_meta = _infer_duckdb_type(first_data)
            dtype_meta = {"mode": "single_column", "columns": {"value": col_meta}}

            if not self._duck._table_exists(table_name):
                self._duck._execute(f"""
                    CREATE TABLE "{table_name}" (
                        schema_id INTEGER NOT NULL,
                        parameter_id INTEGER NOT NULL,
                        version_id INTEGER NOT NULL,
                        "value" {ddb_type}
                    )
                """)
                self._create_variable_view(variable_class)

        timings["1_setup"] = time.perf_counter() - t0

        # --- Early fast path: skip all per-row work for re-saves ---
        t_fast = time.perf_counter()
        vk_json = json.dumps(version_keys or {}, sort_keys=True)
        existing_param = self._duck._fetchall(
            "SELECT parameter_id FROM _variables "
            "WHERE variable_name = ? AND version_keys = ?",
            [table_name, vk_json],
        )
        if existing_param:
            fast_parameter_id = existing_param[0][0]
            existing_count = self._duck._fetchall(
                "SELECT COUNT(*) FROM _record_metadata "
                "WHERE variable_name = ? AND parameter_id = ?",
                [table_name, fast_parameter_id],
            )[0][0]

            if existing_count >= len(data_items):
                # Query existing records with their schema key values
                schema_col_select = ", ".join(
                    f's."{k}"' for k in self.dataset_schema_keys
                )
                existing_records = self._duck._fetchall(
                    f"SELECT r.record_id, r.content_hash, {schema_col_select} "
                    f"FROM _record_metadata r "
                    f"LEFT JOIN _schema s ON r.schema_id = s.schema_id "
                    f"WHERE r.variable_name = ? AND r.parameter_id = ? "
                    f"ORDER BY r.version_id DESC",
                    [table_name, fast_parameter_id],
                )

                # Build lookup: schema key values -> (record_id, content_hash)
                # Latest version wins (ORDER BY version_id DESC + first-match)
                existing_by_keys = {}
                for row in existing_records:
                    rid = row[0]
                    chash = row[1]
                    schema_vals = tuple(
                        _schema_str(v) if v is not None else None for v in row[2:]
                    )
                    if schema_vals not in existing_by_keys:
                        existing_by_keys[schema_vals] = (rid, chash)

                # Map each input row by its schema key values
                result_ids = []
                input_keys = []
                all_matched = True
                for _, flat_meta in data_items:
                    key = tuple(
                        _schema_str(flat_meta[k]) if k in flat_meta else None
                        for k in self.dataset_schema_keys
                    )
                    input_keys.append(key)
                    if key in existing_by_keys:
                        result_ids.append(existing_by_keys[key][0])
                    else:
                        all_matched = False
                        break

                # Verify input keys are unique (otherwise ambiguous mapping)
                if all_matched and len(set(input_keys)) == len(input_keys):
                    # Sample-verify content hashes
                    sample_size = min(
                        max(10, len(data_items) // 100), len(data_items)
                    )
                    sample_indices = random.sample(
                        range(len(data_items)), sample_size
                    )

                    sample_ok = True
                    for idx in sample_indices:
                        data_val = data_items[idx][0]
                        actual_hash = canonical_hash(data_val)
                        expected_hash = existing_by_keys[input_keys[idx]][1]
                        if actual_hash != expected_hash:
                            sample_ok = False
                            break

                    if sample_ok:
                        if profile:
                            timings["fast_path"] = time.perf_counter() - t_fast
                            timings["total"] = time.perf_counter() - t0
                            print(
                                f"\n--- save_batch() FAST PATH "
                                f"({len(data_items)} items, all exist) ---"
                            )
                            for phase, elapsed in timings.items():
                                print(f"  {phase:30s} {elapsed:8.3f}s")
                            print()
                        return result_ids

        timings["1b_fast_path_miss"] = time.perf_counter() - t_fast

        # --- Batch schema_id resolution ---
        # Split all metadata and collect unique schema key combos
        t1 = time.perf_counter()
        all_nested = []
        unique_schema_combos = {}  # {tuple_of_values: schema_keys_dict}
        for data_val, flat_meta in data_items:
            nested = self._split_metadata(flat_meta)
            all_nested.append(nested)
            schema_keys = nested.get("schema", {})
            schema_level = self._infer_schema_level(schema_keys)
            if schema_level is not None and schema_keys:
                key_tuple = tuple(
                    _schema_str(schema_keys.get(k, "")) for k in self.dataset_schema_keys
                    if k in schema_keys
                )
                combo_key = (schema_level, key_tuple)
                if combo_key not in unique_schema_combos:
                    unique_schema_combos[combo_key] = schema_keys
            # else: schema_id = 0 (no schema keys)

        timings["2_split_metadata"] = time.perf_counter() - t1

        # Resolve schema_ids for all unique combos (batch)
        t2 = time.perf_counter()
        schema_id_cache = self._duck.batch_get_or_create_schema_ids(
            {k: {col: _schema_str(v) for col, v in vals.items()}
             for k, vals in unique_schema_combos.items()}
        )
        timings["3_schema_resolution"] = time.perf_counter() - t2

        # --- Resolve parameter_id (shared across all rows) ---
        t3 = time.perf_counter()
        # Use the first item's schema_id to resolve the parameter slot
        first_schema_keys = all_nested[0].get("schema", {})
        first_schema_level = self._infer_schema_level(first_schema_keys)
        if first_schema_level is not None and first_schema_keys:
            first_key_tuple = tuple(
                _schema_str(first_schema_keys.get(k, "")) for k in self.dataset_schema_keys
                if k in first_schema_keys
            )
            first_schema_id = schema_id_cache[(first_schema_level, first_key_tuple)]
        else:
            first_schema_id = 0

        vk_json = json.dumps(version_keys or {}, sort_keys=True)
        existing = self._duck._fetchall(
            "SELECT parameter_id FROM _variables "
            "WHERE variable_name = ? AND version_keys = ?",
            [table_name, vk_json],
        )

        if existing:
            parameter_id = existing[0][0]
            is_new_parameter = False
        else:
            param_rows = self._duck._fetchall(
                "SELECT COALESCE(MAX(parameter_id), 0) FROM _variables "
                "WHERE variable_name = ?",
                [table_name],
            )
            parameter_id = param_rows[0][0] + 1
            is_new_parameter = True

        # Get current max version_id per schema_id for this parameter
        all_schema_ids = set()
        for nested in all_nested:
            schema_keys = nested.get("schema", {})
            schema_level = self._infer_schema_level(schema_keys)
            if schema_level is not None and schema_keys:
                key_tuple = tuple(
                    _schema_str(schema_keys.get(k, "")) for k in self.dataset_schema_keys
                    if k in schema_keys
                )
                all_schema_ids.add(schema_id_cache[(schema_level, key_tuple)])
            else:
                all_schema_ids.add(0)

        # Batch query current max version_ids
        version_id_map = {}  # {schema_id: next_version_id}
        if all_schema_ids and not is_new_parameter:
            placeholders = ", ".join(["?"] * len(all_schema_ids))
            sid_list = list(all_schema_ids)
            version_rows = self._duck._fetchall(
                f"SELECT schema_id, COALESCE(MAX(version_id), 0) "
                f"FROM _record_metadata "
                f"WHERE variable_name = ? AND parameter_id = ? "
                f"AND schema_id IN ({placeholders}) "
                f"GROUP BY schema_id",
                [table_name, parameter_id] + sid_list,
            )
            for sid, max_vid in version_rows:
                version_id_map[sid] = max_vid + 1

        # Default version_id for schema_ids with no existing records
        for sid in all_schema_ids:
            if sid not in version_id_map:
                version_id_map[sid] = 1

        timings["4_parameter_resolution"] = time.perf_counter() - t3

        # --- Per-row Python computation (no SQL) ---
        t4 = time.perf_counter()
        created_at = datetime.now().isoformat()
        record_ids = []
        data_table_rows = []
        metadata_rows = []    # tuples for _record_metadata

        for i, (data_val, flat_meta) in enumerate(data_items):
            nested = all_nested[i]
            schema_keys = nested.get("schema", {})
            schema_level = self._infer_schema_level(schema_keys)

            if schema_level is not None and schema_keys:
                key_tuple = tuple(
                    _schema_str(schema_keys.get(k, "")) for k in self.dataset_schema_keys
                    if k in schema_keys
                )
                schema_id = schema_id_cache[(schema_level, key_tuple)]
            else:
                schema_id = 0

            version_id = version_id_map[schema_id]

            content_hash = canonical_hash(data_val)
            record_id = generate_record_id(
                class_name=type_name,
                schema_version=schema_version,
                content_hash=content_hash,
                metadata=nested,
            )
            record_ids.append(record_id)

            if is_dict_of_arrays:
                # Each dict-of-arrays item expands to N rows (one per array element)
                n_rows = len(next(iter(data_val.values())))
                for row_idx in range(n_rows):
                    row_data = [schema_id, parameter_id, version_id]
                    for col in dict_columns:
                        # Use .flat[] to handle both 1D and Nx1 column vectors
                        row_data.append(data_val[col].flat[row_idx])
                    data_table_rows.append(tuple(row_data))
            else:
                storage_value = _python_to_storage(data_val, col_meta)
                data_table_rows.append(
                    (schema_id, parameter_id, version_id, storage_value)
                )

            metadata_rows.append((
                record_id, table_name, parameter_id, version_id, schema_id,
                content_hash, None, schema_version, user_id, created_at,
            ))

        timings["5_per_row_hashing"] = time.perf_counter() - t4

        # --- Batch idempotency check ---
        t5 = time.perf_counter()
        existing_ids = set()
        # Query in chunks to avoid SQL parameter limits
        chunk_size = 500
        for start in range(0, len(record_ids), chunk_size):
            chunk = record_ids[start:start + chunk_size]
            placeholders = ", ".join(["?"] * len(chunk))
            rows = self._duck._fetchall(
                f"SELECT record_id FROM _record_metadata "
                f"WHERE record_id IN ({placeholders})",
                chunk,
            )
            existing_ids.update(r[0] for r in rows)

        # Filter out already-existing records
        new_data_rows = []
        new_meta_rows = []
        if is_dict_of_arrays:
            # For dict-of-arrays, each record_id maps to N data rows
            data_row_idx = 0
            for i, rid in enumerate(record_ids):
                data_val = data_items[i][0]
                n_rows = len(next(iter(data_val.values())))
                if rid not in existing_ids:
                    new_data_rows.extend(data_table_rows[data_row_idx:data_row_idx + n_rows])
                    new_meta_rows.append(metadata_rows[i])
                data_row_idx += n_rows
        else:
            for i, rid in enumerate(record_ids):
                if rid not in existing_ids:
                    new_data_rows.append(data_table_rows[i])
                    new_meta_rows.append(metadata_rows[i])

        timings["6_idempotency_check"] = time.perf_counter() - t5

        # --- Batch inserts ---
        t6 = time.perf_counter()
        if new_data_rows:
            self._duck._begin()
            try:
                if is_dict_of_arrays:
                    data_columns = ["schema_id", "parameter_id", "version_id"] + dict_columns
                    data_df = pd.DataFrame(new_data_rows, columns=data_columns)
                    col_str = ", ".join(f'"{c}"' for c in data_columns)
                    self._duck.con.execute(
                        f'INSERT INTO "{table_name}" ({col_str}) SELECT * FROM data_df'
                    )
                else:
                    data_df = pd.DataFrame(
                        new_data_rows,
                        columns=["schema_id", "parameter_id", "version_id", "value"],
                    )
                    self._duck.con.execute(
                        f'INSERT INTO "{table_name}" '
                        f'("schema_id", "parameter_id", "version_id", "value") '
                        f'SELECT * FROM data_df'
                    )

                meta_df = pd.DataFrame(
                    new_meta_rows,
                    columns=[
                        "record_id", "variable_name", "parameter_id",
                        "version_id", "schema_id", "content_hash",
                        "lineage_hash", "schema_version", "user_id", "created_at",
                    ],
                )
                self._duck.con.execute(
                    "INSERT INTO _record_metadata ("
                    "record_id, variable_name, parameter_id, version_id, schema_id, "
                    "content_hash, lineage_hash, schema_version, user_id, created_at"
                    ") SELECT * FROM meta_df"
                )

                self._duck._commit()
            except Exception:
                try:
                    self._duck._execute("ROLLBACK")
                except Exception:
                    pass
                raise

        # Register in _variables for new parameter slots
        if is_new_parameter:
            effective_level = (
                self._infer_schema_level(all_nested[0].get("schema", {}))
                or self.dataset_schema_keys[-1]
            )
            self._duck._execute(
                "INSERT INTO _variables (variable_name, parameter_id, schema_level, "
                "dtype, version_keys, description) VALUES (?, ?, ?, ?, ?, ?)",
                [table_name, parameter_id, effective_level,
                 json.dumps(dtype_meta),
                 vk_json, ""],
            )

        timings["7_batch_inserts"] = time.perf_counter() - t6
        timings["total"] = time.perf_counter() - t0

        if profile:
            print(f"\n--- save_batch() profile ({len(data_items)} items, "
                  f"{len(unique_schema_combos)} unique schemas) ---")
            for phase, elapsed in timings.items():
                print(f"  {phase:30s} {elapsed:8.3f}s")
            print()

        return record_ids

    @staticmethod
    def _build_bulk_where(keys: set[tuple[int, int, int]]) -> tuple[str, list]:
        """Build an efficient WHERE clause for bulk (pid, vid, sid) lookup.

        Uses per-dimension IN clauses when the key set is a full cross product,
        otherwise falls back to tuple-IN with VALUES.

        Returns (where_clause, params).
        """
        unique_pids = set(p for p, _, _ in keys)
        unique_vids = set(v for _, v, _ in keys)
        unique_sids = set(s for _, _, s in keys)
        cross_product_size = len(unique_pids) * len(unique_vids) * len(unique_sids)

        if cross_product_size == len(keys):
            pid_ph = ", ".join("?" for _ in unique_pids)
            vid_ph = ", ".join("?" for _ in unique_vids)
            sid_ph = ", ".join("?" for _ in unique_sids)
            where_clause = (
                f"parameter_id IN ({pid_ph}) "
                f"AND version_id IN ({vid_ph}) "
                f"AND schema_id IN ({sid_ph})"
            )
            params = list(unique_pids) + list(unique_vids) + list(unique_sids)
        else:
            values_list = ", ".join(f"({p}, {v}, {s})" for p, v, s in keys)
            where_clause = (
                f"(parameter_id, version_id, schema_id) IN "
                f"(VALUES {values_list})"
            )
            params = []
        return where_clause, params

    @staticmethod
    def _has_custom_serialization(variable_class: type) -> bool:
        """Check if a BaseVariable subclass overrides to_db or from_db."""
        return "to_db" in variable_class.__dict__ or "from_db" in variable_class.__dict__

    def _find_record(
        self,
        table_name: str,
        record_id: str | None = None,
        nested_metadata: dict | None = None,
        version_id: int | list[int] | str = "all",
    ) -> pd.DataFrame:
        """
        Query _record_metadata to find matching records.

        Supports two modes:
        - By record_id: direct primary key lookup (with JOINs for full row data)
        - By metadata: filter by schema keys via JOIN with _schema, optionally
          filter by version_keys JSON from _variables, order by created_at DESC

        version_id controls which versions are returned:
        - "all" (default): no version filtering (return every version)
        - "latest": only the row with max version_id per (parameter_id, schema_id)
        - int: only that specific version_id
        - list[int]: only those version_ids

        Schema key values and version key values may be lists, interpreted as
        "match any" (SQL IN / Python in).

        Returns a DataFrame of matching rows including schema columns and version_keys.
        """
        # Build schema column SELECT list
        schema_col_select = ", ".join(
            f's."{col}"' for col in self.dataset_schema_keys
        )

        if record_id is not None:
            sql = (
                f"SELECT rm.*, {schema_col_select}, v.version_keys "
                f"FROM _record_metadata rm "
                f"LEFT JOIN _schema s ON rm.schema_id = s.schema_id "
                f"LEFT JOIN _variables v ON rm.variable_name = v.variable_name "
                f"AND rm.parameter_id = v.parameter_id "
                f"WHERE rm.record_id = ? AND rm.variable_name = ?"
            )
            return self._duck._fetchdf(sql, [record_id, table_name])

        # By metadata
        schema_keys = nested_metadata.get("schema", {}) if nested_metadata else {}
        version_keys = nested_metadata.get("version", {}) if nested_metadata else {}

        conditions = ["rm.variable_name = ?"]
        params: list[Any] = [table_name]

        # Filter schema keys via _schema columns in SQL (lists → IN)
        for key, value in schema_keys.items():
            if isinstance(value, (list, tuple)):
                placeholders = ", ".join(["?"] * len(value))
                conditions.append(f's."{key}" IN ({placeholders})')
                params.extend([_schema_str(v) for v in value])
            else:
                conditions.append(f's."{key}" = ?')
                params.append(_schema_str(value))

        # Filter by specific version_id value(s) in SQL
        if isinstance(version_id, int):
            conditions.append("rm.version_id = ?")
            params.append(version_id)
        elif isinstance(version_id, (list, tuple)) and all(isinstance(v, int) for v in version_id):
            placeholders = ", ".join(["?"] * len(version_id))
            conditions.append(f"rm.version_id IN ({placeholders})")
            params.extend(version_id)

        where = " AND ".join(conditions)

        if version_id == "latest":
            sql = (
                f"WITH ranked AS ("
                f"SELECT rm.*, {schema_col_select}, v.version_keys, "
                f"ROW_NUMBER() OVER ("
                f"PARTITION BY rm.variable_name, rm.parameter_id, rm.schema_id "
                f"ORDER BY rm.version_id DESC"
                f") as rn "
                f"FROM _record_metadata rm "
                f"LEFT JOIN _schema s ON rm.schema_id = s.schema_id "
                f"LEFT JOIN _variables v ON rm.variable_name = v.variable_name "
                f"AND rm.parameter_id = v.parameter_id "
                f"WHERE {where}"
                f") SELECT * FROM ranked WHERE rn = 1 "
                f"ORDER BY created_at DESC"
            )
        else:
            sql = (
                f"SELECT rm.*, {schema_col_select}, v.version_keys "
                f"FROM _record_metadata rm "
                f"LEFT JOIN _schema s ON rm.schema_id = s.schema_id "
                f"LEFT JOIN _variables v ON rm.variable_name = v.variable_name "
                f"AND rm.parameter_id = v.parameter_id "
                f"WHERE {where} "
                f"ORDER BY rm.created_at DESC"
            )
        df = self._duck._fetchdf(sql, params)

        # Filter by version keys via Python-side JSON parsing (lists → in)
        if version_keys and len(df) > 0:
            for key, value in version_keys.items():
                if isinstance(value, (list, tuple)):
                    mask = df["version_keys"].apply(
                        lambda vk, k=key, vals=value: json.loads(vk).get(k) in vals
                        if vk is not None and isinstance(vk, str) else False
                    )
                else:
                    mask = df["version_keys"].apply(
                        lambda vk, k=key, v=value: json.loads(vk).get(k) == v
                        if vk is not None and isinstance(vk, str) else False
                    )
                df = df[mask]

        return df

    def _reconstruct_metadata_from_row(self, row: pd.Series) -> tuple[dict, dict]:
        """
        Reconstruct flat and nested metadata from a JOINed row.

        The row contains schema columns from _schema and version_keys from
        _variables, which together form the complete metadata.

        Returns (flat_metadata, nested_metadata).
        """
        schema = {}
        for key in self.dataset_schema_keys:
            if key in row.index:
                val = row[key]
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    schema[key] = _from_schema_str(val)

        vk_raw = row.get("version_keys")
        version = {}
        if vk_raw is not None and isinstance(vk_raw, str):
            version = json.loads(vk_raw)

        nested_metadata = {"schema": schema, "version": version}
        flat_metadata = {}
        flat_metadata.update(schema)
        flat_metadata.update(version)
        return flat_metadata, nested_metadata

    def _deserialize_custom_subdf(
        self,
        variable_class: type[BaseVariable],
        sub_df: pd.DataFrame,
        dtype_meta: dict,
    ):
        """Deserialize a sub-DataFrame using custom dtype metadata.

        Handles four sub-paths based on dtype_meta flags:
        - dict_of_arrays: reconstruct dict of numpy arrays
        - from_db: class-level custom deserialization
        - struct_columns: unflatten dot-separated columns
        - raw: return DataFrame as-is

        The sub_df must already have internal columns
        (schema_id, parameter_id, version_id) dropped.
        """
        if dtype_meta.get("dict_of_arrays"):
            ndarray_keys = dtype_meta.get("ndarray_keys", {})
            data = {}
            for col in sub_df.columns:
                arr = sub_df[col].values
                if col in ndarray_keys:
                    col_meta = ndarray_keys[col]
                    arr = arr.astype(np.dtype(col_meta["dtype"]))
                    orig_shape = col_meta.get("shape")
                    if orig_shape and len(orig_shape) == 2:
                        if orig_shape[0] == 1:
                            arr = arr.reshape(1, -1)
                        elif orig_shape[1] == 1:
                            arr = arr.reshape(-1, 1)
                        else:
                            try:
                                arr = arr.reshape(orig_shape)
                            except ValueError:
                                pass
                data[col] = arr
            return data
        elif self._has_custom_serialization(variable_class):
            return variable_class.from_db(sub_df)
        elif dtype_meta.get("struct_columns"):
            return _unflatten_struct_columns(sub_df, dtype_meta["struct_columns"])
        else:
            return sub_df

    def _load_by_record_row(
        self,
        variable_class: type[BaseVariable],
        row: pd.Series,
        loc: Any = None,
        iloc: Any = None,
    ) -> BaseVariable:
        """
        Load a variable instance given a row from _record_metadata.

        Determines native vs custom deserialization from _variables.dtype,
        loads data from SciDuck, and constructs the BaseVariable instance.
        """
        table_name = row["variable_name"]
        parameter_id = int(row["parameter_id"])
        version_id = int(row["version_id"])
        schema_id = int(row["schema_id"])
        record_id = row["record_id"]
        content_hash = row["content_hash"]
        lineage_hash = row["lineage_hash"]
        # Normalize NaN to None (DuckDB may return NaN for NULL in some contexts)
        if lineage_hash is not None and not isinstance(lineage_hash, str):
            lineage_hash = None
        flat_metadata, nested_metadata = self._reconstruct_metadata_from_row(row)

        # Get dtype from _variables to determine deserialization path
        dtype_rows = self._duck._fetchall(
            "SELECT dtype FROM _variables WHERE variable_name = ? AND parameter_id = ?",
            [table_name, parameter_id],
        )

        if not dtype_rows:
            raise NotFoundError(
                f"No parameter_id {parameter_id} found for {table_name} in _variables"
            )

        dtype_meta = json.loads(dtype_rows[0][0])
        is_custom = dtype_meta.get("custom", False)

        if is_custom:
            # Custom path: direct query by parameter_id, version_id, and schema_id
            df = self._duck._fetchdf(
                f'SELECT * FROM "{table_name}" WHERE parameter_id = ? AND version_id = ? AND schema_id = ?',
                [parameter_id, version_id, schema_id],
            )
            # Drop SciDuck internal columns
            df = df.drop(columns=["schema_id", "parameter_id", "version_id"], errors="ignore")

            if loc is not None:
                if not isinstance(loc, (list, range, slice)):
                    loc = [loc]
                df = df.loc[loc]
            elif iloc is not None:
                if not isinstance(iloc, (list, range, slice)):
                    iloc = [iloc]
                df = df.iloc[iloc]

            data = self._deserialize_custom_subdf(variable_class, df, dtype_meta)
        else:
            # Native path: use SciDuck.load() with parameter_id, version_id and raw=True
            schema_keys = nested_metadata.get("schema", {})
            schema_level = self._infer_schema_level(schema_keys)
            if schema_level is not None and schema_keys:
                data = self._duck.load(
                    table_name, parameter_id=parameter_id, version_id=version_id, raw=True,
                    **{k: _schema_str(v) for k, v in schema_keys.items()},
                )
            else:
                # Non-contiguous or no schema — load directly
                data = self._duck.load(
                    table_name, parameter_id=parameter_id, version_id=version_id, raw=True,
                )

        instance = variable_class(data)
        instance.record_id = record_id
        instance.metadata = flat_metadata
        instance.content_hash = content_hash
        instance.lineage_hash = lineage_hash
        instance.version_id = version_id
        instance.parameter_id = parameter_id

        return instance

    def register(self, variable_class: Type[BaseVariable]) -> None:
        """
        Register a variable type for storage.

        Args:
            variable_class: The BaseVariable subclass to register
        """
        type_name = variable_class.__name__
        table_name = variable_class.table_name()
        schema_version = variable_class.schema_version

        # Register in metadata table (skip if already registered)
        existing = self._duck._fetchall(
            "SELECT 1 FROM _registered_types WHERE type_name = ?",
            [type_name],
        )
        if not existing:
            self._duck._execute(
                """
                INSERT INTO _registered_types (type_name, table_name, schema_version)
                VALUES (?, ?, ?)
                """,
                [type_name, table_name, schema_version],
            )

        # Cache locally
        self._registered_types[type_name] = variable_class

    def _ensure_registered(
        self, variable_class: Type[BaseVariable], auto_register: bool = True
    ) -> str:
        """
        Ensure a variable type is registered.

        Returns:
            The table name for this variable type
        """
        type_name = variable_class.__name__

        if type_name in self._registered_types:
            return variable_class.table_name()

        # Check database
        rows = self._duck._fetchall(
            "SELECT table_name FROM _registered_types WHERE type_name = ?",
            [type_name],
        )

        if not rows:
            if auto_register:
                self.register(variable_class)
                return variable_class.table_name()
            else:
                raise NotRegisteredError(
                    f"Variable type '{type_name}' is not registered. "
                    f"No data has been saved for this type yet."
                )

        self._registered_types[type_name] = variable_class
        return rows[0][0]

    def save_variable(
        self,
        variable_class: Type[BaseVariable],
        data: Any,
        index: Any = None,
        **metadata,
    ) -> str:
        """
        Save data as a variable, handling input normalization and lineage.

        Accepts ThunkOutput (from thunked computation), an existing BaseVariable
        instance, or raw data. Lineage is automatically extracted and stored
        when applicable.

        Args:
            variable_class: The BaseVariable subclass to save as
            data: The data to save (ThunkOutput, BaseVariable, or raw data)
            index: Optional index to set on the DataFrame
            **metadata: Addressing metadata (e.g., subject=1, trial=1)

        Returns:
            The record_id of the saved data
        """
        from .thunk import ThunkOutput
        from .lineage import extract_lineage, find_unsaved_variables, get_raw_value
        from .exceptions import UnsavedIntermediateError

        _t_total = time.perf_counter()
        _t0 = time.perf_counter()

        print(f"  [save_variable] ENTRY: variable_class={variable_class.__name__}, "
              f"data type={type(data).__name__}")
        if isinstance(data, pd.DataFrame):
            print(f"  [save_variable] DataFrame columns={data.columns.tolist()}, "
                  f"dtypes={dict(data.dtypes)}, shape={data.shape}")
            for col in data.columns:
                if data[col].dtype == object:
                    first_val = data[col].dropna().iloc[0] if len(data[col].dropna()) > 0 else None
                    print(f"  [save_variable] object column {col!r}: "
                          f"first_val type={type(first_val).__name__}")
                    if isinstance(first_val, dict):
                        print(f"  [save_variable]   dict keys={list(first_val.keys())}")

        # Normalize input: extract raw data and lineage info based on input type
        lineage = None
        lineage_hash = None
        pipeline_lineage_hash = None
        raw_data = None

        if isinstance(data, ThunkOutput):
            # Lineage-only save for side-effect functions (generates_file=True)
            if data.pipeline_thunk.thunk.generates_file:
                lineage = extract_lineage(data)
                pipeline_lineage_hash = data.pipeline_thunk.compute_lineage_hash()
                generated_id = f"generated:{pipeline_lineage_hash[:32]}"
                user_id = get_user_id()
                nested_metadata = self._split_metadata(metadata)
                self._save_lineage(
                    output_record_id=generated_id,
                    output_type=variable_class.__name__,
                    lineage=lineage,
                    lineage_hash=pipeline_lineage_hash,
                    user_id=user_id,
                    schema_keys=nested_metadata.get("schema"),
                    output_content_hash=None,
                )
                return generated_id

            # Data from a thunk computation
            unsaved = find_unsaved_variables(data)

            if self.lineage_mode == "strict" and unsaved:
                var_descriptions = []
                for var, path in unsaved:
                    var_type = type(var).__name__
                    var_descriptions.append(f"  - {var_type} (path: {path})")
                vars_str = "\n".join(var_descriptions)
                raise UnsavedIntermediateError(
                    f"Strict lineage mode requires all intermediate variables to be saved.\n"
                    f"Found {len(unsaved)} unsaved variable(s) in the computation chain:\n"
                    f"{vars_str}\n\n"
                    f"Either save these variables first, or use lineage_mode='ephemeral' "
                    f"in configure_database() to allow unsaved intermediates."
                )

            elif self.lineage_mode == "ephemeral" and unsaved:
                user_id = get_user_id()
                schema_keys = self._split_metadata(metadata).get("schema")
                for var, path in unsaved:
                    inner_data = getattr(var, "data", None)
                    if isinstance(inner_data, ThunkOutput):
                        ephemeral_id = f"ephemeral:{inner_data.hash[:32]}"
                        var_type = type(var).__name__
                        intermediate_lineage = extract_lineage(inner_data)
                        self.save_ephemeral_lineage(
                            ephemeral_id=ephemeral_id,
                            variable_type=var_type,
                            lineage=intermediate_lineage,
                            user_id=user_id,
                            schema_keys=schema_keys,
                        )

            lineage = extract_lineage(data)
            lineage_hash = data.hash
            pipeline_lineage_hash = data.pipeline_thunk.compute_lineage_hash()
            raw_data = get_raw_value(data)

        elif isinstance(data, BaseVariable):
            raw_data = data.data
            lineage_hash = data.lineage_hash

        else:
            raw_data = data

        _t_normalize = time.perf_counter() - _t0
        print(f"  [save_variable] input normalization: {_t_normalize:.4f}s")

        _t0 = time.perf_counter()
        instance = variable_class(raw_data)
        _t_instantiate = time.perf_counter() - _t0
        print(f"  [save_variable] variable instantiation: {_t_instantiate:.4f}s")

        _t0 = time.perf_counter()
        record_id = self.save(
            instance, metadata, lineage=lineage, lineage_hash=lineage_hash,
            pipeline_lineage_hash=pipeline_lineage_hash, index=index,
        )
        _t_save = time.perf_counter() - _t0
        print(f"  [save_variable] self.save() call: {_t_save:.4f}s")

        instance.record_id = record_id
        instance.metadata = metadata
        instance.lineage_hash = lineage_hash

        _t_total_elapsed = time.perf_counter() - _t_total
        print(f"  [save_variable] TOTAL: {_t_total_elapsed:.4f}s")

        return record_id

    def save(
        self,
        variable: BaseVariable,
        metadata: dict,
        lineage: "LineageRecord | None" = None,
        lineage_hash: str | None = None,
        pipeline_lineage_hash: str | None = None,
        index: Any = None,
    ) -> str:
        """
        Save a variable to the database.

        Args:
            variable: The variable instance to save
            metadata: Addressing metadata (flat dict)
            lineage: Optional lineage record if data came from a thunk
            lineage_hash: Optional pre-computed lineage hash (stored in DuckDB
                for input classification when this variable is reused later)
            pipeline_lineage_hash: Optional lineage hash for PipelineDB cache
                lookup. If None, falls back to lineage_hash.
            index: Optional index to set on the DataFrame

        Returns:
            The record_id of the saved data
        """
        _t0 = time.perf_counter()

        table_name = self._ensure_registered(type(variable))
        type_name = variable.__class__.__name__
        user_id = get_user_id()

        # Split metadata
        nested_metadata = self._split_metadata(metadata)

        # Warn if no metadata keys match the schema
        if metadata and not nested_metadata.get("schema"):
            warnings.warn(
                f"None of the metadata keys {list(metadata.keys())} match the "
                f"configured dataset_schema_keys {self.dataset_schema_keys}. "
                f"All keys will be treated as version parameters.",
                UserWarning,
                stacklevel=2,
            )

        _t_setup = time.perf_counter() - _t0
        print(f"    [save] setup (register/metadata/user_id): {_t_setup:.4f}s")

        # Normalize array.array values to numpy arrays (MATLAB bridge can produce these)
        import array as _array_mod
        if isinstance(variable.data, dict):
            for k, v in variable.data.items():
                if isinstance(v, _array_mod.array):
                    print(f"    [save] converting array.array key={k!r} (typecode={v.typecode}, len={len(v)}) -> np.ndarray")
                    variable.data[k] = np.array(v)

        # Compute content hash
        _t0 = time.perf_counter()
        content_hash = canonical_hash(variable.data)
        _t_hash = time.perf_counter() - _t0
        print(f"    [save] canonical_hash: {_t_hash:.4f}s")

        # Generate record_id
        _t0 = time.perf_counter()
        record_id = generate_record_id(
            class_name=type_name,
            schema_version=variable.schema_version,
            content_hash=content_hash,
            metadata=nested_metadata,
        )
        _t_genid = time.perf_counter() - _t0
        print(f"    [save] generate_record_id: {_t_genid:.4f}s")

        # Idempotency: check if record already exists in _record_metadata
        _t0 = time.perf_counter()
        existing = self._duck._fetchall(
            "SELECT 1 FROM _record_metadata WHERE record_id = ?",
            [record_id],
        )
        _t_idempotency = time.perf_counter() - _t0
        print(f"    [save] idempotency check: {_t_idempotency:.4f}s")
        if existing:
            print(f"    [save] record already exists, returning early")
            return record_id

        # Wrap all writes in a single transaction to avoid repeated
        # WAL checkpoints (each auto-committed statement can trigger a
        # checkpoint/fsync, causing random multi-second stalls).
        self._duck._begin()

        try:
            schema_keys = nested_metadata.get("schema", {})
            version_keys = nested_metadata.get("version", {})
            schema_level = self._infer_schema_level(schema_keys)
            created_at = datetime.now().isoformat()

            _t0 = time.perf_counter()
            data_type = type(variable.data).__name__
            print(f"    [save] ENTRY: data_type={data_type}, "
                  f"is_DataFrame={isinstance(variable.data, pd.DataFrame)}, "
                  f"is_dict={isinstance(variable.data, dict)}, "
                  f"has_custom_ser={self._has_custom_serialization(type(variable))}")
            if isinstance(variable.data, pd.DataFrame):
                print(f"    [save] DataFrame cols={variable.data.columns.tolist()}, "
                      f"shape={variable.data.shape}")
                for col in variable.data.columns:
                    if variable.data[col].dtype == object:
                        nonnull = variable.data[col].dropna()
                        if len(nonnull) > 0:
                            fv = nonnull.iloc[0]
                            print(f"    [save] object col {col!r}: "
                                  f"first_val type={type(fv).__name__}, "
                                  f"is_dict={isinstance(fv, dict)}")
                            if isinstance(fv, dict):
                                print(f"    [save]   keys={list(fv.keys())[:10]}")
            elif isinstance(variable.data, dict):
                print(f"    [save] dict keys={list(variable.data.keys())[:10]}")
                for k, v in list(variable.data.items())[:5]:
                    vtype = type(v).__name__
                    extra = ""
                    if isinstance(v, np.ndarray):
                        extra = f" shape={v.shape} dtype={v.dtype}"
                    elif isinstance(v, dict):
                        extra = f" keys={list(v.keys())[:5]}"
                    print(f"    [save]   key={k!r}: {vtype}{extra}")

            if self._has_custom_serialization(type(variable)):
                # Custom serialization: user provides to_db() → DataFrame
                print(f"    [save] path=custom_serialization")
                df = variable.to_db()

                if index is not None:
                    index_list = list(index) if not isinstance(index, list) else index
                    if len(index_list) != len(df):
                        raise ValueError(
                            f"Index length ({len(index_list)}) does not match "
                            f"DataFrame row count ({len(df)})"
                        )
                    df.index = index

                parameter_id, schema_id, version_id = self._save_columnar(
                    table_name, type(variable), df, schema_level, schema_keys, content_hash,
                    version_keys=version_keys,
                )
            elif isinstance(variable.data, pd.DataFrame):
                # Native DataFrame: store directly as multi-column DuckDB table
                print(f"    [save] path=native_dataframe")
                df = variable.data.copy()

                if index is not None:
                    index_list = list(index) if not isinstance(index, list) else index
                    if len(index_list) != len(df):
                        raise ValueError(
                            f"Index length ({len(index_list)}) does not match "
                            f"DataFrame row count ({len(df)})"
                        )
                    df.index = index

                # Flatten nested struct columns (e.g., MATLAB table with struct cells)
                print(f"    [save] calling _flatten_struct_columns on DataFrame "
                      f"cols={df.columns.tolist()}, dtypes={dict(df.dtypes)}")
                df, struct_columns_info = _flatten_struct_columns(df)
                print(f"    [save] after flatten: cols={df.columns.tolist()}, "
                      f"struct_columns_info keys={list(struct_columns_info.keys()) if struct_columns_info else 'none'}")
                if struct_columns_info:
                    print(f"    [save] flattened dtypes={dict(df.dtypes)}")

                parameter_id, schema_id, version_id = self._save_columnar(
                    table_name, type(variable), df, schema_level, schema_keys, content_hash,
                    version_keys=version_keys,
                    struct_columns=struct_columns_info if struct_columns_info else None,
                )
            elif _is_tabular_dict(variable.data):
                # Dict-of-arrays: convert to DataFrame for efficient columnar storage
                n_keys = len(variable.data)
                first_len = len(next(iter(variable.data.values())))
                print(f"    [save] path=dict_of_arrays ({n_keys} fields, {first_len} rows each, data_type={data_type})")
                for k, v in variable.data.items():
                    print(f"      [dict_of_arrays save] key={k!r}: dtype={v.dtype}, shape={v.shape}, ndim={v.ndim}")
                _t_df = time.perf_counter()
                # Squeeze Nx1 column vectors to 1D so DataFrame gets one column per key
                squeezed = {
                    k: v.squeeze() if v.ndim == 2 else v
                    for k, v in variable.data.items()
                }
                df = pd.DataFrame(squeezed)
                print(f"    [save] pd.DataFrame(dict): {time.perf_counter() - _t_df:.4f}s  shape={df.shape}")
                parameter_id, schema_id, version_id = self._save_columnar(
                    table_name, type(variable), df, schema_level, schema_keys, content_hash,
                    version_keys=version_keys,
                    dict_of_arrays=True,
                    ndarray_keys={
                        k: {"dtype": str(v.dtype), "shape": list(v.shape)}
                        for k, v in variable.data.items()
                    },
                )
            else:
                # Native single-value storage (scalars, arrays, lists, dicts)
                print(f"    [save] path=single_value (data_type={data_type})")
                parameter_id, schema_id, version_id = self._save_single_value(
                    table_name, type(variable), variable.data, content_hash,
                    schema_level=schema_level, schema_keys=schema_keys,
                    version_keys=version_keys,
                )
            _t_data_save = time.perf_counter() - _t0
            print(f"    [save] data write (DuckDB): {_t_data_save:.4f}s")

            _t0 = time.perf_counter()
            self._save_record_metadata(
                record_id=record_id,
                variable_name=table_name,
                parameter_id=parameter_id,
                version_id=version_id,
                schema_id=schema_id,
                content_hash=content_hash,
                lineage_hash=lineage_hash,
                schema_version=variable.schema_version,
                user_id=user_id,
                created_at=created_at,
            )
            _t_record_meta = time.perf_counter() - _t0
            print(f"    [save] _save_record_metadata: {_t_record_meta:.4f}s")

            # Save lineage if provided
            _t0 = time.perf_counter()
            if lineage is not None:
                effective_plh = pipeline_lineage_hash if pipeline_lineage_hash is not None else lineage_hash
                self._save_lineage(
                    record_id, type_name, lineage, effective_plh, user_id,
                    schema_keys=nested_metadata.get("schema"),
                    output_content_hash=content_hash,
                )
            _t_lineage = time.perf_counter() - _t0
            print(f"    [save] _save_lineage: {_t_lineage:.4f}s")

            _t0 = time.perf_counter()
            self._duck._commit()
            _t_commit = time.perf_counter() - _t0
            print(f"    [save] transaction commit: {_t_commit:.4f}s")

        except Exception:
            try:
                self._duck._rollback()
            except Exception:
                pass  # Connection may already be closed
            raise

        return record_id

    def _save_lineage(
        self,
        output_record_id: str,
        output_type: str,
        lineage: "LineageRecord",
        lineage_hash: str | None = None,
        user_id: str | None = None,
        schema_keys: dict | None = None,
        output_content_hash: str | None = None,
    ) -> None:
        """Save lineage record for a variable to PipelineDB (SQLite)."""
        self._pipeline_db.save_lineage(
            output_record_id=output_record_id,
            output_type=output_type,
            function_name=lineage.function_name,
            function_hash=lineage.function_hash,
            inputs=lineage.inputs,
            constants=lineage.constants,
            lineage_hash=lineage_hash,
            user_id=user_id,
            schema_keys=schema_keys,
            output_content_hash=output_content_hash,
        )

    def save_ephemeral_lineage(
        self,
        ephemeral_id: str,
        variable_type: str,
        lineage: "LineageRecord",
        user_id: str | None = None,
        schema_keys: dict | None = None,
    ) -> None:
        """Save an ephemeral lineage record for an unsaved intermediate variable."""
        self._pipeline_db.save_ephemeral(
            ephemeral_id=ephemeral_id,
            variable_type=variable_type,
            function_name=lineage.function_name,
            function_hash=lineage.function_hash,
            inputs=lineage.inputs,
            constants=lineage.constants,
            user_id=user_id,
            schema_keys=schema_keys,
        )

    def load(
        self,
        variable_class: Type[BaseVariable],
        metadata: dict,
        version: str = "latest",
        loc: Any = None,
        iloc: Any = None,
        where=None,
    ) -> BaseVariable:
        """
        Load a single variable matching the given metadata.

        Args:
            variable_class: The type to load
            metadata: Flat metadata dict
            version: "latest" for most recent, or specific record_id
            loc: Optional label-based index selection
            iloc: Optional integer position-based index selection

        Returns:
            The matching variable instance
        """
        table_name = self._ensure_registered(variable_class, auto_register=True)

        try:
            if version != "latest" and version is not None:
                # Load by specific record_id
                records = self._find_record(table_name, record_id=version)
                if len(records) == 0:
                    raise NotFoundError(f"No data found with record_id '{version}'")
            else:
                # Load by metadata (latest version per parameter set)
                nested_metadata = self._split_metadata(metadata)
                records = self._find_record(table_name, nested_metadata=nested_metadata, version_id="latest")
                if len(records) == 0:
                    raise NotFoundError(
                        f"No {variable_class.__name__} found matching metadata: {metadata}"
                    )

            # Apply where= filter if provided
            if where is not None:
                allowed_schema_ids = where.resolve(self, variable_class, table_name)
                records = records[records["schema_id"].isin(allowed_schema_ids)]
                if len(records) == 0:
                    raise NotFoundError(
                        f"No {variable_class.__name__} found matching metadata: {metadata} "
                        f"with the given where= filter."
                    )

            # Take the first (latest) record
            row = records.iloc[0]
        except NotFoundError:
            raise
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                raise NotFoundError(
                    f"No {variable_class.__name__} found matching metadata: {metadata}"
                )
            raise

        return self._load_by_record_row(variable_class, row, loc=loc, iloc=iloc)

    def load_all(
        self,
        variable_class: Type[BaseVariable],
        metadata: dict,
        version_id="all",
        where=None,
    ):
        """
        Load all variables matching the given metadata as a generator.

        Args:
            variable_class: The type to load
            metadata: Flat metadata dict
            version_id: Which versions to return:
                - "all" (default): return every version
                - "latest": return only the latest version per parameter set
                - int: return only that specific version_id
                - list[int]: return only those version_ids

        Yields:
            BaseVariable instances matching the metadata
        """
        table_name = self._ensure_registered(variable_class, auto_register=True)
        nested_metadata = self._split_metadata(metadata)

        try:
            records = self._find_record(table_name, nested_metadata=nested_metadata, version_id=version_id)
        except Exception:
            return  # No data

        if len(records) == 0:
            return

        # Apply where= filter if provided
        if where is not None:
            allowed_schema_ids = where.resolve(self, variable_class, table_name)
            records = records[records["schema_id"].isin(allowed_schema_ids)]
            if len(records) == 0:
                return

        # --- Bulk loading path ---

        # 1. Bulk dtype lookup: one query per unique parameter_id
        unique_pids = records["parameter_id"].unique()
        dtype_by_pid = {}
        for pid in unique_pids:
            pid = int(pid)
            dtype_rows = self._duck._fetchall(
                "SELECT dtype FROM _variables WHERE variable_name = ? AND parameter_id = ?",
                [table_name, pid],
            )
            if dtype_rows:
                dtype_by_pid[pid] = json.loads(dtype_rows[0][0])

        if not dtype_by_pid:
            return

        # Partition parameter_ids into custom vs native
        custom_pids = {pid for pid, dm in dtype_by_pid.items() if dm.get("custom", False)}
        native_pids = {pid for pid, dm in dtype_by_pid.items() if not dm.get("custom", False)}

        # Vectorized key extraction
        pids_arr = records["parameter_id"].values.astype(int)
        vids_arr = records["version_id"].values.astype(int)
        sids_arr = records["schema_id"].values.astype(int)
        all_keys = set(zip(pids_arr.tolist(), vids_arr.tolist(), sids_arr.tolist()))
        if not all_keys:
            return

        data_lookup = {}  # (pid, vid, sid) -> (value, dtype_meta)  for native
        custom_data_lookup = {}  # (pid, vid, sid) -> value  for custom

        # 2a. Bulk data fetch for CUSTOM pids
        if custom_pids:
            custom_keys = {k for k in all_keys if k[0] in custom_pids}
            where_clause, params = self._build_bulk_where(custom_keys)

            sql = f'SELECT * FROM "{table_name}" WHERE {where_clause}'
            all_custom_df = self._duck._fetchdf(sql, params)

            # Group by (pid, vid, sid) and deserialize per-record
            if len(all_custom_df) > 0:
                grouped = all_custom_df.groupby(
                    ["parameter_id", "version_id", "schema_id"],
                    sort=False,
                )
                for (pid, vid, sid), sub_df in grouped:
                    pid, vid, sid = int(pid), int(vid), int(sid)
                    sub_df = sub_df.drop(
                        columns=["schema_id", "parameter_id", "version_id"],
                        errors="ignore",
                    ).reset_index(drop=True)
                    dtype_meta = dtype_by_pid[pid]
                    custom_data_lookup[(pid, vid, sid)] = self._deserialize_custom_subdf(
                        variable_class, sub_df, dtype_meta,
                    )

        # 2b. Bulk data fetch for NATIVE pids (existing optimized path)
        if native_pids:
            native_keys = {k for k in all_keys if k[0] in native_pids}

            first_native_dtype = dtype_by_pid[next(iter(native_pids))]
            data_cols = list(first_native_dtype["columns"].keys())
            data_select = ", ".join(f'"{c}"' for c in data_cols)

            where_clause, params = self._build_bulk_where(native_keys)
            sql = (
                f'SELECT parameter_id, version_id, schema_id, {data_select} '
                f'FROM "{table_name}" '
                f'WHERE {where_clause}'
            )
            all_data_df = self._duck._fetchdf(sql, params)

            # Restore types per parameter_id group, then build lookup
            for pid in native_pids:
                pid_mask = all_data_df["parameter_id"] == pid
                pid_df = all_data_df[pid_mask]
                if len(pid_df) == 0:
                    continue
                dtype_meta = dtype_by_pid[pid]
                restored = pid_df[data_cols].copy()
                restored = self._duck._restore_types(restored, dtype_meta)

                pid_keys_p = pid_df["parameter_id"].values.astype(int)
                pid_keys_v = pid_df["version_id"].values.astype(int)
                pid_keys_s = pid_df["schema_id"].values.astype(int)

                mode = dtype_meta.get("mode", "single_column")
                columns_meta = dtype_meta.get("columns", {})

                if mode == "single_column":
                    col_name = next(iter(columns_meta))
                    col_values = restored[col_name].tolist()
                    for i in range(len(col_values)):
                        key = (int(pid_keys_p[i]), int(pid_keys_v[i]), int(pid_keys_s[i]))
                        data_lookup[key] = (col_values[i], dtype_meta)
                else:
                    col_lists = {c: restored[c].tolist() for c in columns_meta}
                    for i in range(len(pid_df)):
                        key = (int(pid_keys_p[i]), int(pid_keys_v[i]), int(pid_keys_s[i]))
                        data_lookup[key] = ({c: col_lists[c][i] for c in columns_meta}, dtype_meta)

        # 3. Construct instances using itertuples + inline metadata
        _count = 0
        schema_keys = self.dataset_schema_keys
        for row in records.itertuples(index=False):
            pid = int(row.parameter_id)
            vid = int(row.version_id)
            sid = int(row.schema_id)
            key = (pid, vid, sid)

            if key in custom_data_lookup:
                data_value = custom_data_lookup[key]
            elif key in data_lookup:
                data_value, _ = data_lookup[key]
            else:
                continue

            record_id = row.record_id
            content_hash = row.content_hash
            lineage_hash = row.lineage_hash
            if lineage_hash is not None and not isinstance(lineage_hash, str):
                lineage_hash = None

            flat_metadata = {}
            for sk in schema_keys:
                val = getattr(row, sk, None)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    flat_metadata[sk] = _from_schema_str(val)
            vk_raw = getattr(row, 'version_keys', None)
            if vk_raw is not None and isinstance(vk_raw, str):
                flat_metadata.update(json.loads(vk_raw))

            instance = variable_class(data_value)
            instance.record_id = record_id
            instance.metadata = flat_metadata
            instance.content_hash = content_hash
            instance.lineage_hash = lineage_hash
            instance.version_id = vid
            instance.parameter_id = pid
            _count += 1

            yield instance

    def list_versions(
        self,
        variable_class: Type[BaseVariable],
        **metadata,
    ) -> list[dict]:
        """
        List all versions at a schema location.

        Args:
            variable_class: The type to query
            **metadata: Schema metadata to match

        Returns:
            List of dicts with record_id, schema, version, created_at
        """
        table_name = self._ensure_registered(variable_class, auto_register=True)
        nested_metadata = self._split_metadata(metadata)

        try:
            records = self._find_record(table_name, nested_metadata=nested_metadata, version_id="all")
        except Exception:
            return []

        results = []
        for _, row in records.iterrows():
            _, nested = self._reconstruct_metadata_from_row(row)
            results.append({
                "record_id": row["record_id"],
                "schema": nested.get("schema", {}),
                "version": nested.get("version", {}),
                "created_at": row["created_at"],
            })

        # Sort by created_at descending
        results.sort(key=lambda x: x["created_at"], reverse=True)
        return results

    def get_provenance(
        self,
        variable_class: Type[BaseVariable],
        version: str | None = None,
        **metadata,
    ) -> dict | None:
        """
        Get the provenance (lineage) of a variable.

        Returns:
            Dict with function_name, function_hash, inputs, constants
            or None if no lineage recorded
        """
        if version:
            record_id = version
        else:
            var = self.load(variable_class, metadata)
            record_id = var.record_id

        lineage = self._pipeline_db.get_lineage(record_id)
        if not lineage:
            return None

        return {
            "function_name": lineage["function_name"],
            "function_hash": lineage["function_hash"],
            "inputs": lineage["inputs"],
            "constants": lineage["constants"],
        }

    def get_provenance_by_schema(self, **schema_keys) -> list[dict]:
        """
        Get all provenance records at a schema location (schema-aware view).

        Args:
            **schema_keys: Schema key filters (e.g., subject="S01", session="1")

        Returns:
            List of lineage record dicts matching the schema keys
        """
        return self._pipeline_db.find_by_schema(**schema_keys)

    def get_pipeline_structure(self) -> list[dict]:
        """
        Get the abstract pipeline structure (schema-blind view).

        Returns unique (function_name, function_hash, output_type, input_types)
        combinations, describing how variable types flow through functions
        without reference to specific data instances or schema locations.

        Returns:
            List of dicts with keys: function_name, function_hash, output_type,
            input_types (list of type names)
        """
        return self._pipeline_db.get_pipeline_structure()

    def has_lineage(self, record_id: str) -> bool:
        """Check if a variable has lineage information."""
        return self._pipeline_db.has_lineage(record_id)

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    def export_to_csv(
        self,
        variable_class: Type[BaseVariable],
        path: str,
        **metadata,
    ) -> int:
        """Export matching variables to a CSV file."""
        results = list(self.load_all(variable_class, metadata))

        if not results:
            raise NotFoundError(
                f"No {variable_class.__name__} found matching metadata: {metadata}"
            )

        all_dfs = []
        for var in results:
            df = variable_class(var.data).to_db()
            df["_record_id"] = var.record_id
            if var.metadata:
                for key, value in var.metadata.items():
                    df[f"_meta_{key}"] = value
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(path, index=False)

        return len(results)

    def find_by_lineage(self, pipeline_thunk) -> list | None:
        """
        Find output values by computation lineage.

        Given a PipelineThunk (function + inputs), finds any previously
        computed outputs that match by:
        1. Querying PipelineDB (SQLite) for matching lineage_hash
        2. Loading data from SciDuck (DuckDB) using the record_ids

        Args:
            pipeline_thunk: The computation to look up

        Returns:
            List of output values if found, None otherwise
        """
        lineage_hash = pipeline_thunk.compute_lineage_hash()

        # Query PipelineDB (SQLite) for matching lineage
        records = self._pipeline_db.find_by_lineage_hash(lineage_hash)
        if not records:
            return None

        results = []
        has_generated = False
        for record in records:
            output_record_id = record["output_record_id"]
            output_type = record["output_type"]

            # Skip ephemeral entries (no data stored in SciDuck)
            if output_record_id.startswith("ephemeral:"):
                continue

            # Track generated entries (lineage-only, no data stored)
            if output_record_id.startswith("generated:"):
                has_generated = True
                continue

            var_class = self._get_variable_class(output_type)
            if var_class is None:
                return None

            try:
                # Load data from SciDuck (DuckDB)
                var = self.load(var_class, {}, version=output_record_id)
                results.append(var.data)
            except (KeyError, NotFoundError):
                # Record not found in SciDuck
                return None

        if results:
            return results
        if has_generated:
            return [None]
        return None

    def _get_variable_class(self, type_name: str):
        """Get a variable class by name."""
        if type_name in self._registered_types:
            return self._registered_types[type_name]

        return BaseVariable.get_subclass_by_name(type_name)

    def distinct_schema_values(self, key: str) -> list:
        """Return all distinct values stored for a schema key.

        Args:
            key: A schema key name (e.g. "subject", "session")

        Returns:
            Sorted list of distinct non-null values for that key
        """
        return self._duck.distinct_schema_values(key)

    # -------------------------------------------------------------------------
    # Variable Groups
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_var_name(v) -> str:
        """Resolve a single variable to its name string.

        Accepts a Python str, a BaseVariable subclass (class object),
        or a MATLAB BaseVariable instance (matlab.object with class name).
        """
        if isinstance(v, str):
            return v
        if isinstance(v, type) and issubclass(v, BaseVariable):
            return v.table_name()
        # MATLAB objects cross the bridge as matlab.object; try str()
        # to extract the class name (e.g. "StepLength").
        s = str(v)
        if s:
            return s
        raise TypeError(
            f"Expected a string or BaseVariable subclass, got {type(v)}"
        )

    @staticmethod
    def _resolve_var_names(variables) -> list:
        """Resolve a single or list/iterable of variables to name strings.

        Each element can be a string, a BaseVariable subclass, or a MATLAB
        object.  Accepts Python lists, MATLAB cell arrays, and MATLAB string
        arrays (any iterable).
        """
        # Scalar: single string or single class
        if isinstance(variables, (str, type)):
            return [DatabaseManager._resolve_var_name(variables)]
        # Any iterable (Python list, MATLAB cell array, MATLAB string array)
        try:
            return [DatabaseManager._resolve_var_name(v) for v in variables]
        except TypeError:
            # Not iterable — treat as a single item
            return [DatabaseManager._resolve_var_name(variables)]

    def add_to_var_group(self, group_name: str, variables):
        """Add one or more variables to a variable group.

        Args:
            group_name: Name of the group.
            variables: A BaseVariable subclass, a variable name string,
                or a list of either.
        """
        self._duck.add_to_group(group_name, self._resolve_var_names(variables))

    def remove_from_var_group(self, group_name: str, variables):
        """Remove one or more variables from a variable group.

        Args:
            group_name: Name of the group.
            variables: A BaseVariable subclass, a variable name string,
                or a list of either.
        """
        self._duck.remove_from_group(group_name, self._resolve_var_names(variables))

    def list_var_groups(self) -> list:
        """List all variable group names.

        Returns:
            Sorted list of distinct group names.
        """
        return self._duck.list_groups()

    def get_var_group(self, group_name: str) -> list:
        """Get all variable classes in a variable group.

        Args:
            group_name: Name of the group.

        Returns:
            Sorted list of BaseVariable subclasses in the group.
        """
        names = self._duck.get_group(group_name)
        classes = []
        for name in names:
            cls = BaseVariable.get_subclass_by_name(name)
            if cls is None:
                raise NotRegisteredError(
                    f"Variable '{name}' in group '{group_name}' has no "
                    f"registered BaseVariable subclass."
                )
            classes.append(cls)
        return classes

    def close(self):
        """Close the database connections and reset Thunk.query if it points to self."""
        from .thunk import Thunk
        if getattr(Thunk, "query", None) is self:
            Thunk.query = None
        self._duck.close()
        self._pipeline_db.close()
        # remove global reference
        if getattr(_local, "database", None) is self:
            self._closed = True

    def reopen(self):
        # reopen DuckDB
        if self._duck is None:
            self._duck = SciDuck(self.dataset_db_path, dataset_schema=self.dataset_schema_keys)
        # reopen PipelineDB
        if self._pipeline_db is None:
            self._pipeline_db = PipelineDB(self.pipeline_db_path)
        self._closed = False

    def set_current_db(self):
        """Set this DatabaseManager as the active global database."""
        from .thunk import Thunk
        Thunk.query = self
        _local.database = self
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
