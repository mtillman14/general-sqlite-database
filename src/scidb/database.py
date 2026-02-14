"""Database connection and management using SciDuck backend."""

import json
import os
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Type, Any

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

# Add sub-package paths
import sys
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root / "sciduck" / "src"))
sys.path.insert(0, str(_project_root / "pipelinedb-lib" / "src"))

from sciduck import SciDuck, _infer_duckdb_type, _python_to_storage
from pipelinedb import PipelineDB


# Global database instance (thread-local for safety)
_local = threading.local()


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
        Infer the schema level from contiguous keys starting at the top.

        Walks dataset_schema_keys top-down. Returns the deepest key where all
        keys from the top are contiguously present.

        Returns None if no schema keys are provided.

        Raises ValueError if schema keys are non-contiguous (e.g. providing
        "subject" and "stage" but not "trial" when the hierarchy is
        ["subject", "trial", "stage"]).
        """
        if not schema_keys:
            return None

        level = None
        found_gap = False
        for key in self.dataset_schema_keys:
            if key in schema_keys:
                if found_gap:
                    # Determine the expected keys for a helpful error message
                    level_idx = self.dataset_schema_keys.index(key)
                    missing = [
                        k for k in self.dataset_schema_keys[:level_idx]
                        if k not in schema_keys
                    ]
                    raise ValueError(
                        f"Non-contiguous schema keys: '{key}' was provided but "
                        f"{missing} are missing. Schema keys must be provided "
                        f"as a contiguous prefix of the hierarchy: "
                        f"{self.dataset_schema_keys}"
                    )
                level = key
            else:
                if level is not None:
                    found_gap = True
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

    def _save_custom_to_sciduck(
        self,
        table_name: str,
        variable_class: Type[BaseVariable],
        df: pd.DataFrame,
        schema_level: str | None,
        schema_keys: dict,
        content_hash: str,
        version_keys: dict | None = None,
    ) -> tuple[int, int, int]:
        """
        Save a custom-serialized DataFrame into SciDuck's table format.

        Creates a table with schema_id, parameter_id, version_id, and data columns.
        Registers in _variables with dtype = {"custom": True}.

        Returns (parameter_id, schema_id, version_id).
        """
        schema_id = (
            self._duck._get_or_create_schema_id(schema_level, schema_keys)
            if schema_level is not None and schema_keys
            else 0
        )

        # Resolve parameter_id and version_id
        parameter_id, version_id, is_new_parameter = self._resolve_parameter_slot(
            table_name, schema_id, version_keys
        )

        # Ensure table exists with SciDuck-compatible schema
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

        # Insert all DataFrame rows with the same (schema_id, parameter_id, version_id)
        for _, row in df.iterrows():
            col_names = ["schema_id", "parameter_id", "version_id"] + list(df.columns)
            col_str = ", ".join(f'"{c}"' for c in col_names)
            placeholders = ", ".join(["?"] * len(col_names))
            values = [schema_id, parameter_id, version_id] + [
                row[c].item() if hasattr(row[c], 'item') else row[c]
                for c in df.columns
            ]
            self._duck._execute(
                f'INSERT INTO "{table_name}" ({col_str}) VALUES ({placeholders})',
                values,
            )

        # Register in _variables only for truly new parameter slots
        effective_level = schema_level or self.dataset_schema_keys[-1]
        if is_new_parameter:
            self._duck._execute(
                "INSERT INTO _variables (variable_name, parameter_id, schema_level, "
                "dtype, version_keys, description) VALUES (?, ?, ?, ?, ?, ?)",
                [table_name, parameter_id, effective_level,
                 json.dumps({"custom": True}),
                 json.dumps(version_keys or {}, sort_keys=True), ""],
            )

        return parameter_id, schema_id, version_id

    def _save_native_direct(
        self,
        table_name: str,
        variable_class: Type[BaseVariable],
        data: Any,
        content_hash: str,
        version_keys: dict | None = None,
    ) -> tuple[int, int, int]:
        """
        Save native data directly without SciDuck's schema-aware save.

        Used when schema keys are non-contiguous or absent. Creates a
        SciDuck-compatible table with schema_id=0, parameter_id, and version_id.

        Returns (parameter_id, schema_id=0, version_id).
        """
        parameter_id, version_id, is_new_parameter = self._resolve_parameter_slot(
            table_name, 0, version_keys
        )

        ddb_type, col_meta = _infer_duckdb_type(data)
        storage_value = _python_to_storage(data, col_meta)
        dtype_meta = {"mode": "single_column", "columns": {"value": col_meta}}

        # Ensure table exists
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

        # Insert data
        self._duck._execute(
            f'INSERT INTO "{table_name}" ("schema_id", "parameter_id", "version_id", "value") VALUES (?, ?, ?, ?)',
            [0, parameter_id, version_id, storage_value],
        )

        # Register in _variables only for truly new parameter slots
        if is_new_parameter:
            self._duck._execute(
                "INSERT INTO _variables (variable_name, parameter_id, schema_level, "
                "dtype, version_keys, description) VALUES (?, ?, ?, ?, ?, ?)",
                [table_name, parameter_id, self.dataset_schema_keys[-1],
                 json.dumps(dtype_meta),
                 json.dumps(version_keys or {}, sort_keys=True), ""],
            )

        return parameter_id, 0, version_id

    def _save_native_with_schema(
        self,
        table_name: str,
        variable_class: Type[BaseVariable],
        data: Any,
        schema_level: str,
        schema_keys: dict,
        content_hash: str,
        version_keys: dict | None = None,
    ) -> tuple[int, int, int]:
        """
        Save native data at a specific schema location.

        Like _save_native_direct but with schema-aware storage and
        parameter_id/version_id tracking.

        Returns (parameter_id, schema_id, version_id).
        """
        schema_id = self._duck._get_or_create_schema_id(
            schema_level, {k: str(v) for k, v in schema_keys.items()}
        )

        # Resolve parameter_id and version_id
        parameter_id, version_id, is_new_parameter = self._resolve_parameter_slot(
            table_name, schema_id, version_keys
        )

        ddb_type, col_meta = _infer_duckdb_type(data)
        storage_value = _python_to_storage(data, col_meta)
        dtype_meta = {"mode": "single_column", "columns": {"value": col_meta}}

        # Ensure table exists
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

        # Insert data
        self._duck._execute(
            f'INSERT INTO "{table_name}" ("schema_id", "parameter_id", "version_id", "value") VALUES (?, ?, ?, ?)',
            [schema_id, parameter_id, version_id, storage_value],
        )

        # Register in _variables only for truly new parameter slots
        if is_new_parameter:
            self._duck._execute(
                "INSERT INTO _variables (variable_name, parameter_id, schema_level, "
                "dtype, version_keys, description) VALUES (?, ?, ?, ?, ?, ?)",
                [table_name, parameter_id, schema_level,
                 json.dumps(dtype_meta),
                 json.dumps(version_keys or {}, sort_keys=True), ""],
            )

        return parameter_id, schema_id, version_id

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
                params.extend([str(v) for v in value])
            else:
                conditions.append(f's."{key}" = ?')
                params.append(str(value))

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
                    schema[key] = str(val)

        vk_raw = row.get("version_keys")
        version = {}
        if vk_raw is not None and isinstance(vk_raw, str):
            version = json.loads(vk_raw)

        nested_metadata = {"schema": schema, "version": version}
        flat_metadata = {}
        flat_metadata.update(schema)
        flat_metadata.update(version)
        return flat_metadata, nested_metadata

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

            if self._has_custom_serialization(variable_class):
                data = variable_class.from_db(df)
            else:
                # Native DataFrame: return raw DataFrame directly
                data = df
        else:
            # Native path: use SciDuck.load() with parameter_id, version_id and raw=True
            schema_keys = nested_metadata.get("schema", {})
            schema_level = self._infer_schema_level(schema_keys)
            if schema_level is not None and schema_keys:
                data = self._duck.load(
                    table_name, parameter_id=parameter_id, version_id=version_id, raw=True,
                    **{k: str(v) for k, v in schema_keys.items()},
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

        instance = variable_class(raw_data)
        record_id = self.save(
            instance, metadata, lineage=lineage, lineage_hash=lineage_hash,
            pipeline_lineage_hash=pipeline_lineage_hash, index=index,
        )
        instance.record_id = record_id
        instance.metadata = metadata
        instance.lineage_hash = lineage_hash

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

        # Compute content hash
        content_hash = canonical_hash(variable.data)

        # Generate record_id
        record_id = generate_record_id(
            class_name=type_name,
            schema_version=variable.schema_version,
            content_hash=content_hash,
            metadata=nested_metadata,
        )

        # Idempotency: check if record already exists in _record_metadata
        existing = self._duck._fetchall(
            "SELECT 1 FROM _record_metadata WHERE record_id = ?",
            [record_id],
        )
        if existing:
            return record_id

        schema_keys = nested_metadata.get("schema", {})
        version_keys = nested_metadata.get("version", {})
        schema_level = self._infer_schema_level(schema_keys)
        created_at = datetime.now().isoformat()

        if self._has_custom_serialization(type(variable)):
            # Custom serialization: user provides to_db() → DataFrame
            df = variable.to_db()

            if index is not None:
                index_list = list(index) if not isinstance(index, list) else index
                if len(index_list) != len(df):
                    raise ValueError(
                        f"Index length ({len(index_list)}) does not match "
                        f"DataFrame row count ({len(df)})"
                    )
                df.index = index

            parameter_id, schema_id, version_id = self._save_custom_to_sciduck(
                table_name, type(variable), df, schema_level, schema_keys, content_hash,
                version_keys=version_keys,
            )
        elif isinstance(variable.data, pd.DataFrame):
            # Native DataFrame: store directly as multi-column DuckDB table
            df = variable.data.copy()

            if index is not None:
                index_list = list(index) if not isinstance(index, list) else index
                if len(index_list) != len(df):
                    raise ValueError(
                        f"Index length ({len(index_list)}) does not match "
                        f"DataFrame row count ({len(df)})"
                    )
                df.index = index

            parameter_id, schema_id, version_id = self._save_custom_to_sciduck(
                table_name, type(variable), df, schema_level, schema_keys, content_hash,
                version_keys=version_keys,
            )
        else:
            # Native storage
            if schema_level is not None and schema_keys:
                # Contiguous keys — save with schema-scoped parameter_id
                parameter_id, schema_id, version_id = self._save_native_with_schema(
                    table_name, type(variable), variable.data, schema_level, schema_keys, content_hash,
                    version_keys=version_keys,
                )
            else:
                # No schema or non-contiguous keys — direct insert
                parameter_id, schema_id, version_id = self._save_native_direct(
                    table_name, type(variable), variable.data, content_hash,
                    version_keys=version_keys,
                )

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

        # Save lineage if provided
        if lineage is not None:
            effective_plh = pipeline_lineage_hash if pipeline_lineage_hash is not None else lineage_hash
            self._save_lineage(
                record_id, type_name, lineage, effective_plh, user_id,
                schema_keys=nested_metadata.get("schema"),
                output_content_hash=content_hash,
            )

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

        for _, row in records.iterrows():
            yield self._load_by_record_row(variable_class, row)

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
