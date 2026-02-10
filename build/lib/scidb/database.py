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

from sciduck import SciDuck, _infer_duckdb_type, _python_to_storage, _storage_to_python
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
    from .thunk import Thunk

    db = DatabaseManager(
        dataset_db_path,
        dataset_schema_keys=dataset_schema_keys,
        pipeline_db_path=pipeline_db_path,
        lineage_mode=lineage_mode,
    )
    for cls in BaseVariable._all_subclasses.values():
        db.register(cls)
    Thunk.query = db
    _local.database = db
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
        self.dataset_schema_keys = list(dataset_schema_keys)
        self._registered_types: dict[str, Type[BaseVariable]] = {}

        # Initialize SciDuck backend for data storage
        self._duck = SciDuck(self.dataset_db_path, dataset_schema=dataset_schema_keys)

        # Initialize PipelineDB for lineage storage (SQLite)
        self._pipeline_db = PipelineDB(pipeline_db_path)

        # Create metadata tables for type registration (in DuckDB)
        self._ensure_meta_tables()

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

    def _flatten_metadata(self, nested_metadata: dict) -> dict:
        """Flatten nested schema/version metadata back to flat dict."""
        flat = {}
        flat.update(nested_metadata.get("schema", {}))
        flat.update(nested_metadata.get("version", {}))
        return flat

    def _save_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Save a DataFrame directly to DuckDB, creating the table if needed."""
        # Check if table exists
        exists = self._duck._fetchall(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        )

        if not exists:
            # Create table from DataFrame schema
            self._duck._conn.execute(
                f'CREATE TABLE "{table_name}" AS SELECT * FROM df WHERE 1=0'
            )

        # Insert data
        self._duck._conn.execute(f'INSERT INTO "{table_name}" SELECT * FROM df')

    def _load_dataframe(self, table_name: str, **filters) -> pd.DataFrame:
        """Load a DataFrame from DuckDB with optional filters."""
        if filters:
            conditions = []
            params = []
            for key, value in filters.items():
                conditions.append(f'"{key}" = ?')
                params.append(value)
            where_clause = " AND ".join(conditions)
            query = f'SELECT * FROM "{table_name}" WHERE {where_clause}'
            return self._duck._conn.execute(query, params).fetchdf()
        else:
            return self._duck._conn.execute(f'SELECT * FROM "{table_name}"').fetchdf()

    @staticmethod
    def _has_custom_serialization(variable_class: type) -> bool:
        """Check if a BaseVariable subclass overrides to_db or from_db."""
        return "to_db" in variable_class.__dict__ or "from_db" in variable_class.__dict__

    def _ensure_native_table(self, table_name: str, ddb_type: str) -> None:
        """Create a table for native (SciDuck) storage if it doesn't exist."""
        exists = self._duck._fetchall(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        )
        if exists:
            return

        schema_cols = ", ".join(
            f'"{k}" VARCHAR' for k in self.dataset_schema_keys
        )

        self._duck._execute(f'''
            CREATE TABLE "{table_name}" (
                "value" {ddb_type},
                "_dtype_meta" VARCHAR,
                "_record_id" VARCHAR,
                "_content_hash" VARCHAR,
                "_lineage_hash" VARCHAR,
                "_schema_version" INTEGER,
                "_metadata" VARCHAR,
                "_user_id" VARCHAR,
                "_created_at" VARCHAR,
                {schema_cols}
            )
        ''')

    def _save_native_row(
        self,
        table_name: str,
        data: Any,
        record_id: str,
        content_hash: str,
        lineage_hash: str | None,
        schema_version: int,
        nested_metadata: dict,
        user_id: str | None,
    ) -> None:
        """Save a single row using SciDuck's native type conversion."""
        ddb_type, col_meta = _infer_duckdb_type(data)
        storage_value = _python_to_storage(data, col_meta)
        dtype_meta_json = json.dumps(col_meta)
        metadata_json = json.dumps(nested_metadata, sort_keys=True)
        created_at = datetime.now().isoformat()

        self._ensure_native_table(table_name, ddb_type)

        col_names = [
            "value", "_dtype_meta", "_record_id", "_content_hash", "_lineage_hash",
            "_schema_version", "_metadata", "_user_id", "_created_at",
        ]
        values: list[Any] = [
            storage_value, dtype_meta_json, record_id, content_hash, lineage_hash,
            schema_version, metadata_json, user_id, created_at,
        ]

        for key, value in nested_metadata.get("schema", {}).items():
            col_names.append(key)
            values.append(value)

        col_str = ", ".join(f'"{c}"' for c in col_names)
        placeholders = ", ".join(["?"] * len(col_names))

        self._duck._execute(
            f'INSERT INTO "{table_name}" ({col_str}) VALUES ({placeholders})',
            values,
        )

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
        raw_data = None

        if isinstance(data, ThunkOutput):
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
                        )

            lineage = extract_lineage(data)
            lineage_hash = data.pipeline_thunk.compute_lineage_hash()
            raw_data = get_raw_value(data)

        elif isinstance(data, BaseVariable):
            raw_data = data.data
            lineage_hash = data.lineage_hash

        else:
            raw_data = data

        instance = variable_class(raw_data)
        record_id = self.save(instance, metadata, lineage=lineage, lineage_hash=lineage_hash, index=index)
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
        index: Any = None,
    ) -> str:
        """
        Save a variable to the database.

        Args:
            variable: The variable instance to save
            metadata: Addressing metadata (flat dict)
            lineage: Optional lineage record if data came from a thunk
            lineage_hash: Optional pre-computed lineage hash
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

            df["_record_id"] = record_id
            df["_content_hash"] = content_hash
            df["_lineage_hash"] = lineage_hash
            df["_schema_version"] = variable.schema_version
            df["_metadata"] = json.dumps(nested_metadata, sort_keys=True)
            df["_user_id"] = user_id
            df["_created_at"] = datetime.now().isoformat()

            for key, value in nested_metadata.get("schema", {}).items():
                df[key] = value

            self._save_dataframe(table_name, df)
        else:
            # Native storage: SciDuck handles type conversion
            self._save_native_row(
                table_name, variable.data, record_id, content_hash,
                lineage_hash, variable.schema_version, nested_metadata, user_id,
            )

        # Save lineage if provided
        if lineage is not None:
            self._save_lineage(record_id, type_name, lineage, lineage_hash, user_id)

        return record_id

    def _save_lineage(
        self,
        output_record_id: str,
        output_type: str,
        lineage: "LineageRecord",
        lineage_hash: str | None = None,
        user_id: str | None = None,
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
        )

    def save_ephemeral_lineage(
        self,
        ephemeral_id: str,
        variable_type: str,
        lineage: "LineageRecord",
        user_id: str | None = None,
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
            # Build query
            if version != "latest" and version is not None:
                # Load by specific record_id
                df = self._load_dataframe(table_name, _record_id=version)
                if len(df) == 0:
                    raise NotFoundError(f"No data found with record_id '{version}'")
            else:
                # Load by metadata
                split = self._split_metadata(metadata)
                schema_filter = split["schema"]

                df = self._load_dataframe(table_name, **schema_filter)

                # Further filter by version keys if provided
                version_filter = split["version"]
                if version_filter:
                    for key, value in version_filter.items():
                        # Filter on _metadata JSON column
                        mask = df["_metadata"].apply(
                            lambda m: json.loads(m).get("version", {}).get(key) == value
                        )
                        df = df[mask]

                if len(df) == 0:
                    raise NotFoundError(
                        f"No {variable_class.__name__} found matching metadata: {metadata}"
                    )

                # Get latest by created_at
                df = df.sort_values("_created_at", ascending=False).head(1)
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                raise NotFoundError(
                    f"No {variable_class.__name__} found matching metadata: {metadata}"
                )
            raise

        return self._df_to_variable(variable_class, df, loc=loc, iloc=iloc)

    def _df_to_variable(
        self,
        variable_class: Type[BaseVariable],
        df: pd.DataFrame,
        loc: Any = None,
        iloc: Any = None,
    ) -> BaseVariable:
        """Convert a DataFrame row to a variable instance."""
        # Extract metadata columns
        record_id = df["_record_id"].iloc[0]
        content_hash = df["_content_hash"].iloc[0]
        lineage_hash = df["_lineage_hash"].iloc[0] if "_lineage_hash" in df.columns else None
        metadata_json = df["_metadata"].iloc[0]
        nested_metadata = json.loads(metadata_json)
        flat_metadata = self._flatten_metadata(nested_metadata)

        # Determine deserialization path from stored data
        use_native = (
            "_dtype_meta" in df.columns
            and df["_dtype_meta"].iloc[0] is not None
            and pd.notna(df["_dtype_meta"].iloc[0])
        )

        if use_native:
            # Native path: SciDuck handles type restoration
            col_meta = json.loads(df["_dtype_meta"].iloc[0])
            raw_value = df["value"].iloc[0]
            data = _storage_to_python(raw_value, col_meta)
        else:
            # Custom path: use from_db()
            internal_cols = [
                "_record_id", "_content_hash", "_lineage_hash",
                "_schema_version", "_metadata", "_user_id", "_created_at",
                "_dtype_meta",
            ] + self.dataset_schema_keys
            data_df = df.drop(columns=[c for c in internal_cols if c in df.columns], errors="ignore")

            if loc is not None:
                if not isinstance(loc, (list, range, slice)):
                    loc = [loc]
                data_df = data_df.loc[loc]
            elif iloc is not None:
                if not isinstance(iloc, (list, range, slice)):
                    iloc = [iloc]
                data_df = data_df.iloc[iloc]

            data = variable_class.from_db(data_df)

        instance = variable_class(data)
        instance.record_id = record_id
        instance.metadata = flat_metadata
        instance.content_hash = content_hash
        instance.lineage_hash = lineage_hash

        return instance

    def load_all(
        self,
        variable_class: Type[BaseVariable],
        metadata: dict,
    ):
        """
        Load all variables matching the given metadata as a generator.

        Args:
            variable_class: The type to load
            metadata: Flat metadata dict

        Yields:
            BaseVariable instances matching the metadata
        """
        table_name = self._ensure_registered(variable_class, auto_register=True)

        split = self._split_metadata(metadata)
        schema_filter = split["schema"]

        try:
            df = self._load_dataframe(table_name, **schema_filter)
        except Exception:
            return  # No data

        # Further filter by version keys if provided
        version_filter = split["version"]
        if version_filter:
            for key, value in version_filter.items():
                mask = df["_metadata"].apply(
                    lambda m: json.loads(m).get("version", {}).get(key) == value
                )
                df = df[mask]

        # Yield each unique record_id
        for record_id in df["_record_id"].unique():
            record_df = df[df["_record_id"] == record_id]
            yield self._df_to_variable(variable_class, record_df)

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

        split = self._split_metadata(metadata)
        schema_filter = split["schema"]

        try:
            df = self._load_dataframe(table_name, **schema_filter)
        except Exception:
            return []

        results = []
        for _, row in df.iterrows():
            nested_metadata = json.loads(row["_metadata"])
            results.append({
                "record_id": row["_record_id"],
                "schema": nested_metadata.get("schema", {}),
                "version": nested_metadata.get("version", {}),
                "created_at": row["_created_at"],
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
        for record in records:
            output_record_id = record["output_record_id"]
            output_type = record["output_type"]

            # Skip ephemeral entries (no data stored in SciDuck)
            if output_record_id.startswith("ephemeral:"):
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

        return results if results else None

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
