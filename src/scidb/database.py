"""Database connection and management using SciDuck backend."""

import json
import os
import threading
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

# Import SciDuck - adjust path as needed
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sciduck.sciduck import SciDuck


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
    db_path: str | Path,
    schema_keys: list[str],
    lineage_mode: str = "strict",
) -> "DatabaseManager":
    """
    Configure the global database connection.

    Also registers the database as a cache backend for the thunk library,
    enabling automatic computation caching.

    Args:
        db_path: Path to the DuckDB database file
        schema_keys: List of metadata keys that define the dataset schema
            (e.g., ["subject", "visit", "channel"]). These keys identify the
            logical location of data and are used for the folder hierarchy.
            Any metadata keys not in this list are treated as version parameters
            that distinguish different computational versions of the same data.
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
    _local.database = DatabaseManager(db_path, schema_keys=schema_keys, lineage_mode=lineage_mode)
    return _local.database


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
    Manages database connection and variable storage using SciDuck backend.

    Example:
        db = DatabaseManager("experiment.duckdb")
        db.register(RotationMatrix)

        record_id = RotationMatrix.save(np.eye(3), db=db, subject=1, trial=1)

        loaded = RotationMatrix.load(db=db, subject=1, trial=1)
    """

    VALID_LINEAGE_MODES = ("strict", "ephemeral")

    def __init__(
        self,
        db_path: str | Path,
        schema_keys: list[str],
        lineage_mode: str = "strict",
    ):
        """
        Initialize database connection.

        Args:
            db_path: Path to DuckDB database file (created if doesn't exist)
            schema_keys: List of metadata keys that define the dataset schema
                (e.g., ["subject", "visit", "channel"]). These keys identify the
                logical location of data. Any other metadata keys are treated as
                version parameters.
            lineage_mode: How to handle intermediate variables ("strict" or "ephemeral")

        Raises:
            ValueError: If lineage_mode is not valid
        """
        if lineage_mode not in self.VALID_LINEAGE_MODES:
            raise ValueError(
                f"lineage_mode must be one of {self.VALID_LINEAGE_MODES}, "
                f"got '{lineage_mode}'"
            )

        self.db_path = Path(db_path)
        self.lineage_mode = lineage_mode
        self.schema_keys = list(schema_keys)
        self._registered_types: dict[str, Type[BaseVariable]] = {}

        # Initialize SciDuck backend
        self._duck = SciDuck(self.db_path, dataset_schema=schema_keys)

        # Create additional metadata tables for lineage tracking
        self._ensure_meta_tables()

    def _ensure_meta_tables(self):
        """Create internal metadata tables for lineage tracking."""
        # Registered types table
        self._duck._execute("""
            CREATE TABLE IF NOT EXISTS _registered_types (
                type_name VARCHAR PRIMARY KEY,
                table_name VARCHAR UNIQUE NOT NULL,
                schema_version INTEGER NOT NULL,
                registered_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # Lineage tracking table
        self._duck._execute("""
            CREATE TABLE IF NOT EXISTS _lineage (
                id INTEGER PRIMARY KEY,
                output_record_id VARCHAR UNIQUE NOT NULL,
                output_type VARCHAR NOT NULL,
                lineage_hash VARCHAR,
                function_name VARCHAR NOT NULL,
                function_hash VARCHAR NOT NULL,
                inputs JSON NOT NULL,
                constants JSON NOT NULL,
                user_id VARCHAR,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # Index for efficient cache lookups by lineage_hash
        try:
            self._duck._execute(
                "CREATE INDEX IF NOT EXISTS idx_lineage_hash ON _lineage(lineage_hash)"
            )
        except Exception:
            pass

        # Create sequence for lineage id
        try:
            self._duck._execute("CREATE SEQUENCE IF NOT EXISTS _lineage_id_seq START 1")
        except Exception:
            pass

    def _split_metadata(self, flat_metadata: dict) -> dict:
        """
        Split flat metadata into nested schema/version structure.

        Keys in schema_keys go to "schema", all other keys go to "version".
        """
        schema = {}
        version = {}
        for key, value in flat_metadata.items():
            if key in self.schema_keys:
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

    def register(self, variable_class: Type[BaseVariable]) -> None:
        """
        Register a variable type for storage.

        Args:
            variable_class: The BaseVariable subclass to register
        """
        type_name = variable_class.__name__
        table_name = variable_class.table_name()
        schema_version = variable_class.schema_version

        # Register in metadata table
        try:
            self._duck._execute(
                """
                INSERT INTO _registered_types (type_name, table_name, schema_version)
                VALUES (?, ?, ?)
                ON CONFLICT (type_name) DO UPDATE SET
                    table_name = EXCLUDED.table_name,
                    schema_version = EXCLUDED.schema_version,
                    registered_at = current_timestamp
                """,
                [type_name, table_name, schema_version],
            )
        except Exception:
            # Handle DuckDB syntax differences
            self._duck._execute(
                """
                INSERT OR REPLACE INTO _registered_types (type_name, table_name, schema_version)
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

        # Convert to DataFrame
        df = variable.to_db()

        # Apply index if provided
        if index is not None:
            index_list = list(index) if not isinstance(index, list) else index
            if len(index_list) != len(df):
                raise ValueError(
                    f"Index length ({len(index_list)}) does not match "
                    f"DataFrame row count ({len(df)})"
                )
            df.index = index

        # Compute content hash
        content_hash = canonical_hash(variable.data)

        # Generate record_id
        record_id = generate_record_id(
            class_name=type_name,
            schema_version=variable.schema_version,
            content_hash=content_hash,
            metadata=nested_metadata,
        )

        # Store metadata as JSON in the data
        df["_record_id"] = record_id
        df["_content_hash"] = content_hash
        df["_lineage_hash"] = lineage_hash
        df["_schema_version"] = variable.schema_version
        df["_metadata"] = json.dumps(nested_metadata, sort_keys=True)
        df["_user_id"] = user_id
        df["_created_at"] = datetime.now().isoformat()

        # Add schema keys as columns for SciDuck
        schema_level = self.schema_keys[-1] if self.schema_keys else None
        for key in self.schema_keys:
            if key in nested_metadata["schema"]:
                df[key] = nested_metadata["schema"][key]

        # Save via SciDuck
        try:
            self._duck.save(
                table_name,
                df,
                schema_level=schema_level,
                force=True,  # Always save (we handle dedup via record_id)
            )
        except Exception as e:
            # If table doesn't exist or schema mismatch, create fresh
            if "not found" in str(e).lower() or "mismatch" in str(e).lower():
                self._duck.save(
                    table_name,
                    df,
                    schema_level=schema_level,
                    force=True,
                )
            else:
                raise

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
        """Save lineage record for a variable."""
        try:
            new_id = self._duck._fetchall("SELECT nextval('_lineage_id_seq')")[0][0]
        except Exception:
            new_id = 0

        try:
            self._duck._execute(
                """
                INSERT INTO _lineage
                (id, output_record_id, output_type, lineage_hash, function_name, function_hash,
                 inputs, constants, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (output_record_id) DO UPDATE SET
                    lineage_hash = EXCLUDED.lineage_hash,
                    function_name = EXCLUDED.function_name,
                    function_hash = EXCLUDED.function_hash,
                    inputs = EXCLUDED.inputs,
                    constants = EXCLUDED.constants,
                    user_id = EXCLUDED.user_id,
                    created_at = current_timestamp
                """,
                [
                    new_id,
                    output_record_id,
                    output_type,
                    lineage_hash,
                    lineage.function_name,
                    lineage.function_hash,
                    json.dumps(lineage.inputs),
                    json.dumps(lineage.constants),
                    user_id,
                ],
            )
        except Exception:
            # Fallback for older DuckDB versions
            self._duck._execute(
                """
                INSERT OR REPLACE INTO _lineage
                (id, output_record_id, output_type, lineage_hash, function_name, function_hash,
                 inputs, constants, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    new_id,
                    output_record_id,
                    output_type,
                    lineage_hash,
                    lineage.function_name,
                    lineage.function_hash,
                    json.dumps(lineage.inputs),
                    json.dumps(lineage.constants),
                    user_id,
                ],
            )

    def save_ephemeral_lineage(
        self,
        ephemeral_id: str,
        variable_type: str,
        lineage: "LineageRecord",
        user_id: str | None = None,
    ) -> None:
        """Save an ephemeral lineage record for an unsaved intermediate variable."""
        # Check if already exists
        rows = self._duck._fetchall(
            "SELECT 1 FROM _lineage WHERE output_record_id = ?",
            [ephemeral_id],
        )
        if rows:
            return

        self._save_lineage(ephemeral_id, variable_type, lineage)

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
        table_name = self._ensure_registered(variable_class, auto_register=False)

        # Build query
        if version != "latest" and version is not None:
            # Load by specific record_id
            df = self._duck.load(table_name)
            df = df[df["_record_id"] == version]
            if len(df) == 0:
                raise NotFoundError(f"No data found with record_id '{version}'")
        else:
            # Load by metadata
            split = self._split_metadata(metadata)
            schema_filter = split["schema"]

            df = self._duck.load(table_name, **schema_filter)

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

        # Remove internal columns for from_db
        internal_cols = [
            "_record_id", "_content_hash", "_lineage_hash",
            "_schema_version", "_metadata", "_user_id", "_created_at"
        ] + self.schema_keys
        data_df = df.drop(columns=[c for c in internal_cols if c in df.columns], errors="ignore")

        # Apply index selection
        if loc is not None:
            if not isinstance(loc, (list, range, slice)):
                loc = [loc]
            data_df = data_df.loc[loc]
        elif iloc is not None:
            if not isinstance(iloc, (list, range, slice)):
                iloc = [iloc]
            data_df = data_df.iloc[iloc]

        # Convert back to native data
        data = variable_class.from_db(data_df)

        instance = variable_class(data)
        instance._record_id = record_id
        instance._metadata = flat_metadata
        instance._content_hash = content_hash
        instance._lineage_hash = lineage_hash

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
        table_name = self._ensure_registered(variable_class, auto_register=False)

        split = self._split_metadata(metadata)
        schema_filter = split["schema"]

        try:
            df = self._duck.load(table_name, **schema_filter)
        except KeyError:
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
        table_name = self._ensure_registered(variable_class, auto_register=False)

        split = self._split_metadata(metadata)
        schema_filter = split["schema"]

        try:
            df = self._duck.load(table_name, **schema_filter)
        except KeyError:
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

        rows = self._duck._fetchall(
            """SELECT function_name, function_hash, inputs, constants
               FROM _lineage WHERE output_record_id = ?""",
            [record_id],
        )

        if not rows:
            return None

        row = rows[0]
        return {
            "function_name": row[0],
            "function_hash": row[1],
            "inputs": json.loads(row[2]),
            "constants": json.loads(row[3]),
        }

    def has_lineage(self, record_id: str) -> bool:
        """Check if a variable has lineage information."""
        rows = self._duck._fetchall(
            "SELECT 1 FROM _lineage WHERE output_record_id = ?",
            [record_id],
        )
        return len(rows) > 0

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

    def close(self):
        """Close the database connection."""
        self._duck.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
