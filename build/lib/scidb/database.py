"""Database connection and management."""

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .lineage import LineageRecord

from .exceptions import (
    DatabaseNotConfiguredError,
    NotFoundError,
    NotRegisteredError,
)
from .hashing import generate_record_id, canonical_hash
from .parquet_storage import (
    compute_parquet_path,
    get_parquet_root,
    read_parquet,
    write_parquet,
)
from .preview import generate_preview
from .storage import register_adapters
from .variable import BaseVariable


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
        db_path: Path to the SQLite database file
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
    from thunk import configure_cache

    _local.database = DatabaseManager(db_path, schema_keys=schema_keys, lineage_mode=lineage_mode)
    # Register as cache backend for thunk library
    configure_cache(_local.database)
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
    Manages SQLite database connection and variable storage.

    Example:
        db = DatabaseManager("experiment.db")
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
            db_path: Path to SQLite database file (created if doesn't exist)
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
        self.schema_keys = list(schema_keys)  # Store a copy
        self._registered_types: dict[str, Type[BaseVariable]] = {}

        # Set up Parquet file storage root
        self.parquet_root = get_parquet_root(self.db_path)

        # Register custom adapters
        register_adapters()

        # Connect with type detection
        self.connection = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,  # Allow multi-threaded access
        )
        self.connection.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        self.connection.execute("PRAGMA journal_mode=WAL")

        # Create metadata tables
        self._ensure_meta_tables()

    def _ensure_meta_tables(self):
        """Create internal metadata tables if they don't exist."""
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS _registered_types (
                type_name TEXT PRIMARY KEY,
                table_name TEXT UNIQUE NOT NULL,
                schema_version INTEGER NOT NULL,
                registered_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS _version_log (
                record_id TEXT PRIMARY KEY,
                type_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata JSON NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Index on content_hash for finding records with identical content
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_version_log_content_hash
            ON _version_log(content_hash)
        """
        )

        # Lineage tracking table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS _lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                output_record_id TEXT UNIQUE NOT NULL,
                output_type TEXT NOT NULL,
                function_name TEXT NOT NULL,
                function_hash TEXT NOT NULL,
                inputs JSON NOT NULL,
                constants JSON NOT NULL,
                user_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Index for querying by function
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_lineage_function
            ON _lineage(function_hash)
        """
        )

        # Index for querying by output
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_lineage_output
            ON _lineage(output_record_id)
        """
        )

        # Computation cache table - maps (function + inputs + output_num) -> output record_id
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS _computation_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT NOT NULL,
                output_num INTEGER NOT NULL DEFAULT 0,
                function_name TEXT NOT NULL,
                function_hash TEXT NOT NULL,
                output_type TEXT NOT NULL,
                output_record_id TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(cache_key, output_num)
            )
        """
        )

        # Index for cache key lookup
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_key
            ON _computation_cache(cache_key)
        """
        )

        self.connection.commit()

    def _split_metadata(self, flat_metadata: dict) -> dict:
        """
        Split flat metadata into nested schema/version structure.

        Keys in schema_keys go to "schema", all other keys go to "version".

        Args:
            flat_metadata: Flat dict of metadata key-value pairs

        Returns:
            Nested dict with "schema" and "version" keys
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
        """
        Flatten nested schema/version metadata back to flat dict.

        Args:
            nested_metadata: Dict with "schema" and "version" keys

        Returns:
            Flat dict combining all keys
        """
        flat = {}
        flat.update(nested_metadata.get("schema", {}))
        flat.update(nested_metadata.get("version", {}))
        return flat

    def _is_nested_metadata(self, metadata: dict) -> bool:
        """Check if metadata is in nested format."""
        return "schema" in metadata and isinstance(metadata.get("schema"), dict)

    def register(self, variable_class: Type[BaseVariable]) -> None:
        """
        Register a variable type for storage.

        Creates the table if it doesn't exist. Must be called before
        save() or load() for this type.

        Args:
            variable_class: The BaseVariable subclass to register
        """
        type_name = variable_class.__name__
        table_name = variable_class.table_name()
        schema_version = variable_class.schema_version

        # Create the metadata table (data is stored in Parquet files)
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                lineage_hash TEXT,
                schema_version INTEGER NOT NULL,
                metadata JSON NOT NULL,
                preview TEXT,
                user_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create index on record_id for fast lookups
        self.connection.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_record_id
            ON {table_name}(record_id)
        """
        )

        # Create index on content_hash for deduplication lookups
        self.connection.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_content_hash
            ON {table_name}(content_hash)
        """
        )

        # Create index on lineage_hash for cache lookups
        self.connection.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_lineage_hash
            ON {table_name}(lineage_hash)
        """
        )

        # Register in metadata table (use space separator for SQLite TIMESTAMP compatibility)
        self.connection.execute(
            """
            INSERT OR REPLACE INTO _registered_types
            (type_name, table_name, schema_version, registered_at)
            VALUES (?, ?, ?, ?)
        """,
            (type_name, table_name, schema_version, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")),
        )

        self.connection.commit()

        # Cache locally
        self._registered_types[type_name] = variable_class

    def _ensure_registered(
        self, variable_class: Type[BaseVariable], auto_register: bool = True
    ) -> str:
        """
        Ensure a variable type is registered.

        Args:
            variable_class: The BaseVariable subclass to check
            auto_register: If True, auto-register if not found. If False, raise error.

        Returns:
            The table name for this variable type

        Raises:
            NotRegisteredError: If not registered and auto_register is False
        """
        type_name = variable_class.__name__

        # Check local cache first
        if type_name in self._registered_types:
            return variable_class.table_name()

        # Check database
        cursor = self.connection.execute(
            "SELECT table_name FROM _registered_types WHERE type_name = ?",
            (type_name,),
        )
        row = cursor.fetchone()

        if row is None:
            if auto_register:
                # Auto-register the type
                self.register(variable_class)
                return variable_class.table_name()
            else:
                raise NotRegisteredError(
                    f"Variable type '{type_name}' is not registered. "
                    f"No data has been saved for this type yet."
                )

        # Update local cache
        self._registered_types[type_name] = variable_class
        return row["table_name"]

    def save(
        self,
        variable: BaseVariable,
        metadata: dict,
        lineage: "LineageRecord | None" = None,
        lineage_hash: str | None = None,
        index: any = None,
    ) -> str:
        """
        Save a variable to the database.

        Data is stored as a Parquet file in the parquet_root directory.
        Metadata entries are stored in the variable-specific SQLite table.

        If the exact same data+metadata already exists (same record_id),
        this is a no-op and returns the existing record_id.

        Args:
            variable: The variable instance to save
            metadata: Addressing metadata (flat dict, will be split into schema/version)
            lineage: Optional lineage record if data came from a thunk
            lineage_hash: Optional pre-computed lineage hash for cache key computation
            index: Optional index to set on the DataFrame after to_db() is called.
                Must match the length of the DataFrame rows.

        Returns:
            The record_id of the saved/existing data

        Raises:
            ValueError: If index length doesn't match DataFrame row count
        """
        table_name = self._ensure_registered(type(variable))
        type_name = variable.__class__.__name__
        user_id = get_user_id()

        # Split metadata into schema (dataset identity) and version (computational identity)
        nested_metadata = self._split_metadata(metadata)

        # Convert to DataFrame and compute content hash
        df = variable.to_db()

        # Apply index if provided
        if index is not None:
            # Convert to list for length checking (handles range, etc.)
            index_list = list(index) if not isinstance(index, list) else index
            if len(index_list) != len(df):
                raise ValueError(
                    f"Index length ({len(index_list)}) does not match "
                    f"DataFrame row count ({len(df)})"
                )
            df.index = index

        content_hash = canonical_hash(variable.data)

        # Generate record_id using the full nested metadata for uniqueness
        record_id = generate_record_id(
            class_name=type_name,
            schema_version=variable.schema_version,
            content_hash=content_hash,
            metadata=nested_metadata,  # Use nested structure for record_id
        )

        # Check if metadata entry already exists (idempotent save)
        cursor = self.connection.execute(
            f"SELECT record_id FROM {table_name} WHERE record_id = ?",
            (record_id,),
        )
        if cursor.fetchone() is not None:
            return record_id  # Already saved

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Write data to Parquet file (use only schema keys for folder structure)
        parquet_path = compute_parquet_path(
            table_name=table_name,
            record_id=record_id,
            metadata=nested_metadata["schema"],  # Only schema keys for folder hierarchy
            parquet_root=self.parquet_root,
        )
        write_parquet(df, parquet_path)

        # Store nested metadata in variable table
        metadata_json = json.dumps(nested_metadata, sort_keys=True)
        preview_str = generate_preview(df)

        self.connection.execute(
            f"""
            INSERT INTO {table_name}
            (record_id, content_hash, lineage_hash, schema_version, metadata, preview, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (record_id, content_hash, lineage_hash, variable.schema_version, metadata_json, preview_str, user_id, now),
        )

        # Log to version history (includes content_hash for finding identical content)
        self.connection.execute(
            """
            INSERT INTO _version_log
            (record_id, type_name, table_name, content_hash, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (record_id, type_name, table_name, content_hash, metadata_json, now),
        )

        # Save lineage if provided
        if lineage is not None:
            self._save_lineage(record_id, type_name, lineage, now, user_id)

        self.connection.commit()
        return record_id

    def _save_lineage(
        self,
        output_record_id: str,
        output_type: str,
        lineage: "LineageRecord",
        timestamp: str,
        user_id: str | None = None,
    ) -> None:
        """Save lineage record for a variable."""
        self.connection.execute(
            """
            INSERT OR REPLACE INTO _lineage
            (output_record_id, output_type, function_name, function_hash,
             inputs, constants, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                output_record_id,
                output_type,
                lineage.function_name,
                lineage.function_hash,
                json.dumps(lineage.inputs),
                json.dumps(lineage.constants),
                user_id,
                timestamp,
            ),
        )

    def save_ephemeral_lineage(
        self,
        ephemeral_id: str,
        variable_type: str,
        lineage: "LineageRecord",
        user_id: str | None = None,
    ) -> None:
        """
        Save an ephemeral lineage record for an unsaved intermediate variable.

        Ephemeral entries store the computation graph without storing actual data.
        They use synthetic IDs prefixed with "ephemeral:" to distinguish from
        real record_ides.

        Args:
            ephemeral_id: Synthetic ID for this ephemeral entry (e.g., "ephemeral:abc123")
            variable_type: The type name of the unsaved variable
            lineage: The LineageRecord for the computation that produced this variable
            user_id: Optional user ID for attribution
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Check if already exists (avoid duplicates)
        cursor = self.connection.execute(
            "SELECT 1 FROM _lineage WHERE output_record_id = ?",
            (ephemeral_id,),
        )
        if cursor.fetchone() is not None:
            return  # Already stored

        self.connection.execute(
            """
            INSERT INTO _lineage
            (output_record_id, output_type, function_name, function_hash,
             inputs, constants, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                ephemeral_id,
                variable_type,
                lineage.function_name,
                lineage.function_hash,
                json.dumps(lineage.inputs),
                json.dumps(lineage.constants),
                user_id,
                timestamp,
            ),
        )
        self.connection.commit()

    def load(
        self,
        variable_class: Type[BaseVariable],
        metadata: dict,
        version: str = "latest",
        loc: any = None,
        iloc: any = None,
    ) -> BaseVariable:
        """
        Load a single variable matching the given metadata.

        Metadata keys are split into schema keys (dataset identity) and version
        keys (computational identity) based on the configured schema_keys.

        - If only schema keys are provided: returns the latest version at that location
        - If version keys are also provided: returns the exact matching version
        - Version keys only match records with explicit version values (records with
          empty version {} do not match version-specific queries)

        Args:
            variable_class: The type to load
            metadata: Flat metadata dict (will be split into schema/version internally)
            version: "latest" for most recent, or specific record_id
            loc: Optional label-based index selection (like pandas df.loc[]).
                Supports single values, lists, ranges, or slices.
            iloc: Optional integer position-based index selection (like pandas df.iloc[]).
                Supports single values, lists, ranges, or slices.

        Returns:
            The matching variable instance (latest if multiple versions exist)

        Raises:
            NotFoundError: If no matches found
            NotRegisteredError: If this type has never been saved
        """
        table_name = self._ensure_registered(variable_class, auto_register=False)

        if version != "latest" and version is not None:
            # Load specific version by record_id
            cursor = self.connection.execute(
                f"""SELECT record_id, content_hash, lineage_hash, metadata
                    FROM {table_name}
                    WHERE record_id = ?""",
                (version,),
            )
            row = cursor.fetchone()
            if row is None:
                raise NotFoundError(f"No data found with record_id '{version}'")

            return self._row_to_variable(variable_class, table_name, row, loc=loc, iloc=iloc)

        # Split incoming metadata into schema and version parts
        split = self._split_metadata(metadata)
        schema_filter = split["schema"]
        version_filter = split["version"]

        # Build query conditions for nested metadata structure
        conditions = []
        params = []

        # Match schema keys
        for key, value in schema_filter.items():
            conditions.append(f"json_extract(metadata, '$.schema.{key}') = ?")
            if isinstance(value, (dict, list)):
                params.append(json.dumps(value))
            else:
                params.append(value)

        # Match version keys (only if provided)
        for key, value in version_filter.items():
            conditions.append(f"json_extract(metadata, '$.version.{key}') = ?")
            if isinstance(value, (dict, list)):
                params.append(json.dumps(value))
            else:
                params.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.connection.execute(
            f"""SELECT record_id, content_hash, lineage_hash, metadata
                FROM {table_name}
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT 1""",
            params,
        )
        row = cursor.fetchone()

        if row is None:
            raise NotFoundError(
                f"No {variable_class.__name__} found matching metadata: {metadata}"
            )

        return self._row_to_variable(variable_class, table_name, row, loc=loc, iloc=iloc)

    def _row_to_variable(
        self,
        variable_class: Type[BaseVariable],
        table_name: str,
        row: sqlite3.Row,
        loc: any = None,
        iloc: any = None,
    ) -> BaseVariable:
        """Convert a database row to a variable instance."""
        # Parse metadata (stored in nested format)
        nested_metadata = json.loads(row["metadata"])
        record_id = row["record_id"]

        # Handle both nested and legacy flat metadata formats
        if self._is_nested_metadata(nested_metadata):
            schema_metadata = nested_metadata["schema"]
            # Flatten for user-facing property
            flat_metadata = self._flatten_metadata(nested_metadata)
        else:
            # Legacy flat format (for backwards compatibility during migration)
            schema_metadata = nested_metadata
            flat_metadata = nested_metadata

        # Compute path using only schema keys and read Parquet file
        parquet_path = compute_parquet_path(
            table_name=table_name,
            record_id=record_id,
            metadata=schema_metadata,
            parquet_root=self.parquet_root,
        )
        df = read_parquet(parquet_path)

        # Apply index selection if provided
        if loc is not None:
            # Normalize scalar to list so from_db always receives a DataFrame
            if not isinstance(loc, (list, range, slice)):
                loc = [loc]
            df = df.loc[loc]
        elif iloc is not None:
            # Normalize scalar to list so from_db always receives a DataFrame
            if not isinstance(iloc, (list, range, slice)):
                iloc = [iloc]
            df = df.iloc[iloc]

        data = variable_class.from_db(df)

        instance = variable_class(data)
        instance._record_id = record_id
        instance._metadata = flat_metadata  # User sees flat metadata
        instance._content_hash = row["content_hash"]
        instance._lineage_hash = row["lineage_hash"]  # May be None for raw data

        return instance

    def load_all(
        self,
        variable_class: Type[BaseVariable],
        metadata: dict,
    ):
        """
        Load all variables matching the given metadata as a generator.

        This is useful for loading multiple records with partial schema keys,
        e.g., all records for a given subject across all visits.

        Args:
            variable_class: The type to load
            metadata: Flat metadata dict (will be split into schema/version internally)

        Yields:
            BaseVariable instances matching the metadata

        Raises:
            NotRegisteredError: If this type has never been saved

        Example:
            # Load all records for subject 1
            for var in db.load_all(ProcessedSignal, {"subject": 1}):
                print(var.metadata, var.data)
        """
        table_name = self._ensure_registered(variable_class, auto_register=False)

        # Split incoming metadata into schema and version parts
        split = self._split_metadata(metadata)
        schema_filter = split["schema"]
        version_filter = split["version"]

        # Build query conditions for nested metadata structure
        conditions = []
        params = []

        # Match schema keys
        for key, value in schema_filter.items():
            conditions.append(f"json_extract(metadata, '$.schema.{key}') = ?")
            if isinstance(value, (dict, list)):
                params.append(json.dumps(value))
            else:
                params.append(value)

        # Match version keys (only if provided)
        for key, value in version_filter.items():
            conditions.append(f"json_extract(metadata, '$.version.{key}') = ?")
            if isinstance(value, (dict, list)):
                params.append(json.dumps(value))
            else:
                params.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.connection.execute(
            f"""SELECT record_id, content_hash, lineage_hash, metadata
                FROM {table_name}
                WHERE {where_clause}
                ORDER BY created_at DESC""",
            params,
        )

        # Yield variables one at a time (generator)
        for row in cursor:
            yield self._row_to_variable(variable_class, table_name, row)

    def list_versions(
        self,
        variable_class: Type[BaseVariable],
        **metadata,
    ) -> list[dict]:
        """
        List all versions at a schema location.

        Shows all saved versions (including those with empty version {})
        at a given schema location. This is useful for seeing what
        computational variants exist for a given dataset location.

        Args:
            variable_class: The type to query
            **metadata: Schema metadata to match (version keys are ignored for listing)

        Returns:
            List of dicts with record_id, schema, version, created_at

        Raises:
            NotRegisteredError: If this type has never been saved
        """
        table_name = self._ensure_registered(variable_class, auto_register=False)

        # Split metadata - only use schema keys for the query
        split = self._split_metadata(metadata)
        schema_filter = split["schema"]

        conditions = []
        params = []
        for key, value in schema_filter.items():
            conditions.append(f"json_extract(metadata, '$.schema.{key}') = ?")
            if isinstance(value, (dict, list)):
                params.append(json.dumps(value))
            else:
                params.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.connection.execute(
            f"""SELECT record_id, metadata, created_at FROM {table_name}
                WHERE {where_clause}
                ORDER BY created_at DESC""",
            params,
        )

        results = []
        for row in cursor.fetchall():
            nested_metadata = json.loads(row["metadata"])
            if self._is_nested_metadata(nested_metadata):
                results.append({
                    "record_id": row["record_id"],
                    "schema": nested_metadata.get("schema", {}),
                    "version": nested_metadata.get("version", {}),
                    "created_at": row["created_at"],
                })
            else:
                # Legacy flat format
                results.append({
                    "record_id": row["record_id"],
                    "schema": nested_metadata,
                    "version": {},
                    "created_at": row["created_at"],
                })
        return results

    def get_provenance(
        self,
        variable_class: Type[BaseVariable],
        version: str | None = None,
        **metadata,
    ) -> dict | None:
        """
        Get the provenance (lineage) of a variable.

        Args:
            variable_class: The type to query
            version: Specific record_id, or None to use metadata lookup
            **metadata: Metadata to match (if version not specified)

        Returns:
            Dict with function_name, function_hash, inputs, constants
            or None if no lineage recorded (data wasn't from a thunk)
        """
        # First, find the record_id
        if version:
            record_id = version
        else:
            var = self.load(variable_class, metadata)
            record_id = var.record_id

        cursor = self.connection.execute(
            """SELECT function_name, function_hash, inputs, constants
               FROM _lineage WHERE output_record_id = ?""",
            (record_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None  # No lineage recorded

        return {
            "function_name": row["function_name"],
            "function_hash": row["function_hash"],
            "inputs": json.loads(row["inputs"]),
            "constants": json.loads(row["constants"]),
        }

    def get_derived_from(
        self,
        variable_class: Type[BaseVariable],
        version: str | None = None,
        **metadata,
    ) -> list[dict]:
        """
        Find all variables that were derived from this one.

        Answers: "What outputs used this variable as an input?"

        Args:
            variable_class: The type to query
            version: Specific record_id, or None to use metadata lookup
            **metadata: Metadata to match (if version not specified)

        Returns:
            List of dicts with record_id, type, function for each derived variable
        """
        # First, find the record_id
        if version:
            record_id = version
        else:
            var = self.load(variable_class, metadata)
            record_id = var.record_id

        # Search for this record_id in any lineage inputs
        cursor = self.connection.execute(
            """SELECT output_record_id, output_type, function_name, inputs
               FROM _lineage
               WHERE EXISTS (
                   SELECT 1 FROM json_each(inputs)
                   WHERE json_extract(value, '$.record_id') = ?
               )""",
            (record_id,),
        )

        return [
            {
                "record_id": row["output_record_id"],
                "type": row["output_type"],
                "function": row["function_name"],
            }
            for row in cursor.fetchall()
        ]

    def has_lineage(self, record_id: str) -> bool:
        """
        Check if a variable has lineage information.

        Args:
            record_id: The version hash to check

        Returns:
            True if lineage exists, False otherwise
        """
        cursor = self.connection.execute(
            "SELECT 1 FROM _lineage WHERE output_record_id = ?",
            (record_id,),
        )
        return cursor.fetchone() is not None

    def get_full_lineage(
        self,
        variable_class: Type[BaseVariable],
        version: str | None = None,
        max_depth: int = 100,
        **metadata,
    ) -> dict:
        """
        Get the complete lineage chain for a variable.

        Recursively traverses the lineage graph to show the full
        computation history from original inputs to final output.

        Args:
            variable_class: The type to query
            version: Specific record_id, or None to use metadata lookup
            max_depth: Maximum recursion depth to prevent infinite loops
            **metadata: Metadata to match (if version not specified)

        Returns:
            Nested dict representing the full computation graph:
            {
                "type": "FinalResult",
                "record_id": "abc123...",
                "function": "compute_stats",
                "function_hash": "def456...",
                "constants": [{"name": "threshold", "value": 0.5}],
                "inputs": [
                    {
                        "type": "FilteredData",
                        "record_id": "...",
                        "function": "filter",
                        ...
                    }
                ]
            }
        """
        # Find the record_id
        if version:
            record_id = version
            type_name = variable_class.__name__
        else:
            var = self.load(variable_class, metadata)
            record_id = var.record_id
            type_name = variable_class.__name__

        return self._build_lineage_tree(record_id, type_name, max_depth, set())

    def _build_lineage_tree(
        self,
        record_id: str,
        type_name: str,
        max_depth: int,
        visited: set,
    ) -> dict:
        """Recursively build the lineage tree for a record_id."""
        # Prevent infinite loops
        if max_depth <= 0 or record_id in visited:
            return {
                "type": type_name,
                "record_id": record_id,
                "truncated": True,
            }

        visited = visited | {record_id}

        # Get lineage for this record_id
        cursor = self.connection.execute(
            """SELECT function_name, function_hash, inputs, constants
               FROM _lineage WHERE output_record_id = ?""",
            (record_id,),
        )
        row = cursor.fetchone()

        if row is None:
            # No lineage - this is a source/leaf node
            return {
                "type": type_name,
                "record_id": record_id,
                "function": None,
                "source": "manual",
            }

        inputs_json = json.loads(row["inputs"])
        constants_json = json.loads(row["constants"])

        # Recursively process inputs
        processed_inputs = []
        for inp in inputs_json:
            if inp.get("source_type") == "variable" and "record_id" in inp:
                # Saved variable - recurse
                child = self._build_lineage_tree(
                    inp["record_id"],
                    inp.get("type", "Unknown"),
                    max_depth - 1,
                    visited,
                )
                child["input_name"] = inp.get("name")
                processed_inputs.append(child)
            elif inp.get("source_type") == "unsaved_variable":
                # Unsaved variable - check if we have ephemeral lineage
                if inp.get("inner_source") == "thunk" and "source_hash" in inp:
                    # Try to find ephemeral lineage by hash
                    ephemeral_id = f"ephemeral:{inp['source_hash'][:32]}"
                    child = self._build_lineage_tree(
                        ephemeral_id,
                        inp.get("type", "Unknown"),
                        max_depth - 1,
                        visited,
                    )
                    child["input_name"] = inp.get("name")
                    child["ephemeral"] = True
                    processed_inputs.append(child)
                else:
                    # Unsaved variable with raw data only (no thunk lineage)
                    processed_inputs.append({
                        "type": inp.get("type", "Unknown"),
                        "input_name": inp.get("name"),
                        "content_hash": inp.get("content_hash"),
                        "ephemeral": True,
                        "source": "unsaved_raw_data",
                    })
            elif inp.get("source_type") == "thunk":
                # Output from another thunk - try to find by hash
                # This is trickier since we need to find the record_id
                processed_inputs.append({
                    "type": "ThunkOutput",
                    "input_name": inp.get("name"),
                    "source_function": inp.get("source_function"),
                    "source_hash": inp.get("source_hash"),
                    "output_num": inp.get("output_num"),
                    "note": "Intermediate thunk output (not saved separately)",
                })
            else:
                # Unknown input type
                processed_inputs.append({
                    "type": "Unknown",
                    "input_name": inp.get("name"),
                    "raw": inp,
                })

        return {
            "type": type_name,
            "record_id": record_id,
            "function": row["function_name"],
            "function_hash": row["function_hash"],
            "constants": constants_json,
            "inputs": processed_inputs,
        }

    def format_lineage(
        self,
        variable_class: Type[BaseVariable],
        version: str | None = None,
        **metadata,
    ) -> str:
        """
        Get a print-friendly representation of the full lineage.

        Args:
            variable_class: The type to query
            version: Specific record_id, or None to use metadata lookup
            **metadata: Metadata to match (if version not specified)

        Returns:
            A formatted string showing the computation graph.

        Example output:
            FinalResult (record_id: abc123...)
            └── compute_stats [hash: def456...]
                ├── constants: threshold=0.5
                └── inputs:
                    └── FilteredData (record_id: ghi789...)
                        └── filter [hash: jkl012...]
                            ├── constants: cutoff=10
                            └── inputs:
                                └── RawData (record_id: mno345...)
                                    └── [source: manual]
        """
        lineage = self.get_full_lineage(variable_class, version, **metadata)
        lines = []
        self._format_lineage_node(lineage, lines, prefix="", is_last=True)
        return "\n".join(lines)

    def _format_lineage_node(
        self,
        node: dict,
        lines: list,
        prefix: str,
        is_last: bool,
    ) -> None:
        """Recursively format a lineage node for printing."""
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        # Format the node header
        type_name = node.get("type", "Unknown")
        is_ephemeral = node.get("ephemeral", False)

        if node.get("truncated"):
            record_id_short = node.get("record_id", "???")[:12]
            lines.append(f"{prefix}{connector}{type_name} (record_id: {record_id_short}...) [truncated]")
            return

        # Handle ephemeral nodes without record_id (unsaved raw data)
        if is_ephemeral and node.get("source") == "unsaved_raw_data":
            content_hash = node.get("content_hash", "???")[:12]
            lines.append(f"{prefix}{connector}{type_name} [ephemeral, hash: {content_hash}...]")
            lines.append(f"{child_prefix}└── [source: unsaved raw data]")
            return

        # Format record_id display
        record_id = node.get("record_id", "???")
        if record_id.startswith("ephemeral:"):
            record_id_display = f"ephemeral:{record_id[10:22]}..."
        else:
            record_id_display = f"{record_id[:12]}..."

        ephemeral_marker = " [ephemeral]" if is_ephemeral else ""
        lines.append(f"{prefix}{connector}{type_name} (record_id: {record_id_display}){ephemeral_marker}")

        # Format function info
        function = node.get("function")
        if function is None:
            source = node.get("source", "unknown")
            lines.append(f"{child_prefix}└── [source: {source}]")
            return

        function_hash_short = node.get("function_hash", "???")[:12]
        lines.append(f"{child_prefix}├── function: {function} [hash: {function_hash_short}...]")

        # Format constants
        constants = node.get("constants", [])
        if constants:
            const_strs = []
            for c in constants:
                name = c.get("name", "?")
                value_repr = c.get("value_repr", str(c.get("value", "?")))
                # Truncate long values
                if len(value_repr) > 50:
                    value_repr = value_repr[:47] + "..."
                const_strs.append(f"{name}={value_repr}")
            lines.append(f"{child_prefix}├── constants: {', '.join(const_strs)}")

        # Format inputs
        inputs = node.get("inputs", [])
        if inputs:
            lines.append(f"{child_prefix}└── inputs:")
            for i, inp in enumerate(inputs):
                is_last_input = (i == len(inputs) - 1)
                input_prefix = child_prefix + "    "

                if inp.get("note"):
                    # Thunk output that wasn't saved
                    connector2 = "└── " if is_last_input else "├── "
                    input_name = inp.get("input_name", "?")
                    src_fn = inp.get("source_function", "?")
                    lines.append(f"{input_prefix}{connector2}[{input_name}] {src_fn}() -> (not saved)")
                else:
                    # Recurse into saved variable
                    self._format_lineage_node(inp, lines, input_prefix, is_last_input)
        else:
            lines.append(f"{child_prefix}└── inputs: (none)")

    # -------------------------------------------------------------------------
    # Computation Cache Methods
    # -------------------------------------------------------------------------

    def get_cached_computation(
        self,
        cache_key: str,
        variable_class: Type[BaseVariable],
    ) -> BaseVariable | None:
        """
        Look up a cached computation result.

        Args:
            cache_key: The cache key (hash of function + inputs)
            variable_class: The expected output type

        Returns:
            The cached variable instance, or None if not cached
        """
        cursor = self.connection.execute(
            """SELECT output_record_id, output_type FROM _computation_cache
               WHERE cache_key = ?""",
            (cache_key,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        # Verify type matches
        if row["output_type"] != variable_class.__name__:
            return None

        # Load the cached result
        try:
            return self.load(variable_class, {}, version=row["output_record_id"])
        except Exception:
            # Cache entry exists but data is missing - stale cache
            return None

    def cache_computation(
        self,
        cache_key: str,
        function_name: str,
        function_hash: str,
        output_type: str,
        output_record_id: str,
        output_num: int = 0,
    ) -> None:
        """
        Store a computation result in the cache.

        Args:
            cache_key: The cache key (hash of function + inputs)
            function_name: Name of the function
            function_hash: Hash of the function bytecode
            output_type: Name of the output variable class
            output_record_id: Version hash of the saved output
            output_num: Output index for multi-output functions (default 0)
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self.connection.execute(
            """
            INSERT OR REPLACE INTO _computation_cache
            (cache_key, output_num, function_name, function_hash, output_type, output_record_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (cache_key, output_num, function_name, function_hash, output_type, output_record_id, now),
        )
        self.connection.commit()

    def load_by_record_id(
        self,
        variable_class: Type[BaseVariable],
        record_id: str,
    ) -> BaseVariable | None:
        """
        Load a variable by its exact record_id.

        Args:
            variable_class: The type to load
            record_id: The version hash

        Returns:
            The variable instance, or None if not found
        """
        try:
            return self.load(variable_class, {}, version=record_id)
        except Exception:
            return None

    def get_cached_by_key(
        self, cache_key: str, n_outputs: int = 1
    ) -> list[tuple] | None:
        """
        Look up cached result by cache key only (without knowing variable type).

        This is used for automatic cache checking in Thunk.__call__().

        Args:
            cache_key: The cache key (hash of function + inputs)
            n_outputs: Number of outputs expected (default 1)

        Returns:
            List of (data, record_id) tuples if ALL outputs found and loadable,
            None otherwise. For n_outputs=1, still returns a list with one tuple.
        """
        cursor = self.connection.execute(
            """SELECT output_num, output_type, output_record_id FROM _computation_cache
               WHERE cache_key = ?
               ORDER BY output_num""",
            (cache_key,),
        )
        rows = cursor.fetchall()

        if len(rows) != n_outputs:
            return None  # Not all outputs cached

        # Verify we have all expected output_nums (0 to n_outputs-1)
        output_nums = {row["output_num"] for row in rows}
        expected_nums = set(range(n_outputs))
        if output_nums != expected_nums:
            return None  # Missing some outputs

        results = []
        for row in rows:
            output_type = row["output_type"]
            output_record_id = row["output_record_id"]

            # Look up class from database registry first
            var_class = self._registered_types.get(output_type)
            if var_class is None:
                # Try global registry and auto-register
                var_class = BaseVariable.get_subclass_by_name(output_type)
                if var_class is None:
                    return None  # Class not found anywhere
                self.register(var_class)

            # Load the variable by record_id
            var = self.load_by_record_id(var_class, output_record_id)
            if var is None:
                return None  # Data was deleted or corrupted

            results.append((var.data, output_record_id))

        return results

    def get_cached(
        self, cache_key: str, n_outputs: int = 1
    ) -> list[tuple] | None:
        """
        Look up cached results by cache key.

        This method implements the thunk.CacheBackend protocol.

        Args:
            cache_key: The cache key (hash of function + inputs)
            n_outputs: Number of outputs expected

        Returns:
            List of (data, identifier) tuples if ALL outputs found,
            None otherwise.
        """
        return self.get_cached_by_key(cache_key, n_outputs)

    def invalidate_cache(
        self,
        function_name: str | None = None,
        function_hash: str | None = None,
    ) -> int:
        """
        Invalidate cached computations.

        Args:
            function_name: If provided, only invalidate for this function
            function_hash: If provided, only invalidate for this function version

        Returns:
            Number of cache entries invalidated
        """
        if function_hash:
            cursor = self.connection.execute(
                "DELETE FROM _computation_cache WHERE function_hash = ?",
                (function_hash,),
            )
        elif function_name:
            cursor = self.connection.execute(
                "DELETE FROM _computation_cache WHERE function_name = ?",
                (function_name,),
            )
        else:
            cursor = self.connection.execute("DELETE FROM _computation_cache")

        self.connection.commit()
        return cursor.rowcount

    def get_cache_stats(self) -> dict:
        """
        Get statistics about the computation cache.

        Returns:
            Dict with total_entries, functions, and entries_by_function
        """
        cursor = self.connection.execute(
            "SELECT COUNT(*) as total FROM _computation_cache"
        )
        total = cursor.fetchone()["total"]

        cursor = self.connection.execute(
            """SELECT function_name, COUNT(*) as count
               FROM _computation_cache
               GROUP BY function_name"""
        )
        by_function = {row["function_name"]: row["count"] for row in cursor.fetchall()}

        return {
            "total_entries": total,
            "functions": len(by_function),
            "entries_by_function": by_function,
        }

    # -------------------------------------------------------------------------
    # Export and Preview Methods
    # -------------------------------------------------------------------------

    def export_to_csv(
        self,
        variable_class: Type[BaseVariable],
        path: str,
        **metadata,
    ) -> int:
        """
        Export matching variables to a CSV file.

        Exports the underlying DataFrame representation (from to_db()) for
        each matching variable. If multiple variables match, they are
        concatenated with a 'record_id' column to distinguish them.

        Args:
            variable_class: The type to export
            path: Output file path (will be overwritten if exists)
            **metadata: Metadata filter for selecting records

        Returns:
            Number of records exported

        Raises:
            NotFoundError: If no matching data found
            NotRegisteredError: If this type has never been saved
        """
        import pandas as pd

        # Load all matching variables using the generator
        results = list(self.load_all(variable_class, metadata))

        if not results:
            raise NotFoundError(
                f"No {variable_class.__name__} found matching metadata: {metadata}"
            )

        # Build combined DataFrame
        all_dfs = []
        for var in results:
            df = variable_class(var.data).to_db()
            df["_record_id"] = var.record_id
            # Add metadata columns (already flattened by load_all)
            if var.metadata:
                for key, value in var.metadata.items():
                    df[f"_meta_{key}"] = value
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(path, index=False)

        return len(results)

    def preview_data(
        self,
        variable_class: Type[BaseVariable],
        **metadata,
    ) -> str:
        """
        Get a formatted preview of matching variables.

        This retrieves the preview strings stored in the database,
        providing a quick look at the data without full deserialization.

        Args:
            variable_class: The type to preview
            **metadata: Metadata filter for selecting records

        Returns:
            Formatted string with previews of matching records

        Raises:
            NotFoundError: If no matching data found
            NotRegisteredError: If this type has never been saved
        """
        table_name = self._ensure_registered(variable_class, auto_register=False)

        # Split metadata into schema and version parts
        split = self._split_metadata(metadata)
        schema_filter = split["schema"]
        version_filter = split["version"]

        # Build query with nested metadata structure
        conditions = []
        params = []

        for key, value in schema_filter.items():
            conditions.append(f"json_extract(metadata, '$.schema.{key}') = ?")
            if isinstance(value, (dict, list)):
                params.append(json.dumps(value))
            else:
                params.append(value)

        for key, value in version_filter.items():
            conditions.append(f"json_extract(metadata, '$.version.{key}') = ?")
            if isinstance(value, (dict, list)):
                params.append(json.dumps(value))
            else:
                params.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.connection.execute(
            f"""SELECT record_id, metadata, preview, created_at FROM {table_name}
                WHERE {where_clause}
                ORDER BY created_at DESC""",
            params,
        )
        rows = cursor.fetchall()

        if not rows:
            raise NotFoundError(
                f"No {variable_class.__name__} found matching metadata: {metadata}"
            )

        # Format output
        lines = [f"=== {variable_class.__name__} ({len(rows)} records) ===\n"]
        for row in rows:
            record_id_short = row["record_id"][:12]
            nested_meta = json.loads(row["metadata"])

            # Format metadata for display
            if self._is_nested_metadata(nested_meta):
                schema_str = ", ".join(f"{k}={v}" for k, v in nested_meta.get("schema", {}).items())
                version_str = ", ".join(f"{k}={v}" for k, v in nested_meta.get("version", {}).items())
            else:
                schema_str = ", ".join(f"{k}={v}" for k, v in nested_meta.items())
                version_str = ""

            preview = row["preview"] or "(no preview)"

            lines.append(f"record_id: {record_id_short}...")
            if schema_str:
                lines.append(f"  schema: {schema_str}")
            if version_str:
                lines.append(f"  version: {version_str}")
            lines.append(f"  preview: {preview}")
            lines.append(f"  created: {row['created_at']}")
            lines.append("")

        return "\n".join(lines)

    def close(self):
        """Close the database connection."""
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
