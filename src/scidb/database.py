"""Database connection and management."""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Type

from .exceptions import (
    DatabaseNotConfiguredError,
    NotFoundError,
    NotRegisteredError,
)
from .hashing import generate_vhash
from .storage import deserialize_dataframe, register_adapters, serialize_dataframe
from .variable import BaseVariable


# Global database instance (thread-local for safety)
_local = threading.local()


def configure_database(db_path: str | Path) -> "DatabaseManager":
    """
    Configure the global database connection.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        The DatabaseManager instance
    """
    _local.database = DatabaseManager(db_path)
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

        var = RotationMatrix(np.eye(3))
        vhash = var.save(db=db, subject=1, trial=1)

        loaded = RotationMatrix.load(db=db, subject=1, trial=1)
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
        """
        self.db_path = Path(db_path)
        self._registered_types: dict[str, Type[BaseVariable]] = {}

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
                vhash TEXT PRIMARY KEY,
                type_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                metadata JSON NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        self.connection.commit()

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

        # Create the data table
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vhash TEXT UNIQUE NOT NULL,
                schema_version INTEGER NOT NULL,
                metadata JSON NOT NULL,
                data BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create index on vhash for fast lookups
        self.connection.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_vhash
            ON {table_name}(vhash)
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

    def _check_registered(self, variable_class: Type[BaseVariable]) -> str:
        """
        Check if a variable type is registered, return table name.

        Raises:
            NotRegisteredError: If not registered
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
            raise NotRegisteredError(
                f"Variable type '{type_name}' is not registered. "
                f"Call db.register({type_name}) first."
            )

        # Update local cache
        self._registered_types[type_name] = variable_class
        return row["table_name"]

    def save(self, variable: BaseVariable, metadata: dict) -> str:
        """
        Save a variable to the database.

        If the exact same data+metadata already exists (same vhash),
        this is a no-op and returns the existing vhash.

        Args:
            variable: The variable instance to save
            metadata: Addressing metadata

        Returns:
            The vhash of the saved/existing data
        """
        table_name = self._check_registered(type(variable))
        type_name = variable.__class__.__name__

        # Generate vhash
        vhash = generate_vhash(
            class_name=type_name,
            schema_version=variable.schema_version,
            data=variable.data,
            metadata=metadata,
        )

        # Check if already exists (idempotent save)
        cursor = self.connection.execute(
            f"SELECT vhash FROM {table_name} WHERE vhash = ?",
            (vhash,),
        )
        if cursor.fetchone() is not None:
            return vhash  # Already saved

        # Serialize and save
        df = variable.to_db()
        data_blob = serialize_dataframe(df)
        metadata_json = json.dumps(metadata, sort_keys=True)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self.connection.execute(
            f"""
            INSERT INTO {table_name}
            (vhash, schema_version, metadata, data, created_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            (vhash, variable.schema_version, metadata_json, data_blob, now),
        )

        # Log to version history
        self.connection.execute(
            """
            INSERT INTO _version_log
            (vhash, type_name, table_name, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            (vhash, type_name, table_name, metadata_json, now),
        )

        self.connection.commit()
        return vhash

    def load(
        self,
        variable_class: Type[BaseVariable],
        metadata: dict,
        version: str = "latest",
    ) -> BaseVariable | list[BaseVariable]:
        """
        Load variable(s) matching the given metadata.

        Args:
            variable_class: The type to load
            metadata: Metadata to match (partial matching supported)
            version: "latest" or specific vhash

        Returns:
            Single instance if one match, list if multiple matches

        Raises:
            NotFoundError: If no matches found
        """
        table_name = self._check_registered(variable_class)

        if version != "latest" and version is not None:
            # Load specific version by vhash
            cursor = self.connection.execute(
                f"SELECT vhash, metadata, data FROM {table_name} WHERE vhash = ?",
                (version,),
            )
            row = cursor.fetchone()
            if row is None:
                raise NotFoundError(f"No data found with vhash '{version}'")

            return self._row_to_variable(variable_class, row)

        # Build query for metadata matching
        # Match all provided metadata keys
        conditions = []
        params = []
        for key, value in metadata.items():
            conditions.append(f"json_extract(metadata, '$.{key}') = ?")
            if isinstance(value, (dict, list)):
                params.append(json.dumps(value))
            else:
                params.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.connection.execute(
            f"""SELECT vhash, metadata, data FROM {table_name}
                WHERE {where_clause}
                ORDER BY created_at DESC""",
            params,
        )
        rows = cursor.fetchall()

        if not rows:
            raise NotFoundError(
                f"No {variable_class.__name__} found matching metadata: {metadata}"
            )

        # Return single or list based on count
        results = [self._row_to_variable(variable_class, row) for row in rows]

        if len(results) == 1:
            return results[0]
        return results

    def _row_to_variable(
        self,
        variable_class: Type[BaseVariable],
        row: sqlite3.Row,
    ) -> BaseVariable:
        """Convert a database row to a variable instance."""
        df = deserialize_dataframe(row["data"])
        data = variable_class.from_db(df)

        instance = variable_class(data)
        instance._vhash = row["vhash"]
        instance._metadata = json.loads(row["metadata"])

        return instance

    def list_versions(
        self,
        variable_class: Type[BaseVariable],
        **metadata,
    ) -> list[dict]:
        """
        List all versions matching the metadata.

        Args:
            variable_class: The type to query
            **metadata: Metadata to match

        Returns:
            List of dicts with vhash, metadata, created_at
        """
        table_name = self._check_registered(variable_class)

        conditions = []
        params = []
        for key, value in metadata.items():
            conditions.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.connection.execute(
            f"""SELECT vhash, metadata, created_at FROM {table_name}
                WHERE {where_clause}
                ORDER BY created_at DESC""",
            params,
        )

        return [
            {
                "vhash": row["vhash"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
            }
            for row in cursor.fetchall()
        ]

    def close(self):
        """Close the database connection."""
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
