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
from .hashing import generate_vhash, canonical_hash
from .preview import generate_preview
from .storage import deserialize_dataframe, register_adapters, serialize_dataframe
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

        # Content-addressed data storage (deduplicated)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS _data (
                content_hash TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT
            )
        """
        )

        # Lineage tracking table
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS _lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                output_vhash TEXT UNIQUE NOT NULL,
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
            ON _lineage(output_vhash)
        """
        )

        # Computation cache table - maps (function + inputs + output_num) -> output vhash
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS _computation_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT NOT NULL,
                output_num INTEGER NOT NULL DEFAULT 0,
                function_name TEXT NOT NULL,
                function_hash TEXT NOT NULL,
                output_type TEXT NOT NULL,
                output_vhash TEXT NOT NULL,
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

        # Create the metadata table (data is stored in _data table)
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vhash TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                lineage_hash TEXT,
                schema_version INTEGER NOT NULL,
                metadata JSON NOT NULL,
                preview TEXT,
                user_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_hash) REFERENCES _data(content_hash)
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
    ) -> str:
        """
        Save a variable to the database.

        Data is stored in the shared _data table (content-addressed, deduplicated).
        Metadata entries are stored in the variable-specific table.

        If the exact same data+metadata already exists (same vhash),
        this is a no-op and returns the existing vhash.

        Args:
            variable: The variable instance to save
            metadata: Addressing metadata
            lineage: Optional lineage record if data came from a thunk
            lineage_hash: Optional pre-computed lineage hash for cache key computation

        Returns:
            The vhash of the saved/existing data
        """
        table_name = self._ensure_registered(type(variable))
        type_name = variable.__class__.__name__
        user_id = get_user_id()

        # Serialize data and compute content hash
        df = variable.to_db()
        data_blob = serialize_dataframe(df)
        content_hash = canonical_hash(variable.data)

        # Generate vhash (using content_hash, not raw data)
        vhash = generate_vhash(
            class_name=type_name,
            schema_version=variable.schema_version,
            content_hash=content_hash,
            metadata=metadata,
        )

        # Check if metadata entry already exists (idempotent save)
        cursor = self.connection.execute(
            f"SELECT vhash FROM {table_name} WHERE vhash = ?",
            (vhash,),
        )
        if cursor.fetchone() is not None:
            return vhash  # Already saved

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Store data in _data table if not already there (deduplicated)
        cursor = self.connection.execute(
            "SELECT content_hash FROM _data WHERE content_hash = ?",
            (content_hash,),
        )
        if cursor.fetchone() is None:
            self.connection.execute(
                """
                INSERT INTO _data (content_hash, data, created_at, created_by)
                VALUES (?, ?, ?, ?)
            """,
                (content_hash, data_blob, now, user_id),
            )

        # Store metadata entry in variable table
        metadata_json = json.dumps(metadata, sort_keys=True)
        preview_str = generate_preview(df)

        self.connection.execute(
            f"""
            INSERT INTO {table_name}
            (vhash, content_hash, lineage_hash, schema_version, metadata, preview, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (vhash, content_hash, lineage_hash, variable.schema_version, metadata_json, preview_str, user_id, now),
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

        # Save lineage if provided
        if lineage is not None:
            self._save_lineage(vhash, type_name, lineage, now, user_id)

        self.connection.commit()
        return vhash

    def _save_lineage(
        self,
        output_vhash: str,
        output_type: str,
        lineage: "LineageRecord",
        timestamp: str,
        user_id: str | None = None,
    ) -> None:
        """Save lineage record for a variable."""
        self.connection.execute(
            """
            INSERT OR REPLACE INTO _lineage
            (output_vhash, output_type, function_name, function_hash,
             inputs, constants, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                output_vhash,
                output_type,
                lineage.function_name,
                lineage.function_hash,
                json.dumps(lineage.inputs),
                json.dumps(lineage.constants),
                user_id,
                timestamp,
            ),
        )

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
            NotRegisteredError: If this type has never been saved
        """
        table_name = self._ensure_registered(variable_class, auto_register=False)

        if version != "latest" and version is not None:
            # Load specific version by vhash
            cursor = self.connection.execute(
                f"""SELECT v.vhash, v.content_hash, v.lineage_hash, v.metadata, d.data
                    FROM {table_name} v
                    JOIN _data d ON v.content_hash = d.content_hash
                    WHERE v.vhash = ?""",
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
            conditions.append(f"json_extract(v.metadata, '$.{key}') = ?")
            if isinstance(value, (dict, list)):
                params.append(json.dumps(value))
            else:
                params.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.connection.execute(
            f"""SELECT v.vhash, v.content_hash, v.lineage_hash, v.metadata, d.data
                FROM {table_name} v
                JOIN _data d ON v.content_hash = d.content_hash
                WHERE {where_clause}
                ORDER BY v.created_at DESC""",
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
        instance._content_hash = row["content_hash"]
        instance._lineage_hash = row["lineage_hash"]  # May be None for raw data

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

        Raises:
            NotRegisteredError: If this type has never been saved
        """
        table_name = self._ensure_registered(variable_class, auto_register=False)

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
            version: Specific vhash, or None to use metadata lookup
            **metadata: Metadata to match (if version not specified)

        Returns:
            Dict with function_name, function_hash, inputs, constants
            or None if no lineage recorded (data wasn't from a thunk)
        """
        # First, find the vhash
        if version:
            vhash = version
        else:
            var = self.load(variable_class, metadata)
            if isinstance(var, list):
                var = var[0]  # Latest
            vhash = var.vhash

        cursor = self.connection.execute(
            """SELECT function_name, function_hash, inputs, constants
               FROM _lineage WHERE output_vhash = ?""",
            (vhash,),
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
            version: Specific vhash, or None to use metadata lookup
            **metadata: Metadata to match (if version not specified)

        Returns:
            List of dicts with vhash, type, function for each derived variable
        """
        # First, find the vhash
        if version:
            vhash = version
        else:
            var = self.load(variable_class, metadata)
            if isinstance(var, list):
                var = var[0]
            vhash = var.vhash

        # Search for this vhash in any lineage inputs
        cursor = self.connection.execute(
            """SELECT output_vhash, output_type, function_name, inputs
               FROM _lineage
               WHERE EXISTS (
                   SELECT 1 FROM json_each(inputs)
                   WHERE json_extract(value, '$.vhash') = ?
               )""",
            (vhash,),
        )

        return [
            {
                "vhash": row["output_vhash"],
                "type": row["output_type"],
                "function": row["function_name"],
            }
            for row in cursor.fetchall()
        ]

    def has_lineage(self, vhash: str) -> bool:
        """
        Check if a variable has lineage information.

        Args:
            vhash: The version hash to check

        Returns:
            True if lineage exists, False otherwise
        """
        cursor = self.connection.execute(
            "SELECT 1 FROM _lineage WHERE output_vhash = ?",
            (vhash,),
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
            version: Specific vhash, or None to use metadata lookup
            max_depth: Maximum recursion depth to prevent infinite loops
            **metadata: Metadata to match (if version not specified)

        Returns:
            Nested dict representing the full computation graph:
            {
                "type": "FinalResult",
                "vhash": "abc123...",
                "function": "compute_stats",
                "function_hash": "def456...",
                "constants": [{"name": "threshold", "value": 0.5}],
                "inputs": [
                    {
                        "type": "FilteredData",
                        "vhash": "...",
                        "function": "filter",
                        ...
                    }
                ]
            }
        """
        # Find the vhash
        if version:
            vhash = version
            type_name = variable_class.__name__
        else:
            var = self.load(variable_class, metadata)
            if isinstance(var, list):
                var = var[0]  # Latest
            vhash = var.vhash
            type_name = variable_class.__name__

        return self._build_lineage_tree(vhash, type_name, max_depth, set())

    def _build_lineage_tree(
        self,
        vhash: str,
        type_name: str,
        max_depth: int,
        visited: set,
    ) -> dict:
        """Recursively build the lineage tree for a vhash."""
        # Prevent infinite loops
        if max_depth <= 0 or vhash in visited:
            return {
                "type": type_name,
                "vhash": vhash,
                "truncated": True,
            }

        visited = visited | {vhash}

        # Get lineage for this vhash
        cursor = self.connection.execute(
            """SELECT function_name, function_hash, inputs, constants
               FROM _lineage WHERE output_vhash = ?""",
            (vhash,),
        )
        row = cursor.fetchone()

        if row is None:
            # No lineage - this is a source/leaf node
            return {
                "type": type_name,
                "vhash": vhash,
                "function": None,
                "source": "manual",
            }

        inputs_json = json.loads(row["inputs"])
        constants_json = json.loads(row["constants"])

        # Recursively process inputs
        processed_inputs = []
        for inp in inputs_json:
            if inp.get("source_type") == "variable" and "vhash" in inp:
                # Saved variable - recurse
                child = self._build_lineage_tree(
                    inp["vhash"],
                    inp.get("type", "Unknown"),
                    max_depth - 1,
                    visited,
                )
                child["input_name"] = inp.get("name")
                processed_inputs.append(child)
            elif inp.get("source_type") == "thunk":
                # Output from another thunk - try to find by hash
                # This is trickier since we need to find the vhash
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
            "vhash": vhash,
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
            version: Specific vhash, or None to use metadata lookup
            **metadata: Metadata to match (if version not specified)

        Returns:
            A formatted string showing the computation graph.

        Example output:
            FinalResult (vhash: abc123...)
            └── compute_stats [hash: def456...]
                ├── constants: threshold=0.5
                └── inputs:
                    └── FilteredData (vhash: ghi789...)
                        └── filter [hash: jkl012...]
                            ├── constants: cutoff=10
                            └── inputs:
                                └── RawData (vhash: mno345...)
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
        vhash_short = node.get("vhash", "???")[:12]
        type_name = node.get("type", "Unknown")

        if node.get("truncated"):
            lines.append(f"{prefix}{connector}{type_name} (vhash: {vhash_short}...) [truncated]")
            return

        lines.append(f"{prefix}{connector}{type_name} (vhash: {vhash_short}...)")

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
            """SELECT output_vhash, output_type FROM _computation_cache
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
            return self.load(variable_class, {}, version=row["output_vhash"])
        except Exception:
            # Cache entry exists but data is missing - stale cache
            return None

    def cache_computation(
        self,
        cache_key: str,
        function_name: str,
        function_hash: str,
        output_type: str,
        output_vhash: str,
        output_num: int = 0,
    ) -> None:
        """
        Store a computation result in the cache.

        Args:
            cache_key: The cache key (hash of function + inputs)
            function_name: Name of the function
            function_hash: Hash of the function bytecode
            output_type: Name of the output variable class
            output_vhash: Version hash of the saved output
            output_num: Output index for multi-output functions (default 0)
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self.connection.execute(
            """
            INSERT OR REPLACE INTO _computation_cache
            (cache_key, output_num, function_name, function_hash, output_type, output_vhash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (cache_key, output_num, function_name, function_hash, output_type, output_vhash, now),
        )
        self.connection.commit()

    def load_by_vhash(
        self,
        variable_class: Type[BaseVariable],
        vhash: str,
    ) -> BaseVariable | None:
        """
        Load a variable by its exact vhash.

        Args:
            variable_class: The type to load
            vhash: The version hash

        Returns:
            The variable instance, or None if not found
        """
        try:
            return self.load(variable_class, {}, version=vhash)
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
            List of (data, vhash) tuples if ALL outputs found and loadable,
            None otherwise. For n_outputs=1, still returns a list with one tuple.
        """
        cursor = self.connection.execute(
            """SELECT output_num, output_type, output_vhash FROM _computation_cache
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
            output_vhash = row["output_vhash"]

            # Look up class from database registry first
            var_class = self._registered_types.get(output_type)
            if var_class is None:
                # Try global registry and auto-register
                var_class = BaseVariable.get_subclass_by_name(output_type)
                if var_class is None:
                    return None  # Class not found anywhere
                self.register(var_class)

            # Load the variable by vhash
            var = self.load_by_vhash(var_class, output_vhash)
            if var is None:
                return None  # Data was deleted or corrupted

            results.append((var.data, output_vhash))

        return results

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
        concatenated with a 'vhash' column to distinguish them.

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

        # Load matching variables
        results = self.load(variable_class, metadata)
        if not isinstance(results, list):
            results = [results]

        # Build combined DataFrame
        all_dfs = []
        for var in results:
            df = variable_class(var.data).to_db()
            df["_vhash"] = var.vhash
            # Add metadata columns
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

        # Build query
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
            f"""SELECT vhash, metadata, preview, created_at FROM {table_name}
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
            vhash_short = row["vhash"][:12]
            meta = json.loads(row["metadata"])
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
            preview = row["preview"] or "(no preview)"

            lines.append(f"vhash: {vhash_short}...")
            if meta_str:
                lines.append(f"  metadata: {meta_str}")
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
