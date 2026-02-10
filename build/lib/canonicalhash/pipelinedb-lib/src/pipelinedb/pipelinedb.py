"""SQLite-based lineage persistence layer."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class PipelineDB:
    """
    SQLite-based lineage persistence layer.

    Stores computation lineage (provenance) separately from data storage.
    Uses record_id references to link to data in external storage (e.g., SciDuck).

    Supports two levels of provenance:

    **Schema-aware (instance provenance):** Each lineage record includes schema
    keys (e.g., subject, session) and content hashes, enabling queries like
    "what inputs at subject=1, session=1 produced this output?"

    **Schema-blind (pipeline structure):** Derived from schema-aware records by
    grouping on (function_name, output_type, input_types). Answers "how is the
    pipeline generally structured?" without reference to specific data instances.

    Example:
        db = PipelineDB("pipeline.db")

        db.save_lineage(
            output_record_id="abc123",
            output_type="ProcessedData",
            function_name="process_data",
            function_hash="ghi789",
            inputs=[{"name": "arg_0", "record_id": "xyz000", "type": "RawData"}],
            constants=[],
            lineage_hash="def456",
            schema_keys={"subject": "S01", "session": "1"},
            output_content_hash="contenthash123",
        )

        # Schema-aware: find by schema location
        records = db.find_by_schema(subject="S01")

        # Schema-blind: get pipeline structure
        structure = db.get_pipeline_structure()

        # Cache lookup (optionally scoped to schema keys)
        records = db.find_by_lineage_hash("def456", schema_keys={"subject": "S01"})
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize SQLite connection.

        Args:
            db_path: Path to the SQLite database file (created if doesn't exist)
        """
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the lineage table and indexes if they don't exist."""
        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                output_record_id TEXT UNIQUE NOT NULL,
                output_type TEXT NOT NULL,
                lineage_hash TEXT,
                function_name TEXT NOT NULL,
                function_hash TEXT NOT NULL,
                inputs TEXT NOT NULL,
                constants TEXT NOT NULL,
                user_id TEXT,
                schema_keys TEXT,
                output_content_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add columns if upgrading from older schema
        self._add_column_if_missing(cursor, "schema_keys", "TEXT")
        self._add_column_if_missing(cursor, "output_content_hash", "TEXT")

        # Create indexes for efficient lookups
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_lineage_hash ON lineage(lineage_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_output_type ON lineage(output_type)"
        )

        self._conn.commit()

    @staticmethod
    def _add_column_if_missing(cursor, column_name: str, column_type: str) -> None:
        """Add a column to the lineage table if it doesn't already exist."""
        cursor.execute("PRAGMA table_info(lineage)")
        existing = {row[1] for row in cursor.fetchall()}
        if column_name not in existing:
            cursor.execute(
                f"ALTER TABLE lineage ADD COLUMN {column_name} {column_type}"
            )

    def save_lineage(
        self,
        output_record_id: str,
        output_type: str,
        function_name: str,
        function_hash: str,
        inputs: list[dict[str, Any]],
        constants: list[dict[str, Any]],
        lineage_hash: str | None = None,
        user_id: str | None = None,
        schema_keys: dict[str, Any] | None = None,
        output_content_hash: str | None = None,
    ) -> None:
        """
        Save a lineage record (upsert).

        If a record with the same output_record_id already exists, all fields
        are overwritten with the new values.

        Args:
            output_record_id: Unique ID referencing data in external storage (e.g., SciDuck)
                             or "ephemeral:..." prefix for unsaved intermediates
            output_type: Variable class name (e.g., "ProcessedData")
            function_name: Name of the function that produced this output
            function_hash: Hash of the function's source code
            inputs: List of input specifications, each with record_id references
            constants: List of constant values used in the computation
            lineage_hash: Pre-computed hash of the full lineage (for cache lookups)
            user_id: Optional user ID for attribution
            schema_keys: Optional dict of schema keys (e.g., {"subject": "S01", "session": "1"})
            output_content_hash: Optional content hash of the output data
        """
        cursor = self._conn.cursor()

        cursor.execute(
            """
            INSERT INTO lineage
            (output_record_id, output_type, lineage_hash, function_name, function_hash,
             inputs, constants, user_id, schema_keys, output_content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (output_record_id) DO UPDATE SET
                output_type = excluded.output_type,
                lineage_hash = excluded.lineage_hash,
                function_name = excluded.function_name,
                function_hash = excluded.function_hash,
                inputs = excluded.inputs,
                constants = excluded.constants,
                user_id = excluded.user_id,
                schema_keys = excluded.schema_keys,
                output_content_hash = excluded.output_content_hash,
                created_at = excluded.created_at
            """,
            [
                output_record_id,
                output_type,
                lineage_hash,
                function_name,
                function_hash,
                json.dumps(inputs, sort_keys=True),
                json.dumps(constants, sort_keys=True),
                user_id,
                json.dumps(schema_keys, sort_keys=True) if schema_keys else None,
                output_content_hash,
                datetime.now().isoformat(),
            ],
        )

        self._conn.commit()

    def find_by_lineage_hash(
        self,
        lineage_hash: str,
        schema_keys: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | None:
        """
        Find outputs by lineage hash.

        Used for cache lookups: given a computation (function + inputs),
        find any previously computed outputs.

        Args:
            lineage_hash: The hash of the computation to look up
            schema_keys: Optional schema keys to scope the search

        Returns:
            List of matching record dicts, or None if no matches found
        """
        cursor = self._conn.cursor()

        sql = """SELECT output_record_id, output_type, function_name, function_hash,
                        inputs, constants, user_id, schema_keys, output_content_hash,
                        created_at
                 FROM lineage
                 WHERE lineage_hash = ?"""
        params: list[Any] = [lineage_hash]

        if schema_keys:
            for key, value in schema_keys.items():
                sql += f" AND json_extract(schema_keys, '$.{key}') = ?"
                params.append(str(value))

        cursor.execute(sql, params)

        rows = cursor.fetchall()
        if not rows:
            return None

        return [self._row_to_dict(row) for row in rows]

    def get_lineage(self, output_record_id: str) -> dict[str, Any] | None:
        """
        Get lineage for a specific output.

        Args:
            output_record_id: The record ID to look up

        Returns:
            Dict with lineage info, or None if not found
        """
        cursor = self._conn.cursor()

        cursor.execute(
            """SELECT output_record_id, output_type, lineage_hash, function_name,
                      function_hash, inputs, constants, user_id, schema_keys,
                      output_content_hash, created_at
               FROM lineage
               WHERE output_record_id = ?""",
            [output_record_id],
        )

        row = cursor.fetchone()
        if not row:
            return None

        result = self._row_to_dict(row)
        result["lineage_hash"] = row["lineage_hash"]
        return result

    def find_by_schema(self, **schema_keys) -> list[dict[str, Any]]:
        """
        Find all lineage records matching the given schema keys.

        This enables schema-aware provenance queries like "show me all
        computations for subject=S01, session=1".

        Args:
            **schema_keys: Schema key filters (e.g., subject="S01", session="1")

        Returns:
            List of matching lineage record dicts (may be empty)
        """
        cursor = self._conn.cursor()

        sql = """SELECT output_record_id, output_type, lineage_hash, function_name,
                        function_hash, inputs, constants, user_id, schema_keys,
                        output_content_hash, created_at
                 FROM lineage
                 WHERE schema_keys IS NOT NULL"""
        params: list[Any] = []

        for key, value in schema_keys.items():
            sql += f" AND json_extract(schema_keys, '$.{key}') = ?"
            params.append(str(value))

        sql += " ORDER BY created_at DESC"
        cursor.execute(sql, params)

        rows = cursor.fetchall()
        results = []
        for row in rows:
            result = self._row_to_dict(row)
            result["lineage_hash"] = row["lineage_hash"]
            results.append(result)
        return results

    def get_pipeline_structure(self) -> list[dict[str, Any]]:
        """
        Get the abstract pipeline structure (schema-blind view).

        Returns unique (function_name, function_hash, output_type, input_types)
        combinations, describing how variable types flow through functions
        without reference to specific data instances or schema locations.

        Returns:
            List of dicts with keys: function_name, function_hash, output_type,
            input_types (list of type names)
        """
        cursor = self._conn.cursor()

        cursor.execute(
            """SELECT DISTINCT function_name, function_hash, output_type, inputs
               FROM lineage
               ORDER BY function_name"""
        )

        seen = set()
        structure = []
        for row in cursor.fetchall():
            inputs_data = json.loads(row["inputs"])
            input_types = tuple(
                sorted(inp.get("type", inp.get("value_type", "constant"))
                       for inp in inputs_data)
            )
            key = (row["function_name"], row["function_hash"], row["output_type"], input_types)
            if key not in seen:
                seen.add(key)
                structure.append({
                    "function_name": row["function_name"],
                    "function_hash": row["function_hash"],
                    "output_type": row["output_type"],
                    "input_types": list(input_types),
                })

        return structure

    def save_ephemeral(
        self,
        ephemeral_id: str,
        variable_type: str,
        function_name: str,
        function_hash: str,
        inputs: list[dict[str, Any]],
        constants: list[dict[str, Any]],
        user_id: str | None = None,
        schema_keys: dict[str, Any] | None = None,
    ) -> None:
        """
        Save ephemeral lineage (no data stored in external storage).

        Ephemeral entries track the computation graph for unsaved intermediate
        variables. The ephemeral_id should have an "ephemeral:" prefix.

        This method is idempotent: if a record with the same ephemeral_id
        already exists, it is left unchanged (no update is performed).

        Args:
            ephemeral_id: ID with "ephemeral:" prefix (e.g., "ephemeral:abc123")
            variable_type: Variable class name
            function_name: Name of the function that produced this output
            function_hash: Hash of the function's source code
            inputs: List of input specifications
            constants: List of constant values
            user_id: Optional user ID
            schema_keys: Optional dict of schema keys
        """
        # Check if already exists
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT 1 FROM lineage WHERE output_record_id = ?",
            [ephemeral_id],
        )
        if cursor.fetchone():
            return

        self.save_lineage(
            output_record_id=ephemeral_id,
            output_type=variable_type,
            function_name=function_name,
            function_hash=function_hash,
            inputs=inputs,
            constants=constants,
            lineage_hash=None,  # Ephemeral entries don't need lineage hash
            user_id=user_id,
            schema_keys=schema_keys,
        )

    def has_lineage(self, output_record_id: str) -> bool:
        """
        Check if a record has lineage information.

        Args:
            output_record_id: The record ID to check

        Returns:
            True if lineage exists, False otherwise
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT 1 FROM lineage WHERE output_record_id = ?",
            [output_record_id],
        )
        return cursor.fetchone() is not None

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a lineage dict."""
        schema_keys_raw = row["schema_keys"]
        return {
            "output_record_id": row["output_record_id"],
            "output_type": row["output_type"],
            "function_name": row["function_name"],
            "function_hash": row["function_hash"],
            "inputs": json.loads(row["inputs"]),
            "constants": json.loads(row["constants"]),
            "user_id": row["user_id"],
            "schema_keys": json.loads(schema_keys_raw) if schema_keys_raw else None,
            "output_content_hash": row["output_content_hash"],
            "created_at": row["created_at"],
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PipelineDB":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing the database connection."""
        self.close()
