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
        )

        # Cache lookup
        records = db.find_by_lineage_hash("def456")
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for efficient lookups
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_lineage_hash ON lineage(lineage_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_output_type ON lineage(output_type)"
        )

        self._conn.commit()

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
        """
        cursor = self._conn.cursor()

        cursor.execute(
            """
            INSERT INTO lineage
            (output_record_id, output_type, lineage_hash, function_name, function_hash,
             inputs, constants, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (output_record_id) DO UPDATE SET
                output_type = excluded.output_type,
                lineage_hash = excluded.lineage_hash,
                function_name = excluded.function_name,
                function_hash = excluded.function_hash,
                inputs = excluded.inputs,
                constants = excluded.constants,
                user_id = excluded.user_id,
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
                datetime.now().isoformat(),
            ],
        )

        self._conn.commit()

    def find_by_lineage_hash(self, lineage_hash: str) -> list[dict[str, Any]] | None:
        """
        Find outputs by lineage hash.

        Used for cache lookups: given a computation (function + inputs),
        find any previously computed outputs.

        Args:
            lineage_hash: The hash of the computation to look up

        Returns:
            List of matching record dicts (with keys: output_record_id,
            output_type, function_name, function_hash, inputs, constants,
            user_id, created_at), or None if no matches found
        """
        cursor = self._conn.cursor()

        cursor.execute(
            """SELECT output_record_id, output_type, function_name, function_hash,
                      inputs, constants, user_id, created_at
               FROM lineage
               WHERE lineage_hash = ?""",
            [lineage_hash],
        )

        rows = cursor.fetchall()
        if not rows:
            return None

        results = []
        for row in rows:
            results.append({
                "output_record_id": row["output_record_id"],
                "output_type": row["output_type"],
                "function_name": row["function_name"],
                "function_hash": row["function_hash"],
                "inputs": json.loads(row["inputs"]),
                "constants": json.loads(row["constants"]),
                "user_id": row["user_id"],
                "created_at": row["created_at"],
            })

        return results

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
                      function_hash, inputs, constants, user_id, created_at
               FROM lineage
               WHERE output_record_id = ?""",
            [output_record_id],
        )

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "output_record_id": row["output_record_id"],
            "output_type": row["output_type"],
            "lineage_hash": row["lineage_hash"],
            "function_name": row["function_name"],
            "function_hash": row["function_hash"],
            "inputs": json.loads(row["inputs"]),
            "constants": json.loads(row["constants"]),
            "user_id": row["user_id"],
            "created_at": row["created_at"],
        }

    def save_ephemeral(
        self,
        ephemeral_id: str,
        variable_type: str,
        function_name: str,
        function_hash: str,
        inputs: list[dict[str, Any]],
        constants: list[dict[str, Any]],
        user_id: str | None = None,
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
