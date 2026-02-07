"""Metadata-driven query layer over SciDuck.

This module provides QueryByMetadata, a query layer that uses SciDB metadata
to enable powerful queries against the SciDuck data store.
"""

from typing import TYPE_CHECKING, Any

from .exceptions import NotFoundError

if TYPE_CHECKING:
    from thunk import PipelineThunk
    from .database import DatabaseManager


class QueryByMetadata:
    """
    Metadata-driven query layer over SciDuck.

    Uses SciDB metadata (lineage, relationships) to enable rich queries
    against the barebones SciDuck data store.

    Example:
        from scidb.query_by_metadata import QueryByMetadata
        from scidb.database import configure_database
        from thunk import Thunk

        db = configure_database("db.duckdb", schema_keys=["subject", "session"])
        Thunk.query = QueryByMetadata(db)
    """

    def __init__(self, db: "DatabaseManager"):
        """
        Initialize the query layer.

        Args:
            db: DatabaseManager instance (provides access to metadata and SciDuck)
        """
        self.db = db

    def find_by_lineage(self, pipeline_thunk: "PipelineThunk") -> list[Any] | None:
        """
        Find output values by computation lineage.

        Given a PipelineThunk (function + inputs), finds any previously
        computed outputs that match.

        Args:
            pipeline_thunk: The computation to look up

        Returns:
            List of output values if found, None otherwise
        """
        lineage_hash = pipeline_thunk.compute_lineage_hash()

        rows = self.db._duck._fetchall(
            """SELECT output_record_id, output_type
               FROM _lineage
               WHERE lineage_hash = ?""",
            [lineage_hash],
        )

        if not rows:
            return None

        results = []
        for output_record_id, output_type in rows:
            var_class = self._get_variable_class(output_type)
            if var_class is None:
                return None

            try:
                var = self.db.load(var_class, {}, version=output_record_id)
                results.append(var.data)
            except (KeyError, NotFoundError):
                # Record not found in database
                return None

        return results if results else None

    def _get_variable_class(self, type_name: str):
        """Get a variable class by name."""
        from .variable import BaseVariable

        if type_name in self.db._registered_types:
            return self.db._registered_types[type_name]

        return BaseVariable.get_subclass_by_name(type_name)
