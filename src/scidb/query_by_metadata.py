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

    Uses PipelineDB (SQLite) for lineage lookups and SciDuck (DuckDB)
    for data retrieval. This bridges the two-database architecture.

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
            db: DatabaseManager instance (provides access to PipelineDB and SciDuck)
        """
        self.db = db

    def find_by_lineage(self, pipeline_thunk: "PipelineThunk") -> list[Any] | None:
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
        records = self.db._pipeline_db.find_by_lineage_hash(lineage_hash)
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
                var = self.db.load(var_class, {}, version=output_record_id)
                results.append(var.data)
            except (KeyError, NotFoundError):
                # Record not found in SciDuck
                return None

        return results if results else None

    def _get_variable_class(self, type_name: str):
        """Get a variable class by name."""
        from .variable import BaseVariable

        if type_name in self.db._registered_types:
            return self.db._registered_types[type_name]

        return BaseVariable.get_subclass_by_name(type_name)
