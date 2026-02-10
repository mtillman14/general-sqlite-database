"""PipelineDB: SQLite-based lineage persistence layer for data pipelines.

This package provides a lightweight SQLite database for storing computation
lineage (provenance) information separately from the actual data.

Example:
    from pipelinedb import PipelineDB

    db = PipelineDB("pipeline.db")

    # Save lineage for a computation
    db.save_lineage(
        output_record_id="abc123",
        output_type="ProcessedData",
        function_name="process_data",
        function_hash="ghi789",
        inputs=[{"name": "arg_0", "record_id": "xyz000", "type": "RawData"}],
        constants=[],
        lineage_hash="def456",
    )

    # Look up by lineage hash (for cache hits)
    records = db.find_by_lineage_hash("def456")
"""

from .pipelinedb import PipelineDB

__version__ = "0.1.0"

__all__ = [
    "PipelineDB",
]
