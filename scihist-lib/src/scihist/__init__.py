"""SciHist: Lineage-tracked batch execution for scientific data pipelines.

This package adds Thunk-based lineage tracking on top of scidb.
It provides the same for_each() interface as scidb, but automatically
wraps functions in Thunk for provenance recording.

Example:
    from scihist import for_each, Fixed, configure_database
    from thunk import thunk

    configure_database("experiment.duckdb", ["subject", "session"])

    @thunk
    def process_data(raw, calibration):
        return raw * calibration

    for_each(
        process_data,
        inputs={"raw": RawData, "calibration": Fixed(Calibration, session="baseline")},
        outputs=[ProcessedData],
        subject=[1, 2, 3],
        session=["A", "B", "C"],
    )
"""

from .foreach import for_each, save
from .database import configure_database, find_by_lineage

# Re-export DB wrappers from scidb
from scidb import Fixed, Merge, ColumnSelection, ForEachConfig

# Re-export scifor helpers
from scifor import Col, set_schema, get_schema, PathInput

# Re-export thunk system
from thunk import thunk, Thunk, ThunkOutput, PipelineThunk

__version__ = "0.1.0"

__all__ = [
    # Core batch execution
    "for_each",
    "save",
    # Configuration
    "configure_database",
    # Lineage query
    "find_by_lineage",
    # DB wrappers
    "Fixed",
    "Merge",
    "ColumnSelection",
    "ForEachConfig",
    "PathInput",
    # Schema helpers
    "Col",
    "set_schema",
    "get_schema",
    # Thunk system
    "thunk",
    "Thunk",
    "ThunkOutput",
    "PipelineThunk",
]
