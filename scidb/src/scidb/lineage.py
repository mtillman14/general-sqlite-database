"""Lineage extraction and storage for provenance tracking.

This module re-exports the thunk library's lineage functionality and provides
scidb-specific extensions for database integration.

Example:
    from scidb.lineage import extract_lineage, get_raw_value

    # After running a thunked pipeline:
    result = process_signal(raw_data, calibration)

    # Extract lineage for storage
    lineage = extract_lineage(result)
    print(lineage.function_name)  # 'process_signal'
    print(lineage.inputs)  # [{'name': 'arg_0', 'type': 'RawSignal', 'record_id': '...'}]

    # Get the raw value for storage
    raw_value = get_raw_value(result)  # The actual numpy array, DataFrame, etc.
"""

# Re-export from thunk library
from thunk import (
    LineageRecord,
    extract_lineage,
    find_unsaved_variables,
    get_raw_value,
    get_upstream_lineage,
)

__all__ = [
    "LineageRecord",
    "extract_lineage",
    "find_unsaved_variables",
    "get_raw_value",
    "get_upstream_lineage",
]
