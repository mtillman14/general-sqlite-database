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

from typing import TYPE_CHECKING

# Re-export from thunk library
from thunk import (
    LineageRecord,
    extract_lineage,
    find_unsaved_variables,
    get_lineage_chain,
    get_raw_value,
    get_upstream_lineage,
)
from thunk import ThunkOutput

if TYPE_CHECKING:
    from .database import DatabaseManager
    from thunk import PipelineThunk


def check_cache(
    pipeline_thunk: "PipelineThunk",
    variable_class: type,
    db: "DatabaseManager | None" = None,
) -> "ThunkOutput | None":
    """
    Check if a computation result is cached in the scidb database.

    This allows skipping execution when a result already exists.

    Example:
        @thunk(n_outputs=1)
        def process(data):
            return data * 2

        # Create pipeline thunk without executing
        pt = PipelineThunk(process, input_data)
        cached = check_cache(pt, MyVar, db=db)
        if cached:
            result = cached
        else:
            result = process(input_data)
            MyVar.save(result, db=db, ...)

    Args:
        pipeline_thunk: The PipelineThunk to check cache for
        variable_class: The expected output type
        db: Database to check (uses global if not provided)

    Returns:
        ThunkOutput with cached value, or None if not cached
    """
    if db is None:
        from .database import get_database
        try:
            db = get_database()
        except Exception:
            return None

    cache_key = pipeline_thunk.compute_cache_key()
    cached_var = db.get_cached_computation(cache_key, variable_class)

    if cached_var is None:
        return None

    # Create an ThunkOutput representing the cached result
    return ThunkOutput(
        pipeline_thunk=pipeline_thunk,
        output_num=0,
        is_complete=True,
        data=cached_var.data,
        was_cached=True,
        cached_id=cached_var.record_id,
    )


__all__ = [
    "LineageRecord",
    "extract_lineage",
    "find_unsaved_variables",
    "get_lineage_chain",
    "get_raw_value",
    "get_upstream_lineage",
    "check_cache",
]
