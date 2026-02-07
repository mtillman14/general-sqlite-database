"""Thunk system for automatic lineage tracking.

This module re-exports the thunk library and provides scidb-specific integration.
The core thunk functionality is provided by the standalone `thunk` package.

Example:
    @thunk
    def process_signal(raw: np.ndarray, cal_factor: float) -> np.ndarray:
        return raw * cal_factor

    result = process_signal(raw_data, 2.5)  # Returns ThunkOutput
    print(result.data)  # The actual computed data
    print(result.pipeline_thunk.inputs)  # Captured inputs for lineage
"""

# Re-export from thunk library
from thunk import (
    ThunkOutput,
    PipelineThunk,
    Thunk,
    thunk,
)

# Re-export query layer from scidb
from .query_by_metadata import QueryByMetadata

__all__ = [
    "Thunk",
    "PipelineThunk",
    "ThunkOutput",
    "thunk",
    "QueryByMetadata",
]
