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

# Re-export everything from the thunk library
from thunk import (
    CacheBackend,
    ThunkOutput,
    PipelineThunk,
    Thunk,
    configure_cache,
    get_cache_backend,
    thunk,
)

__all__ = [
    "Thunk",
    "PipelineThunk",
    "ThunkOutput",
    "thunk",
    "CacheBackend",
    "configure_cache",
    "get_cache_backend",
]
