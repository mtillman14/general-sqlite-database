"""Thunk: Lazy Evaluation and Lineage Tracking for Python.

A lightweight library for building data processing pipelines with automatic
provenance tracking, inspired by Haskell's thunk concept.

Features:
- Lazy evaluation with automatic memoization
- Full lineage tracking for reproducibility
- Pluggable caching backends
- Zero dependencies on heavy frameworks

Example:
    from thunk import thunk

    @thunk(n_outputs=1)
    def process(data, factor):
        return data * factor

    result = process(input_data, 2.5)  # Returns ThunkOutput
    print(result.data)  # The computed value
    print(result.pipeline_thunk.inputs)  # Captured inputs for provenance

For multi-output functions:

    @thunk(n_outputs=2)
    def split(data):
        return data[:len(data)//2], data[len(data)//2:]

    first_half, second_half = split(my_data)

To enable caching:

    from thunk import configure_cache

    class MyCache:
        def get_cached(self, cache_key, n_outputs):
            # Return list of (data, id) tuples or None
            return None

    configure_cache(MyCache())
"""

from .core import (
    CacheBackend,
    ThunkOutput,
    PipelineThunk,
    Thunk,
    configure_cache,
    get_cache_backend,
    thunk,
)
from .hashing import canonical_hash
from .inputs import InputKind, ClassifiedInput, classify_input, is_trackable_variable
from .lineage import (
    LineageRecord,
    extract_lineage,
    find_unsaved_variables,
    get_raw_value,
    get_upstream_lineage,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Thunk",
    "PipelineThunk",
    "ThunkOutput",
    # Decorator
    "thunk",
    # Cache configuration
    "CacheBackend",
    "configure_cache",
    "get_cache_backend",
    # Input classification
    "InputKind",
    "ClassifiedInput",
    "classify_input",
    "is_trackable_variable",
    # Lineage
    "LineageRecord",
    "extract_lineage",
    "find_unsaved_variables",
    "get_raw_value",
    "get_upstream_lineage",
    # Hashing
    "canonical_hash",
]
