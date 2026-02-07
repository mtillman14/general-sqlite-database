"""Thunk: Lineage Tracking for Python.

A lightweight library for building data processing pipelines with automatic
provenance tracking, inspired by Haskell's thunk concept.

Features:
- Full lineage tracking for reproducibility
- Automatic input capture and output wrapping
- Zero dependencies on heavy frameworks

Example:
    from thunk import thunk

    @thunk
    def process(data, factor):
        return data * factor

    result = process(input_data, 2.5)  # Returns ThunkOutput
    print(result.data)  # The computed value
    print(result.pipeline_thunk.inputs)  # Captured inputs for provenance

For multi-output functions, use unpack_output=True:

    @thunk(unpack_output=True)
    def split(data):
        return data[:len(data)//2], data[len(data)//2:]

    first_half, second_half = split(my_data)  # Each is a ThunkOutput
"""

from .core import (
    ThunkOutput,
    PipelineThunk,
    Thunk,
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
