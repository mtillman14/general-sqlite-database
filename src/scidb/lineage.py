"""Lineage extraction and storage for provenance tracking.

This module provides utilities for extracting lineage information from
OutputThunks and converting it to a storable format.

Example:
    from scidb.lineage import extract_lineage, get_raw_value

    # After running a thunked pipeline:
    result = process_signal(raw_data, calibration)

    # Extract lineage for storage
    lineage = extract_lineage(result)
    print(lineage.function_name)  # 'process_signal'
    print(lineage.inputs)  # [{'name': 'arg_0', 'type': 'RawSignal', 'vhash': '...'}]

    # Get the raw value for storage
    raw_value = get_raw_value(result)  # The actual numpy array, DataFrame, etc.
"""

from dataclasses import dataclass, field
from typing import Any

from .thunk import OutputThunk


@dataclass
class LineageRecord:
    """
    Represents the provenance of a single variable.

    Attributes:
        function_name: Name of the function that produced the output
        function_hash: Hash of the function bytecode
        inputs: List of input descriptors (variables with vhashes)
        constants: List of constant input descriptors (literals)
    """

    function_name: str
    function_hash: str
    inputs: list[dict] = field(default_factory=list)
    constants: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "function_name": self.function_name,
            "function_hash": self.function_hash,
            "inputs": self.inputs,
            "constants": self.constants,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LineageRecord":
        """Create a LineageRecord from a dictionary."""
        return cls(
            function_name=data["function_name"],
            function_hash=data["function_hash"],
            inputs=data.get("inputs", []),
            constants=data.get("constants", []),
        )


def extract_lineage(output_thunk: OutputThunk) -> LineageRecord:
    """
    Extract lineage information from an OutputThunk.

    Traverses the input graph to capture:
    - Function name and hash
    - Input variables (with their vhashes if saved)
    - Constant values

    Args:
        output_thunk: The OutputThunk to extract lineage from

    Returns:
        LineageRecord containing the provenance information
    """
    from .hashing import canonical_hash

    pt = output_thunk.pipeline_thunk

    inputs = []
    constants = []

    for name, value in pt.inputs.items():
        if isinstance(value, OutputThunk):
            # Input came from another thunk
            inputs.append(
                {
                    "name": name,
                    "source_type": "thunk",
                    "source_function": value.pipeline_thunk.thunk.fcn.__name__,
                    "source_hash": value.hash,
                    "output_num": value.output_num,
                }
            )
        elif hasattr(value, "_vhash") and value._vhash is not None:
            # Input is a saved BaseVariable (using private attr)
            inputs.append(
                {
                    "name": name,
                    "source_type": "variable",
                    "type": type(value).__name__,
                    "vhash": value._vhash,
                    "metadata": value._metadata,
                }
            )
        elif hasattr(value, "vhash") and value.vhash is not None:
            # Input is a saved BaseVariable (using property)
            inputs.append(
                {
                    "name": name,
                    "source_type": "variable",
                    "type": type(value).__name__,
                    "vhash": value.vhash,
                    "metadata": getattr(value, "metadata", None),
                }
            )
        else:
            # Input is a constant/literal
            try:
                value_hash = canonical_hash(value)
            except Exception:
                value_hash = "unhashable"

            constants.append(
                {
                    "name": name,
                    "value_hash": value_hash,
                    "value_repr": repr(value)[:200],  # Truncate for storage
                    "value_type": type(value).__name__,
                }
            )

    return LineageRecord(
        function_name=pt.thunk.fcn.__name__,
        function_hash=pt.thunk.hash,
        inputs=inputs,
        constants=constants,
    )


def get_raw_value(data: Any) -> Any:
    """
    Unwrap OutputThunk to get raw value, or return as-is.

    This is used when saving a variable whose data might be wrapped
    in an OutputThunk from a thunked computation.

    Args:
        data: Either an OutputThunk or a raw value

    Returns:
        The raw value (unwrapped if OutputThunk, otherwise unchanged)
    """
    if isinstance(data, OutputThunk):
        return data.value
    return data


def get_lineage_chain(output_thunk: OutputThunk, max_depth: int = 100) -> list[LineageRecord]:
    """
    Extract the full lineage chain from an OutputThunk.

    Recursively traverses the input graph to capture the complete
    provenance history up to max_depth.

    Args:
        output_thunk: The OutputThunk to extract lineage from
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        List of LineageRecords from output back to inputs
    """
    if max_depth <= 0:
        return []

    chain = [extract_lineage(output_thunk)]

    # Recursively extract lineage from input OutputThunks
    for name, value in output_thunk.pipeline_thunk.inputs.items():
        if isinstance(value, OutputThunk):
            chain.extend(get_lineage_chain(value, max_depth - 1))

    return chain
