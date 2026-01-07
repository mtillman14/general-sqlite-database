"""Lineage extraction for provenance tracking.

This module provides utilities for extracting lineage information from
OutputThunks and converting it to a storable format.

Example:
    from thunk import thunk
    from thunk.lineage import extract_lineage, get_raw_value

    @thunk(n_outputs=1)
    def process_signal(raw_data, calibration):
        return raw_data * calibration

    result = process_signal(data, 2.5)

    # Extract lineage for storage
    lineage = extract_lineage(result)
    print(lineage.function_name)  # 'process_signal'
    print(lineage.inputs)  # Input descriptors
    print(lineage.constants)  # Constant values like 2.5

    # Get the raw value for storage
    raw_value = get_raw_value(result)  # The actual computed array
"""

from dataclasses import dataclass, field
from typing import Any

from .core import OutputThunk, PipelineThunk
from .hashing import canonical_hash


@dataclass
class LineageRecord:
    """
    Represents the provenance of a single computed value.

    Attributes:
        function_name: Name of the function that produced the output
        function_hash: Hash of the function bytecode
        inputs: List of input descriptors (variables with identifiers)
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
    - Input variables (with their identifiers if saved)
    - Constant values

    Args:
        output_thunk: The OutputThunk to extract lineage from

    Returns:
        LineageRecord containing the provenance information
    """
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
        elif _is_trackable_variable(value):
            # Input is a trackable variable (e.g., scidb BaseVariable)
            vhash = getattr(value, "_vhash", None) or getattr(value, "vhash", None)
            metadata = getattr(value, "_metadata", None) or getattr(
                value, "metadata", None
            )

            if vhash is not None:
                inputs.append(
                    {
                        "name": name,
                        "source_type": "variable",
                        "type": type(value).__name__,
                        "vhash": vhash,
                        "metadata": metadata,
                    }
                )
            else:
                # Unsaved variable - treat as constant
                try:
                    data = getattr(value, "data", value)
                    value_hash = canonical_hash(data)
                except Exception:
                    value_hash = "unhashable"

                constants.append(
                    {
                        "name": name,
                        "value_hash": value_hash,
                        "value_repr": repr(value)[:200],
                        "value_type": type(value).__name__,
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
        The raw data (unwrapped if OutputThunk, otherwise unchanged)
    """
    if isinstance(data, OutputThunk):
        return data.data
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


def _is_trackable_variable(obj: Any) -> bool:
    """Check if obj is a trackable variable with lineage support."""
    return (
        hasattr(obj, "data")
        and hasattr(obj, "_vhash")
        and hasattr(obj, "to_db")
        and hasattr(obj, "from_db")
    )
