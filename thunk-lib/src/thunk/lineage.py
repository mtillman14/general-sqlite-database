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
                # Saved variable - can be dereferenced later
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
                # Unsaved variable - still an input, but no vhash to reference
                inner_data = getattr(value, "data", None)
                if isinstance(inner_data, OutputThunk):
                    # Unsaved variable wrapping an OutputThunk - can trace lineage
                    inputs.append(
                        {
                            "name": name,
                            "source_type": "unsaved_variable",
                            "type": type(value).__name__,
                            "inner_source": "thunk",
                            "source_function": inner_data.pipeline_thunk.thunk.fcn.__name__,
                            "source_hash": inner_data.hash,
                            "output_num": inner_data.output_num,
                        }
                    )
                else:
                    # Unsaved variable with raw data - store content hash
                    try:
                        value_hash = canonical_hash(inner_data if inner_data is not None else value)
                    except Exception:
                        value_hash = "unhashable"

                    inputs.append(
                        {
                            "name": name,
                            "source_type": "unsaved_variable",
                            "type": type(value).__name__,
                            "content_hash": value_hash,
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


def find_unsaved_variables(
    output_thunk: OutputThunk,
    max_depth: int = 100,
) -> list[tuple[Any, str]]:
    """
    Find all unsaved BaseVariables in the upstream chain of an OutputThunk.

    Traverses the in-memory lineage chain to find any BaseVariable that
    hasn't been saved (has no vhash). This is used by strict lineage mode
    to validate that all intermediates are saved.

    Args:
        output_thunk: The OutputThunk to start traversal from
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        List of (variable, path) tuples where:
        - variable: The unsaved BaseVariable instance
        - path: String describing how we got there (e.g., "filter() -> arg_0")
    """
    unsaved = []
    visited = set()

    def traverse(thunk_or_var: Any, path: str, depth: int) -> None:
        if depth <= 0:
            return

        # Avoid cycles
        obj_id = id(thunk_or_var)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(thunk_or_var, OutputThunk):
            # Traverse into the pipeline thunk's inputs
            pt = thunk_or_var.pipeline_thunk
            func_name = pt.thunk.fcn.__name__
            for name, value in pt.inputs.items():
                input_path = f"{func_name}() -> {name}" if not path else f"{path} -> {func_name}() -> {name}"
                traverse(value, input_path, depth - 1)

        elif _is_trackable_variable(thunk_or_var):
            vhash = getattr(thunk_or_var, "_vhash", None) or getattr(thunk_or_var, "vhash", None)
            if vhash is None:
                # Found an unsaved variable
                unsaved.append((thunk_or_var, path))

                # Check if it wraps an OutputThunk - if so, continue traversing
                inner_data = getattr(thunk_or_var, "data", None)
                if isinstance(inner_data, OutputThunk):
                    traverse(inner_data, path, depth - 1)

    traverse(output_thunk, "", max_depth)
    return unsaved


def get_upstream_lineage(
    output_thunk: OutputThunk,
    max_depth: int = 100,
) -> list[dict]:
    """
    Get lineage information for all upstream computations, including through
    unsaved variables.

    This traverses the full in-memory chain, extracting lineage even for
    unsaved intermediate variables (ephemeral mode).

    Args:
        output_thunk: The OutputThunk to start from
        max_depth: Maximum recursion depth

    Returns:
        List of lineage dicts, one per upstream computation
    """
    lineages = []
    visited = set()

    def traverse(thunk: OutputThunk, depth: int) -> None:
        if depth <= 0:
            return

        thunk_id = id(thunk)
        if thunk_id in visited:
            return
        visited.add(thunk_id)

        # Extract lineage for this thunk
        lineage = extract_lineage(thunk)
        lineages.append(lineage.to_dict())

        # Recurse into input OutputThunks
        for name, value in thunk.pipeline_thunk.inputs.items():
            if isinstance(value, OutputThunk):
                traverse(value, depth - 1)
            elif _is_trackable_variable(value):
                # Check if unsaved variable wraps an OutputThunk
                inner = getattr(value, "data", None)
                if isinstance(inner, OutputThunk):
                    traverse(inner, depth - 1)

    traverse(output_thunk, max_depth)
    return lineages
