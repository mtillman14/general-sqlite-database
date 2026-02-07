"""Core thunk system for lineage tracking.

This module provides a Python adaptation of Haskell's thunk concept for building
data processing pipelines with automatic provenance tracking.

Example:
    @thunk
    def process_signal(raw: np.ndarray, cal_factor: float) -> np.ndarray:
        return raw * cal_factor

    result = process_signal(raw_data, 2.5)  # Returns ThunkOutput
    print(result.data)  # The actual computed data
    print(result.pipeline_thunk.inputs)  # Captured inputs for lineage

For multi-output functions:
    @thunk(unpack_output=True)
    def split(data):
        return data[:len(data)//2], data[len(data)//2:]

    first, second = split(data)  # Each is a separate ThunkOutput
"""

from functools import wraps
from hashlib import sha256
from typing import Any, Callable

from .inputs import classify_inputs, is_trackable_variable

STRING_REPR_DELIMITER = "-"


# -----------------------------------------------------------------------------
# Thunk Classes
# -----------------------------------------------------------------------------


class Thunk:
    """
    Wraps a function to enable lineage tracking.

    When called, creates a PipelineThunk that tracks inputs.
    The function's bytecode is hashed for reproducibility checking.

    Attributes:
        fcn: The wrapped function
        unpack_output: Whether to unpack tuple returns into separate ThunkOutputs
        unwrap: Whether to unwrap special input types to raw data
        hash: SHA-256 hash of function bytecode + unpack_output
        pipeline_thunks: All PipelineThunks created from this Thunk
    """

    def __init__(self, fcn: Callable, unpack_output: bool = False, unwrap: bool = True):
        """
        Initialize a Thunk wrapper of a function.

        Args:
            fcn: The function to wrap
            unpack_output: If True, unpack tuple returns into separate ThunkOutputs.
                          If False (default), the return value is wrapped as a single
                          ThunkOutput (even if it's a tuple).
            unwrap: If True (default), unwrap ThunkOutput inputs to their raw
                   data before calling the function. If False, pass the wrapper
                   objects directly (useful for debugging/inspection).
        """
        self.fcn = fcn
        self.unpack_output = unpack_output
        self.unwrap = unwrap
        self.pipeline_thunks: tuple[PipelineThunk, ...] = ()

        # Hash function bytecode + constants for reproducibility
        # Include co_consts to distinguish functions with same structure but different literals
        fcn_code = fcn.__code__.co_code
        fcn_consts = str(fcn.__code__.co_consts).encode()
        combined_code = fcn_code + fcn_consts
        fcn_hash = sha256(combined_code).hexdigest()

        string_repr = f"{fcn_hash}{STRING_REPR_DELIMITER}{unpack_output}"
        self.hash = sha256(string_repr.encode()).hexdigest()

    def __repr__(self) -> str:
        return f"Thunk(fcn={self.fcn.__name__}, unpack_output={self.unpack_output}, unwrap={self.unwrap})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Thunk):
            return False
        return self.hash == other.hash

    def __hash__(self) -> int:
        return int(self.hash[:16], 16)

    def __call__(self, *args, **kwargs) -> "ThunkOutput | tuple[ThunkOutput, ...]":
        """
        Create a PipelineThunk and execute.

        Returns:
            ThunkOutput or tuple of ThunkOutputs wrapping the result(s)
        """
        pipeline_thunk = PipelineThunk(self, *args, **kwargs)

        # Check for existing equivalent PipelineThunk in the tuple of PipelineThunks
        for existing in self.pipeline_thunks:
            if pipeline_thunk._matches(existing):
                pipeline_thunk = existing
                break
        else:
            self.pipeline_thunks = (*self.pipeline_thunks, pipeline_thunk)

        return pipeline_thunk(*args, **kwargs)


class PipelineThunk:
    """
    Represents a specific invocation of a Thunk with captured inputs.

    Tracks:
    - The parent Thunk (function definition)
    - All input arguments (positional and keyword)
    - Output(s) after execution

    Attributes:
        thunk: The parent Thunk that created this
        inputs: Dict mapping argument names to values
        outputs: Tuple of ThunkOutputs after execution
        unwrap: Whether to unwrap inputs before calling the function
    """

    def __init__(self, thunk: Thunk, *args, **kwargs):
        """
        Initialize a PipelineThunk.

        Args:
            thunk: The parent Thunk
            *args: Positional arguments passed to the function
            **kwargs: Keyword arguments passed to the function
        """
        self.thunk = thunk
        self.unwrap = thunk.unwrap
        self.inputs: dict[str, Any] = {}

        # Capture positional args
        for i, arg in enumerate(args):
            self.inputs[f"arg_{i}"] = arg

        # Capture keyword args
        self.inputs.update(kwargs)

        self.outputs: tuple[ThunkOutput, ...] = ()

    @property
    def hash(self) -> str:
        """Dynamic hash based on thunk + inputs (lineage-based, metadata-agnostic)."""
        return self.compute_lineage_hash()

    def __hash__(self) -> int:
        return int(self.hash[:16], 16)

    def __repr__(self) -> str:
        return (
            f"PipelineThunk(fcn={self.thunk.fcn.__name__}, "
            f"n_inputs={len(self.inputs)}, unpack_output={self.thunk.unpack_output})"
        )

    @property
    def is_complete(self) -> bool:
        """True if all inputs are concrete values (not pending thunks)."""
        for value in self.inputs.values():
            if isinstance(value, ThunkOutput) and not value.is_complete:
                return False
        return True

    def __call__(self, *args, **kwargs) -> "ThunkOutput | tuple[ThunkOutput, ...]":
        """
        Execute the function if complete, return ThunkOutput(s).

        Returns:
            ThunkOutput or tuple of ThunkOutputs wrapping the result(s)
        """
        if self.is_complete:
            # Resolve arguments - unwrap if self.unwrap is True
            resolved_args = []
            for arg in args:
                if self.unwrap:
                    resolved_args.append(self._deep_unwrap(arg))
                else:
                    # Pass through as-is (for debugging/inspection)
                    resolved_args.append(arg)

            resolved_kwargs = {}
            for k, v in kwargs.items():
                if self.unwrap:
                    resolved_kwargs[k] = self._deep_unwrap(v)
                else:
                    resolved_kwargs[k] = v

            result = self.thunk.fcn(*resolved_args, **resolved_kwargs)

            # Handle output based on unpack_output setting
            if self.thunk.unpack_output:
                # Unpack tuple into separate ThunkOutputs
                if not isinstance(result, tuple):
                    raise ValueError(
                        f"Function {self.thunk.fcn.__name__} has unpack_output=True "
                        f"but did not return a tuple."
                    )
                outputs = tuple(
                    ThunkOutput(self, i, True, val) for i, val in enumerate(result)
                )
            else:
                # Wrap entire result as single ThunkOutput
                outputs = (ThunkOutput(self, 0, True, result),)
        else:
            # Not complete - create placeholder output(s)
            outputs = (ThunkOutput(self, 0, False, None),)

        self.outputs = outputs

        if len(outputs) == 1:
            return outputs[0]
        if self.thunk.unpack_output:
            return *outputs
        return outputs

    def _deep_unwrap(self, value: Any) -> Any:
        """Recursively unwrap ThunkOutputs and trackable variables to raw data.

        Handles cases like:
        - ThunkOutput -> raw data
        - BaseVariable -> raw data
        - BaseVariable wrapping ThunkOutput -> raw data (recursive)
        """
        # Unwrap ThunkOutput
        if isinstance(value, ThunkOutput):
            return value.data

        # Unwrap trackable variable (e.g., BaseVariable)
        if is_trackable_variable(value):
            inner = getattr(value, "data", value)
            # If the variable's data is itself an ThunkOutput, unwrap that too
            if isinstance(inner, ThunkOutput):
                return inner.data
            return inner

        return value

    def compute_lineage_hash(self) -> str:
        """
        Compute a hash representing this computation's lineage.

        The hash is based on:
        - Function hash (bytecode + constants)
        - For ThunkOutput inputs: use lineage-based hash
        - For trackable variable inputs: use lineage_hash if computed, else content+type
        - For raw values: use content hash

        Returns:
            SHA-256 hash encoding the computation lineage
        """
        classified = classify_inputs(self.inputs)
        input_tuples = [c.to_cache_tuple() for c in classified]
        hash_input = f"{self.thunk.hash}{STRING_REPR_DELIMITER}{input_tuples}"
        return sha256(hash_input.encode()).hexdigest()

    def _matches(self, other: "PipelineThunk") -> bool:
        """Check if this is equivalent to another PipelineThunk."""
        if self.thunk.hash != other.thunk.hash:
            return False

        # Check if inputs match
        if set(self.inputs.keys()) != set(other.inputs.keys()):
            return False

        for key in self.inputs:
            self_val = self.inputs[key]
            other_val = other.inputs[key]

            # Compare ThunkOutputs by hash
            if isinstance(self_val, ThunkOutput) and isinstance(other_val, ThunkOutput):
                if self_val.hash != other_val.hash:
                    return False
            elif isinstance(self_val, ThunkOutput) or isinstance(other_val, ThunkOutput):
                return False
            else:
                # Compare other values directly
                try:
                    if self_val != other_val:
                        return False
                except (ValueError, TypeError):
                    # numpy arrays raise ValueError for == comparison
                    try:
                        import numpy as np

                        if isinstance(self_val, np.ndarray) and isinstance(
                            other_val, np.ndarray
                        ):
                            if not np.array_equal(self_val, other_val):
                                return False
                        else:
                            return False
                    except ImportError:
                        return False

        return True


class ThunkOutput:
    """
    Wraps a function output with lineage information.

    Contains:
    - Reference to the PipelineThunk that produced it
    - Output index (for multi-output functions)
    - The actual computed data

    This is the key to provenance: every ThunkOutput knows its parent
    PipelineThunk, which knows its inputs (possibly other ThunkOutputs).

    Attributes:
        pipeline_thunk: The PipelineThunk that produced this output
        output_num: Index of this output (0-based)
        is_complete: True if the data has been computed
        data: The actual computed data (None if not complete)
        hash: SHA-256 hash encoding the full lineage
    """

    def __init__(
        self,
        pipeline_thunk: PipelineThunk,
        output_num: int,
        is_complete: bool,
        data: Any,
    ):
        """
        Initialize a ThunkOutput.

        Args:
            pipeline_thunk: The PipelineThunk that produced this
            output_num: Index of this output
            is_complete: Whether the data has been computed
            data: The computed data (or None if not complete)
        """
        self.pipeline_thunk = pipeline_thunk
        self.output_num = output_num
        self.is_complete = is_complete
        self.data = data if is_complete else None

        string_repr = (
            f"{pipeline_thunk.hash}{STRING_REPR_DELIMITER}"
            f"output{STRING_REPR_DELIMITER}{output_num}"
        )
        self.hash = sha256(string_repr.encode()).hexdigest()

    def __repr__(self) -> str:
        fcn_name = self.pipeline_thunk.thunk.fcn.__name__
        return f"ThunkOutput(fn={fcn_name}, output={self.output_num}, complete={self.is_complete})"

    def __str__(self) -> str:
        """Show only the data when printed."""
        return str(self.data)

    def __eq__(self, other: object) -> bool:
        """Compare by hash if ThunkOutput, otherwise compare data."""
        if isinstance(other, ThunkOutput):
            return self.hash == other.hash
        return self.data == other

    def __hash__(self) -> int:
        return int(self.hash[:16], 16)


# -----------------------------------------------------------------------------
# Decorator
# -----------------------------------------------------------------------------


def thunk(
    func: Callable | None = None,
    *,
    unpack_output: bool = False,
    unwrap: bool = True,
) -> Callable[[Callable], Thunk] | Thunk:
    """
    Decorator to convert a function into a Thunk for lineage tracking.

    Can be used with or without parentheses:

        @thunk
        def process_signal(raw, calibration):
            return raw * calibration

        @thunk()
        def another_function(x):
            return x * 2

    For multi-output functions, use unpack_output=True:

        @thunk(unpack_output=True)
        def split(data):
            return data[:len(data)//2], data[len(data)//2:]

        first_half, second_half = split(my_data)  # Each is a ThunkOutput

    For debugging/inspection, use unwrap=False to receive the full objects:

        @thunk(unwrap=False)
        def debug_process(var):
            print(f"Input record_id: {var.record_id}")
            print(f"Input metadata: {var.metadata}")
            return var.data * 2

    Args:
        func: The function to wrap (when used without parentheses)
        unpack_output: If True, unpack tuple returns into separate ThunkOutputs.
                      If False (default), the return value is wrapped as a single
                      ThunkOutput (even if it's a tuple).
        unwrap: If True (default), unwrap ThunkOutput and variable inputs
               to their raw data (.data). If False, pass the wrapper
               objects directly for inspection/debugging.

    Returns:
        A Thunk wrapping the function, or a decorator that creates one
    """

    def decorator(fcn: Callable) -> Thunk:
        t = Thunk(fcn, unpack_output, unwrap)
        return wraps(fcn)(t)

    if func is not None:
        # Called without parentheses: @thunk
        return decorator(func)
    # Called with parentheses: @thunk() or @thunk(unpack_output=True)
    return decorator
