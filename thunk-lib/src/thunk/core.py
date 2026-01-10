"""Core thunk system for lazy evaluation and lineage tracking.

This module provides a Python adaptation of Haskell's thunk concept for building
data processing pipelines with automatic provenance tracking.

Example:
    @thunk(n_outputs=1)
    def process_signal(raw: np.ndarray, cal_factor: float) -> np.ndarray:
        return raw * cal_factor

    result = process_signal(raw_data, 2.5)  # Returns OutputThunk
    print(result.data)  # The actual computed data
    print(result.pipeline_thunk.inputs)  # Captured inputs for lineage
"""

from functools import wraps
from hashlib import sha256
from typing import Any, Callable, Protocol, runtime_checkable

from .hashing import canonical_hash

STRING_REPR_DELIMITER = "-"


# -----------------------------------------------------------------------------
# Cache Backend Protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol for cache backends that can be plugged into the thunk system.

    Implement this protocol to provide custom caching behavior.
    """

    def get_cached(
        self, cache_key: str, n_outputs: int
    ) -> list[tuple[Any, str]] | None:
        """
        Look up cached results by cache key.

        Args:
            cache_key: The cache key (hash of function + inputs)
            n_outputs: Number of outputs expected

        Returns:
            List of (data, identifier) tuples if ALL outputs found,
            None otherwise.
        """
        ...


# Global cache backend (None means no caching)
_cache_backend: CacheBackend | None = None


def configure_cache(backend: CacheBackend | None) -> None:
    """
    Configure the global cache backend.

    Args:
        backend: A CacheBackend implementation, or None to disable caching.

    Example:
        from thunk import configure_cache

        class MyCache:
            def get_cached(self, cache_key, n_outputs):
                # Custom cache lookup logic
                return None

        configure_cache(MyCache())
    """
    global _cache_backend
    _cache_backend = backend


def get_cache_backend() -> CacheBackend | None:
    """Get the currently configured cache backend."""
    return _cache_backend


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
        n_outputs: Number of outputs the function returns
        unwrap: Whether to unwrap special input types to raw data
        hash: SHA-256 hash of function bytecode + n_outputs
        pipeline_thunks: All PipelineThunks created from this Thunk
    """

    def __init__(self, fcn: Callable, n_outputs: int = 1, unwrap: bool = True):
        """
        Initialize a Thunk wrapper.

        Args:
            fcn: The function to wrap
            n_outputs: Number of output values the function returns
            unwrap: If True (default), unwrap OutputThunk inputs to their raw
                   data before calling the function. If False, pass the wrapper
                   objects directly (useful for debugging/inspection).
        """
        self.fcn = fcn
        self.n_outputs = n_outputs
        self.unwrap = unwrap
        self.pipeline_thunks: tuple[PipelineThunk, ...] = ()

        # Hash function bytecode + constants for reproducibility
        # Include co_consts to distinguish functions with same structure but different literals
        fcn_code = fcn.__code__.co_code
        fcn_consts = str(fcn.__code__.co_consts).encode()
        combined_code = fcn_code + fcn_consts
        fcn_hash = sha256(combined_code).hexdigest()

        string_repr = f"{fcn_hash}{STRING_REPR_DELIMITER}{n_outputs}"
        self.hash = sha256(string_repr.encode()).hexdigest()

    def __repr__(self) -> str:
        return f"Thunk(fcn={self.fcn.__name__}, n_outputs={self.n_outputs}, unwrap={self.unwrap})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Thunk):
            return False
        return self.hash == other.hash

    def __hash__(self) -> int:
        return int(self.hash[:16], 16)

    def __call__(self, *args, **kwargs) -> "OutputThunk | tuple[OutputThunk, ...]":
        """
        Create a PipelineThunk and execute or defer.

        Automatically checks the computation cache if configured. If all outputs
        are cached, returns them without re-executing the function.

        Returns:
            OutputThunk or tuple of OutputThunks wrapping the result(s)
        """
        pipeline_thunk = PipelineThunk(self, *args, **kwargs)

        # Check for existing equivalent PipelineThunk
        for existing in self.pipeline_thunks:
            if pipeline_thunk._matches(existing):
                pipeline_thunk = existing
                break
        else:
            self.pipeline_thunks = (*self.pipeline_thunks, pipeline_thunk)

        # Auto-cache check if backend configured
        try:
            backend = get_cache_backend()
            if backend is not None:
                cache_key = pipeline_thunk.compute_cache_key()
                cached = backend.get_cached(cache_key, self.n_outputs)

                if cached is not None:
                    # Build OutputThunks from cached results
                    outputs = tuple(
                        OutputThunk(
                            pipeline_thunk=pipeline_thunk,
                            output_num=i,
                            is_complete=True,
                            data=data,
                            was_cached=True,
                            cached_id=cached_id,
                        )
                        for i, (data, cached_id) in enumerate(cached)
                    )
                    pipeline_thunk.outputs = outputs

                    if len(outputs) == 1:
                        return outputs[0]
                    return outputs
        except Exception:
            pass  # No cache or error, execute normally

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
        outputs: Tuple of OutputThunks after execution
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

        self.outputs: tuple[OutputThunk, ...] = ()

    @property
    def hash(self) -> str:
        """Dynamic hash based on thunk + inputs (lineage-based, metadata-agnostic)."""
        # Create a stable hash of inputs based on lineage
        input_parts = []
        for name in sorted(self.inputs.keys()):
            value = self.inputs[name]
            if isinstance(value, OutputThunk):
                # Freshly computed: use pipeline hash (already lineage-based)
                input_parts.append((name, "output", value.hash))
            elif self._is_trackable_variable(value):
                # Variable with lineage info: use lineage_hash if available
                lineage_hash = getattr(value, "_lineage_hash", None) or getattr(
                    value, "lineage_hash", None
                )
                if lineage_hash is not None:
                    input_parts.append((name, "lineage", lineage_hash))
                else:
                    # Raw data with no lineage: check if it wraps an OutputThunk
                    inner_data = getattr(value, "data", value)
                    if isinstance(inner_data, OutputThunk):
                        # Unsaved variable wrapping OutputThunk - use the thunk's hash
                        input_parts.append((name, "unsaved_thunk", value.__class__.__name__, inner_data.hash))
                    else:
                        # Raw data: use content + type
                        content_hash = canonical_hash(inner_data)
                        type_name = value.__class__.__name__
                        input_parts.append((name, "raw", type_name, content_hash))
            else:
                # Literal value: use content hash
                input_parts.append((name, "value", canonical_hash(value)))

        input_str = str(input_parts)
        combined = f"{self.thunk.hash}{STRING_REPR_DELIMITER}{input_str}"
        return sha256(combined.encode()).hexdigest()

    def __hash__(self) -> int:
        return int(self.hash[:16], 16)

    def __repr__(self) -> str:
        return (
            f"PipelineThunk(fcn={self.thunk.fcn.__name__}, "
            f"n_inputs={len(self.inputs)}, n_outputs={self.thunk.n_outputs})"
        )

    @property
    def is_complete(self) -> bool:
        """True if all inputs are concrete values (not pending thunks)."""
        for value in self.inputs.values():
            if isinstance(value, OutputThunk) and not value.is_complete:
                return False
        return True

    def __call__(self, *args, **kwargs) -> "OutputThunk | tuple[OutputThunk, ...]":
        """
        Execute the function if complete, return OutputThunk(s).

        Returns:
            OutputThunk or tuple of OutputThunks wrapping the result(s)
        """
        result: tuple = tuple(None for _ in range(self.thunk.n_outputs))

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

            # Normalize to tuple
            if not isinstance(result, tuple):
                result = (result,)

            if len(result) != self.thunk.n_outputs:
                raise ValueError(
                    f"Function {self.thunk.fcn.__name__} returned {len(result)} outputs, "
                    f"but {self.thunk.n_outputs} were expected."
                )

        # Wrap outputs in OutputThunks
        outputs = tuple(
            OutputThunk(self, i, self.is_complete, val) for i, val in enumerate(result)
        )
        self.outputs = outputs

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    @staticmethod
    def _is_trackable_variable(obj: Any) -> bool:
        """Check if obj is a trackable variable with lineage support.

        Looks for characteristic attributes of variables that support lineage
        tracking (like scidb's BaseVariable).
        """
        return (
            hasattr(obj, "data")
            and hasattr(obj, "_vhash")
            and hasattr(obj, "to_db")
            and hasattr(obj, "from_db")
        )

    def _deep_unwrap(self, value: Any) -> Any:
        """Recursively unwrap OutputThunks and trackable variables to raw data.

        Handles cases like:
        - OutputThunk -> raw data
        - BaseVariable -> raw data
        - BaseVariable wrapping OutputThunk -> raw data (recursive)
        """
        # Unwrap OutputThunk
        if isinstance(value, OutputThunk):
            return value.data

        # Unwrap trackable variable (e.g., BaseVariable)
        if self._is_trackable_variable(value):
            inner = getattr(value, "data", value)
            # If the variable's data is itself an OutputThunk, unwrap that too
            if isinstance(inner, OutputThunk):
                return inner.data
            return inner

        return value

    def compute_cache_key(self) -> str:
        """
        Compute a cache key for this pipeline invocation.

        The cache key is based on LINEAGE (how values were computed), not metadata:
        - Function hash (bytecode + constants)
        - For OutputThunk inputs: use lineage-based hash
        - For trackable variable inputs: use lineage_hash if computed, else content+type
        - For raw values: use content hash

        This enables cross-user caching (different metadata, same computation)
        while preventing false cache hits (same content, different computation).

        Returns:
            SHA-256 hash suitable for cache lookup
        """
        input_hashes = []
        for name in sorted(self.inputs.keys()):
            value = self.inputs[name]
            if isinstance(value, OutputThunk):
                # Freshly computed: use pipeline hash (already lineage-based)
                input_hashes.append((name, "output", value.hash))
            elif self._is_trackable_variable(value):
                # Variable with lineage: use stored lineage_hash if available
                lineage_hash = getattr(value, "_lineage_hash", None) or getattr(
                    value, "lineage_hash", None
                )
                if lineage_hash is not None:
                    input_hashes.append((name, "lineage", lineage_hash))
                else:
                    # Raw data with no lineage: check if it wraps an OutputThunk
                    inner_data = getattr(value, "data", value)
                    if isinstance(inner_data, OutputThunk):
                        # Unsaved variable wrapping OutputThunk - use the thunk's hash
                        input_hashes.append((name, "unsaved_thunk", value.__class__.__name__, inner_data.hash))
                    else:
                        # Raw data: use content + type
                        content_hash = canonical_hash(inner_data)
                        type_name = value.__class__.__name__
                        input_hashes.append((name, "raw", type_name, content_hash))
            else:
                # Literal value: use content hash
                input_hashes.append((name, "value", canonical_hash(value)))

        cache_input = f"{self.thunk.hash}{STRING_REPR_DELIMITER}{input_hashes}"
        return sha256(cache_input.encode()).hexdigest()

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

            # Compare OutputThunks by hash
            if isinstance(self_val, OutputThunk) and isinstance(other_val, OutputThunk):
                if self_val.hash != other_val.hash:
                    return False
            elif isinstance(self_val, OutputThunk) or isinstance(other_val, OutputThunk):
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


class OutputThunk:
    """
    Wraps a function output with lineage information.

    Contains:
    - Reference to the PipelineThunk that produced it
    - Output index (for multi-output functions)
    - The actual computed data

    This is the key to provenance: every OutputThunk knows its parent
    PipelineThunk, which knows its inputs (possibly other OutputThunks).

    Attributes:
        pipeline_thunk: The PipelineThunk that produced this output
        output_num: Index of this output (0-based)
        is_complete: True if the data has been computed
        data: The actual computed data (None if not complete)
        hash: SHA-256 hash encoding the full lineage
        was_cached: True if this result was loaded from cache
        cached_id: Identifier of the cached result (if was_cached)
    """

    def __init__(
        self,
        pipeline_thunk: PipelineThunk,
        output_num: int,
        is_complete: bool,
        data: Any,
        was_cached: bool = False,
        cached_id: str | None = None,
    ):
        """
        Initialize an OutputThunk.

        Args:
            pipeline_thunk: The PipelineThunk that produced this
            output_num: Index of this output
            is_complete: Whether the data has been computed
            data: The computed data (or None if not complete)
            was_cached: Whether this result was loaded from cache
            cached_id: Identifier if loaded from cache (e.g., vhash)
        """
        self.pipeline_thunk = pipeline_thunk
        self.output_num = output_num
        self.is_complete = is_complete
        self.data = data if is_complete else None
        self._was_cached = was_cached
        self._cached_id = cached_id

        string_repr = (
            f"{pipeline_thunk.hash}{STRING_REPR_DELIMITER}"
            f"output{STRING_REPR_DELIMITER}{output_num}"
        )
        self.hash = sha256(string_repr.encode()).hexdigest()

    @property
    def was_cached(self) -> bool:
        """True if this result was loaded from the computation cache."""
        return self._was_cached

    @property
    def cached_id(self) -> str | None:
        """The identifier of the cached result, if loaded from cache."""
        return self._cached_id

    def __repr__(self) -> str:
        fcn_name = self.pipeline_thunk.thunk.fcn.__name__
        return f"OutputThunk(fn={fcn_name}, output={self.output_num}, complete={self.is_complete})"

    def __str__(self) -> str:
        """Show only the data when printed."""
        return str(self.data)

    def __eq__(self, other: object) -> bool:
        """Compare by hash if OutputThunk, otherwise compare data."""
        if isinstance(other, OutputThunk):
            return self.hash == other.hash
        return self.data == other

    def __hash__(self) -> int:
        return int(self.hash[:16], 16)


# -----------------------------------------------------------------------------
# Decorator
# -----------------------------------------------------------------------------


def thunk(n_outputs: int = 1, unwrap: bool = True) -> Callable[[Callable], Thunk]:
    """
    Decorator to convert a function into a Thunk.

    Example:
        @thunk(n_outputs=1)
        def process_signal(raw, calibration):
            return raw * calibration

        result = process_signal(raw_data, cal_data)  # Returns OutputThunk
        print(result.data)  # The actual computed data
        print(result.pipeline_thunk.inputs)  # Captured inputs

    For debugging/inspection, use unwrap=False to receive the full objects:

        @thunk(n_outputs=1, unwrap=False)
        def debug_process(var):
            print(f"Input vhash: {var.vhash}")
            print(f"Input metadata: {var.metadata}")
            return var.data * 2

    Args:
        n_outputs: Number of outputs the function returns (default 1)
        unwrap: If True (default), unwrap OutputThunk and variable inputs
               to their raw data (.data). If False, pass the wrapper
               objects directly for inspection/debugging.

    Returns:
        A decorator that converts functions to Thunks
    """

    def decorator(fcn: Callable) -> Thunk:
        t = Thunk(fcn, n_outputs, unwrap)
        # Preserve function metadata
        t.__doc__ = fcn.__doc__
        t.__name__ = fcn.__name__  # type: ignore
        return t

    return decorator
