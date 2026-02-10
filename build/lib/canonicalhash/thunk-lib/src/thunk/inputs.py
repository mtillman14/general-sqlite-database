"""Input classification for lineage tracking.

This module provides utilities for classifying function inputs
to determine how they should be tracked for lineage/provenance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import ThunkOutput

from .hashing import canonical_hash


class InputKind(Enum):
    """Classification of a function input for lineage tracking."""
    THUNK_OUTPUT = "thunk"        # ThunkOutput from another computation
    SAVED_VARIABLE = "variable"   # Variable with record_id (saved to DB)
    UNSAVED_THUNK = "unsaved"     # Variable wrapping an ThunkOutput (not saved)
    RAW_DATA = "raw"              # Variable with raw data (no lineage)
    CONSTANT = "constant"         # Literal value (int, float, str, etc.)


@dataclass
class ClassifiedInput:
    """A classified function input with lineage-relevant information."""
    kind: InputKind
    name: str
    hash: str
    type_name: str | None = None

    # For THUNK_OUTPUT and UNSAVED_THUNK
    source_function: str | None = None
    output_num: int | None = None

    # For SAVED_VARIABLE
    record_id: str | None = None
    metadata: dict | None = None
    content_hash: str | None = None

    # For CONSTANT
    value_repr: str | None = None

    def to_lineage_dict(self) -> dict:
        """Convert to dict format for LineageRecord storage."""
        if self.kind == InputKind.THUNK_OUTPUT:
            return {
                "name": self.name,
                "source_type": "thunk",
                "source_function": self.source_function,
                "source_hash": self.hash,
                "output_num": self.output_num,
            }
        elif self.kind == InputKind.SAVED_VARIABLE:
            d = {
                "name": self.name,
                "source_type": "variable",
                "type": self.type_name,
                "record_id": self.record_id,
                "metadata": self.metadata,
            }
            if self.content_hash is not None:
                d["content_hash"] = self.content_hash
            return d
        elif self.kind == InputKind.UNSAVED_THUNK:
            return {
                "name": self.name,
                "source_type": "unsaved_variable",
                "type": self.type_name,
                "inner_source": "thunk",
                "source_function": self.source_function,
                "source_hash": self.hash,
                "output_num": self.output_num,
            }
        elif self.kind == InputKind.RAW_DATA:
            return {
                "name": self.name,
                "source_type": "unsaved_variable",
                "type": self.type_name,
                "content_hash": self.hash,
            }
        else:  # CONSTANT
            return {
                "name": self.name,
                "value_hash": self.hash,
                "value_repr": self.value_repr,
                "value_type": self.type_name,
            }

    def to_cache_tuple(self) -> tuple:
        """Convert to tuple format for cache key computation."""
        if self.kind == InputKind.THUNK_OUTPUT:
            return (self.name, "output", self.hash)
        elif self.kind == InputKind.SAVED_VARIABLE:
            return (self.name, "lineage", self.hash)
        elif self.kind == InputKind.UNSAVED_THUNK:
            return (self.name, "unsaved_thunk", self.type_name, self.hash)
        elif self.kind == InputKind.RAW_DATA:
            return (self.name, "raw", self.type_name, self.hash)
        else:  # CONSTANT
            return (self.name, "value", self.hash)


def is_trackable_variable(obj: Any) -> bool:
    """
    Check if obj is a trackable variable with lineage support.

    Looks for characteristic attributes of variables that support
    lineage tracking (like scidb's BaseVariable).
    """
    return (
        hasattr(obj, "data")
        and hasattr(obj, "_record_id")
        and hasattr(obj, "to_db")
        and hasattr(obj, "from_db")
    )


def classify_input(name: str, value: Any) -> ClassifiedInput:
    """
    Classify a function input for lineage tracking.

    Determines what kind of input this is and extracts the
    relevant information for lineage/caching.

    Args:
        name: The argument name
        value: The argument value

    Returns:
        ClassifiedInput with kind and relevant metadata
    """
    # Import here to avoid circular imports
    from .core import ThunkOutput

    if isinstance(value, ThunkOutput):
        # Input came from another thunk computation
        return ClassifiedInput(
            kind=InputKind.THUNK_OUTPUT,
            name=name,
            hash=value.hash,
            source_function=value.pipeline_thunk.thunk.fcn.__name__,
            output_num=value.output_num,
        )

    if is_trackable_variable(value):
        # Input is a trackable variable (e.g., scidb BaseVariable)
        record_id = getattr(value, "_record_id", None) or getattr(value, "record_id", None)
        metadata = getattr(value, "_metadata", None) or getattr(value, "metadata", None)
        type_name = type(value).__name__

        if record_id is not None:
            lineage_hash = getattr(value, "_lineage_hash", None) or getattr(value, "lineage_hash", None)
            content_hash = getattr(value, "_content_hash", None) or getattr(value, "content_hash", None)

            if lineage_hash is not None:
                # Variable was produced by a thunk — classify it the same way
                # so that to_cache_tuple() matches the original ThunkOutput
                return ClassifiedInput(
                    kind=InputKind.THUNK_OUTPUT,
                    name=name,
                    hash=lineage_hash,
                )

            return ClassifiedInput(
                kind=InputKind.SAVED_VARIABLE,
                name=name,
                hash=_safe_hash(getattr(value, "data", value)),
                type_name=type_name,
                record_id=record_id,
                metadata=metadata,
                content_hash=content_hash,
            )

        # Unsaved variable - check if it wraps an ThunkOutput
        inner_data = getattr(value, "data", None)
        if isinstance(inner_data, ThunkOutput):
            return ClassifiedInput(
                kind=InputKind.UNSAVED_THUNK,
                name=name,
                hash=inner_data.hash,
                type_name=type_name,
                source_function=inner_data.pipeline_thunk.thunk.fcn.__name__,
                output_num=inner_data.output_num,
            )

        # Unsaved variable with raw data
        return ClassifiedInput(
            kind=InputKind.RAW_DATA,
            name=name,
            hash=_safe_hash(inner_data if inner_data is not None else value),
            type_name=type_name,
        )

    # Constant/literal value
    return ClassifiedInput(
        kind=InputKind.CONSTANT,
        name=name,
        hash=_safe_hash(value),
        type_name=type(value).__name__,
        value_repr=repr(value)[:200],
    )


def classify_inputs(inputs: dict[str, Any]) -> list[ClassifiedInput]:
    """
    Classify all inputs in a dict.

    Args:
        inputs: Dict mapping argument names to values

    Returns:
        List of ClassifiedInput, sorted by name for determinism
    """
    return [classify_input(name, value) for name in sorted(inputs.keys()) for value in [inputs[name]]]


def _safe_hash(value: Any) -> str:
    """Hash a value, returning 'unhashable' on failure."""
    try:
        return canonical_hash(value)
    except Exception:
        return "unhashable"
