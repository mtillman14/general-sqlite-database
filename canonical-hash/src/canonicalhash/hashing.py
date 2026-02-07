"""Deterministic hashing for arbitrary Python objects.

This module provides utilities for creating stable, deterministic hashes
of Python objects, which is essential for cache key computation, data
versioning, and reproducibility.
"""

import hashlib
import json
from typing import Any


def canonical_hash(obj: Any) -> str:
    """
    Generate a deterministic hash for arbitrary Python objects.

    Strategy:
    1. For JSON-serializable primitives (None, bool, int, float, str): use JSON
    2. For numpy ndarrays: use shape + dtype + raw bytes
    3. For pandas DataFrames: use columns + index + array serialization
    4. For pandas Series: use name + array serialization
    5. For dicts: sort keys, recursively serialize
    6. For lists/tuples: preserve order, recursively serialize
    7. For other objects: raise ValueError

    Args:
        obj: Any Python object to hash

    Returns:
        16-character hex string (first 64 bits of SHA-256)

    Raises:
        ValueError: If an unserializable object is provided

    Example:
        >>> h = canonical_hash(42)
        >>> len(h) == 16 and all(c in '0123456789abcdef' for c in h)
        True
        >>> canonical_hash(42) == canonical_hash(42)  # Deterministic
        True
        >>> canonical_hash([1, 2, 3]) != canonical_hash([1, 2, 4])  # Content-sensitive
        True
    """
    serialized = _serialize_for_hash(obj)
    return hashlib.sha256(serialized).hexdigest()[:16]


def _serialize_for_hash(obj: Any) -> bytes:
    """Convert object to bytes for hashing."""

    # Primitives - use JSON for stability
    if isinstance(obj, (type(None), bool, int, float, str)):
        return json.dumps(obj).encode("utf-8")

    # Dicts - sort keys for determinism
    if isinstance(obj, dict):
        sorted_items = sorted(obj.items(), key=lambda x: str(x[0]))
        parts = []
        for k, v in sorted_items:
            parts.append(_serialize_for_hash(k))
            parts.append(_serialize_for_hash(v))
        return b"dict:" + b"|".join(parts)

    # Lists/tuples - preserve order
    if isinstance(obj, (list, tuple)):
        type_prefix = b"list:" if isinstance(obj, list) else b"tuple:"
        parts = [_serialize_for_hash(item) for item in obj]
        return type_prefix + b"|".join(parts)

    # Numpy arrays - use shape, dtype, and raw bytes
    if hasattr(obj, "tobytes") and hasattr(obj, "dtype") and hasattr(obj, "shape"):
        return (
            b"ndarray:"
            + str(obj.shape).encode()
            + b":"
            + str(obj.dtype).encode()
            + b":"
            + obj.tobytes()
        )

    # Pandas DataFrame
    if hasattr(obj, "to_numpy") and hasattr(obj, "columns"):
        arr = obj.to_numpy()
        cols = list(obj.columns)
        idx = list(obj.index) if hasattr(obj, "index") else []
        return (
            b"dataframe:"
            + _serialize_for_hash(cols)
            + b":"
            + _serialize_for_hash(idx)
            + b":"
            + _serialize_for_hash(arr)
        )

    # Pandas Series
    if hasattr(obj, "to_numpy") and hasattr(obj, "name") and not hasattr(obj, "columns"):
        return (
            b"series:"
            + _serialize_for_hash(obj.name)
            + b":"
            + _serialize_for_hash(obj.to_numpy())
        )

    # Unsupported type
    raise ValueError(f"Unserializable data type: {type(obj)}")


def generate_record_id(
    class_name: str,
    schema_version: int,
    content_hash: str,
    metadata: dict,
) -> str:
    """
    Generate a unique record ID from components.

    The record_id uniquely identifies a record by its type, schema, content,
    and metadata. Useful for addressing/querying versioned data.

    Args:
        class_name: The record type (e.g., "RotationMatrix")
        schema_version: Integer version of the serialization schema
        content_hash: Pre-computed hash of the data content
        metadata: The addressing metadata (subject, trial, etc.)

    Returns:
        16-character hex string

    Example:
        >>> rid = generate_record_id("MyData", 1, "abc123", {"subject": 1})
        >>> len(rid) == 16 and all(c in '0123456789abcdef' for c in rid)
        True
    """
    components = [
        f"class:{class_name}",
        f"schema:{schema_version}",
        f"content:{content_hash}",
        f"meta:{canonical_hash(metadata)}",
    ]
    combined = "|".join(components).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()[:16]
