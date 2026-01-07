"""Deterministic hashing for arbitrary Python objects."""

import hashlib
import json
import pickle
from typing import Any


def canonical_hash(obj: Any) -> str:
    """
    Generate a deterministic hash for arbitrary Python objects.

    Strategy:
    1. For JSON-serializable primitives: use JSON (deterministic)
    2. For numpy/pandas: use shape + dtype + raw bytes
    3. For other objects: use pickle protocol 4

    Returns:
        16-character hex string (first 64 bits of SHA-256)
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

    # Fallback: pickle (less ideal but handles arbitrary objects)
    return b"pickle:" + pickle.dumps(obj, protocol=4)


def generate_vhash(
    class_name: str,
    schema_version: int,
    content_hash: str,
    metadata: dict,
) -> str:
    """
    Generate a version hash for a variable.

    The vhash uniquely identifies a variable by its type, schema, content,
    and metadata. It is used for addressing/querying data.

    Components:
    - class_name: The variable type (e.g., "RotationMatrix")
    - schema_version: Integer version of the to_db/from_db schema
    - content_hash: Pre-computed hash of the data content
    - metadata: The addressing metadata (subject, trial, etc.)

    Returns:
        16-character hex string
    """
    components = [
        f"class:{class_name}",
        f"schema:{schema_version}",
        f"content:{content_hash}",
        f"meta:{canonical_hash(metadata)}",
    ]
    combined = "|".join(components).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()[:16]
