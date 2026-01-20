"""Deterministic hashing for arbitrary Python objects.

This module re-exports the thunk library's hashing functionality and provides
scidb-specific extensions for record ID generation.
"""

import hashlib

# Re-export canonical_hash from thunk library
from thunk import canonical_hash

__all__ = ["canonical_hash", "generate_record_id"]


def generate_record_id(
    class_name: str,
    schema_version: int,
    content_hash: str,
    metadata: dict,
) -> str:
    """
    Generate a unique record ID for a variable.

    The record_id uniquely identifies a saved record by its type, schema, content,
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
