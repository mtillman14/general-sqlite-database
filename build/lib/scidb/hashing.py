"""Deterministic hashing for arbitrary Python objects.

This module re-exports the thunk library's hashing functionality and provides
scidb-specific extensions for version hash generation.
"""

import hashlib

# Re-export canonical_hash from thunk library
from thunk import canonical_hash

__all__ = ["canonical_hash", "generate_vhash"]


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
