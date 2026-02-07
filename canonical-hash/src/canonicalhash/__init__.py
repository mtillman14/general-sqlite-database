"""Deterministic hashing for arbitrary Python objects.

This package provides utilities for creating stable, deterministic hashes
of Python objects, essential for cache key computation, data versioning,
and reproducibility in data pipelines.
"""

from canonicalhash.hashing import canonical_hash, generate_record_id

__all__ = ["canonical_hash", "generate_record_id"]
__version__ = "0.1.0"
