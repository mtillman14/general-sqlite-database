"""Deterministic hashing for arbitrary Python objects.

This module re-exports the canonical hashing functionality from the
canonicalhash package for backwards compatibility.
"""

from canonicalhash import canonical_hash

__all__ = ["canonical_hash"]
