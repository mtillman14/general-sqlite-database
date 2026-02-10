"""Deterministic hashing for arbitrary Python objects.

This module re-exports hashing functionality from the canonicalhash package
for convenience within scidb.
"""

from canonicalhash import canonical_hash, generate_record_id

__all__ = ["canonical_hash", "generate_record_id"]
