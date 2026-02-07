"""Deterministic hashing for arbitrary Python objects.

This module re-exports hashing functionality from the canonicalhash package
for backwards compatibility within scidb.
"""

from canonicalhash import canonical_hash, generate_record_id

__all__ = ["canonical_hash", "generate_record_id"]
