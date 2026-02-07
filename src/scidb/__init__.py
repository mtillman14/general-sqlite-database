"""
SciDB: Scientific Data Versioning Framework

A lightweight database framework for scientific computing that provides:
- Type-safe serialization of numpy arrays, DataFrames, and custom types
- Automatic content-based versioning
- Flexible metadata-based addressing
- Portable single-file DuckDB storage
- Automatic lineage tracking via thunks

Example:
    from scidb import configure_database, BaseVariable, thunk
    import numpy as np

    # One-line setup (auto-registers types, enables caching)
    db = configure_database("experiment.duckdb", ["subject", "session"], "pipeline.db")

    class RawSignal(BaseVariable):
        schema_version = 1

    @thunk
    def calibrate(signal, factor):
        return signal * factor

    # Save/load (no db= parameter needed)
    RawSignal.save(np.array([1, 2, 3]), subject=1, session="A")
    raw = RawSignal.load(subject=1, session="A")

    # Thunk caching works automatically
    result = calibrate(raw, 2.5)
    CalibratedSignal.save(result, subject=1, session="A")
"""

from .database import configure_database, get_database, get_user_id
from .exceptions import (
    DatabaseNotConfiguredError,
    NotFoundError,
    NotRegisteredError,
    ReservedMetadataKeyError,
    SciDBError,
    UnsavedIntermediateError,
)

# Re-export from scirun for backwards compatibility
# The original foreach.py is kept for now but scirun is the canonical source
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scirun-lib" / "src"))
from scirun import Fixed, for_each

from .thunk import ThunkOutput, Thunk, thunk
from .variable import BaseVariable

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "BaseVariable",
    # Configuration
    "configure_database",
    "get_database",
    # Batch execution
    "for_each",
    "Fixed",
    # Thunk system
    "thunk",
    "Thunk",
    "ThunkOutput",
    # Exceptions
    "SciDBError",
    "NotRegisteredError",
    "NotFoundError",
    "DatabaseNotConfiguredError",
    "ReservedMetadataKeyError",
    "UnsavedIntermediateError",
]
