"""
SciDB: Scientific Data Versioning Framework

A lightweight database framework for scientific computing that provides:
- Type-safe serialization of numpy arrays, DataFrames, and custom types
- Automatic content-based versioning
- Flexible metadata-based addressing
- Portable single-file SQLite storage
- Automatic lineage tracking via thunks

Example:
    from scidb import configure_database, BaseVariable, thunk
    import numpy as np
    import pandas as pd

    class RotationMatrix(BaseVariable):
        schema_version = 1

        def to_db(self) -> pd.DataFrame:
            return pd.DataFrame({'value': self.data.flatten()})

        @classmethod
        def from_db(cls, df: pd.DataFrame) -> np.ndarray:
            return df['value'].values.reshape(3, 3)

    # Setup
    db = configure_database("experiment.db")
    db.register(RotationMatrix)

    # Save
    record_id = RotationMatrix.save(np.eye(3), subject=1, trial=1)

    # Load
    loaded = RotationMatrix.load(subject=1, trial=1)

    # With lineage tracking
    @thunk(n_outputs=1)
    def process(data):
        return data * 2

    result = process(loaded)  # Returns OutputThunk with lineage
    RotationMatrix.save(result, subject=1, trial=1, stage="processed")

    # Query provenance
    provenance = db.get_provenance(RotationMatrix, subject=1, trial=1, stage="processed")
"""

from .database import DatabaseManager, configure_database, get_database, get_user_id
from .exceptions import (
    DatabaseNotConfiguredError,
    NotFoundError,
    NotRegisteredError,
    ReservedMetadataKeyError,
    SciDBError,
    UnsavedIntermediateError,
)
from .foreach import Fixed, for_each
from .lineage import (
    LineageRecord,
    check_cache,
    extract_lineage,
    find_unsaved_variables,
    get_raw_value,
)
from .paths import PathGenerator
from .thunk import OutputThunk, PipelineThunk, Thunk, thunk
from .variable import BaseVariable

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "DatabaseManager",
    "BaseVariable",
    # Configuration
    "configure_database",
    "get_database",
    "get_user_id",
    # Path utilities
    "PathGenerator",
    # Batch execution
    "for_each",
    "Fixed",
    # Thunk system
    "thunk",
    "Thunk",
    "PipelineThunk",
    "OutputThunk",
    # Lineage & Caching
    "LineageRecord",
    "extract_lineage",
    "find_unsaved_variables",
    "get_raw_value",
    "check_cache",
    # Exceptions
    "SciDBError",
    "NotRegisteredError",
    "NotFoundError",
    "DatabaseNotConfiguredError",
    "ReservedMetadataKeyError",
    "UnsavedIntermediateError",
]
