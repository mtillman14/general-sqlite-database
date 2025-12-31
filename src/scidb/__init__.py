"""
SciDB: Scientific Data Versioning Framework

A lightweight database framework for scientific computing that provides:
- Type-safe serialization of numpy arrays, DataFrames, and custom types
- Automatic content-based versioning
- Flexible metadata-based addressing
- Portable single-file SQLite storage

Example:
    from scidb import configure_database, BaseVariable
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
    rot = RotationMatrix(np.eye(3))
    vhash = rot.save(subject=1, trial=1)

    # Load
    loaded = RotationMatrix.load(subject=1, trial=1)
"""

from .database import DatabaseManager, configure_database, get_database
from .exceptions import (
    DatabaseNotConfiguredError,
    NotFoundError,
    NotRegisteredError,
    ReservedMetadataKeyError,
    SciDBError,
)
from .variable import BaseVariable

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "DatabaseManager",
    "BaseVariable",
    # Configuration
    "configure_database",
    "get_database",
    # Exceptions
    "SciDBError",
    "NotRegisteredError",
    "NotFoundError",
    "DatabaseNotConfiguredError",
    "ReservedMetadataKeyError",
]
