# SciDB: Scientific Data Versioning Framework - Implementation Plan

## Overview

A lightweight, portable database framework for scientific computing that:
- Wraps arbitrary data types (numpy arrays, DataFrames, etc.) in serializable classes
- Stores data in SQLite with automatic versioning via content hashing
- Addresses data by flexible metadata kwargs (subject, trial, condition, etc.)
- **Tracks data provenance via thunk-based lineage** (what inputs + code produced each output)
- Prioritizes portability and reproducibility over raw performance

---

## Design Decisions

| Aspect | Decision |
|--------|----------|
| Storage backend | SQLite (single-file, portable) |
| Table naming | One table per variable type, snake_case class name |
| Hash determinism | Content + metadata hash; timestamp excluded |
| Large array storage | BLOB in SQLite (pickle serialized) |
| Version retention | Keep all versions (pruning deferred) |
| Serialization | Generic approach using pickle/numpy/pandas introspection |
| DB connection | Both global singleton and explicit passing supported |
| Multi-match on load | Return list of all matches |
| Table creation | Explicit registration required before save |
| **Lineage tracking** | **Thunk-based: functions wrapped with `@thunk` decorator capture inputs automatically** |

---

## Module Structure

```
src/scidb/
├── __init__.py              # Public API exports, configure_database()
├── database.py              # DatabaseManager class
├── variable.py              # BaseVariable ABC
├── hashing.py               # canonical_hash(), vhash generation
├── storage.py               # SQLite BLOB serialization helpers
├── exceptions.py            # Custom exceptions
├── thunk.py                 # Thunk, PipelineThunk, OutputThunk classes
├── lineage.py               # Lineage extraction and querying
└── variables/               # Built-in variable types (examples)
    ├── __init__.py
    └── array.py             # Generic numpy array wrapper
```

---

## Core Components

### 1. `scidb/hashing.py` - Deterministic Hashing

Provides content-based hashing that is deterministic across runs.

```python
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
        return json.dumps(obj).encode('utf-8')

    # Dicts - sort keys for determinism
    if isinstance(obj, dict):
        sorted_items = sorted(obj.items(), key=lambda x: str(x[0]))
        parts = []
        for k, v in sorted_items:
            parts.append(_serialize_for_hash(k))
            parts.append(_serialize_for_hash(v))
        return b'dict:' + b'|'.join(parts)

    # Lists/tuples - preserve order
    if isinstance(obj, (list, tuple)):
        type_prefix = b'list:' if isinstance(obj, list) else b'tuple:'
        parts = [_serialize_for_hash(item) for item in obj]
        return type_prefix + b'|'.join(parts)

    # Numpy arrays - use shape, dtype, and raw bytes
    if hasattr(obj, 'tobytes') and hasattr(obj, 'dtype') and hasattr(obj, 'shape'):
        return (
            b'ndarray:' +
            str(obj.shape).encode() + b':' +
            str(obj.dtype).encode() + b':' +
            obj.tobytes()
        )

    # Pandas DataFrame
    if hasattr(obj, 'to_numpy') and hasattr(obj, 'columns'):
        arr = obj.to_numpy()
        cols = list(obj.columns)
        idx = list(obj.index) if hasattr(obj, 'index') else []
        return (
            b'dataframe:' +
            _serialize_for_hash(cols) + b':' +
            _serialize_for_hash(idx) + b':' +
            _serialize_for_hash(arr)
        )

    # Pandas Series
    if hasattr(obj, 'to_numpy') and hasattr(obj, 'name'):
        return (
            b'series:' +
            _serialize_for_hash(obj.name) + b':' +
            _serialize_for_hash(obj.to_numpy())
        )

    # Fallback: pickle (less ideal but handles arbitrary objects)
    return b'pickle:' + pickle.dumps(obj, protocol=4)


def generate_vhash(
    class_name: str,
    schema_version: int,
    data: Any,
    metadata: dict
) -> str:
    """
    Generate a version hash for a variable.

    Components:
    - class_name: The variable type (e.g., "RotationMatrix")
    - schema_version: Integer version of the to_db/from_db schema
    - data: The actual data being stored
    - metadata: The addressing metadata (subject, trial, etc.)

    Returns:
        16-character hex string
    """
    components = [
        f"class:{class_name}",
        f"schema:{schema_version}",
        f"data:{canonical_hash(data)}",
        f"meta:{canonical_hash(metadata)}"
    ]
    combined = "|".join(components).encode('utf-8')
    return hashlib.sha256(combined).hexdigest()[:16]
```

---

### 2. `scidb/exceptions.py` - Custom Exceptions

```python
"""Custom exceptions for scidb."""

class SciDBError(Exception):
    """Base exception for all scidb errors."""
    pass


class NotRegisteredError(SciDBError):
    """Raised when trying to save/load an unregistered variable type."""
    pass


class NotFoundError(SciDBError):
    """Raised when no matching data is found for the given metadata."""
    pass


class DatabaseNotConfiguredError(SciDBError):
    """Raised when trying to use implicit database before configuration."""
    pass


class ReservedMetadataKeyError(SciDBError):
    """Raised when user tries to use a reserved metadata key."""
    pass
```

---

### 3. `scidb/storage.py` - Serialization Helpers

```python
"""SQLite BLOB serialization helpers."""

import pickle
import sqlite3
from typing import Any

import pandas as pd


def serialize_dataframe(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to bytes for BLOB storage."""
    return pickle.dumps(df, protocol=4)


def deserialize_dataframe(blob: bytes) -> pd.DataFrame:
    """Deserialize bytes back to a DataFrame."""
    return pickle.loads(blob)


def adapt_dataframe(df: pd.DataFrame) -> bytes:
    """SQLite adapter for DataFrame."""
    return serialize_dataframe(df)


def convert_dataframe(blob: bytes) -> pd.DataFrame:
    """SQLite converter for DataFrame."""
    return deserialize_dataframe(blob)


def register_adapters():
    """Register custom SQLite adapters/converters."""
    sqlite3.register_adapter(pd.DataFrame, adapt_dataframe)
    sqlite3.register_converter("DATAFRAME", convert_dataframe)
```

---

### 4. `scidb/variable.py` - Base Variable Class

```python
"""Base class for database-storable variables."""

from abc import ABC, abstractmethod
from typing import Any, Self, TYPE_CHECKING
import re

import pandas as pd

if TYPE_CHECKING:
    from .database import DatabaseManager


class BaseVariable(ABC):
    """
    Abstract base class for all database-storable variable types.

    Subclasses must implement:
    - to_db(): Convert native data to DataFrame
    - from_db(): Convert DataFrame back to native data

    Example:
        class RotationMatrix(BaseVariable):
            schema_version = 1

            def to_db(self) -> pd.DataFrame:
                data = self.data
                return pd.DataFrame({
                    'row': [0, 0, 0, 1, 1, 1, 2, 2, 2],
                    'col': [0, 1, 2, 0, 1, 2, 0, 1, 2],
                    'value': data.flatten().tolist()
                })

            @classmethod
            def from_db(cls, df: pd.DataFrame) -> np.ndarray:
                values = df.sort_values(['row', 'col'])['value'].values
                return values.reshape(3, 3)
    """

    schema_version: int = 1

    # Reserved metadata keys that users cannot use
    _reserved_keys = frozenset({'vhash', 'id', 'created_at', 'schema_version', 'data'})

    def __init__(self, data: Any):
        """
        Initialize with native data.

        Args:
            data: The native Python object (numpy array, etc.)
        """
        self.data = data
        self._vhash: str | None = None
        self._metadata: dict | None = None

    @property
    def vhash(self) -> str | None:
        """The version hash, set after save() or load()."""
        return self._vhash

    @property
    def metadata(self) -> dict | None:
        """The metadata, set after save() or load()."""
        return self._metadata

    @abstractmethod
    def to_db(self) -> pd.DataFrame:
        """
        Convert native data to a DataFrame for storage.

        The DataFrame will be serialized and stored as a BLOB.
        This method defines the schema for this variable type.

        Returns:
            pd.DataFrame: Tabular representation of the data
        """
        pass

    @classmethod
    @abstractmethod
    def from_db(cls, df: pd.DataFrame) -> Any:
        """
        Convert a DataFrame back to the native data type.

        Args:
            df: The DataFrame retrieved from storage

        Returns:
            The native Python object
        """
        pass

    @classmethod
    def table_name(cls) -> str:
        """
        Get the SQLite table name for this variable type.

        Converts CamelCase class name to snake_case.

        Returns:
            str: Table name (e.g., "rotation_matrix")
        """
        name = cls.__name__
        # Insert underscore before uppercase letters, then lowercase
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

    def save(self, db: 'DatabaseManager | None' = None, **metadata) -> str:
        """
        Save this variable to the database.

        Args:
            db: Optional explicit database. If None, uses global database.
            **metadata: Addressing metadata (e.g., subject=1, trial=1)

        Returns:
            str: The vhash of the saved data

        Raises:
            ReservedMetadataKeyError: If metadata contains reserved keys
            NotRegisteredError: If this variable type is not registered
            DatabaseNotConfiguredError: If no database is available
        """
        from .database import get_database
        from .exceptions import ReservedMetadataKeyError

        # Validate metadata keys
        reserved_used = set(metadata.keys()) & self._reserved_keys
        if reserved_used:
            raise ReservedMetadataKeyError(
                f"Cannot use reserved metadata keys: {reserved_used}"
            )

        db = db or get_database()
        vhash = db.save(self, metadata)
        self._vhash = vhash
        self._metadata = metadata
        return vhash

    @classmethod
    def load(
        cls,
        db: 'DatabaseManager | None' = None,
        version: str = "latest",
        **metadata
    ) -> Self | list[Self]:
        """
        Load variable(s) from the database.

        Args:
            db: Optional explicit database. If None, uses global database.
            version: "latest" for most recent, or specific vhash
            **metadata: Addressing metadata to match

        Returns:
            If exactly one match: returns that instance
            If multiple matches: returns list of instances

        Raises:
            NotFoundError: If no matching data found
            NotRegisteredError: If this variable type is not registered
            DatabaseNotConfiguredError: If no database is available
        """
        from .database import get_database

        db = db or get_database()
        return db.load(cls, metadata, version=version)
```

---

### 5. `scidb/database.py` - Database Manager

```python
"""Database connection and management."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Type
import threading

import pandas as pd

from .variable import BaseVariable
from .hashing import generate_vhash
from .storage import serialize_dataframe, deserialize_dataframe, register_adapters
from .exceptions import (
    NotRegisteredError,
    NotFoundError,
    DatabaseNotConfiguredError,
    SciDBError
)


# Global database instance (thread-local for safety)
_local = threading.local()


def configure_database(db_path: str | Path) -> 'DatabaseManager':
    """
    Configure the global database connection.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        The DatabaseManager instance
    """
    _local.database = DatabaseManager(db_path)
    return _local.database


def get_database() -> 'DatabaseManager':
    """
    Get the global database connection.

    Returns:
        The DatabaseManager instance

    Raises:
        DatabaseNotConfiguredError: If configure_database() hasn't been called
    """
    db = getattr(_local, 'database', None)
    if db is None:
        raise DatabaseNotConfiguredError(
            "Database not configured. Call configure_database(path) first."
        )
    return db


class DatabaseManager:
    """
    Manages SQLite database connection and variable storage.

    Example:
        db = DatabaseManager("experiment.db")
        db.register(RotationMatrix)

        var = RotationMatrix(np.eye(3))
        vhash = var.save(db=db, subject=1, trial=1)

        loaded = RotationMatrix.load(db=db, subject=1, trial=1)
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
        """
        self.db_path = Path(db_path)
        self._registered_types: dict[str, Type[BaseVariable]] = {}

        # Register custom adapters
        register_adapters()

        # Connect with type detection
        self.connection = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False  # Allow multi-threaded access
        )
        self.connection.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        self.connection.execute("PRAGMA journal_mode=WAL")

        # Create metadata tables
        self._ensure_meta_tables()

    def _ensure_meta_tables(self):
        """Create internal metadata tables if they don't exist."""
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS _registered_types (
                type_name TEXT PRIMARY KEY,
                table_name TEXT UNIQUE NOT NULL,
                schema_version INTEGER NOT NULL,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS _version_log (
                vhash TEXT PRIMARY KEY,
                type_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                metadata JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.connection.commit()

    def register(self, variable_class: Type[BaseVariable]) -> None:
        """
        Register a variable type for storage.

        Creates the table if it doesn't exist. Must be called before
        save() or load() for this type.

        Args:
            variable_class: The BaseVariable subclass to register

        Raises:
            SciDBError: If registration fails
        """
        type_name = variable_class.__name__
        table_name = variable_class.table_name()
        schema_version = variable_class.schema_version

        # Create the data table
        self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vhash TEXT UNIQUE NOT NULL,
                schema_version INTEGER NOT NULL,
                metadata JSON NOT NULL,
                data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on vhash for fast lookups
        self.connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_vhash
            ON {table_name}(vhash)
        """)

        # Register in metadata table
        self.connection.execute("""
            INSERT OR REPLACE INTO _registered_types
            (type_name, table_name, schema_version, registered_at)
            VALUES (?, ?, ?, ?)
        """, (type_name, table_name, schema_version, datetime.now()))

        self.connection.commit()

        # Cache locally
        self._registered_types[type_name] = variable_class

    def _check_registered(self, variable_class: Type[BaseVariable]) -> str:
        """
        Check if a variable type is registered, return table name.

        Raises:
            NotRegisteredError: If not registered
        """
        type_name = variable_class.__name__

        # Check local cache first
        if type_name in self._registered_types:
            return variable_class.table_name()

        # Check database
        cursor = self.connection.execute(
            "SELECT table_name FROM _registered_types WHERE type_name = ?",
            (type_name,)
        )
        row = cursor.fetchone()

        if row is None:
            raise NotRegisteredError(
                f"Variable type '{type_name}' is not registered. "
                f"Call db.register({type_name}) first."
            )

        # Update local cache
        self._registered_types[type_name] = variable_class
        return row['table_name']

    def save(self, variable: BaseVariable, metadata: dict) -> str:
        """
        Save a variable to the database.

        If the exact same data+metadata already exists (same vhash),
        this is a no-op and returns the existing vhash.

        Args:
            variable: The variable instance to save
            metadata: Addressing metadata

        Returns:
            The vhash of the saved/existing data
        """
        table_name = self._check_registered(type(variable))
        type_name = variable.__class__.__name__

        # Generate vhash
        vhash = generate_vhash(
            class_name=type_name,
            schema_version=variable.schema_version,
            data=variable.data,
            metadata=metadata
        )

        # Check if already exists (idempotent save)
        cursor = self.connection.execute(
            f"SELECT vhash FROM {table_name} WHERE vhash = ?",
            (vhash,)
        )
        if cursor.fetchone() is not None:
            return vhash  # Already saved

        # Serialize and save
        df = variable.to_db()
        data_blob = serialize_dataframe(df)
        metadata_json = json.dumps(metadata, sort_keys=True)

        self.connection.execute(f"""
            INSERT INTO {table_name}
            (vhash, schema_version, metadata, data, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (vhash, variable.schema_version, metadata_json, data_blob, datetime.now()))

        # Log to version history
        self.connection.execute("""
            INSERT INTO _version_log
            (vhash, type_name, table_name, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (vhash, type_name, table_name, metadata_json, datetime.now()))

        self.connection.commit()
        return vhash

    def load(
        self,
        variable_class: Type[BaseVariable],
        metadata: dict,
        version: str = "latest"
    ) -> BaseVariable | list[BaseVariable]:
        """
        Load variable(s) matching the given metadata.

        Args:
            variable_class: The type to load
            metadata: Metadata to match (partial matching supported)
            version: "latest" or specific vhash

        Returns:
            Single instance if one match, list if multiple matches

        Raises:
            NotFoundError: If no matches found
        """
        table_name = self._check_registered(variable_class)

        if version != "latest" and version is not None:
            # Load specific version by vhash
            cursor = self.connection.execute(
                f"SELECT vhash, metadata, data FROM {table_name} WHERE vhash = ?",
                (version,)
            )
            row = cursor.fetchone()
            if row is None:
                raise NotFoundError(f"No data found with vhash '{version}'")

            return self._row_to_variable(variable_class, row)

        # Build query for metadata matching
        # Match all provided metadata keys
        conditions = []
        params = []
        for key, value in metadata.items():
            conditions.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(json.dumps(value) if isinstance(value, (dict, list)) else value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.connection.execute(
            f"""SELECT vhash, metadata, data FROM {table_name}
                WHERE {where_clause}
                ORDER BY created_at DESC""",
            params
        )
        rows = cursor.fetchall()

        if not rows:
            raise NotFoundError(
                f"No {variable_class.__name__} found matching metadata: {metadata}"
            )

        # Return single or list based on count
        results = [self._row_to_variable(variable_class, row) for row in rows]

        if len(results) == 1:
            return results[0]
        return results

    def _row_to_variable(
        self,
        variable_class: Type[BaseVariable],
        row: sqlite3.Row
    ) -> BaseVariable:
        """Convert a database row to a variable instance."""
        df = deserialize_dataframe(row['data'])
        data = variable_class.from_db(df)

        instance = variable_class(data)
        instance._vhash = row['vhash']
        instance._metadata = json.loads(row['metadata'])

        return instance

    def list_versions(
        self,
        variable_class: Type[BaseVariable],
        **metadata
    ) -> list[dict]:
        """
        List all versions matching the metadata.

        Args:
            variable_class: The type to query
            **metadata: Metadata to match

        Returns:
            List of dicts with vhash, metadata, created_at
        """
        table_name = self._check_registered(variable_class)

        conditions = []
        params = []
        for key, value in metadata.items():
            conditions.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.connection.execute(
            f"""SELECT vhash, metadata, created_at FROM {table_name}
                WHERE {where_clause}
                ORDER BY created_at DESC""",
            params
        )

        return [
            {
                'vhash': row['vhash'],
                'metadata': json.loads(row['metadata']),
                'created_at': row['created_at']
            }
            for row in cursor.fetchall()
        ]

    def close(self):
        """Close the database connection."""
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

---

### 6. `scidb/__init__.py` - Public API

```python
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
from .variable import BaseVariable
from .exceptions import (
    SciDBError,
    NotRegisteredError,
    NotFoundError,
    DatabaseNotConfiguredError,
    ReservedMetadataKeyError,
)

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
```

---

### 7. Example Variable: `scidb/variables/array.py`

```python
"""Generic numpy array variable type."""

import numpy as np
import pandas as pd

from ..variable import BaseVariable


class Array(BaseVariable):
    """
    Generic numpy array storage.

    Stores arrays of any shape by flattening with shape metadata.

    Example:
        arr = np.random.rand(10, 20, 30)
        var = Array(arr)
        var.save(subject=1, measurement="eeg")

        loaded = Array.load(subject=1, measurement="eeg")
        assert loaded.data.shape == (10, 20, 30)
    """

    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        """Store flattened array with shape/dtype metadata."""
        arr = np.asarray(self.data)
        return pd.DataFrame({
            'values': [arr.tobytes()],  # Raw bytes
            'shape': [str(arr.shape)],
            'dtype': [str(arr.dtype)],
        })

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        """Reconstruct array from stored bytes and metadata."""
        row = df.iloc[0]
        shape = eval(row['shape'])  # e.g., "(10, 20, 30)"
        dtype = np.dtype(row['dtype'])
        arr = np.frombuffer(row['values'], dtype=dtype)
        return arr.reshape(shape)
```

---

## Implementation Order

1. **Phase 1: Core Infrastructure**
   - [ ] `exceptions.py` - Custom exceptions
   - [ ] `hashing.py` - Deterministic hashing
   - [ ] `storage.py` - Serialization helpers
   - [ ] `variable.py` - BaseVariable ABC
   - [ ] `database.py` - DatabaseManager
   - [ ] `__init__.py` - Public API

2. **Phase 2: Testing**
   - [ ] Unit tests for hashing determinism
   - [ ] Unit tests for save/load round-trip
   - [ ] Integration tests for multi-version scenarios
   - [ ] Test idempotent saves

3. **Phase 3: Example Variables**
   - [ ] `Array` - Generic numpy array
   - [ ] `DataFrame` - Pandas DataFrame wrapper
   - [ ] `RotationMatrix` - Example from spec

4. **Phase 4: Enhancements (Later)**
   - [ ] Query interface (`db.query("subject > 5")`)
   - [ ] Version pruning/deletion
   - [ ] Schema migration helpers
   - [ ] CLI tools for database inspection

---

## Potential Pitfalls & Mitigations

| Pitfall | Mitigation |
|---------|------------|
| Pickle security (arbitrary code execution) | Only load from trusted databases; document this risk |
| Pickle version compatibility | Use protocol 4; document Python version requirements |
| Large BLOB performance | Monitor; hybrid storage is future option |
| Float precision in hashing | Use `tobytes()` directly, not string conversion |
| Thread safety | WAL mode + `check_same_thread=False` + thread-local globals |
| Schema evolution | `schema_version` field; migration methods in future phase |

---

## Thunk-Based Lineage System

The lineage system is built on a Python adaptation of Haskell's thunk concept. Thunks wrap functions to capture their inputs and outputs, enabling automatic provenance tracking.

### Core Thunk Classes

The thunk system consists of three main classes (adapted from `ross_thunk`):

```python
"""Thunk system for automatic lineage tracking."""

from typing import Callable, Any
from hashlib import sha256
from functools import wraps

STRING_REPR_DELIMITER = "-"


class Thunk:
    """
    Wraps a function to enable lineage tracking.

    When called, creates a PipelineThunk that tracks inputs.
    The function's bytecode is hashed for reproducibility checking.

    Example:
        @thunk(n_outputs=1)
        def process(raw, calibration):
            return raw * calibration
    """

    def __init__(self, fcn: Callable, n_outputs: int = 1):
        self.fcn = fcn
        self.n_outputs = n_outputs
        self.pipeline_thunks: tuple = ()

        # Hash function bytecode for reproducibility
        fcn_code = fcn.__code__.co_code
        fcn_hash = sha256(fcn_code).hexdigest()

        string_repr = f"{fcn_hash}{STRING_REPR_DELIMITER}{n_outputs}"
        self.hash = sha256(string_repr.encode()).hexdigest()

    def __call__(self, *args, **kwargs) -> 'OutputThunk':
        """Create a PipelineThunk and execute or defer."""
        pipeline_thunk = PipelineThunk(self, *args, **kwargs)

        # Check for existing equivalent PipelineThunk
        for existing in self.pipeline_thunks:
            if pipeline_thunk._matches(existing):
                pipeline_thunk = existing
                break
        else:
            self.pipeline_thunks = (*self.pipeline_thunks, pipeline_thunk)

        return pipeline_thunk(*args, **kwargs)


class PipelineThunk:
    """
    Represents a specific invocation of a Thunk with captured inputs.

    Tracks:
    - The parent Thunk (function definition)
    - All input arguments (positional and keyword)
    - Output(s) after execution
    """

    def __init__(self, thunk: Thunk, *args, **kwargs):
        self.thunk = thunk
        self.inputs: dict[str, Any] = {}

        # Capture positional args
        for i, arg in enumerate(args):
            self.inputs[f"arg_{i}"] = arg

        # Capture keyword args
        self.inputs.update(kwargs)

        self.outputs: tuple = ()

    @property
    def hash(self) -> str:
        """Dynamic hash based on thunk + inputs."""
        input_hash = sha256(str(sorted(self.inputs.items())).encode()).hexdigest()
        return sha256(f"{self.thunk.hash}{STRING_REPR_DELIMITER}{input_hash}".encode()).hexdigest()

    @property
    def is_complete(self) -> bool:
        """True if all inputs are concrete values (not pending thunks)."""
        for value in self.inputs.values():
            if isinstance(value, OutputThunk) and not value.is_complete:
                return False
        return True

    def __call__(self, *args, **kwargs) -> 'OutputThunk':
        """Execute the function if complete, return OutputThunk(s)."""
        result = (None,) * self.thunk.n_outputs

        if self.is_complete:
            # Unwrap any OutputThunk inputs to get raw values
            resolved_args = [
                arg.value if isinstance(arg, OutputThunk) else arg
                for arg in args
            ]
            result = self.thunk.fcn(*resolved_args, **kwargs)

            # Normalize to tuple
            if not isinstance(result, tuple):
                result = (result,)

        # Wrap outputs in OutputThunks
        outputs = tuple(
            OutputThunk(self, i, self.is_complete, val)
            for i, val in enumerate(result)
        )
        self.outputs = outputs

        return outputs[0] if len(outputs) == 1 else outputs

    def _matches(self, other: 'PipelineThunk') -> bool:
        """Check if this is equivalent to another PipelineThunk."""
        if self.thunk.hash != other.thunk.hash:
            return False
        return self.inputs == other.inputs


class OutputThunk:
    """
    Wraps a function output with lineage information.

    Contains:
    - Reference to the PipelineThunk that produced it
    - Output index (for multi-output functions)
    - The actual computed value

    This is the key to provenance: every OutputThunk knows its parent
    PipelineThunk, which knows its inputs (possibly other OutputThunks).
    """

    def __init__(
        self,
        pipeline_thunk: PipelineThunk,
        output_num: int,
        is_complete: bool,
        value: Any
    ):
        self.pipeline_thunk = pipeline_thunk
        self.output_num = output_num
        self.is_complete = is_complete
        self.value = value if is_complete else None

        self.hash = sha256(
            f"{pipeline_thunk.hash}{STRING_REPR_DELIMITER}output{STRING_REPR_DELIMITER}{output_num}".encode()
        ).hexdigest()

    def __eq__(self, other):
        if isinstance(other, OutputThunk):
            return self.hash == other.hash
        return self.value == other

    def __repr__(self):
        return f"OutputThunk(fn={self.pipeline_thunk.thunk.fcn.__name__}, value={self.value})"


def thunk(n_outputs: int = 1):
    """
    Decorator to convert a function into a Thunk.

    Example:
        @thunk(n_outputs=1)
        def process_signal(raw, calibration):
            return raw * calibration

        result = process_signal(raw_data, cal_data)  # Returns OutputThunk
        print(result.value)  # The actual computed value
        print(result.pipeline_thunk.inputs)  # {'arg_0': raw_data, 'arg_1': cal_data}
    """
    def decorator(fcn: Callable) -> Thunk:
        return Thunk(fcn, n_outputs)
    return decorator
```

---

### Lineage Extraction and Storage

When saving a variable whose data came from an `OutputThunk`, we extract and store the full lineage:

```python
"""Lineage extraction and storage."""

from typing import Any
from dataclasses import dataclass


@dataclass
class LineageRecord:
    """Represents the provenance of a single variable."""
    function_name: str
    function_hash: str
    inputs: list[dict]  # [{name, type, vhash, value_hash}, ...]
    constants: list[dict]  # [{name, value_hash, value_repr}, ...]


def extract_lineage(output_thunk: 'OutputThunk') -> LineageRecord:
    """
    Extract lineage information from an OutputThunk.

    Traverses the input graph to capture:
    - Function name and hash
    - Input variables (with their vhashes if saved)
    - Constant values
    """
    pt = output_thunk.pipeline_thunk

    inputs = []
    constants = []

    for name, value in pt.inputs.items():
        if isinstance(value, OutputThunk):
            # Input came from another thunk
            inputs.append({
                'name': name,
                'source_function': value.pipeline_thunk.thunk.fcn.__name__,
                'source_hash': value.hash,
                'output_num': value.output_num,
            })
        elif hasattr(value, '_vhash') and value._vhash is not None:
            # Input is a saved BaseVariable
            inputs.append({
                'name': name,
                'type': type(value).__name__,
                'vhash': value._vhash,
                'metadata': value._metadata,
            })
        else:
            # Input is a constant/literal
            from .hashing import canonical_hash
            constants.append({
                'name': name,
                'value_hash': canonical_hash(value),
                'value_repr': repr(value)[:200],  # Truncate for storage
            })

    return LineageRecord(
        function_name=pt.thunk.fcn.__name__,
        function_hash=pt.thunk.hash,
        inputs=inputs,
        constants=constants,
    )


def get_raw_value(data: Any) -> Any:
    """Unwrap OutputThunk to get raw value, or return as-is."""
    if isinstance(data, OutputThunk):
        return data.value
    return data
```

---

### Database Schema for Lineage

Add a lineage table and update the variable tables:

```sql
-- Lineage records table
CREATE TABLE _lineage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    output_vhash TEXT NOT NULL,           -- The variable that was produced
    output_type TEXT NOT NULL,            -- Variable type name
    function_name TEXT NOT NULL,          -- Name of function that produced it
    function_hash TEXT NOT NULL,          -- Hash of function bytecode
    inputs JSON NOT NULL,                 -- [{name, type, vhash, ...}, ...]
    constants JSON NOT NULL,              -- [{name, value_hash, value_repr}, ...]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(output_vhash)
);

-- Index for querying "what used this input?"
CREATE INDEX idx_lineage_inputs ON _lineage(json_extract(inputs, '$[*].vhash'));

-- Index for querying by function
CREATE INDEX idx_lineage_function ON _lineage(function_hash);
```

---

### Updated Variable.save() with Lineage

```python
def save(self, db: 'DatabaseManager | None' = None, **metadata) -> str:
    """
    Save this variable to the database, with automatic lineage extraction.

    If self.data is an OutputThunk, lineage is captured automatically.
    """
    from .database import get_database
    from .thunk import OutputThunk
    from .lineage import extract_lineage, get_raw_value

    db = db or get_database()

    # Extract lineage if data came from a thunk
    lineage = None
    if isinstance(self.data, OutputThunk):
        lineage = extract_lineage(self.data)
        # Unwrap to get actual value for storage
        raw_data = get_raw_value(self.data)
    else:
        raw_data = self.data

    # Store the raw data (not the OutputThunk wrapper)
    self._raw_data = raw_data

    vhash = db.save(self, metadata, lineage=lineage)
    self._vhash = vhash
    self._metadata = metadata
    return vhash
```

---

### Querying Provenance

```python
# In DatabaseManager class

def get_provenance(
    self,
    variable_class: Type[BaseVariable],
    version: str = None,
    **metadata
) -> dict | None:
    """
    Get the provenance (lineage) of a variable.

    Returns:
        {
            'function_name': 'process_signal',
            'function_hash': 'abc123...',
            'inputs': [
                {'name': 'arg_0', 'type': 'RawSignal', 'vhash': '...', 'metadata': {...}},
                {'name': 'arg_1', 'type': 'Calibration', 'vhash': '...', 'metadata': {...}},
            ],
            'constants': [
                {'name': 'arg_2', 'value_repr': '0.5', 'value_hash': '...'},
            ]
        }
    """
    # First, find the vhash
    if version:
        vhash = version
    else:
        var = self.load(variable_class, metadata)
        if isinstance(var, list):
            var = var[0]  # Latest
        vhash = var.vhash

    cursor = self.connection.execute(
        """SELECT function_name, function_hash, inputs, constants
           FROM _lineage WHERE output_vhash = ?""",
        (vhash,)
    )
    row = cursor.fetchone()

    if row is None:
        return None  # No lineage recorded (data wasn't from a thunk)

    return {
        'function_name': row['function_name'],
        'function_hash': row['function_hash'],
        'inputs': json.loads(row['inputs']),
        'constants': json.loads(row['constants']),
    }


def get_derived_from(
    self,
    variable_class: Type[BaseVariable],
    version: str = None,
    **metadata
) -> list[dict]:
    """
    Find all variables that were derived from this one.

    Answers: "What outputs used this variable as an input?"
    """
    if version:
        vhash = version
    else:
        var = self.load(variable_class, metadata)
        if isinstance(var, list):
            var = var[0]
        vhash = var.vhash

    # Search for this vhash in any lineage inputs
    cursor = self.connection.execute(
        """SELECT output_vhash, output_type, function_name, inputs
           FROM _lineage
           WHERE EXISTS (
               SELECT 1 FROM json_each(inputs)
               WHERE json_extract(value, '$.vhash') = ?
           )""",
        (vhash,)
    )

    return [
        {
            'vhash': row['output_vhash'],
            'type': row['output_type'],
            'function': row['function_name'],
        }
        for row in cursor.fetchall()
    ]
```

---

### Complete Usage Example

```python
from scidb import configure_database, BaseVariable
from scidb.thunk import thunk
import numpy as np
import pandas as pd

# === Setup ===
db = configure_database("experiment.db")

# === Define variable types ===
class RawSignal(BaseVariable):
    schema_version = 1
    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({'value': self.data.flatten()})
    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        return df['value'].values

class Calibration(BaseVariable):
    schema_version = 1
    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({'factor': [self.data]})
    @classmethod
    def from_db(cls, df: pd.DataFrame) -> float:
        return df['factor'].iloc[0]

class ProcessedSignal(BaseVariable):
    schema_version = 1
    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({'value': self.data.flatten()})
    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        return df['value'].values

db.register(RawSignal)
db.register(Calibration)
db.register(ProcessedSignal)

# === Define processing pipeline with thunks ===
@thunk(n_outputs=1)
def calibrate_signal(raw: np.ndarray, cal_factor: float) -> np.ndarray:
    return raw * cal_factor

@thunk(n_outputs=1)
def normalize(signal: np.ndarray) -> np.ndarray:
    return (signal - signal.mean()) / signal.std()

# === Save raw inputs ===
raw = RawSignal(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
raw.save(subject=1, trial=1)

cal = Calibration(2.5)
cal.save(subject=1)

# === Run pipeline (lineage captured automatically) ===
calibrated = calibrate_signal(raw, cal)  # Returns OutputThunk
normalized = normalize(calibrated)        # Chains lineage

# === Save output (lineage extracted automatically) ===
result = ProcessedSignal(normalized)
result.save(subject=1, trial=1, stage="final")

# === Query: What are all versions of ProcessedSignal for subject 1? ===
versions = db.list_versions(ProcessedSignal, subject=1)
# [{'vhash': 'abc123', 'metadata': {...}, 'created_at': ...}]

# === Query: What produced this ProcessedSignal? ===
provenance = db.get_provenance(ProcessedSignal, subject=1, trial=1, stage="final")
# {
#     'function_name': 'normalize',
#     'function_hash': 'def456...',
#     'inputs': [
#         {'name': 'arg_0', 'source_function': 'calibrate_signal', 'source_hash': '...'}
#     ],
#     'constants': []
# }

# === Query: What was derived from the raw signal? ===
derived = db.get_derived_from(RawSignal, subject=1, trial=1)
# [{'vhash': '...', 'type': 'ProcessedSignal', 'function': 'calibrate_signal'}]
```

---

## Computation Caching (Memoization)

The framework supports **persistent memoization**: if a computation has already been performed with the same inputs and function, the cached result is returned without re-executing. This enables efficient pipeline re-runs where only changed computations are executed.

### Core Concept

Before executing a thunked function, we compute a **cache key** from:
- Function bytecode hash (detects code changes)
- Input content hashes (detects data changes)

If this cache key exists in the database, we skip execution and return the cached result.

```
Pipeline Re-run Flow:

1. User calls: result = process_signal(raw, cal)
2. Compute cache_key = hash(function_hash + input_hashes)
3. Query: SELECT output_vhash FROM _computation_cache WHERE cache_key = ?
4. If found → Load cached result, skip execution
5. If not found → Execute function, cache result on save()
```

### Database Schema

```sql
-- Computation cache: maps (function + inputs) → output
CREATE TABLE _computation_cache (
    cache_key TEXT PRIMARY KEY,           -- hash(function_hash + input_hashes)
    function_name TEXT NOT NULL,          -- For debugging/inspection
    function_hash TEXT NOT NULL,          -- Bytecode hash
    input_summary JSON NOT NULL,          -- Brief description of inputs for debugging
    output_vhash TEXT NOT NULL,           -- The cached result's vhash
    output_type TEXT NOT NULL,            -- Variable type name
    hit_count INTEGER DEFAULT 0,          -- Number of cache hits (for analytics)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_hit_at TIMESTAMP,                -- Last time this cache entry was used

    FOREIGN KEY (output_vhash) REFERENCES _version_log(vhash)
);

CREATE INDEX idx_cache_function ON _computation_cache(function_hash);
CREATE INDEX idx_cache_output ON _computation_cache(output_vhash);
```

### Updated Thunk Classes

```python
class PipelineThunk:
    """Updated with cache-aware execution."""

    def __init__(self, thunk: Thunk, *args, **kwargs):
        self.thunk = thunk
        self.inputs: dict[str, Any] = {}
        self._cache_key: str | None = None
        self._cached_result: OutputThunk | None = None

        # Capture inputs
        for i, arg in enumerate(args):
            self.inputs[f"arg_{i}"] = arg
        self.inputs.update(kwargs)

        self.outputs: tuple = ()

    def _compute_cache_key(self) -> str | None:
        """
        Compute cache key from function + inputs.

        Returns None if inputs cannot be hashed (e.g., contain unhashable objects
        that aren't OutputThunks or saved Variables).
        """
        try:
            input_hashes = []
            for name, value in sorted(self.inputs.items()):
                if isinstance(value, OutputThunk):
                    # Use OutputThunk's hash (encodes its full lineage)
                    input_hashes.append((name, 'output', value.hash))
                elif hasattr(value, '_vhash') and value._vhash is not None:
                    # Saved BaseVariable - use its vhash
                    input_hashes.append((name, 'variable', value._vhash))
                elif hasattr(value, 'vhash') and value.vhash is not None:
                    # Also check property-style vhash
                    input_hashes.append((name, 'variable', value.vhash))
                else:
                    # Raw value - compute content hash
                    from .hashing import canonical_hash
                    input_hashes.append((name, 'constant', canonical_hash(value)))

            combined = f"{self.thunk.hash}|{input_hashes}"
            return hashlib.sha256(combined.encode()).hexdigest()
        except Exception:
            # If hashing fails, return None (will skip cache lookup)
            return None

    def _lookup_cache(self) -> 'OutputThunk | tuple[OutputThunk, ...] | None':
        """
        Check if this computation is cached.

        Returns cached OutputThunk(s) if found, None otherwise.
        """
        from .database import get_database

        cache_key = self._compute_cache_key()
        if cache_key is None:
            return None

        self._cache_key = cache_key

        try:
            db = get_database()
            cached = db.get_cached_computation(cache_key)
            if cached is not None:
                # Build OutputThunk(s) from cached data
                return self._build_cached_output(cached)
        except Exception:
            # If database not configured or lookup fails, proceed without cache
            pass

        return None

    def _build_cached_output(self, cached: dict) -> 'OutputThunk | tuple[OutputThunk, ...]':
        """Reconstruct OutputThunk(s) from cached database result."""
        # Load the actual data
        from .database import get_database
        db = get_database()

        # Get the variable class and load it
        var = db.load_by_vhash(cached['output_vhash'], cached['output_type'])

        # Wrap in OutputThunk for API consistency
        output = OutputThunk(
            pipeline_thunk=self,
            output_num=0,
            is_complete=True,
            value=var.data
        )
        output._from_cache = True
        output._cached_vhash = cached['output_vhash']

        self.outputs = (output,)
        return output if self.thunk.n_outputs == 1 else self.outputs

    def __call__(self, *args, force: bool = False, **kwargs) -> 'OutputThunk':
        """
        Execute the function, using cache if available.

        Args:
            *args: Function arguments
            force: If True, bypass cache and force re-execution
            **kwargs: Function keyword arguments

        Returns:
            OutputThunk wrapping the result
        """
        # Check cache first (unless force=True)
        if not force and self.is_complete:
            cached = self._lookup_cache()
            if cached is not None:
                return cached

        # Cache miss or force - execute the function
        result = (None,) * self.thunk.n_outputs

        if self.is_complete:
            # Unwrap any OutputThunk inputs to get raw values
            resolved_args = []
            for arg in args:
                if isinstance(arg, OutputThunk):
                    resolved_args.append(arg.value)
                elif hasattr(arg, 'data'):
                    # BaseVariable - use its data
                    resolved_args.append(arg.data)
                else:
                    resolved_args.append(arg)

            resolved_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, OutputThunk):
                    resolved_kwargs[k] = v.value
                elif hasattr(v, 'data'):
                    resolved_kwargs[k] = v.data
                else:
                    resolved_kwargs[k] = v

            result = self.thunk.fcn(*resolved_args, **resolved_kwargs)

            if not isinstance(result, tuple):
                result = (result,)

        # Wrap outputs in OutputThunks
        outputs = tuple(
            OutputThunk(self, i, self.is_complete, val)
            for i, val in enumerate(result)
        )
        self.outputs = outputs

        return outputs[0] if len(outputs) == 1 else outputs


class OutputThunk:
    """Updated with cache awareness."""

    def __init__(self, pipeline_thunk, output_num, is_complete, value):
        self.pipeline_thunk = pipeline_thunk
        self.output_num = output_num
        self.is_complete = is_complete
        self.value = value if is_complete else None

        # Cache metadata
        self._from_cache: bool = False
        self._cached_vhash: str | None = None

        self.hash = sha256(
            f"{pipeline_thunk.hash}-output-{output_num}".encode()
        ).hexdigest()

    @property
    def was_cached(self) -> bool:
        """True if this result was loaded from cache (not computed)."""
        return self._from_cache
```

### Database Methods for Caching

```python
# In DatabaseManager class

def get_cached_computation(self, cache_key: str) -> dict | None:
    """
    Look up a cached computation result.

    Returns:
        {'output_vhash': '...', 'output_type': '...'} if cached, None otherwise
    """
    cursor = self.connection.execute(
        """SELECT output_vhash, output_type
           FROM _computation_cache
           WHERE cache_key = ?""",
        (cache_key,)
    )
    row = cursor.fetchone()

    if row is None:
        return None

    # Update hit count and last_hit_at
    self.connection.execute(
        """UPDATE _computation_cache
           SET hit_count = hit_count + 1, last_hit_at = ?
           WHERE cache_key = ?""",
        (datetime.now(), cache_key)
    )
    self.connection.commit()

    return {
        'output_vhash': row['output_vhash'],
        'output_type': row['output_type'],
    }

def cache_computation(
    self,
    cache_key: str,
    function_name: str,
    function_hash: str,
    input_summary: dict,
    output_vhash: str,
    output_type: str
) -> None:
    """
    Cache a computation result for future lookups.
    """
    self.connection.execute(
        """INSERT OR REPLACE INTO _computation_cache
           (cache_key, function_name, function_hash, input_summary,
            output_vhash, output_type, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (cache_key, function_name, function_hash, json.dumps(input_summary),
         output_vhash, output_type, datetime.now())
    )
    self.connection.commit()

def load_by_vhash(self, vhash: str, type_name: str) -> BaseVariable:
    """
    Load a variable by its vhash and type name.

    Used for cache retrieval when we know the exact version.
    """
    # Look up the variable class from registered types
    if type_name not in self._registered_types:
        # Try to find it in the database
        cursor = self.connection.execute(
            "SELECT table_name FROM _registered_types WHERE type_name = ?",
            (type_name,)
        )
        row = cursor.fetchone()
        if row is None:
            raise NotRegisteredError(f"Type '{type_name}' not registered")

    variable_class = self._registered_types[type_name]
    table_name = variable_class.table_name()

    cursor = self.connection.execute(
        f"SELECT vhash, metadata, data FROM {table_name} WHERE vhash = ?",
        (vhash,)
    )
    row = cursor.fetchone()

    if row is None:
        raise NotFoundError(f"No data found with vhash '{vhash}'")

    return self._row_to_variable(variable_class, row)

def invalidate_cache(
    self,
    function_name: str = None,
    function_hash: str = None,
    output_vhash: str = None
) -> int:
    """
    Invalidate (delete) cached computations.

    Args:
        function_name: Invalidate all caches for this function
        function_hash: Invalidate all caches for this specific function version
        output_vhash: Invalidate cache for this specific output

    Returns:
        Number of cache entries invalidated
    """
    conditions = []
    params = []

    if function_name:
        conditions.append("function_name = ?")
        params.append(function_name)
    if function_hash:
        conditions.append("function_hash = ?")
        params.append(function_hash)
    if output_vhash:
        conditions.append("output_vhash = ?")
        params.append(output_vhash)

    if not conditions:
        return 0

    where_clause = " AND ".join(conditions)
    cursor = self.connection.execute(
        f"DELETE FROM _computation_cache WHERE {where_clause}",
        params
    )
    self.connection.commit()

    return cursor.rowcount

def get_cache_stats(self) -> dict:
    """
    Get cache statistics.

    Returns:
        {
            'total_entries': int,
            'total_hits': int,
            'top_functions': [{'name': str, 'entries': int, 'hits': int}, ...]
        }
    """
    cursor = self.connection.execute(
        """SELECT COUNT(*) as total, SUM(hit_count) as hits
           FROM _computation_cache"""
    )
    row = cursor.fetchone()

    cursor2 = self.connection.execute(
        """SELECT function_name, COUNT(*) as entries, SUM(hit_count) as hits
           FROM _computation_cache
           GROUP BY function_name
           ORDER BY hits DESC
           LIMIT 10"""
    )

    return {
        'total_entries': row['total'] or 0,
        'total_hits': row['hits'] or 0,
        'top_functions': [
            {'name': r['function_name'], 'entries': r['entries'], 'hits': r['hits']}
            for r in cursor2.fetchall()
        ]
    }
```

### Updated save() with Cache Population

```python
# In BaseVariable class

def save(self, db: 'DatabaseManager | None' = None, **metadata) -> str:
    """
    Save variable and cache computation if from thunk.
    """
    from .database import get_database
    from .thunk import OutputThunk
    from .lineage import extract_lineage, get_raw_value

    db = db or get_database()

    # Extract lineage if data came from a thunk
    lineage = None
    cache_info = None

    if isinstance(self.data, OutputThunk):
        output_thunk = self.data
        lineage = extract_lineage(output_thunk)

        # Prepare cache info (will be saved after we have the vhash)
        if output_thunk.pipeline_thunk._cache_key:
            cache_info = {
                'cache_key': output_thunk.pipeline_thunk._cache_key,
                'function_name': output_thunk.pipeline_thunk.thunk.fcn.__name__,
                'function_hash': output_thunk.pipeline_thunk.thunk.hash,
                'input_summary': self._summarize_inputs(output_thunk.pipeline_thunk.inputs),
            }

        # Unwrap to get actual value
        raw_data = get_raw_value(self.data)
    else:
        raw_data = self.data

    self._raw_data = raw_data

    # Save the variable
    vhash = db.save(self, metadata, lineage=lineage)
    self._vhash = vhash
    self._metadata = metadata

    # Populate cache if this came from a thunk
    if cache_info:
        db.cache_computation(
            cache_key=cache_info['cache_key'],
            function_name=cache_info['function_name'],
            function_hash=cache_info['function_hash'],
            input_summary=cache_info['input_summary'],
            output_vhash=vhash,
            output_type=self.__class__.__name__
        )

    return vhash

def _summarize_inputs(self, inputs: dict) -> dict:
    """Create a human-readable summary of inputs for debugging."""
    summary = {}
    for name, value in inputs.items():
        if isinstance(value, OutputThunk):
            summary[name] = f"OutputThunk({value.pipeline_thunk.thunk.fcn.__name__})"
        elif hasattr(value, '__class__'):
            summary[name] = f"{value.__class__.__name__}"
        else:
            summary[name] = repr(value)[:50]
    return summary
```

### Complete Pipeline Example with Caching

```python
from scidb import configure_database, BaseVariable
from scidb.thunk import thunk
import numpy as np
import time

# Setup
db = configure_database("experiment.db")

class RawSignal(BaseVariable):
    schema_version = 1
    def to_db(self): return pd.DataFrame({'v': self.data.flatten()})
    @classmethod
    def from_db(cls, df): return df['v'].values

class ProcessedSignal(BaseVariable):
    schema_version = 1
    def to_db(self): return pd.DataFrame({'v': self.data.flatten()})
    @classmethod
    def from_db(cls, df): return df['v'].values

db.register(RawSignal)
db.register(ProcessedSignal)

@thunk(n_outputs=1)
def expensive_processing(data: np.ndarray) -> np.ndarray:
    print(f"  [Executing expensive_processing...]")
    time.sleep(2)  # Simulate expensive computation
    return data * 2 + np.sin(data)

# === First run: All computations execute ===
print("=== First Pipeline Run ===")

for subject in [1, 2, 3]:
    for trial in [1, 2]:
        raw = RawSignal(np.random.rand(100))
        raw.save(subject=subject, trial=trial)

        result = expensive_processing(raw)  # Executes (cache miss)
        ProcessedSignal(result).save(subject=subject, trial=trial)

print(f"Cache stats: {db.get_cache_stats()}")

# === Second run: All computations skip (cache hits) ===
print("\n=== Second Pipeline Run (Re-run) ===")

for subject in [1, 2, 3]:
    for trial in [1, 2]:
        raw = RawSignal.load(subject=subject, trial=trial)

        result = expensive_processing(raw)  # Skipped! (cache hit)

        print(f"  Subject {subject}, Trial {trial}: cached={result.was_cached}")

        # Save is idempotent - same vhash, no duplicate
        ProcessedSignal(result).save(subject=subject, trial=trial)

print(f"Cache stats: {db.get_cache_stats()}")

# === Third run: Only new data executes ===
print("\n=== Third Pipeline Run (New trial added) ===")

for subject in [1, 2, 3]:
    for trial in [1, 2, 3]:  # Trial 3 is new!
        try:
            raw = RawSignal.load(subject=subject, trial=trial)
        except NotFoundError:
            raw = RawSignal(np.random.rand(100))
            raw.save(subject=subject, trial=trial)

        result = expensive_processing(raw)  # Only trial 3 executes
        ProcessedSignal(result).save(subject=subject, trial=trial)

# === Force re-execution ===
print("\n=== Force Re-execution ===")
raw = RawSignal.load(subject=1, trial=1)
result = expensive_processing(raw, force=True)  # Forces execution despite cache
print(f"  Forced execution: cached={result.was_cached}")  # False
```

**Output:**
```
=== First Pipeline Run ===
  [Executing expensive_processing...]
  [Executing expensive_processing...]
  ... (6 executions)
Cache stats: {'total_entries': 6, 'total_hits': 0, 'top_functions': [...]}

=== Second Pipeline Run (Re-run) ===
  Subject 1, Trial 1: cached=True
  Subject 1, Trial 2: cached=True
  ... (no "[Executing...]" prints - all cached!)
Cache stats: {'total_entries': 6, 'total_hits': 6, 'top_functions': [...]}

=== Third Pipeline Run (New trial added) ===
  [Executing expensive_processing...]  # Only trial 3 for each subject
  [Executing expensive_processing...]
  [Executing expensive_processing...]

=== Force Re-execution ===
  [Executing expensive_processing...]
  Forced execution: cached=False
```

---

## Implementation Order (Updated)

1. **Phase 1: Core Infrastructure**
   - [ ] `exceptions.py` - Custom exceptions
   - [ ] `hashing.py` - Deterministic hashing
   - [ ] `storage.py` - Serialization helpers
   - [ ] `variable.py` - BaseVariable ABC
   - [ ] `database.py` - DatabaseManager (with `_registered_types`, `_version_log` tables)
   - [ ] `__init__.py` - Public API

2. **Phase 2: Thunk System & Lineage**
   - [ ] `thunk.py` - Thunk, PipelineThunk, OutputThunk classes
   - [ ] `lineage.py` - LineageRecord, extract_lineage(), get_raw_value()
   - [ ] Update `database.py` - Add `_lineage` table, save with lineage
   - [ ] Update `variable.py` - Lineage extraction in save()

3. **Phase 3: Computation Caching**
   - [ ] Add `_computation_cache` table to database schema
   - [ ] `PipelineThunk._compute_cache_key()` - Pre-execution cache key
   - [ ] `PipelineThunk._lookup_cache()` - Check for cached result
   - [ ] `DatabaseManager.get_cached_computation()` - Cache lookup
   - [ ] `DatabaseManager.cache_computation()` - Cache population
   - [ ] `DatabaseManager.load_by_vhash()` - Load specific version
   - [ ] Update `variable.py` save() - Populate cache after saving
   - [ ] `OutputThunk.was_cached` property
   - [ ] `force=True` parameter to bypass cache

4. **Phase 4: Provenance Queries**
   - [ ] `database.py` - get_provenance() method
   - [ ] `database.py` - get_derived_from() method
   - [ ] `database.py` - get_lineage_graph() for full DAG visualization

5. **Phase 5: Cache Management**
   - [ ] `DatabaseManager.invalidate_cache()` - Clear cache entries
   - [ ] `DatabaseManager.get_cache_stats()` - Cache analytics
   - [ ] Cache warming utilities (optional)

6. **Phase 6: Testing**
   - [ ] Unit tests for hashing determinism
   - [ ] Unit tests for thunk hash stability
   - [ ] Unit tests for lineage extraction
   - [ ] Unit tests for cache key computation
   - [ ] Integration tests for cache hit/miss scenarios
   - [ ] Integration tests for full pipeline save/load
   - [ ] Test provenance queries
   - [ ] Test `force=True` bypass

7. **Phase 7: Example Variables**
   - [ ] `Array` - Generic numpy array
   - [ ] `DataFrame` - Pandas DataFrame wrapper
   - [ ] Example pipeline with lineage and caching

8. **Phase 8: Enhancements (Later)**
   - [ ] Query interface (`db.query("subject > 5")`)
   - [ ] Version pruning/deletion
   - [ ] Schema migration helpers
   - [ ] CLI tools for database inspection
   - [ ] Lineage visualization (DAG rendering)
   - [ ] Cache TTL / automatic expiration

---

## Potential Pitfalls & Mitigations (Updated)

| Pitfall | Mitigation |
|---------|------------|
| Pickle security (arbitrary code execution) | Only load from trusted databases; document this risk |
| Pickle version compatibility | Use protocol 4; document Python version requirements |
| Large BLOB performance | Monitor; hybrid storage is future option |
| Float precision in hashing | Use `tobytes()` directly, not string conversion |
| Thread safety | WAL mode + `check_same_thread=False` + thread-local globals |
| Schema evolution | `schema_version` field; migration methods in future phase |
| **Function bytecode changes** | Hash includes bytecode; changing function = new hash (intended behavior) |
| **Circular lineage** | DAG structure prevents cycles by construction |
| **Large lineage graphs** | Store only immediate parents; reconstruct full graph on query |
| **Unsaved inputs in lineage** | Store value_hash + repr for constants; warn if OutputThunk input not saved |
| **Stale cache after function edit** | Function bytecode hash changes → different cache key → cache miss (correct!) |
| **Cache grows unbounded** | Add cache TTL, LRU eviction, or manual `invalidate_cache()` in future |
| **Cache hit but output deleted** | `load_by_vhash()` fails → fall back to re-execution |
| **Non-deterministic functions** | User responsibility; document that cached results assume determinism |
| **Side effects in thunked functions** | Side effects won't re-run on cache hit; document this limitation |

---

## Key Questions This Design Answers

| Question | How It's Answered |
|----------|-------------------|
| "What are all values MyVar has had?" | `db.list_versions(MyVar, subject=1, trial=1)` |
| "Under what conditions was MyVar generated?" | `db.get_provenance(MyVar, ...)` → function, inputs, constants |
| "What was derived from this input?" | `db.get_derived_from(InputVar, ...)` |
| "Has this computation been done before?" | Automatic cache lookup before execution |
| "Can I skip redundant computations?" | Yes - cache hit returns stored result without re-execution |
| "Did the function code change?" | Function bytecode hash in cache key; code change = cache miss |

---

## Open Questions for Future

1. **Schema Migration**: How to handle loading old schema versions?
2. **Garbage Collection**: When to auto-prune old versions?
3. **Compression**: Should BLOBs be compressed (zlib)?
4. **Export**: How to export subsets of data for sharing?
5. **Lineage Visualization**: How to render the DAG (graphviz, mermaid, etc.)?
6. **Function Source Storage**: Should we store the full function source code for reproducibility?
7. **Cross-Database Lineage**: How to track lineage when inputs come from a different database file?
8. **Cache Eviction Policy**: LRU, TTL, or manual-only?
9. **Distributed Caching**: Share cache across multiple machines/users?
10. **Partial Re-execution**: If one input changes, can we reuse intermediate results?
