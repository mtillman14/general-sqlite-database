"""Base class for database-storable variables."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self
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
    _reserved_keys = frozenset(
        {"vhash", "id", "created_at", "schema_version", "data"}
    )

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
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def save(self, db: "DatabaseManager | None" = None, **metadata) -> str:
        """
        Save this variable to the database.

        If self.data is an OutputThunk (from a thunked computation),
        lineage is automatically extracted and stored, and the computation
        is cached for future reuse.

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
        from .thunk import OutputThunk
        from .lineage import extract_lineage, get_raw_value

        # Validate metadata keys
        reserved_used = set(metadata.keys()) & self._reserved_keys
        if reserved_used:
            raise ReservedMetadataKeyError(
                f"Cannot use reserved metadata keys: {reserved_used}"
            )

        db = db or get_database()

        # Extract lineage and cache info if data came from a thunk
        lineage = None
        output_thunk = None
        if isinstance(self.data, OutputThunk):
            output_thunk = self.data
            lineage = extract_lineage(output_thunk)
            # Unwrap to get actual value for hashing and storage
            self.data = get_raw_value(output_thunk)

        vhash = db.save(self, metadata, lineage=lineage)
        self._vhash = vhash
        self._metadata = metadata

        # Populate computation cache if this came from a thunk
        if output_thunk is not None:
            cache_key = output_thunk.pipeline_thunk.compute_cache_key()
            db.cache_computation(
                cache_key=cache_key,
                function_name=output_thunk.pipeline_thunk.thunk.fcn.__name__,
                function_hash=output_thunk.pipeline_thunk.thunk.hash,
                output_type=self.__class__.__name__,
                output_vhash=vhash,
            )

        return vhash

    @classmethod
    def load(
        cls,
        db: "DatabaseManager | None" = None,
        version: str = "latest",
        **metadata,
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
