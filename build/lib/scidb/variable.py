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

    # Global registry of all subclasses (for auto-registration with database)
    _all_subclasses: dict[str, type["BaseVariable"]] = {}

    def __init_subclass__(cls, **kwargs):
        """Register subclass in global registry when defined."""
        super().__init_subclass__(**kwargs)
        cls._all_subclasses[cls.__name__] = cls

    @classmethod
    def get_subclass_by_name(cls, name: str) -> type["BaseVariable"] | None:
        """Look up a subclass by its name from the global registry."""
        return cls._all_subclasses.get(name)

    def __init__(self, data: Any):
        """
        Initialize with native data.

        Args:
            data: The native Python object (numpy array, etc.)
        """
        self.data = data
        self._vhash: str | None = None
        self._metadata: dict | None = None
        self._content_hash: str | None = None
        self._lineage_hash: str | None = None

    @property
    def vhash(self) -> str | None:
        """The version hash, set after save() or load()."""
        return self._vhash

    @property
    def metadata(self) -> dict | None:
        """The metadata, set after save() or load()."""
        return self._metadata

    @property
    def content_hash(self) -> str | None:
        """The content hash (data identity), set after save() or load()."""
        return self._content_hash

    @property
    def lineage_hash(self) -> str | None:
        """
        The lineage hash (computational identity), set after save() or load().

        For raw data (not from a thunk), this is None.
        For computed data, this captures how the value was computed.
        """
        return self._lineage_hash

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
            str: Table name (e.g., "rotation_matrix", "emg_data")
        """
        # Convert CamelCase to snake_case
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

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
            UnsavedIntermediateError: If strict mode and unsaved intermediates exist
        """
        from .database import get_database, get_user_id
        from .exceptions import ReservedMetadataKeyError, UnsavedIntermediateError
        from .thunk import OutputThunk
        from .lineage import extract_lineage, find_unsaved_variables, get_raw_value

        # Validate metadata keys
        reserved_used = set(metadata.keys()) & self._reserved_keys
        if reserved_used:
            raise ReservedMetadataKeyError(
                f"Cannot use reserved metadata keys: {reserved_used}"
            )

        db = db or get_database()

        # Extract lineage and cache info if data came from a thunk
        lineage = None
        lineage_hash = None
        output_thunk = None
        if isinstance(self.data, OutputThunk):
            output_thunk = self.data

            # Check for unsaved intermediates based on lineage mode
            unsaved = find_unsaved_variables(output_thunk)

            if db.lineage_mode == "strict" and unsaved:
                # Strict mode: raise error for unsaved intermediates
                var_descriptions = []
                for var, path in unsaved:
                    var_type = type(var).__name__
                    var_descriptions.append(f"  - {var_type} (path: {path})")
                vars_str = "\n".join(var_descriptions)
                raise UnsavedIntermediateError(
                    f"Strict lineage mode requires all intermediate variables to be saved.\n"
                    f"Found {len(unsaved)} unsaved variable(s) in the computation chain:\n"
                    f"{vars_str}\n\n"
                    f"Either save these variables first, or use lineage_mode='ephemeral' "
                    f"in configure_database() to allow unsaved intermediates."
                )

            elif db.lineage_mode == "ephemeral" and unsaved:
                # Ephemeral mode: save lineage for unsaved intermediates without data
                user_id = get_user_id()
                for var, path in unsaved:
                    inner_data = getattr(var, "data", None)
                    if isinstance(inner_data, OutputThunk):
                        # Generate ephemeral ID from the OutputThunk's hash
                        ephemeral_id = f"ephemeral:{inner_data.hash[:32]}"
                        var_type = type(var).__name__

                        # Extract lineage for this intermediate computation
                        intermediate_lineage = extract_lineage(inner_data)

                        # Save ephemeral lineage record
                        db.save_ephemeral_lineage(
                            ephemeral_id=ephemeral_id,
                            variable_type=var_type,
                            lineage=intermediate_lineage,
                            user_id=user_id,
                        )

            lineage = extract_lineage(output_thunk)
            # Compute lineage_hash for cache key computation on reload
            lineage_hash = output_thunk.pipeline_thunk.compute_cache_key()
            # Unwrap to get actual value for hashing and storage
            self.data = get_raw_value(output_thunk)

        vhash = db.save(self, metadata, lineage=lineage, lineage_hash=lineage_hash)
        self._vhash = vhash
        self._metadata = metadata

        # Populate computation cache if this came from a thunk
        if output_thunk is not None:
            cache_key = lineage_hash  # Already computed above
            db.cache_computation(
                cache_key=cache_key,
                function_name=output_thunk.pipeline_thunk.thunk.fcn.__name__,
                function_hash=output_thunk.pipeline_thunk.thunk.hash,
                output_type=self.__class__.__name__,
                output_vhash=vhash,
                output_num=output_thunk.output_num,
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

    @classmethod
    def save_from_dataframe(
        cls,
        df: pd.DataFrame,
        data_column: str,
        metadata_columns: list[str],
        db: "DatabaseManager | None" = None,
        **common_metadata,
    ) -> list[str]:
        """
        Save each row of a DataFrame as a separate database record.

        Use this when a DataFrame contains multiple independent data items,
        each with its own metadata (e.g., different subjects/trials per row).

        Args:
            df: DataFrame where each row is a separate data item
            data_column: Column name containing the data to store
            metadata_columns: Column names to use as metadata for each row
            db: Optional explicit database. If None, uses global database.
            **common_metadata: Additional metadata applied to all rows

        Returns:
            List of vhashes for each saved record

        Example:
            # DataFrame with 10 rows (2 subjects x 5 trials)
            #   Subject  Trial  MyVar
            #   1        1      0.5
            #   1        2      0.6
            #   ...

            vhashes = ScalarValue.save_from_dataframe(
                df=results_df,
                data_column="MyVar",
                metadata_columns=["Subject", "Trial"],
                experiment="exp1"  # Applied to all rows
            )
        """
        from .database import get_database

        db = db or get_database()
        vhashes = []

        for _, row in df.iterrows():
            # Extract row-specific metadata from columns
            # Convert numpy types to native Python types for JSON serialization
            row_metadata = {}
            for col in metadata_columns:
                val = row[col]
                # Convert numpy types to native Python types
                if hasattr(val, "item"):
                    val = val.item()
                row_metadata[col] = val

            # Combine with common metadata
            full_metadata = {**common_metadata, **row_metadata}

            # Extract data and save
            data = row[data_column]
            instance = cls(data)
            vhash = instance.save(db=db, **full_metadata)
            vhashes.append(vhash)

        return vhashes

    @classmethod
    def load_to_dataframe(
        cls,
        db: "DatabaseManager | None" = None,
        include_vhash: bool = False,
        **metadata,
    ) -> pd.DataFrame:
        """
        Load matching records and reconstruct a DataFrame.

        This is the inverse of save_from_dataframe(). Loads all matching
        records and returns them as a DataFrame with metadata as columns.

        Args:
            db: Optional explicit database. If None, uses global database.
            include_vhash: If True, include vhash as a column
            **metadata: Metadata filter (loads all matching records)

        Returns:
            DataFrame with columns for each metadata key plus 'data' column

        Example:
            # Load all records for experiment
            df = ScalarValue.load_to_dataframe(experiment="exp1")
            # Returns:
            #   Subject  Trial  data   (vhash)
            #   1        1      0.5    abc123...
            #   1        2      0.6    def456...
            #   ...

        Raises:
            NotFoundError: If no matching data found
            NotRegisteredError: If this variable type is not registered
        """
        from .database import get_database

        db = db or get_database()

        # Load all matching records
        results = db.load(cls, metadata, version="latest")

        # Ensure it's a list
        if not isinstance(results, list):
            results = [results]

        # Build DataFrame rows
        rows = []
        for var in results:
            row = dict(var.metadata) if var.metadata else {}
            row["data"] = var.data
            if include_vhash:
                row["vhash"] = var.vhash
            rows.append(row)

        return pd.DataFrame(rows)

    def to_csv(self, path: str) -> None:
        """
        Export this variable's data to a CSV file.

        Exports the DataFrame representation (from to_db()) to CSV format
        for viewing in external tools.

        Args:
            path: Output file path (will be overwritten if exists)

        Example:
            var = TimeSeries.load(db=db, subject=1)
            var.to_csv("subject1_data.csv")
        """
        df = self.to_db()
        df.to_csv(path, index=False)

    def get_preview(self) -> str:
        """
        Get a human-readable preview of this variable's data.

        Returns:
            A string summarizing the data (shape, sample values, stats)

        Example:
            var = TimeSeries.load(db=db, subject=1)
            print(var.get_preview())
        """
        from .preview import generate_preview

        df = self.to_db()
        return generate_preview(df)
