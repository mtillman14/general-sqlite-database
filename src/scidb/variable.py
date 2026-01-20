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
        {"record_id", "id", "created_at", "schema_version"}
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
        self._record_id: str | None = None
        self._metadata: dict | None = None
        self._content_hash: str | None = None
        self._lineage_hash: str | None = None

    @property
    def record_id(self) -> str | None:
        """The unique record ID, set after save() or load()."""
        return self._record_id

    @property
    def metadata(self) -> dict | None:
        """The metadata, set after save() or load()."""
        return self._metadata

    @property
    def content_hash(self) -> str | None:
        """The content hash (data identity). Always computed fresh from current data."""
        if self.data is None:
            return None
        from .hashing import canonical_hash
        from .thunk import OutputThunk
        data = self.data
        if isinstance(data, OutputThunk):
            from .lineage import get_raw_value
            data = get_raw_value(data)
        return canonical_hash(data)

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

    @classmethod
    def save(
        cls,
        data: Any,
        db: "DatabaseManager | None" = None,
        **metadata,
    ) -> str:
        """
        Save data to the database as this variable type.

        Accepts OutputThunk (from thunked computation), an existing BaseVariable
        instance, or raw data. Lineage is automatically extracted and stored
        when applicable, and computations are cached for future reuse.

        Args:
            data: The data to save. Can be:
                - OutputThunk: result from a @thunk decorated function
                - BaseVariable: an existing variable instance
                - Any other type: raw data (numpy array, etc.)
            db: Optional explicit database. If None, uses global database.
            **metadata: Addressing metadata (e.g., subject=1, trial=1)

        Returns:
            str: The record_id of the saved data

        Raises:
            ReservedMetadataKeyError: If metadata contains reserved keys
            NotRegisteredError: If this variable type is not registered
            DatabaseNotConfiguredError: If no database is available
            UnsavedIntermediateError: If strict mode and unsaved intermediates exist

        Example:
            # Save from a thunk computation
            result = process(input_data)
            record_id = CleanData.save(result, subject=1, trial=1)

            # Save raw data
            record_id = CleanData.save(np.array([1, 2, 3]), subject=1, trial=1)

            # Re-save an existing variable with new metadata
            var = CleanData.load(subject=1, trial=1)
            record_id = CleanData.save(var, subject=1, trial=2)
        """
        from .database import get_database, get_user_id
        from .exceptions import ReservedMetadataKeyError, UnsavedIntermediateError
        from .thunk import OutputThunk
        from .lineage import extract_lineage, find_unsaved_variables, get_raw_value

        # Validate metadata keys
        reserved_used = set(metadata.keys()) & cls._reserved_keys
        if reserved_used:
            raise ReservedMetadataKeyError(
                f"Cannot use reserved metadata keys: {reserved_used}"
            )

        db = db or get_database()

        # Normalize input: extract raw data and lineage info based on input type
        lineage = None
        lineage_hash = None
        output_thunk = None
        raw_data = None

        if isinstance(data, OutputThunk):
            # Data from a thunk computation
            output_thunk = data

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
            lineage_hash = output_thunk.pipeline_thunk.compute_cache_key()
            raw_data = get_raw_value(output_thunk)

        elif isinstance(data, BaseVariable):
            # Existing variable instance - extract its data and lineage
            raw_data = data.data
            lineage_hash = data._lineage_hash  # Preserve lineage if it had one

        else:
            # Raw data (numpy array, etc.)
            raw_data = data

        # Create instance with the raw data
        instance = cls(raw_data)

        # Save to database
        record_id = db.save(instance, metadata, lineage=lineage, lineage_hash=lineage_hash)
        instance._record_id = record_id
        instance._metadata = metadata

        # Populate computation cache if this came from a thunk
        if output_thunk is not None:
            cache_key = lineage_hash
            db.cache_computation(
                cache_key=cache_key,
                function_name=output_thunk.pipeline_thunk.thunk.fcn.__name__,
                function_hash=output_thunk.pipeline_thunk.thunk.hash,
                output_type=cls.__name__,
                output_record_id=record_id,
                output_num=output_thunk.output_num,
            )

        return record_id

    @classmethod
    def load(
        cls,
        db: "DatabaseManager | None" = None,
        version: str = "latest",
        **metadata,
    ) -> Self:
        """
        Load a single variable from the database.

        Metadata keys are split into schema keys (dataset identity) and version
        keys (computational identity) based on the configured schema_keys.

        - If only schema keys are provided: returns the latest version at that location
        - If version keys are also provided: returns the exact matching version

        Args:
            db: Optional explicit database. If None, uses global database.
            version: "latest" for most recent, or specific record_id
            **metadata: Addressing metadata to match

        Returns:
            The matching variable instance (latest if multiple versions exist)

        Raises:
            NotFoundError: If no matching data found
            NotRegisteredError: If this variable type is not registered
            DatabaseNotConfiguredError: If no database is available
        """
        from .database import get_database

        db = db or get_database()
        return db.load(cls, metadata, version=version)

    @classmethod
    def load_all(
        cls,
        db: "DatabaseManager | None" = None,
        **metadata,
    ):
        """
        Load all matching variables from the database as a generator.

        This is useful for loading multiple records with partial schema keys,
        e.g., all records for a given subject across all visits.

        Args:
            db: Optional explicit database. If None, uses global database.
            **metadata: Addressing metadata to match (partial matching supported)

        Yields:
            Variable instances matching the metadata

        Raises:
            NotRegisteredError: If this variable type is not registered
            DatabaseNotConfiguredError: If no database is available

        Example:
            # Load all ProcessedSignal records for subject 1
            for signal in ProcessedSignal.load_all(subject=1):
                print(signal.metadata, signal.data.shape)
        """
        from .database import get_database

        db = db or get_database()
        yield from db.load_all(cls, metadata)

    @classmethod
    def list_versions(
        cls,
        db: "DatabaseManager | None" = None,
        **metadata,
    ) -> list[dict]:
        """
        List all versions at a schema location.

        Shows all saved versions (including those with empty version {})
        at a given schema location. This is useful for seeing what
        computational variants exist for a given dataset location.

        Args:
            db: Optional explicit database. If None, uses global database.
            **metadata: Schema metadata to match

        Returns:
            List of dicts with record_id, schema, version, created_at

        Raises:
            NotRegisteredError: If this variable type is not registered
            DatabaseNotConfiguredError: If no database is available

        Example:
            versions = ProcessedSignal.list_versions(subject=1, visit=2)
            for v in versions:
                print(f"record_id: {v['record_id']}, version: {v['version']}")
        """
        from .database import get_database

        db = db or get_database()
        return db.list_versions(cls, **metadata)

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
            List of record_ides for each saved record

        Example:
            # DataFrame with 10 rows (2 subjects x 5 trials)
            #   Subject  Trial  MyVar
            #   1        1      0.5
            #   1        2      0.6
            #   ...

            record_ides = ScalarValue.save_from_dataframe(
                df=results_df,
                data_column="MyVar",
                metadata_columns=["Subject", "Trial"],
                experiment="exp1"  # Applied to all rows
            )
        """
        from .database import get_database

        db = db or get_database()
        record_ides = []

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
            record_id = cls.save(data, db=db, **full_metadata)
            record_ides.append(record_id)

        return record_ides

    @classmethod
    def load_to_dataframe(
        cls,
        db: "DatabaseManager | None" = None,
        include_record_id: bool = False,
        **metadata,
    ) -> pd.DataFrame:
        """
        Load matching records and reconstruct a DataFrame.

        This is the inverse of save_from_dataframe(). Loads all matching
        records and returns them as a DataFrame with metadata as columns.

        Args:
            db: Optional explicit database. If None, uses global database.
            include_record_id: If True, include record_id as a column
            **metadata: Metadata filter (loads all matching records)

        Returns:
            DataFrame with columns for each metadata key plus 'data' column

        Example:
            # Load all records for experiment
            df = ScalarValue.load_to_dataframe(experiment="exp1")
            # Returns:
            #   Subject  Trial  data   (record_id)
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
            if include_record_id:
                row["record_id"] = var.record_id
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
