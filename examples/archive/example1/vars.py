

import pandas as pd
import numpy as np

from scidb import BaseVariable

# -----------------------------------------------------------------------------
# STEP 2: Define Custom Variable Types
# Documentation: See docs/guide/variables.md for patterns and best practices
#
# Key concept: Each variable type needs to implement:
#   - to_db(): Convert native data to pandas DataFrame for storage
#   - from_db(): Convert DataFrame back to native type
#
# The schema_version attribute enables future schema migrations.
# -----------------------------------------------------------------------------

class SensorReading(BaseVariable):
    """
    A time series of sensor readings.

    Implementation notes (docs/guide/variables.md):
    - to_db() must return a DataFrame with your data
    - from_db() receives that DataFrame and reconstructs your data
    - Use schema_version for future-proofing migrations
    """
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        # Convert numpy array to DataFrame for Parquet serialization
        # The 'value' column name is arbitrary - you control the schema
        return pd.DataFrame({'value': self.data})

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        # Reconstruct the numpy array from the stored DataFrame
        return df['value'].values


class ProcessedSignal(BaseVariable):
    """
    A processed signal (filtered, normalized, etc.)

    This is a separate type from SensorReading to demonstrate:
    - Each type gets its own table in the database
    - Lineage can track transformations between types
    """
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({'value': self.data})

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        return df['value'].values


class SignalStatistics(BaseVariable):
    """
    Summary statistics for a signal.

    Demonstrates storing a dictionary as a DataFrame.
    Documentation: See docs/guide/variables.md section "Dictionary Pattern"
    """
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        # Store dict as single-row DataFrame with columns for each key
        return pd.DataFrame([self.data])

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> dict:
        # Convert single-row DataFrame back to dict
        return df.iloc[0].to_dict()