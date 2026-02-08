"""This pipeline is intended to show how replicating my existing pipelines would look with the scidb framework."""

import json
from pathlib import Path

import pandas as pd

from scidb import BaseVariable, configure_database, thunk


# =============================================================================
# Variable Type Definition
# =============================================================================

class EMGData(BaseVariable):
    """EMG data stored as a DataFrame."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return self.data

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df


# Create specialized types for each processing stage
class RawEMGData(EMGData):
    """Raw EMG data before filtering."""
    pass


class FilteredEMGData(EMGData):
    """Filtered EMG data."""
    pass


class MaxEMGData(EMGData):
    """Maximum EMG values."""
    pass


# =============================================================================
# Processing Functions (with @thunk for lineage tracking and caching)
# =============================================================================

def load_emg_data(intervention: str) -> pd.DataFrame:
    """Load EMG data for a specific intervention.

    Note: Not thunked because it's called in a loop inside load_emg_data_all_interventions.
    Caching happens at the outer function level instead.
    """
    print(f"Loading EMG data for intervention: {intervention}")
    # Placeholder for actual data loading logic
    # e.g., read from files, preprocess, etc.
    a_dict = {"intervention": intervention, "data": [1, 2, 3, 4, 5]}
    df = pd.DataFrame(a_dict)
    return df


@thunk()
def load_emg_data_all_interventions(config: dict) -> pd.DataFrame:
    """Load EMG data for all interventions specified in the config."""
    interventions = config.get("interventions", [])
    all_df = []
    for intervention in interventions:
        df = load_emg_data(intervention)
        all_df.append(df)
    df = pd.concat(all_df, ignore_index=True)
    return df


@thunk()
def filter_emg_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter EMG data."""
    # Placeholder for filtering logic
    return df


@thunk()
def compute_max_emg(df: pd.DataFrame) -> pd.DataFrame:
    """Compute maximum EMG values."""
    max_values = df.max()
    return pd.DataFrame(max_values).transpose()


# =============================================================================
# Pipeline Execution
# =============================================================================

if __name__ == "__main__":
    # Load config
    config = json.load(open("examples/pipeline_example1_config.json"))

    # Configure database (types are auto-registered on first save)
    db_path = Path("examples/pipeline_example1.db")
    db = configure_database(db_path)

    # Load the data
    emg_df = load_emg_data_all_interventions(config)

    # Filter the data
    filtered_emg_df = filter_emg_data(emg_df)

    # Compute maximum EMG values
    max_emg_df = compute_max_emg(filtered_emg_df)

    # Save results with metadata (lineage is captured automatically)
    RawEMGData.save(emg_df, db=db, subject=1, intervention="all")
    FilteredEMGData.save(filtered_emg_df, db=db, subject=1, intervention="all")
    MaxEMGData.save(max_emg_df, db=db, subject=1, intervention="all")

    print("Pipeline complete. Data saved to database.")
