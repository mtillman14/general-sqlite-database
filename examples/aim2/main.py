"""
GaitRite data processing pipeline.

This pipeline loads GaitRite .xlsx files, splits them into individual walks,
preprocesses each walk to extract step-level measurements, and saves them
to the database with full metadata for querying.
"""

from typing import Any
import tomllib

import pandas as pd
import numpy as np

from scidb import PathGenerator, BaseVariable, configure_database
from thunk import thunk


# Load config - tomllib requires binary mode
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# Configure database
db = configure_database("aim2.db", schema_keys=["subject", "session", "speed", "repetition"])


# =============================================================================
# Path Generation
# =============================================================================

sessions = ["BL", "MID", "POST", "MO1FU", "MO3FU"]

# Note: walks are WITHIN each file, not separate files, so not in path template
paths = PathGenerator("{subject}/{session}/{speed}.xlsx",
    subject=[f"SS{sub}" for sub in range(10)],
    session=sessions,
    speed=config["speeds"],
)


# =============================================================================
# Variable Definitions
# =============================================================================

class ScalarList(BaseVariable):
    """Base class for scalar measurements."""

    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": [self.data]})

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> Any:
        return [v for v in df["value"]]


class StepLength(ScalarList):
    """Step length measurement in meters."""

    pass


class StepWidth(ScalarList):
    """Step width measurement in meters."""

    pass


class Side(ScalarList):
    """Left or right side"""

    pass


db.register(StepLength)
db.register(StepWidth)
db.register(Side)


# =============================================================================
# Loading & Preprocessing Functions
# =============================================================================

def load_gaitrite_file(path: str) -> pd.DataFrame:
    """Load raw GaitRite .xlsx file."""
    return pd.read_excel(path)


def split_into_walks(raw_df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split a raw GaitRite file into individual walks.

    Each GaitRite file contains 3 walks. This function parses the file
    structure to separate them.
    """
    # Placeholder - actual implementation would parse the file structure
    walks = [raw_df.copy() for _ in range(3)]
    return walks


@thunk(n_outputs=3)
def preprocess_walk(walk_df: pd.DataFrame) -> tuple[list, list, list]:
    """
    Preprocess a single GaitRite walk.

    Returns:
        step_lengths: List of step length values
        step_widths: List of step width values
        sides: List of "L" and "R" values
    """
    # Placeholder processing
    n_steps = 10
    step_lengths = np.random.rand(n_steps)
    step_widths = np.random.rand(n_steps)
    sides = ["L", "R"] * n_steps / 2

    return step_lengths, step_widths, sides


# =============================================================================
# Main Pipeline
# =============================================================================

for filepath, metadata in paths:

    # Load and split file into walks
    raw_df = load_gaitrite_file(filepath)
    walks = split_into_walks(raw_df)

    for walk_number, walk_df in enumerate(walks):
        # Preprocess this walk (lineage tracked by @thunk)
        step_outputs = preprocess_walk(walk_df)

        file_metadata["repetition"] = walk_number

        n_steps = len(step_outputs[0])
        index=range(0, n_steps)
        StepLength.save(step_outputs[0], **file_metadata, index=index)
        StepWidth.save(step_outputs[1], **file_metadata, index=index)
        Side.save(step_outputs[2], **file_metadata, index=index)
