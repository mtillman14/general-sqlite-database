"""
GaitRite data processing pipeline.

This pipeline loads GaitRite .xlsx files, splits them into individual walks,
preprocesses each walk to extract step-level measurements, and saves them
to the database with full metadata for querying.
"""

import tomllib

from scidb import PathGenerator, configure_database, for_each, for_each
from vars import *

from lib.load_gaitrite import load_gaitrite_file
from lib.preprocess_gaitrite import preprocess_walk, split_into_walks


# Load config - tomllib requires binary mode
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# Configure database
db = configure_database("aim2.db", schema_keys=["subject", "session", "speed", "repetition"])


# =============================================================================
# Path Generation
# =============================================================================

sessions = ["BL", "MID", "POST", "MO1FU", "MO3FU"]
subjects = [f"SS{sub}" for sub in range(10)]
speeds = config["speeds"]

# Note: walks are WITHIN each file, not separate files, so not in path template
paths = PathGenerator("{subject}/{session}/{speed}.xlsx",
    subject=subjects,
    session=sessions,
    speed=speeds,
)

db.register(StepLength)
db.register(StepWidth)
db.register(Side)

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

        metadata["repetition"] = walk_number

        n_steps = len(step_outputs[0])
        index=range(0, n_steps)
        StepLength.save(step_outputs[0], **metadata, index=index)
        StepWidth.save(step_outputs[1], **metadata, index=index)
        Side.save(step_outputs[2], **metadata, index=index)

filter_data = Thunk() # Placeholder
for_each(filter_data, subject=subjects, session=sessions, speed=speeds,
    inputs={"arg0": SomeVariable},
    outputs=[OtherVariable])
