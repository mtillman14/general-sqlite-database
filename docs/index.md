# SciDB

**Scientific Data Versioning Framework**

SciDB is a lightweight database framework for scientific computing that provides automatic versioning, provenance tracking, and computation caching in a portable single-file SQLite database.

## Key Features

- **Type-safe storage** - Define custom variable types with explicit serialization
- **Content-based versioning** - Automatic deduplication via deterministic hashing
- **Metadata addressing** - Query data by flexible key-value metadata
- **Lineage tracking** - Automatic provenance capture via `@thunk` decorator
- **External library support** - Wrap functions from scipy, sklearn, etc. with `Thunk(fn)`
- **Computation caching** - Skip redundant computations automatically
- **Portable storage** - Single SQLite file with Parquet-serialized data

## Installation

```bash
pip install scidb
```

## Quick Example

```python
from scidb import BaseVariable, configure_database, thunk
import numpy as np
import pandas as pd

# Define a variable type
class TimeSeries(BaseVariable):
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.data})

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        return df["value"].values

# Setup
db = configure_database("experiment.db")

# Save with metadata
data = np.array([1.0, 2.0, 3.0])
TimeSeries(data).save(subject=1, session="baseline")

# Load by metadata
loaded = TimeSeries.load(subject=1, session="baseline")

# Track lineage with @thunk
@thunk
def normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / arr.std()

result = normalize(loaded.data)
TimeSeries(result).save(subject=1, session="normalized")

# Query provenance
provenance = db.get_provenance(TimeSeries, subject=1, session="normalized")
print(provenance["function_name"])  # "normalize"
```

## Why SciDB?

| Problem                                   | SciDB Solution                                |
| ----------------------------------------- | --------------------------------------------- |
| "Which version of this data did I use?"   | Content-based hashing ensures reproducibility |
| "What processing produced this result?"   | Automatic lineage tracking via `@thunk`       |
| "I already computed this, why recompute?" | Computation caching skips redundant work      |
| "How do I organize my experimental data?" | Flexible metadata addressing                  |
| "I need to share this database"           | Single portable SQLite file                   |

## Documentation

- [Quickstart](quickstart.md) - Get up and running in 5 minutes
- [User Guide](guide/variables.md) - Detailed documentation
- [API Reference](api.md) - Complete API documentation
