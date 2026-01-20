# Quickstart

Get started with SciDB in 5 minutes.

## Installation

```bash
pip install scidb
```

## 1. Define a Variable Type

Every data type you want to store needs a `BaseVariable` subclass that defines how to serialize/deserialize it:

```python
from scidb import BaseVariable
import pandas as pd
import numpy as np

class SignalData(BaseVariable):
    schema_version = 1  # Increment when changing the schema

    def to_db(self) -> pd.DataFrame:
        """Convert numpy array to DataFrame for storage."""
        return pd.DataFrame({
            "index": range(len(self.data)),
            "value": self.data
        })

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        """Convert DataFrame back to numpy array."""
        return df.sort_values("index")["value"].values
```

## 2. Configure the Database

```python
from scidb import configure_database

db = configure_database("my_experiment.db")
```

## 3. Save Data with Metadata

```python
# Save data with metadata
signal = np.sin(np.linspace(0, 2*np.pi, 100))
record_id = SignalData.save(signal,
    subject=1,
    trial=1,
    condition="control"
)
print(f"Saved with hash: {record_id[:16]}...")
```

## 4. Load Data by Metadata

```python
# Load by metadata query
loaded = SignalData.load(subject=1, trial=1)
print(loaded.data)       # The numpy array
print(loaded.record_id)      # Version hash
print(loaded.metadata)   # {"subject": 1, "trial": 1, "condition": "control"}
```

## 5. Track Processing Lineage

Use `@thunk` to automatically track what processing produced each result:

```python
from scidb import thunk

@thunk(n_outputs=1)
def bandpass_filter(signal: np.ndarray, low: float, high: float) -> np.ndarray:
    # Your filtering logic here
    return filtered_signal

@thunk(n_outputs=1)
def compute_power(signal: np.ndarray) -> float:
    return np.mean(signal ** 2)

# Run pipeline - lineage tracked automatically
raw = SignalData.load(subject=1, trial=1)
filtered = bandpass_filter(raw.data, low=1.0, high=40.0)
power = compute_power(filtered)

# Save result - lineage captured
class PowerValue(BaseVariable):
    schema_version = 1
    def to_db(self):
        return pd.DataFrame({"power": [self.data]})
    @classmethod
    def from_db(cls, df):
        return df["power"].iloc[0]

PowerValue.save(power, subject=1, trial=1, stage="power")

# Query what produced this result
provenance = db.get_provenance(PowerValue, subject=1, trial=1, stage="power")
print(provenance["function_name"])  # "compute_power"
```

## 6. Wrap External Functions

Leverage existing libraries with lineage tracking:

```python
from scidb import Thunk
from scipy.signal import butter, filtfilt

# Wrap external functions
thunked_butter = Thunk(butter, n_outputs=2)
thunked_filtfilt = Thunk(filtfilt, n_outputs=1)

# Use with full lineage tracking
b, a = thunked_butter(N=4, Wn=0.1, btype='low')
filtered = thunked_filtfilt(b.data, a.data, raw_data)

SignalData.save(filtered, subject=1, stage="filtered")
```

## 7. Specialized Types via Subclassing

When one variable class represents multiple data types, create subclasses:

```python
# Create specialized types - each gets its own table
class Temperature(SignalData):
    pass  # Table: temperature

class Humidity(SignalData):
    pass  # Table: humidity

# Data stored in separate tables (auto-registered on first save)
Temperature.save(temp_array, sensor=1, day="monday")
Humidity.save(humid_array, sensor=1, day="monday")
```

## Next Steps

- [Variables Guide](guide/variables.md) - Deep dive into variable types
- [Database Guide](guide/database.md) - All database operations
- [Lineage Guide](guide/lineage.md) - Full lineage tracking details
- [Caching Guide](guide/caching.md) - Computation caching
