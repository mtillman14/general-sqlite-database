# Variables

Variables are the core data type in SciDB. Every piece of data you store is wrapped in a `BaseVariable` subclass that defines its serialization format.

## Defining a Variable Type

All variable types use `pd.DataFrame` as the intermediate storage format. Your `to_db()` method must return a DataFrame, and your `from_db()` method receives a DataFrame.

```python
from scidb import BaseVariable
import pandas as pd

class MyVariable(BaseVariable):
    schema_version = 1  # Required

    def to_db(self) -> pd.DataFrame:
        """Convert self.data to a DataFrame for storage."""
        # Must return a pd.DataFrame
        ...

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> Any:
        """Convert DataFrame back to native type."""
        # df is always a pd.DataFrame
        ...
```

### Required Components

| Component | Purpose |
|-----------|---------|
| `schema_version` | Integer version for schema migrations |
| `to_db()` | Instance method converting `self.data` to `pd.DataFrame` |
| `from_db()` | Class method converting `pd.DataFrame` to native type |

## Common Patterns

### Scalar Values

```python
class ScalarValue(BaseVariable):
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": [self.data]})

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> float:
        return df["value"].iloc[0]
```

### 1D Arrays

```python
class ArrayValue(BaseVariable):
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({
            "index": range(len(self.data)),
            "value": self.data
        })

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        return df.sort_values("index")["value"].values
```

### 2D Arrays / Matrices

```python
class MatrixValue(BaseVariable):
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        rows, cols = self.data.shape
        return pd.DataFrame({
            "row": np.repeat(range(rows), cols),
            "col": np.tile(range(cols), rows),
            "value": self.data.flatten()
        })

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        df = df.sort_values(["row", "col"])
        rows = df["row"].max() + 1
        cols = df["col"].max() + 1
        return df["value"].values.reshape(rows, cols)
```

### DataFrames

```python
class DataFrameValue(BaseVariable):
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return self.data  # Already a DataFrame

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df
```

### Dictionaries

```python
class DictValue(BaseVariable):
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame([self.data])  # Single row

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> dict:
        return df.iloc[0].to_dict()
```

## Specialized Types via Subclassing

When one variable class can represent multiple logical data types, create subclasses to store each in separate tables:

```python
class TimeSeries(BaseVariable):
    schema_version = 1
    # ... to_db/from_db implementation

# Create specialized types - each gets its own table
class Temperature(TimeSeries):
    """Temperature time series data."""
    pass  # Table: temperature

class Humidity(TimeSeries):
    """Humidity time series data."""
    pass  # Table: humidity

class Pressure(TimeSeries):
    """Pressure time series data."""
    pass  # Table: pressure
```

Each subclass:
- Inherits `to_db()` and `from_db()` from the parent
- Gets its own table named after the class (CamelCase → snake_case)
- Can define custom methods specific to that data type

## Instance Properties

After `save()` or `load()`:

```python
# Save returns the vhash
vhash = MyVariable.save(data, subject=1)

# Load returns a variable instance with populated properties
var = MyVariable.load(subject=1)
var.data      # The native data
var.vhash     # Content hash (set after load)
var.metadata  # Metadata dict (set after load)
```

## Batch Operations: DataFrames with Multiple Records

When a DataFrame contains multiple independent data items (e.g., one row per subject/trial), use `save_from_dataframe()` and `load_to_dataframe()`:

### Saving Each Row Separately

```python
# DataFrame with results for multiple subjects/trials
#   Subject  Trial  Value
#   1        1      0.52
#   1        2      0.61
#   2        1      0.48
#   2        2      0.55

class ScalarResult(BaseVariable):
    schema_version = 1
    def to_db(self):
        return pd.DataFrame({"value": [self.data]})
    @classmethod
    def from_db(cls, df):
        return df["value"].iloc[0]

# Save each row as a separate record
vhashes = ScalarResult.save_from_dataframe(
    df=results_df,
    data_column="Value",
    metadata_columns=["Subject", "Trial"],
    experiment="exp1"  # Additional common metadata
)
# Creates 4 separate database records
```

### Loading Back to DataFrame

```python
# Load all records matching criteria
df = ScalarResult.load_to_dataframe(experiment="exp1")
#   Subject  Trial  data
#   1        1      0.52
#   1        2      0.61
#   2        1      0.48
#   2        2      0.55

# Include vhash for traceability
df = ScalarResult.load_to_dataframe(experiment="exp1", include_vhash=True)
#   Subject  Trial  data   vhash
#   1        1      0.52   abc123...
#   ...
```

### When to Use Each Pattern

| Scenario | Method |
|----------|--------|
| DataFrame is ONE unit of data (e.g., time series) | `MyVar.save(df, ...)` |
| Each row is SEPARATE data (e.g., subject/trial results) | `MyVar.save_from_dataframe(df, ...)` |

## Metadata Reflects Dataset Structure

The metadata keys you use in `save()` should reflect the natural structure of your dataset. Common patterns include subject/trial designs, session-based recordings, or hierarchical experimental structures.

### Example: Subject × Trial Design

```python
class TrialResult(BaseVariable):
    schema_version = 1
    def to_db(self):
        return pd.DataFrame({"value": [self.data]})
    @classmethod
    def from_db(cls, df):
        return df["value"].iloc[0]

# Save results for each subject and trial
subjects = [1, 2, 3]
trials = ["baseline", "treatment", "followup"]

for subject in subjects:
    for trial in trials:
        # Process data for this subject/trial
        result = analyze_trial(subject, trial)

        # Metadata mirrors dataset structure
        TrialResult.save(result,
            subject=subject,
            trial=trial,
            experiment="exp_2024"
        )

# Later: load specific combinations
baseline_s1 = TrialResult.load(subject=1, trial="baseline")
all_baselines = TrialResult.load(trial="baseline")  # All subjects
```

### Example: Session-Based Recordings

```python
class Recording(BaseVariable):
    schema_version = 1
    # ... to_db/from_db

sessions = ["morning", "afternoon", "evening"]
days = ["day1", "day2", "day3"]

for day in days:
    for session in sessions:
        data = record_session(day, session)
        Recording.save(data, day=day, session=session, device="sensor_A")
```

The key insight: your metadata structure should make it easy to query the data the way you'll need to access it later.

## Reserved Metadata Keys

These keys cannot be used in metadata:

- `vhash` - Reserved for version hash
- `id` - Reserved for database ID
- `created_at` - Reserved for timestamp
- `schema_version` - Reserved for schema version

Using these raises `ReservedMetadataKeyError`.
