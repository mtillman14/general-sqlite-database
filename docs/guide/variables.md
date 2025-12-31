# Variables

Variables are the core data type in SciDB. Every piece of data you store is wrapped in a `BaseVariable` subclass that defines its serialization format.

## Defining a Variable Type

```python
from scidb import BaseVariable
import pandas as pd

class MyVariable(BaseVariable):
    schema_version = 1  # Required

    def to_db(self) -> pd.DataFrame:
        """Convert self.data to a DataFrame for storage."""
        ...

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> Any:
        """Convert DataFrame back to native type."""
        ...
```

### Required Components

| Component | Purpose |
|-----------|---------|
| `schema_version` | Integer version for schema migrations |
| `to_db()` | Instance method converting `self.data` to DataFrame |
| `from_db()` | Class method converting DataFrame to native type |

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

## Specialized Types with `for_type()`

When one variable class can represent multiple logical data types, use `for_type()` to create specialized subclasses with separate tables:

```python
class TimeSeries(BaseVariable):
    schema_version = 1
    # ... to_db/from_db implementation

# Create specialized types
Temperature = TimeSeries.for_type("temperature")
Humidity = TimeSeries.for_type("humidity")
Pressure = TimeSeries.for_type("pressure")

# Each gets its own table
# time_series_temperature, time_series_humidity, time_series_pressure
```

### Migrating from One-to-One to One-to-Many

If you start with a single type and later need specialization:

```python
# Original usage (one-to-one)
TimeSeries(data).save(db=db, sensor=1)  # -> time_series table

# Later, migrate to one-to-many
DefaultSeries = TimeSeries.for_type()  # Access old data
Temperature = TimeSeries.for_type("temperature")

# Old data still accessible via TimeSeries or DefaultSeries
# New typed data goes to time_series_temperature
```

### Type Suffix Normalization

Type names are normalized automatically:

| Input | Table Suffix |
|-------|--------------|
| `"temperature"` | `_temperature` |
| `"Temperature"` | `_temperature` |
| `"ambient temperature"` | `_ambient_temperature` |
| `"air-quality"` | `_air_quality` |

## Instance Properties

After `save()` or `load()`:

```python
var = MyVariable(data)
vhash = var.save(subject=1)

var.data      # The native data
var.vhash     # Content hash (set after save/load)
var.metadata  # Metadata dict (set after save/load)
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
| DataFrame is ONE unit of data (e.g., time series) | `MyVar(df).save(...)` |
| Each row is SEPARATE data (e.g., subject/trial results) | `MyVar.save_from_dataframe(df, ...)` |

## Reserved Metadata Keys

These keys cannot be used in metadata:

- `vhash` - Reserved for version hash
- `id` - Reserved for database ID
- `created_at` - Reserved for timestamp
- `schema_version` - Reserved for schema version
- `data` - Reserved for data storage

Using these raises `ReservedMetadataKeyError`.
