# Browsing and Exporting Data

SciDB stores data efficiently using Parquet-serialized BLOBs inside SQLite. While this provides excellent type fidelity and portability, it means you can't directly view data in standard SQLite browsers.

This guide covers how to browse and export your data.

## Preview Column

Every saved variable includes a human-readable `preview` column that's visible in any SQLite viewer:

```
preview: [1000 rows x 2 cols] Columns: index, value | value: min=0.1234, max=9.876, mean=5.432 | Sample: [0.1234, 0.5678, 0.9012, ...]
```

The preview shows:
- Shape (rows x columns)
- Column names
- Statistics for numeric data (min, max, mean)
- Sample values

## Viewing Data in SQLite Browsers

Open your `.sqlite` file in SQLite Studio, DBeaver, or any SQLite viewer. Each variable type has its own table with these columns:

| Column | Description |
|--------|-------------|
| `record_id` | Unique content hash |
| `metadata` | JSON with addressing keys |
| `preview` | Human-readable summary |
| `data` | Parquet BLOB (not viewable) |
| `created_at` | Timestamp |

The `preview` column gives you a quick understanding of the data without needing Python.

## Previewing Data in Python

### Single Variable

```python
var = TimeSeries.load(db=db, subject=1)
print(var.get_preview())
# [1000 rows x 2 cols] Columns: index, value | value: min=0.12, max=9.87, mean=5.43 | Sample: [0.12, 0.56, ...]
```

### Multiple Variables

```python
print(db.preview_data(TimeSeries, experiment="exp1"))
# === TimeSeries (3 records) ===
#
# record_id: abc123def456...
#   metadata: subject=1, trial=1
#   preview: [1000 rows x 2 cols] ...
#   created: 2024-01-15 10:30:00
#
# record_id: def456ghi789...
#   metadata: subject=1, trial=2
#   preview: [1000 rows x 2 cols] ...
#   created: 2024-01-15 10:31:00
```

## Exporting to CSV

### Single Variable

```python
var = TimeSeries.load(db=db, subject=1, trial=1)
var.to_csv("subject1_trial1.csv")
```

### Multiple Variables

```python
# Export all matching records to a single CSV
count = db.export_to_csv(
    TimeSeries,
    "all_experiment_data.csv",
    experiment="exp1"
)
print(f"Exported {count} records")
```

The exported CSV includes:
- All columns from `to_db()` output
- `_record_id` column for traceability
- `_meta_*` columns for each metadata key

### Example CSV Output

```csv
index,value,_record_id,_meta_subject,_meta_trial
0,0.123,abc123...,1,1
1,0.456,abc123...,1,1
2,0.789,abc123...,1,1
0,0.234,def456...,1,2
1,0.567,def456...,1,2
```

## Workflow Recommendations

1. **Quick inspection**: Use SQLite browser to view `preview` column
2. **Detailed analysis**: Use `var.to_csv()` or `db.export_to_csv()` to export, then open in Excel/pandas
3. **Programmatic access**: Use `load()` and work with data directly in Python

## Example: Full Workflow

```python
from scidb import configure_database, BaseVariable
import pandas as pd
import numpy as np

class Measurement(BaseVariable):
    schema_version = 1

    def to_db(self):
        return pd.DataFrame({
            "timestamp": range(len(self.data)),
            "value": self.data
        })

    @classmethod
    def from_db(cls, df):
        return df.sort_values("timestamp")["value"].values

# Setup
db = configure_database("experiment.db")

# Save some data
for subject in [1, 2, 3]:
    data = np.random.randn(100)
    Measurement(data).save(subject=subject, experiment="demo")

# Quick preview in Python
print(db.preview_data(Measurement, experiment="demo"))

# Export for external analysis
db.export_to_csv(Measurement, "demo_data.csv", experiment="demo")

# Or view in SQLite browser - open experiment.db and look at
# the 'measurement' table's 'preview' column
```
