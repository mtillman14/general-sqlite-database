# Multi-Database Workflow

## Problem

Some projects need to work with multiple databases that have different schemas. For example, a study with two aims where each aim organizes data differently:

- **AIM 1**: `["subject", "intervention", "timepoint", "speed", "trial", "cycle"]`
- **AIM 2**: `["subject", "session", "speed", "trial", "cycle"]`

The goal is to process each database independently using the full framework (thunks, caching, lineage, `for_each`), then merge results for cross-database comparison.

## API for Multi-Database Support

### `db.set_current_db()`

Any `DatabaseManager` instance can become the active global database:

```python
db1 = configure_database("aim1.duckdb", ["subject", "intervention"], "aim1_pipeline.db")
db2 = configure_database("aim2.duckdb", ["subject", "session"], "aim2_pipeline.db")

# db2 is now the global default (most recent configure_database call)
# Switch back to db1:
db1.set_current_db()
```

This sets `Thunk.query` and the thread-local `_local.database` to point to this instance.

### `db=` Parameter on User-Facing Methods

All user-facing methods accept an optional `db=` parameter for one-shot operations against a specific database without changing the global default:

```python
# Save to a specific database
RawEMG.save(data, db=db1, subject=1, trial=1)

# Load from a specific database
var = RawEMG.load(db=db2, subject=1, session="baseline")

# Load all from a specific database
df = MaxActivation.load_all(db=db1, as_df=True)

# List versions in a specific database
versions = RawEMG.list_versions(db=db1, subject=1)

# Batch save from DataFrame to a specific database
ScalarValue.save_from_dataframe(df, "MyVar", ["Subject"], db=db2)
```

The `for_each` function also accepts `db=`:

```python
for_each(
    bandpass_filter,
    inputs={"signal": RawEMG},
    outputs=[FilteredEMG],
    db=db1,
    subject=[1, 2, 3],
    trial=[1, 2, 3],
)
```

When `db=None` (the default), all methods fall back to `get_database()` which returns the current global database — preserving full backwards compatibility.

## Usage Patterns

### Pattern 1: Sequential Processing with `set_current_db()`

Process each database sequentially, switching the global default:

```python
from scidb import configure_database, BaseVariable, thunk, for_each

class RawEMG(BaseVariable):
    pass
class FilteredEMG(BaseVariable):
    pass

@thunk
def bandpass_filter(signal, low_hz, high_hz):
    return filtered_signal

# Create both databases
db1 = configure_database(
    "aim1_data.duckdb",
    dataset_schema_keys=["subject", "intervention", "timepoint", "speed", "trial", "cycle"],
)

db2 = configure_database(
    "aim2_data.duckdb",
    dataset_schema_keys=["subject", "session", "speed", "trial", "cycle"],
)

# --- Process AIM 1 ---
db1.set_current_db()

for_each(
    bandpass_filter,
    inputs={"signal": RawEMG},
    outputs=[FilteredEMG],
    subject=[1, 2, 3],
    intervention=["stim", "sham"],
    timepoint=["pre", "post"],
    speed=["slow", "fast"],
    trial=[1, 2, 3],
    cycle=[1, 2, 3, 4, 5],
)

# --- Process AIM 2 ---
db2.set_current_db()

for_each(
    bandpass_filter,
    inputs={"signal": RawEMG},
    outputs=[FilteredEMG],
    subject=[1, 2, 3],
    session=["baseline", "post"],
    speed=["slow", "fast"],
    trial=[1, 2, 3],
    cycle=[1, 2, 3, 4, 5],
)

# Clean up
db1.close()
db2.close()
```

### Pattern 2: Explicit `db=` for Cross-Database Operations

Use `db=` to read/write across databases without switching the global:

```python
db1 = configure_database("aim1.duckdb", ["subject", "intervention"], "aim1.db")
db2 = configure_database("aim2.duckdb", ["subject", "session"], "aim2.db")

# Load from both databases without switching globals
aim1_df = MaxActivation.load_all(db=db1, as_df=True)
aim2_df = MaxActivation.load_all(db=db2, as_df=True)

# Merge in pandas
aim1_df["aim"] = "aim1"
aim2_df["aim"] = "aim2"
merged = pd.concat([aim1_df, aim2_df], ignore_index=True)
```

### Pattern 3: DuckDB ATTACH for Cross-Database SQL

For advanced SQL queries across databases, use DuckDB's `ATTACH`:

```python
import duckdb

con = duckdb.connect()
con.execute("ATTACH 'aim1_data.duckdb' AS aim1")
con.execute("ATTACH 'aim2_data.duckdb' AS aim2")

merged_df = con.execute("""
    SELECT 'aim1' AS aim, s.subject, s.speed, t.value
    FROM aim1.FilteredEMG_data t
    JOIN aim1._schema s ON t.schema_id = s.schema_id

    UNION ALL

    SELECT 'aim2' AS aim, s.subject, s.speed, t.value
    FROM aim2.FilteredEMG_data t
    JOIN aim2._schema s ON t.schema_id = s.schema_id
""").fetchdf()
```

## Key Points

- Each database gets its own `.duckdb` and `.db` file pair
- `set_current_db()` switches the global database explicitly
- `db=` parameter allows one-shot operations without changing the global
- Variable type classes (e.g., `RawEMG`) are shared and reused across databases
- All framework features (caching, lineage, versioning, `for_each`) work normally within each database
- When `db=None` (default), behavior is identical to before — full backwards compatibility
