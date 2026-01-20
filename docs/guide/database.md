# Database

The `DatabaseManager` handles all storage operations. SciDB uses SQLite with Parquet-serialized data for portability and long-term archival stability.

## Configuration

### Global Database

```python
from scidb import configure_database, get_database

# Configure once at startup
# schema_keys defines which metadata keys identify dataset location
db = configure_database(
    "experiment.db",
    schema_keys=["subject", "trial", "condition"]
)

# Access anywhere
db = get_database()
```

### Schema Keys

The `schema_keys` parameter is **required** and defines which metadata keys identify the "location" in your dataset (e.g., subject, trial, sensor) versus computational variants at that location.

- **Schema keys**: Identify the dataset location (used for folder structure)
- **Version keys**: Everything else - distinguish computational variants at the same location

## Registration

Variable types are **auto-registered** on first save or load. Manual registration is optional:

```python
# Automatic (preferred) - registers on first save
MyVariable.save(data, subject=1)

# Manual (optional) - register explicitly
db.register(MyVariable)
```

Registration creates the table if it doesn't exist. Re-registering is safe (idempotent).

## Save Operations

### Basic Save

```python
record_id = MyVariable.save(data, subject=1, trial=1, condition="A")
```

### Idempotent Saves

Saving identical data+metadata returns the existing record_id without duplicating:

```python
record_id1 = MyVar.save(data, subject=1)
record_id2 = MyVar.save(data, subject=1)  # Same data+metadata
assert record_id1 == record_id2  # No duplicate created
```

## Load Operations

### Load Single Result

`load()` returns the **latest version** at the specified schema location:

```python
# Load by schema keys - returns latest version
var = MyVariable.load(subject=1, trial=1, condition="A")

# Partial match on schema - returns latest at that location
var = MyVariable.load(subject=1)
```

### Load by Version Hash

```python
var = MyVariable.load(version="abc123...")
```

### Load All Matching

Use `load_all()` to iterate over all matching records:

```python
# Generator (memory-efficient)
for var in MyVariable.load_all(condition="A"):
    print(var.data)

# Load all into DataFrame
df = MyVariable.load_all(condition="A", as_df=True)
df = MyVariable.load_all(condition="A", as_df=True, include_record_id=True)
```

## Version History

### List All Versions

```python
versions = db.list_versions(MyVariable, subject=1)
for v in versions:
    print(f"{v['record_id'][:16]} - {v['created_at']}")
    print(f"  Schema: {v['schema']}")    # Dataset location keys
    print(f"  Version: {v['version']}")  # Computational variant keys
```

### Load Specific Version

```python
# By record_id
var = MyVariable.load(version="abc123...")

# By metadata (returns latest matching)
var = MyVariable.load(subject=1, trial=1)
```

## Provenance Queries

### What Produced This?

```python
provenance = db.get_provenance(MyVariable, subject=1, stage="processed")
if provenance:
    print(f"Function: {provenance['function_name']}")
    print(f"Inputs: {provenance['inputs']}")
    print(f"Constants: {provenance['constants']}")
```

### What Used This?

```python
derived = db.get_derived_from(MyVariable, subject=1, stage="raw")
for d in derived:
    print(f"{d['type']} via {d['function']}")
```

### Check Lineage Exists

```python
if db.has_lineage(record_id):
    print("This variable was produced by a thunked function")
```

## Cache Operations

See [Caching Guide](caching.md) for details.

```python
# Get cache statistics
stats = db.get_cache_stats()
print(f"Cached computations: {stats['total_entries']}")

# Invalidate cache
db.invalidate_cache()  # All
db.invalidate_cache(function_name="process")  # By function
db.invalidate_cache(function_hash="abc...")  # By version
```

## Database Schema

SciDB creates these internal tables:

| Table | Purpose |
|-------|---------|
| `_registered_types` | Type registry |
| `_version_log` | All saved versions |
| `_lineage` | Provenance records |
| `_computation_cache` | Cached computation results |
| `{variable_table}` | One per registered type |

## Storage Format

Data is serialized using **Parquet** format, providing:

- Long-term archival stability (not tied to Python versions)
- Cross-language compatibility (R, Julia, Spark, etc.)
- Schema preservation and efficient compression
- Readable in 10+ years

## Exceptions

| Exception | Cause |
|-----------|-------|
| `NotRegisteredError` | Loading a type that has never been saved |
| `NotFoundError` | No data matches the query |
| `DatabaseNotConfiguredError` | `get_database()` called before `configure_database()` |
| `ReservedMetadataKeyError` | Using reserved key in metadata |
