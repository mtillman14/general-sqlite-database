# Database

The `DatabaseManager` handles all storage operations. SciDB uses SQLite with Parquet-serialized data for portability and long-term archival stability.

## Configuration

### Global Database

```python
from scidb import configure_database, get_database

# Configure once at startup
db = configure_database("experiment.db")

# Access anywhere
db = get_database()
```

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
vhash = MyVariable.save(data, subject=1, trial=1, condition="A")
```

### Idempotent Saves

Saving identical data+metadata returns the existing vhash without duplicating:

```python
vhash1 = MyVar.save(data, subject=1)
vhash2 = MyVar.save(data, subject=1)  # Same data+metadata
assert vhash1 == vhash2  # No duplicate created
```

## Load Operations

### Load by Metadata

```python
# Exact match
var = MyVariable.load(subject=1, trial=1, condition="A")

# Partial match (returns latest matching)
var = MyVariable.load(subject=1)

# Multiple matches returns list
vars = MyVariable.load(condition="A")  # May return list
```

### Load by Version Hash

```python
var = MyVariable.load(version="abc123...")
```

## Version History

### List All Versions

```python
versions = db.list_versions(MyVariable, subject=1)
for v in versions:
    print(f"{v['vhash'][:16]} - {v['created_at']}")
    print(f"  Metadata: {v['metadata']}")
```

### Load Specific Version

```python
# By vhash
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
if db.has_lineage(vhash):
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
