# API Reference

## Core Classes

### `BaseVariable`

Abstract base class for storable data types.

```python
class MyVariable(BaseVariable):
    schema_version: int = 1  # Required class attribute

    def to_db(self) -> pd.DataFrame:
        """Convert self.data to DataFrame."""
        ...

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> Any:
        """Convert DataFrame to native type."""
        ...
```

**Instance Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `Any` | The native data |
| `vhash` | `str \| None` | Version hash (after save/load) |
| `metadata` | `dict \| None` | Metadata (after save/load) |

**Methods:**

| Method | Description |
|--------|-------------|
| `save(db=None, **metadata)` | Save to database, returns vhash |
| `load(db=None, version="latest", **metadata)` | Load from database |
| `save_from_dataframe(df, data_column, metadata_columns, db=None, **common_metadata)` | Save each row as separate record |
| `load_to_dataframe(db=None, include_vhash=False, **metadata)` | Load matching records as DataFrame |
| `to_csv(path)` | Export data to CSV file |
| `get_preview()` | Get human-readable data summary |
| `for_type(type_name=None)` | Create specialized subclass |
| `table_name()` | Get SQLite table name |
| `get_type_suffix()` | Get type suffix (if specialized) |

---

### `DatabaseManager`

Manages database connection and operations.

```python
db = DatabaseManager("path/to/db.sqlite")
```

**Methods:**

| Method | Description |
|--------|-------------|
| `register(variable_class)` | Register a variable type (optional, auto-registers on save/load) |
| `save(variable, metadata, lineage=None)` | Save variable (internal) |
| `load(variable_class, metadata, version="latest")` | Load variable(s) |
| `list_versions(variable_class, **metadata)` | List all matching versions |
| `get_provenance(variable_class, version=None, **metadata)` | Get immediate lineage info |
| `get_full_lineage(variable_class, version=None, max_depth=100, **metadata)` | Get complete lineage chain |
| `format_lineage(variable_class, version=None, **metadata)` | Get print-friendly lineage |
| `get_derived_from(variable_class, version=None, **metadata)` | Find derived variables |
| `has_lineage(vhash)` | Check if lineage exists |
| `export_to_csv(variable_class, path, **metadata)` | Export matching records to CSV |
| `preview_data(variable_class, **metadata)` | Get formatted preview of records |
| `get_cached_computation(cache_key, variable_class)` | Look up cached result |
| `cache_computation(...)` | Store computation in cache |
| `invalidate_cache(function_name=None, function_hash=None)` | Clear cache entries |
| `get_cache_stats()` | Get cache statistics |
| `close()` | Close connection |

---

## Configuration Functions

### `configure_database(db_path)`

Configure the global database.

```python
db = configure_database("experiment.db")
```

**Returns:** `DatabaseManager`

---

### `get_database()`

Get the global database.

```python
db = get_database()
```

**Returns:** `DatabaseManager`
**Raises:** `DatabaseNotConfiguredError` if not configured

---

## Thunk System

### `@thunk(n_outputs=1)`

Decorator for lineage-tracked functions.

```python
@thunk(n_outputs=1)
def process(data: np.ndarray) -> np.ndarray:
    return data * 2

result = process(data)  # Returns OutputThunk
result.value  # The actual result
```

---

### `Thunk`

Wrapper for a function with lineage tracking.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `fcn` | `Callable` | The wrapped function |
| `n_outputs` | `int` | Number of outputs |
| `hash` | `str` | SHA-256 of bytecode |

---

### `PipelineThunk`

A specific invocation with captured inputs.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `thunk` | `Thunk` | Parent thunk |
| `inputs` | `dict` | Captured inputs |
| `outputs` | `tuple[OutputThunk]` | Results after execution |
| `hash` | `str` | Hash of thunk + inputs |

**Methods:**

| Method | Description |
|--------|-------------|
| `compute_cache_key()` | Generate cache lookup key |

---

### `OutputThunk`

Wraps a function output with lineage.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `pipeline_thunk` | `PipelineThunk` | Producer |
| `output_num` | `int` | Output index |
| `value` | `Any` | Computed result |
| `is_complete` | `bool` | Whether computed |
| `was_cached` | `bool` | From cache |
| `cached_vhash` | `str \| None` | Cached version hash |
| `hash` | `str` | Lineage hash |

---

## Lineage Functions

### `extract_lineage(output_thunk)`

Extract lineage from an OutputThunk.

```python
lineage = extract_lineage(result)
print(lineage.function_name)
```

**Returns:** `LineageRecord`

---

### `get_raw_value(data)`

Unwrap OutputThunk to raw value.

```python
raw = get_raw_value(output_thunk)  # Returns output_thunk.value
raw = get_raw_value(plain_data)    # Returns plain_data unchanged
```

---

### `check_cache(pipeline_thunk, variable_class, db=None)`

Check if computation is cached.

```python
cached = check_cache(result.pipeline_thunk, MyVar, db=db)
if cached:
    print(cached.value)
```

**Returns:** `OutputThunk | None`

---

### `LineageRecord`

Provenance data structure.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `function_name` | `str` | Function name |
| `function_hash` | `str` | Function bytecode hash |
| `inputs` | `list[dict]` | Input descriptors |
| `constants` | `list[dict]` | Constant descriptors |

---

## Exceptions

| Exception | Description |
|-----------|-------------|
| `SciDBError` | Base exception |
| `NotRegisteredError` | Loading a type that was never saved |
| `NotFoundError` | No matching data |
| `DatabaseNotConfiguredError` | Global DB not configured |
| `ReservedMetadataKeyError` | Using reserved metadata key |

---

## Reserved Metadata Keys

Cannot be used in `save()` metadata:

- `vhash`
- `id`
- `created_at`
- `schema_version`
- `data`
