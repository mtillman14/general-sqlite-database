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

**Class Methods:**

| Method | Description |
|--------|-------------|
| `save(data, db=None, **metadata)` | Save data to database, returns vhash |
| `load(db=None, version="latest", **metadata)` | Load from database |
| `save_from_dataframe(df, data_column, metadata_columns, db=None, **common_metadata)` | Save each row as separate record |
| `load_to_dataframe(db=None, include_vhash=False, **metadata)` | Load matching records as DataFrame |
| `table_name()` | Get SQLite table name |

**Instance Methods:**

| Method | Description |
|--------|-------------|
| `to_csv(path)` | Export data to CSV file |
| `get_preview()` | Get human-readable data summary |

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
| `get_cached_computation(cache_key, variable_class)` | Look up cached result by key and type |
| `get_cached_by_key(cache_key)` | Look up cached result by key only (used for auto-caching) |
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

### `@thunk(n_outputs=1, unwrap=True)`

Decorator for lineage-tracked functions with automatic caching.

```python
@thunk(n_outputs=1)
def process(data: np.ndarray) -> np.ndarray:
    return data * 2

result = process(data)  # Returns OutputThunk
result.data  # The actual result
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_outputs` | `1` | Number of outputs the function returns |
| `unwrap` | `True` | If True, unwrap `BaseVariable` and `OutputThunk` inputs to raw data. If False, pass wrapper objects directly (useful for debugging). |

**Automatic caching:**

Results are cached automatically. Once saved, subsequent calls with the same inputs skip execution:

```python
result = process(data)
MyVar.save(result, ...)  # Populates cache

result2 = process(data)  # Cache hit! No execution
result2.was_cached       # True
```

For multi-output functions, all outputs must be saved before caching takes effect:

```python
@thunk(n_outputs=2)
def split(data):
    return data[:5], data[5:]

left, right = split(data)
LeftVar.save(left, ...)   # Save all outputs
RightVar.save(right, ...)

left2, right2 = split(data)  # Cache hit for both!
```

**Cross-script lineage:**

```python
# step1.py
result = process(raw_data)
Intermediate.save(result, db=db, subject=1)

# step2.py
loaded = Intermediate.load(db=db, subject=1)

@thunk(n_outputs=1)
def analyze(data):  # Receives raw data (loaded.data)
    return data.mean()

result = analyze(loaded)  # Pass the variable, lineage is captured
```

**Debugging with unwrap=False:**

```python
@thunk(n_outputs=1, unwrap=False)
def debug_process(var):
    print(f"Input vhash: {var.vhash}")
    print(f"Input metadata: {var.metadata}")
    return var.data * 2
```

---

### `Thunk`

Wrapper for a function with lineage tracking.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `fcn` | `Callable` | The wrapped function |
| `n_outputs` | `int` | Number of outputs |
| `unwrap` | `bool` | Whether to unwrap inputs |
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
| `data` | `Any` | Computed result |
| `is_complete` | `bool` | Whether computed |
| `was_cached` | `bool` | From cache |
| `cached_id` | `str \| None` | Cached version hash |
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
raw = get_raw_value(output_thunk)  # Returns output_thunk.data
raw = get_raw_value(plain_data)    # Returns plain_data unchanged
```

---

### `check_cache(pipeline_thunk, variable_class, db=None)`

Check if computation is cached.

```python
cached = check_cache(result.pipeline_thunk, MyVar, db=db)
if cached:
    print(cached.data)
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
