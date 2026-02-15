# Schema Hierarchy and Non-Contiguous Key Support

## How the Schema Works

SciDuck models dataset metadata as a **linear hierarchy**. When you configure a database with:

```python
dataset_schema = ["subject", "intervention", "timepoint", "speed", "trial", "cycle"]
```

...this defines an ordered list of categorical dimensions. The `_schema` table in DuckDB has one column per schema key, plus `schema_id` and `schema_level`.

## Contiguous vs Non-Contiguous Keys

### Contiguous (prefix) keys

The method `_schema_key_columns(schema_level)` returns a **contiguous prefix** of the schema:

```python
def _schema_key_columns(self, schema_level: str) -> List[str]:
    idx = self.dataset_schema.index(schema_level)
    return self.dataset_schema[: idx + 1]
```

This is used by Mode A (DataFrame) and Mode C (tuple-dict) saves, where the key set is derived from the schema level.

### Non-contiguous (arbitrary subset) keys

Mode B saves (keyword arguments) accept **any subset** of schema keys. The `schema_level` is set to the deepest provided key in the hierarchy, and unprovided intermediate keys are stored as NULL in `_schema`.

For example, with the schema above:

```python
db.save("cohens_d", 0.85, subject="S01", intervention="RMT30", speed="SSV")
```

- `schema_level` = `"speed"` (deepest provided key)
- `_schema` row: `subject="S01", intervention="RMT30", timepoint=NULL, speed="SSV", trial=NULL, cycle=NULL`

This enables cross-cutting analyses where results collapse across intermediate dimensions (e.g., computing Cohen's d across timepoints for each speed).

## Where Key Resolution Happens

### `_infer_schema_level` (`database.py`)

Returns the deepest provided key in the hierarchy. Accepts any subset of schema keys — no contiguity requirement.

### `_get_or_create_schema_id` (`sciduck.py`)

Derives `key_cols` from the provided `key_values` dict (not from `_schema_key_columns`). Matches on provided keys and requires NULLs for all other schema columns, ensuring distinct `_schema` rows for different key subsets.

### `batch_get_or_create_schema_ids` (`sciduck.py`)

Groups by `(schema_level, frozenset(key_values.keys()))` so that entries with different key sets are handled separately, even if they share the same `schema_level`.

### `save()` Mode B (`sciduck.py`)

Computes `key_cols` as `[k for k in self.dataset_schema if k in schema_keys]` — the provided keys in schema order. No missing-key validation; the key set is exactly what the caller provides.

### `load()` (`sciduck.py`)

- Selects **all** schema columns (not just the prefix), so non-contiguous keys appear in results
- Filters on any provided schema key
- Resolves `parameter_id` to the latest version that has data matching the provided schema keys (via `_latest_parameter_id`)

### `_schema_key_columns()` — still used

Mode A (DataFrame) and Mode C (tuple-dict) saves still use `_schema_key_columns` to derive key columns from the schema level. Their contiguous-prefix semantics remain appropriate since the key set is implicit from the level.

## What Does NOT Change

- **`for_each`** — No contiguity logic; passes flat metadata through
- **Bridge layer** (`scidb-matlab/bridge.py`) — No contiguity logic
- **Variable views** (`_create_variable_view`) — Already SELECT all schema columns
- **`_split_metadata()`** — Already a flat dict partition, no prefix logic

## Design Rationale

The original contiguous-prefix design was chosen because it maps cleanly to hierarchical experimental designs (subjects contain sessions contain trials). Non-contiguous key support was added to handle cross-cutting analyses where the natural grouping doesn't follow the linear order — a common pattern in scientific data processing (e.g., collapsing across intermediate dimensions like timepoints).

The `schema_level` is retained as the deepest provided key, preserving backward compatibility. NULLs in the `_schema` table now appear at any position (not just trailing), distinguishing non-contiguous entries from contiguous ones.
