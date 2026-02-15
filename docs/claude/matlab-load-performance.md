# MATLAB Load Performance Optimizations

## Problem

`for_each()` was slow due to two compounding bottlenecks in the MATLAB-Python bridge:

1. **Per-iteration database queries**: Each `load()` call in the `for_each` loop ran a separate database query. For 108 iterations with 1 input, that was 108 queries (~86s total, ~800ms each).

2. **Per-variable wrapping overhead**: Converting each Python BaseVariable to a MATLAB ThunkOutput required ~34 MATLAB-Python boundary crossings (attribute accesses for record_id, content_hash, lineage_hash, version_id, parameter_id, plus ~4 crossings per metadata key). For 8328 variables with 6 metadata keys each, this was ~250K boundary crossings (~20.7s).

## Solution: Two-Part Optimization

### Optimization 1: Batch Wrapping via Python Bridge

**Files**: `bridge.py`, `BaseVariable.m`

Instead of crossing the MATLAB-Python boundary ~34 times per variable, a Python bridge function (`wrap_batch_bridge`) extracts all scalar fields in bulk:

- Record IDs, content hashes, lineage hashes, version/parameter IDs are packed into newline-joined strings (1 crossing each to transfer, then parsed in pure MATLAB via `splitlines`)
- All metadata dicts are serialized as a single JSON array string, parsed by MATLAB's native C `jsondecode` (1 crossing total for all metadata)
- Data values and Python object references are returned as Python lists (still 1 crossing per element for data conversion, which is unavoidable since data types vary)

This reduces boundary crossings from ~34 per variable to ~3 per variable (data, py_obj reference, and the fixed overhead of bulk field extraction).

**Used by**: `BaseVariable.load()` (multi-result path), `BaseVariable.load_all()`, and `for_each` preloading.

### Optimization 2: Pre-loading in for_each

**File**: `for_each.m`

Instead of calling `load()` per iteration (one DB query each), `for_each` now pre-loads all data for each input variable type in a single query before the main loop:

1. For each loadable input (BaseVariable or Fixed, not PathInput), the iteration metadata values are passed as arrays to `load_all()`, which uses SQL `IN` clauses for matching.
2. Results are batch-wrapped via Optimization 1.
3. A `containers.Map` lookup table is built, keyed by a sorted metadata string (e.g., `"session=A|subject=1"`).
4. In the main loop, results are looked up from the map instead of querying the database.

**Fixed inputs** are handled by overriding specific metadata keys in the query (e.g., `session="BL"` replaces the iterated session values) and adjusting the lookup key accordingly.

**Preload option**: `preload` defaults to `true`. Users can disable it with `preload=false` for datasets too large to fit in memory simultaneously:

```matlab
scidb.for_each(@fn, inputs, outputs, preload=false, subject=[1 2 3], ...);
```

When `preload=false`, the original per-iteration `load()` behavior is used (also the fallback for PathInput and empty metadata).

## Key Implementation Details

### Lookup Key Construction

The lookup key is built from sorted metadata key=value pairs joined by `|`. Three helper functions handle this:

- `build_meta_key(keys, vals)` — core key builder from parallel arrays
- `result_meta_key(metadata_struct, query_keys)` — extracts values from a loaded result's metadata struct using only the query keys
- `combo_meta_key(meta_keys, combo, fixed_meta, query_keys)` — builds the key for a specific iteration, applying Fixed overrides

The key only includes the metadata keys that were part of the query, not all metadata keys on the result. This ensures Fixed inputs (which query with different keys) produce matching lookup keys.

### JSON Metadata Type Consistency

`jsondecode` returns string values as `char` arrays, while the original `pydict_to_struct` returns MATLAB `string` type. The batch wrapper post-processes jsondecoded structs to convert `char` fields to `string` for consistency.

### Fallback Behavior

The preloaded path is skipped (falls back to per-iteration `load()`) when:
- `preload=false`
- `dry_run=true`
- No metadata keys are specified (empty iteration space)
- The input is a `PathInput` (not a database load)

## Performance Impact

| Component | Before | After |
|---|---|---|
| DB queries | ~86s (108 queries) | ~1-2s (1 query per input type) |
| Variable wrapping | ~20.7s (~250K crossings) | ~5-8s (~17K crossings) |
| **Total** | **~107s** | **~7-10s** |
