# for_each Caching / skip_computed (NOT YET IMPLEMENTED)

This document describes a planned feature for `scidb.for_each` that would allow it to skip iterations where the computation has already been run with the same inputs, same outputs, and the same function content. **This feature has not been implemented yet.** It is a design plan for future implementation.

## Motivation

Currently, `for_each` always re-runs every iteration:

```matlab
scidb.for_each(@computeCohensD, ...
    struct('tableIn', var, 'catToSplitBy', 'timepoint', 'dataColName', varName), ...
    {effectSizeVar()}, ...
    'subject', subjects, 'intervention', interventions, 'speed', speeds, ...
    'as_table', true);
```

Re-running this code re-executes `computeCohensD` for every (subject, intervention, speed) combination, even if the results are already in the database from a previous run. The Thunk system supports caching via lineage hashing, but requiring Thunks just to skip already-computed iterations is heavyweight. A simpler, `for_each`-native mechanism is needed.

## Planned API

Add a `skip_computed` parameter (default `false`) to `for_each` in both Python and MATLAB:

```python
# Python
for_each(
    compute_cohens_d,
    inputs={"table_in": StepLengths_GR_Sym, ...},
    outputs=[StepLengths_GR_Sym_cohensd],
    skip_computed=True,       # <-- new
    subject=subjects, intervention=interventions, speed=speeds,
)
```

```matlab
% MATLAB
scidb.for_each(@computeCohensD, ...
    struct('tableIn', var, ...), ...
    {effectSizeVar()}, ...
    'skip_computed', true, ... % <-- new
    'subject', subjects, 'intervention', interventions, 'speed', speeds);
```

## Skip Logic (per iteration)

For each (subject, intervention, speed) combination, before running the function:

1. **Check the for_each cache** (see "Cache Table" below) for a record matching each output variable name + the iteration metadata key.
2. If a cache record exists, compare its stored `fn_hash` to the current function's hash.
3. If the hash matches, verify the output variable still exists in DuckDB for that metadata.
4. If all three checks pass for ALL output variables, print `[cached]` and skip.
5. Otherwise, run the function, save outputs, and update the cache.

```
for each (subject, intervention, speed) combo:
    metadata_key = sorted "intervention=A|speed=1.0|subject=1"
    fn_hash = sha256(function source code)

    all_cached = true
    for each output_var in outputs:
        cache_row = cache.lookup(output_var.name, metadata_key)
        if cache_row is None or cache_row.fn_hash != fn_hash:
            all_cached = false
            break
        if not output_var.exists(**iteration_metadata, **constant_metadata):
            all_cached = false
            break

    if all_cached:
        print("[cached] subject=1, intervention=A, speed=1.0")
        continue

    # ... load inputs, run function, save outputs ...

    for each output_var in outputs:
        cache.upsert(output_var.name, metadata_key, fn_hash)
```

## Cache Table: `_for_each_cache` in PipelineDB

The function hash is stored in PipelineDB (the existing SQLite lineage database), NOT as a version key on the output variable. Storing it as a version key would create separate `parameter_id` entries per function hash, causing normal `MyVar.load(subject=1, speed=2)` calls to return multiple results from different function versions.

### Schema

```sql
CREATE TABLE IF NOT EXISTS _for_each_cache (
    output_var   TEXT NOT NULL,
    metadata_key TEXT NOT NULL,   -- sorted "intervention=A|speed=1.0|subject=1"
    fn_hash      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (output_var, metadata_key)
);
```

- `output_var`: The output variable class name (e.g., `"StepLengths_GR_Sym_cohensd"`)
- `metadata_key`: A deterministic string built from the iteration metadata + constant inputs, sorted by key name and pipe-delimited. Example: `"catToSplitBy=timepoint|dataColName=StepLengths_GR_Sym|intervention=A|speed=1.0|subject=1"`. Constants are included because a change in constant value (e.g., `catToSplitBy` changing from `'timepoint'` to `'condition'`) should trigger re-computation.
- `fn_hash`: SHA-256 hex digest of the function's source code.
- `created_at`: ISO timestamp of when the cache entry was written.

### Why PipelineDB (not DuckDB version keys)

If `_fn_hash` were stored as a version key on the output variable:
- Each distinct `_fn_hash` value creates a different `parameter_id` (parameter_id is allocated per unique version_keys JSON).
- A normal `MyVar.load(subject=1, speed=2)` would match multiple parameter_ids (one per function version), returning multiple results instead of one.
- This would break user expectations for loads outside of `for_each`.

By storing the cache in PipelineDB's SQLite database (a separate file), the data database stays clean.

## Function Hashing

### Python

```python
import inspect, hashlib

def _compute_fn_hash(fn):
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None  # Can't get source; disable caching for this call
    return hashlib.sha256(source.encode()).hexdigest()
```

`inspect.getsource` returns the full source text of the function, including the `def` line, docstring, and body. Any edit to the function file that changes the function's source will produce a different hash.

### MATLAB

```matlab
function h = compute_fn_hash(fn)
    if isa(fn, 'function_handle')
        info = functions(fn);
        fpath = info.file;
    elseif isa(fn, 'scidb.Thunk')
        info = functions(fn.fcn);
        fpath = info.file;
    else
        h = '';
        return;
    end
    if isempty(fpath)
        h = '';
        return;
    end
    content = uint8(fileread(fpath));
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(content);
    hash_bytes = typecast(md.digest(), 'uint8');
    h = sprintf('%02x', hash_bytes);
end
```

Note: MATLAB hashes the entire file, not just the function body. This means editing any function in a multi-function file triggers re-computation. This is conservative but simple. If more precise hashing is needed later, extract just the relevant function text.

## Metadata Key Construction

The metadata key is a deterministic string used as a cache lookup key. It must be identical for the same logical computation regardless of argument order.

**Algorithm:**
1. Collect all iteration metadata key-value pairs (subject=1, intervention="A", speed=1.0).
2. Collect all constant input key-value pairs (catToSplitBy="timepoint", dataColName="StepLengths_GR_Sym").
3. Merge into one dict (constants override iteration keys if names collide, though this shouldn't happen in practice).
4. Sort by key name alphabetically.
5. Format each as `key=value` and join with `|`.

```
"catToSplitBy=timepoint|dataColName=StepLengths_GR_Sym|intervention=A|speed=1.0|subject=1"
```

This reuses the same `build_meta_key` pattern already used for preload lookup maps in the MATLAB `for_each`.

## MATLAB Performance: Preloading Output Existence

The MATLAB `for_each` already has a preload phase that bulk-loads all input data in 1 query per variable type. The same approach should be used for output existence checks:

1. During the preload phase, for each output variable type, run a single bulk query across all metadata combinations to find which records already exist.
2. Build a lookup map: `metadata_key -> true`.
3. During iteration, the existence check is an O(1) map lookup instead of a per-iteration database round-trip.

For the cache table check, similarly: query all `_for_each_cache` rows for each `output_var` name up front, and build a `metadata_key -> fn_hash` map.

This means the total number of database queries added by `skip_computed` is:
- 1 query to `_for_each_cache` per output variable type (SQLite)
- 1 query to DuckDB per output variable type (existence check)

These happen once during preload, not per-iteration.

## What This Does NOT Cover

- **Input data changes**: If someone re-saves new data for an input variable under the same metadata, the cache will still show the output as computed. The function won't re-run automatically. The user must pass `skip_computed=false` to force re-computation. Detecting input data changes would require hashing input content and storing it in the cache, which adds significant complexity.
- **Non-deterministic functions**: If the function produces different outputs for the same inputs (e.g., uses random seeds), skipping will return stale results. This is inherent to any caching mechanism.
- **Thunk integration**: This feature is independent of the Thunk lineage system. They can coexist but don't interact.

## Files to Modify

| File | Change |
|------|--------|
| `pipelinedb-lib/src/pipelinedb/pipelinedb.py` | Add `_for_each_cache` table creation in `_ensure_schema()`. Add `check_for_each_cache(output_var, metadata_key)` and `upsert_for_each_cache(output_var, metadata_key, fn_hash)` methods. |
| `src/scidb/foreach.py` | Add `skip_computed=False` parameter. Compute fn_hash. Before each iteration, check cache + output existence. After saving, update cache. |
| `src/scidb/database.py` | Add a `record_exists(cls, **metadata) -> bool` convenience method that checks `_find_record` without loading data. Expose `PipelineDB.check_for_each_cache` through `DatabaseManager`. |
| `scidb-matlab/src/scidb_matlab/matlab/+scidb/for_each.m` | Add `skip_computed` to `split_options`. Compute fn_hash. During preload phase, bulk-query output existence and cache table. During iteration, check maps. After saving, update cache via Python bridge call. |
| `scidb-matlab/src/scidb_matlab/bridge.py` (if needed) | Add bridge functions for cache check/update and output existence check, callable from MATLAB. |
