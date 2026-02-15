# for_each Input Unwrapping and as_table Behavior

## Context

`scidb.for_each` iterates over metadata combinations, loads inputs, calls a function, and saves outputs. When a `load()` call matches multiple records (e.g., 76 subjects at a given session), the loaded value is an array of `scidb.ThunkOutput` objects. How this array is delivered to the user's function depends on the `as_table` option.

## The Two Delivery Modes

### as_table=false (default)

The ThunkOutput array is passed through `scidb.internal.unwrap_input()`, which extracts the `.data` property from each element:

- **Single ThunkOutput**: returns `arg.data` directly (a matrix, scalar, etc.)
- **Array of N ThunkOutput objects**: returns a `cell(1, N)` where each cell contains one element's `.data`

This means the user's function receives a cell array of raw data values when there are multiple matches.

### as_table=true

The ThunkOutput array is converted to a MATLAB table via `fe_multi_result_to_table()` before the function is called. The table contains:

- One column per metadata field (from the first result's `.metadata` struct)
- A `version_id` column
- A data column named after the variable type (e.g., `StepLength` for a `StepLength` variable)

The data column is a cell array (one cell per row). `unwrap_input` is skipped for table inputs (`~istable(loaded{p})` guard on line 261).

### as_table=["input_name1", "input_name2"]

You can also pass a string array of specific input names to convert to tables, rather than converting all loadable inputs.

## Code Flow in for_each.m

1. **Parse as_table option** (split_options, line 354): stored as `opts.as_table`
2. **Resolve as_table_set** (lines 141-147):
   - `true` (logical) -> all loadable input names
   - `false` (logical) -> `string.empty` (no conversion)
   - string array -> those specific input names
   - not specified -> `string.empty` (default, no conversion)
3. **Table conversion** (lines 239-243): after loading, if the input name is in `as_table_set` AND the loaded value is a multi-element ThunkOutput array, convert to table
4. **Unwrap for plain functions** (lines 259-264): for non-Thunk function handles, unwrap ThunkOutput/BaseVariable inputs to raw data via `scidb.internal.unwrap_input()`, but skip table inputs

## unwrap_input Details

Located at `+scidb/+internal/unwrap_input.m`. Handles:

- `scidb.ThunkOutput` (single or array)
- `scidb.BaseVariable` (single or array)
- Everything else passes through unchanged

For arrays (numel > 1), it loops and builds a cell array of `.data` values. This is critical because MATLAB's dot-indexing on a handle object array (`arr.data`) produces a comma-separated list, and assigning that to a single variable only captures the first element.

## Thunk vs Plain Function

- **Plain function handles**: `unwrap_input` is called on all loadable inputs so the function receives raw MATLAB data (not ThunkOutput wrappers)
- **scidb.Thunk**: `unwrap_input` is NOT called (Thunks handle their own input processing via the Python bridge and need the ThunkOutput wrappers for lineage tracking)
