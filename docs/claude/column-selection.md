# Column Selection for BaseVariable

## Purpose

When a BaseVariable stores a wide table (e.g. 50+ columns loaded from Excel), loading the entire table to extract one column is wasteful. Column selection lets users request specific columns at the point where the variable is used as a `for_each()` input, without changing how data is stored or loaded from the database.

## Python Syntax

```python
# Single column → function receives numpy array
for_each(fn, inputs={"x": MyVar["col_a"]}, outputs=[Result], subject=[1, 2, 3])

# Multiple columns → function receives DataFrame subset
for_each(fn, inputs={"x": MyVar[["col_a", "col_b"]]}, outputs=[Result], subject=[1])

# Works inside Fixed too
for_each(fn, inputs={"x": Fixed(MyVar["col_a"], session="BL")}, outputs=[Result], subject=[1])
```

`MyVar["col"]` uses `BaseVariable.__class_getitem__` (defined in `src/scidb/variable.py`) to return a `ColumnSelection` wrapper (defined in `scirun-lib/src/scirun/column_selection.py`).

## MATLAB Syntax

```matlab
% Single column → function receives array
scidb.for_each(@fn, struct('x', MyVar("col_a")), {Result()}, subject=[1 2 3]);

% Multiple columns → function receives subtable
scidb.for_each(@fn, struct('x', MyVar(["col_a", "col_b"])), {Result()}, subject=[1]);
```

The column names are passed to the `BaseVariable` constructor and stored in the `selected_columns` property (added to `+scidb/BaseVariable.m`).

## Return Behavior

| Selection | Python return type | MATLAB return type |
|-----------|-------------------|-------------------|
| Single column | `numpy.ndarray` (`.values` of the column) | numeric/cell array (table column) |
| Multiple columns | `pandas.DataFrame` (subset of columns) | MATLAB `table` (subtable) |

## How It Works (Python)

1. `MyVar["col"]` creates `ColumnSelection(MyVar, ["col"])`.
2. `_is_loadable()` in `foreach.py` recognizes `ColumnSelection` as a loadable type.
3. In the loading loop, `ColumnSelection` is unwrapped: `var_type = var_spec.var_type`, `column_selection = var_spec.columns`.
4. `var_type.load(...)` runs normally — the full record is loaded.
5. `_apply_column_selection(loaded_value, columns, param_name)` extracts the columns from the loaded `.data` DataFrame.
6. The result (numpy array or DataFrame subset) is stored in `loaded_inputs[param_name]`.
7. `_unwrap()` then passes it to the function — numpy arrays and DataFrames pass through unchanged (important: numpy arrays have a `.data` memoryview attribute that must NOT be accessed via `_unwrap`, so `_unwrap` explicitly skips numpy arrays, DataFrames, Series, and lists).

## How It Works (MATLAB)

1. `MyVar("col_a")` stores `"col_a"` in `selected_columns` property.
2. In `for_each`, after loading and after the `as_table` conversion, `apply_column_selection()` checks `var_inst.selected_columns`.
3. If non-empty, it extracts the column(s) from the loaded ThunkOutput's `.data` table.
4. Single column → `tbl.(col_name)` (array). Multiple columns → `tbl(:, cols)` (subtable).

## Key Files

| File | Role |
|------|------|
| `scirun-lib/src/scirun/column_selection.py` | `ColumnSelection` wrapper class |
| `src/scidb/variable.py` | `BaseVariable.__class_getitem__` |
| `scirun-lib/src/scirun/foreach.py` | Loading loop + `_apply_column_selection()` + fixed `_unwrap()` |
| `scirun-lib/tests/test_foreach.py` | `TestForEachColumnSelection` test class |
| `scidb-matlab/src/scidb_matlab/matlab/+scidb/BaseVariable.m` | `selected_columns` property + constructor |
| `scidb-matlab/src/scidb_matlab/matlab/+scidb/for_each.m` | `apply_column_selection()` helper + call sites |

## Important: `_unwrap()` Fix

Before this feature, `_unwrap()` naively checked `hasattr(value, 'data')`. This broke for numpy arrays (which have a `.data` memoryview attribute). The fix — added when implementing column selection — makes `_unwrap()` skip numpy arrays, DataFrames, Series, and lists before falling back to `.data` extraction.
