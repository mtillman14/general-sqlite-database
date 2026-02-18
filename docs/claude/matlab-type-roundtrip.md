# MATLAB ↔ Python Type Round-Trip (to_python / from_python)

## Overview

The files `+scidb/+internal/to_python.m` and `+scidb/+internal/from_python.m` handle
bidirectional conversion between MATLAB native types and Python objects. These are used
when saving/loading data through the scidb-matlab bridge. The goal is 100% fidelity:
saving a MATLAB value and loading it back should produce the identical type and shape.

## Type Mapping Table

| MATLAB type | Python type | Storage | Round-trip fidelity |
|---|---|---|---|
| `double` / numeric scalar | `py.float` / `py.int` | native | Exact |
| Numeric vector (Nx1 or 1xN) | 1-D numpy ndarray | native | Exact |
| Numeric matrix (NxM, M>1) | list of 1-D ndarrays (object column in DataFrame) | JSON list-of-lists when inside dict; object column when in table | Exact — `try_stack_numeric` reconstructs the matrix on load |
| `string` scalar | `py.str` (via `char`) | native | Exact |
| `string` array | `py.list` of strings | native | Exact — all-string lists are collapsed back to string arrays |
| `logical` scalar | `py.bool` | native | Exact |
| `logical` array | numpy bool ndarray | native | Exact |
| `char` | `py.str` | native | Loads back as `string`, not `char` |
| `datetime` | ISO 8601 string | VARCHAR | Exact (if format matches) |
| `cell` array | `py.list` (recursive) | varies | Exact per element |
| `table` | `pandas.DataFrame` | multi-column DuckDB or custom | See per-column rules below |
| Scalar `struct` | `py.dict` (recursive) | JSON | Exact |
| Struct array | `py.list` of `py.dict` | JSON list | Loads back as cell array of structs (not struct array) |
| `categorical` | converted to `string` before save | VARCHAR | Loads back as `string`, not `categorical` |

## Table Column Handling (to_python.m → pandas DataFrame)

When a MATLAB table is converted to a pandas DataFrame, each column is extracted
with `data.(col_name)` and processed:

| Column type | Conversion | pandas dtype | Notes |
|---|---|---|---|
| Numeric Nx1 vector | `to_python(col)` → 1-D ndarray | float64/int64 | Standard path |
| Numeric NxM matrix (M>1) | Split into cell of 1xM row vectors → list of 1-D ndarrays | object | pandas rejects 2-D ndarrays as column values (see below) |
| String array | `to_python(col)` → list of strings | object | — |
| Datetime array | Format to ISO string first | object (string) | — |
| Categorical | Convert to string first | object | — |
| Struct array | `to_python(col)` → list of dicts | object | Each struct → dict recursively |
| Cell array | `to_python(col)` → list (recursive) | object | — |

### Why NxM matrices need special handling

pandas.DataFrame rejects 2-D numpy arrays as column values:

```python
# pandas source check:
elif isinstance(val, np.ndarray) and val.ndim > 1:
    raise ValueError("Per-column arrays must each be 1-dimensional")
```

The fix in `to_python.m` detects multi-column numeric table variables
(`isnumeric && ismatrix && ~isvector && ~isscalar`) and splits them into a
cell array of row vectors before conversion. Each 1xM row vector becomes a
1-D ndarray, which pandas accepts as an object-dtype column element.

## DataFrame → Table Load Path (from_python.m)

When loading a pandas DataFrame back to a MATLAB table:

| pandas dtype | Conversion | MATLAB result |
|---|---|---|
| numeric (float64, int64, etc.) | `from_python(col.to_numpy())` → `double()` | Numeric column vector |
| datetime64 | ISO string extraction → `datetime()` | Datetime column |
| object (dicts) | Element-by-element `from_python` → cell of structs | Cell column of structs |
| object (uniform numeric lists) | Element-by-element `from_python` → cell of row vectors → `try_stack_numeric` → `vertcat` | NxM numeric matrix |
| object (mixed types) | Element-by-element `from_python` | Cell column |

### try_stack_numeric

Local helper function in `from_python.m` that enables matrix round-tripping.
After an object column is converted to a cell array, this function checks whether
ALL elements are numeric with identical `size()`. If so, it `vertcat`s them into
a matrix. Otherwise it returns the cell array unchanged.

The `args{i} = args{i}(:)` column-vector enforcement is also guarded with
`isvector(args{i})` so that reconstructed NxM matrices are not flattened.

## Struct Round-Trip Details

### Scalar struct (direct save, not inside a table)

Save path: `to_python` → `py.dict` with recursive field conversion →
`_infer_duckdb_type` returns `("JSON", {python_type: "dict", ndarray_keys: {...}})` →
`_python_to_storage` calls `json.dumps(_convert_for_json(value))` which converts
all ndarrays to nested lists.

Load path: JSON string → `json.loads` → dict → `_storage_to_python` restores
top-level `ndarray_keys` to numpy arrays → `from_python(py.dict)` →
`pydict_to_struct` → MATLAB struct.

**Nested ndarrays caveat:** Only top-level dict keys listed in `ndarray_keys`
are restored to numpy arrays. Arrays inside nested dicts come back as Python
lists, which `from_python` converts to cell arrays (not double matrices).
This is a sciduck limitation in `_infer_duckdb_type` — it only tracks one level
of `ndarray_keys`.

### Struct array (inside a table column)

Save: struct array → `py.list` of `py.dict` (new non-scalar struct handler) →
object-dtype column in DataFrame.

Load: object column → cell array of structs via element-by-element `from_python`.
Note: the original struct array becomes a **cell array of structs**, not a struct
array. This is a minor fidelity loss (cell vs struct array), but field contents
are preserved exactly.

### Struct metadata columns (save_from_table)

In `BaseVariable.save_from_table`, struct metadata columns are converted to
JSON strings via `jsonencode` before crossing to Python. Each struct element
becomes a separate JSON string. This is handled in the metadata column loop
in `BaseVariable.m`.

## Key Files

- `scidb-matlab/src/scidb_matlab/matlab/+scidb/+internal/to_python.m` — MATLAB → Python
- `scidb-matlab/src/scidb_matlab/matlab/+scidb/+internal/from_python.m` — Python → MATLAB
- `scidb-matlab/src/scidb_matlab/matlab/+scidb/BaseVariable.m` — save_from_table struct metadata
- `sciduck/src/sciduck/sciduck.py` — `_convert_for_json`, `_python_to_storage`, `_storage_to_python`, `_infer_duckdb_type`
