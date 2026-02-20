# as_table Column Type Normalization

## Context

When `as_table=true` is used in `load()`, `load_all()`, or `for_each()`, multiple
database results are combined into a single MATLAB table. Each result's metadata
(schema keys like `subject`, `session`) becomes a column in that table.

## The Problem (fixed)

Metadata values arrive from Python as individual scalars stored in struct fields.
When building table columns from these, the original code wrapped every value in a
cell: `repmat({val}, nr, 1)` or collected into `cell(n, 1)`. This meant string
metadata like `session="A"` appeared as a cell array of strings (`{'A'; 'B'}`)
rather than a proper MATLAB string column (`["A"; "B"]`).

MATLAB treats cell columns differently from typed columns — they display with
braces, don't support vectorized string operations, and generally confuse users
who expect `tbl.session == "A"` to work.

## How It Works Now

Three code paths build metadata columns for `as_table` tables:

### 1. `fe_multi_result_to_table` — Flatten mode (for_each.m)

Used when all data items are tables. Metadata is replicated per row via `repmat`.
Now type-dispatched:

- Numeric values: `repmat(double(val), nr, 1)` — numeric column
- String/char values: `repmat(string(val), nr, 1)` — string column
- Other: `repmat({val}, nr, 1)` — cell fallback

### 2. `fe_multi_result_to_table` — Non-table mode (for_each.m)

Used when data items are scalars/arrays (not tables). Metadata collected into a
cell array, then normalized via `normalize_meta_column()`:

- All numeric → `cell2mat()` → double column
- All string/char → `string()` → string column
- Mixed → cell column (fallback)

### 3. `multi_result_to_table` (BaseVariable.m)

Used by `load(as_table=true)` and `load_all(as_table=true)`. Same normalization
logic inline: checks all-numeric and all-string before assigning to table column.

## Data Column Normalization

The same normalization applies to the **data column** (named after the variable
type, e.g. `ScalarVar`) in non-table/non-flatten mode:

- All scalar numeric values → `double` column
- All string/char values → `string` column
- Non-scalar data (arrays, tables, etc.) → `cell` column (each cell holds one item)

The scalar check is important: `cell2mat` would fail on a cell of differently-sized
arrays, so only scalar numerics are collapsed. The helper functions are named
`normalize_cell_column` (for_each.m) and `normalize_data_column` (BaseVariable.m).

## Key Invariant

After `as_table` conversion, every metadata column in the returned table is:
- `double` if all values for that key are scalar numeric
- `string` if all values for that key are string/char
- `cell` only if the column has mixed types or non-scalar values

The data column follows the same rule — scalar numerics become a `double` column,
strings become a `string` column, and anything else stays as a `cell` column.

## Test Coverage

`TestAsTable.m` includes:
- `test_as_table_column_types`: Checks both `load(as_table=true)` and
  `for_each(as_table=true)` return `isnumeric` / `isstring` metadata columns
  AND `isnumeric` data columns for scalar data
- `test_as_table_array_data_stays_cell`: Verifies non-scalar array data
  correctly remains as a cell column
- `test_for_each_as_table_flatten_column_types`: Checks the flatten mode
  (table data) also produces typed metadata columns
