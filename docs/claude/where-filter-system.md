# `where=` Filter System

## Purpose

Users frequently need to load one variable conditioned on the value of another. Example: load `StepLength` only where `Side == "L"`. The `where=` parameter provides a composable, Pythonic filter system that requires no SQL knowledge for common cases.

## Syntax

```python
# Equality filter on a whole-variable value
StepLength.load_all(where=Side == "L")

# Comparison filter
StepLength.load_all(where=Speed > 1.2)

# Compound AND / OR
StepLength.load_all(where=(Side == "L") & (Speed > 1.2))
StepLength.load_all(where=(Side == "L") | (Side == "R"))

# NOT
StepLength.load_all(where=~(Side == "L"))

# Column-level filter (tabular variable)
StepLength.load_all(where=GaitData["Side"] == "L")

# Set membership
StepLength.load_all(where=GaitData["Side"].isin(["L", "R"]))

# Raw SQL escape hatch (applied to the target variable's data table)
StepLength.load_all(where=raw_sql('"value" > 0.70'))

# Works with load() too
StepLength.load(subject=1, where=Side == "L")
```

## How It Works

Filters are resolved **before** data is fetched, at the schema_id level:

1. `load()` / `load_all()` calls `db.load_all()` / `db.load()`.
2. `_find_record()` returns the full set of matching `_record_metadata` rows.
3. If `where` is provided, `where.resolve(db, target_class, table_name)` is called.
4. `resolve()` returns `allowed_schema_ids: set[int]`.
5. The records DataFrame is filtered: `records = records[records["schema_id"].isin(allowed_schema_ids)]`.
6. Bulk data loading proceeds as normal over the filtered records.

This is efficient — no unnecessary data is loaded from DuckDB.

## Filter Class Hierarchy

All classes are in `src/scidb/filters.py`.

| Class | Created by | Queries |
|-------|-----------|---------|
| `VariableFilter` | `Side == "L"` (metaclass) | Filter variable's `"value"` column |
| `ColumnFilter` | `GaitData["Side"] == "L"` (ColumnSelection) | Filter variable's named column |
| `InFilter` | `.isin([...])` | Filter variable's column via SQL `IN` |
| `CompoundFilter` | `f1 & f2` or `f1 \| f2` | Set intersection / union of schema_ids |
| `NotFilter` | `~f` | Complement: all target schema_ids minus inner set |
| `RawFilter` | `raw_sql("...")` | Raw SQL applied to the **target** variable's data table |

### resolve() contract

Every filter implements:
```python
def resolve(self, db, target_variable_class, target_table_name) -> set[int]:
    ...
```

Returns the set of `schema_id` integers that pass the filter. `CompoundFilter` combines two sets; `NotFilter` subtracts from the full target schema_id set.

### Latest-version semantics for filter variables

`VariableFilter`, `ColumnFilter`, and `InFilter` all query the filter variable using "latest version per parameter set" semantics (a `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY version_id DESC)` CTE). This mirrors exactly how `load_all(version_id="latest")` works for the target variable — no special casing.

## Metaclass: `VariableMeta`

Defined in `src/scidb/variable.py` and applied to `BaseVariable`.

Overrides `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__` at the **class** level so that `Side == "L"` (where `Side` is a class, not an instance) produces a `VariableFilter`.

Two important preservation rules:
- **Class-to-class equality is preserved**: `Side == Side` returns `True` (standard `type.__eq__` is called when `other` is a `type`).
- **Hashing is preserved**: `__hash__ = type.__hash__` is explicitly set. Without this, overriding `__eq__` on a metaclass would make classes unhashable, breaking dict/set use.

## ColumnSelection operators

`ColumnSelection` (in `scirun-lib/src/scirun/column_selection.py`) gained comparison operators and `isin()`. Each returns a `ColumnFilter` using `self.columns[0]` as the column name. Multi-column selections only use the first column for filtering (consistent with single-column typical usage).

`__hash__` was also added to `ColumnSelection` since `__eq__` was overridden.

## Schema-Level Validation

### Filter at same or coarser level — OK
- **Same level** (e.g., both at `trial`): direct schema_id set match.
- **Coarser level** (e.g., filter at `subject`, target at `trial`): the filter's matching `schema_id`s are **expanded** to all finer-level `schema_id`s in the target that share the same coarse key values. This is a natural hierarchical filter.

### Filter at finer level — Error
Raises `ValueError`:
```
Filter variable 'Side' is stored at schema level 'trial' which is finer than
target 'StepLength' at level 'subject'. Filters must be at the same or coarser
level than the target.
```

### Coverage validation
Every schema location in the target must have a corresponding filter value. If incomplete coverage is detected, raises:
```
Filter variable 'Side' is missing data at 2 schema locations that 'StepLength'
has data for. Ensure the filter variable covers all target locations.
```

### Level detection
Level is inferred by inspecting `_schema` rows for each variable's schema_ids. The deepest non-null column in `dataset_schema_keys` order is the variable's level. This is the same logic used by `_infer_schema_level()` in `DatabaseManager`.

## RawFilter special behavior

`RawFilter` (created by `raw_sql()`) applies the SQL fragment to the **target** variable's own data table (not to the filter variable). This is the only filter type that works this way. It uses a latest-version CTE over the target and injects the raw SQL into the WHERE clause. DuckDB errors propagate wrapped as `ValueError("Invalid where= SQL: ...")`.

## Error messages

| Scenario | Error |
|----------|-------|
| Filter variable has no saved data | `"Filter variable 'X' is not registered. Save data to it first."` |
| Filter at finer level than target | `"Filter variable 'X' is stored at schema level 'y' which is finer than target 'Z' at level 'w'. Filters must be at the same or coarser level than the target."` |
| Filter missing coverage | `"Filter variable 'X' is missing data at N schema locations that 'Z' has data for. Ensure the filter variable covers all target locations."` |
| Raw SQL syntax error | `"Invalid where= SQL: {duckdb_error}"` |

## Key Files

| File | Role |
|------|------|
| `src/scidb/filters.py` | All filter classes + `raw_sql()` factory |
| `src/scidb/variable.py` | `VariableMeta` metaclass + `where=` in `load()`, `load_all()` |
| `src/scidb/__init__.py` | Exports `raw_sql` |
| `src/scidb/database.py` | `where=` in `DatabaseManager.load()` and `load_all()` — calls `resolve()` and filters the records DataFrame |
| `scirun-lib/src/scirun/column_selection.py` | Comparison operators on `ColumnSelection` |
| `tests/test_filters.py` | Unit tests (no DB required) |
| `tests/test_where.py` | Integration tests with real DuckDB |

## MATLAB Implementation

### Filter construction (scidb.BaseVariable operators)

Operator overloads on `scidb.BaseVariable` produce `scidb.Filter` objects wrapping the corresponding Python filter:

```matlab
filt = Side() == "L";           % VariableFilter
filt = Side() ~= "L";           % negated VariableFilter
filt = ScalarVar() > 1.2;       % comparison VariableFilter
filt = (Side() == "L") & (ScalarVar() > 1.0);  % CompoundFilter AND
filt = (Side() == "L") | (Side() == "R");       % CompoundFilter OR
filt = ~(Side() == "L");        % NotFilter
filt = scidb.raw_sql('"value" > 0.70');         % RawFilter
```

`scidb.Filter` holds a `.py_filter` property containing the Python filter object. The `&`, `|`, and `~` operators on `scidb.Filter` delegate to Python via `__and__`, `__or__`, `__invert__`.

Key file: `scidb-matlab/src/scidb_matlab/matlab/+scidb/Filter.m`

### where= in load() / load_all()

```matlab
StepLength().load(where=Side() == "L", subject=1, session='A')
StepLength().load_all(where=(Side() == "L") & (ScalarVar() > 1.0), db=db)
```

The filter is forwarded to Python as `where_filter.py_filter` via the bridge. Schema-level validation and coverage checks behave identically to the Python API.

## `where=` in `scidb.for_each` (MATLAB)

### How it applies — the preload path (default, `preload=true`)

For each non-Merge, non-PathInput input variable, `for_each` bulk-preloads all values for all iteration combos in a single Python call. The `where=` filter is passed to `load_and_extract`:

```matlab
bulk = py.scidb_matlab.bridge.load_and_extract( ...
    py_class, py_metadata, ...
    pyargs('version_id', 'latest', 'db', py_db, 'where', where_filter.py_filter));
```

Results are stored in a `containers.Map` keyed by metadata. Any combo whose key is absent from the map is silently skipped.

### How it applies — the per-iteration path (`preload=false` or no metadata keys)

When preloading is disabled, the filter is passed directly to the per-iteration load call:

```matlab
loaded{p} = var_inst.load(load_nv{:}, db_nv{:}, where_nv{:});
```

Behavior is identical; any iteration where the load returns nothing is skipped.

### where= with `parallel=true`

Parallel mode also uses the preload path (Phase A is serial). The `where=` filter is applied to `load_and_extract` exactly as in the serial preload. Combos missing from the preloaded map are skipped before Phase B (parfor compute).

### where= with `scidb.Fixed` inputs

`Fixed` inputs are preloaded with their overridden metadata substituted in. The `where=` filter is applied to those pinned metadata bulk-loads too.

Practical implication: if `where=Side()=="L"` is used with `Fixed(Baseline, session='BL')`, the filter checks Side at `session='BL'` (not the iteration's session). Save Side data at all sessions that any input will be queried at, or the Fixed preload may yield empty results and skip the iteration.

### where= with `scidb.Merge` (IMPORTANT: filter is NOT applied)

`Merge` inputs are **excluded from the preload phase** and are always loaded via `merge_constituents()`, which does not pass `where_nv`. This is by design — Merge uses its own schema-key inner-join logic and cannot be filtered by an external where= predicate.

**Consequence**: if **all** inputs to `for_each` are `Merge` objects, the `where=` parameter has no effect. The iteration runs for every combo where all Merge constituents have data, regardless of the filter.

If at least one non-Merge input is present, that input is filtered by `where=` and the combo is skipped if it yields no data. The Merge constituents still load without filtering.

```matlab
% where= is IGNORED — only input is a Merge:
scidb.for_each(@fn, struct('d', scidb.Merge(A(), B())), {Out()}, ...
    'subject', [1 2], where=Side() == "L");
% → Runs for every subject that has A and B data, Side value irrelevant.

% where= IS applied — non-Merge input present:
scidb.for_each(@fn, struct('x', RawSignal(), 'd', scidb.Merge(A(), B())), {Out()}, ...
    'subject', [1 2], where=Side() == "L");
% → Runs only where RawSignal passes the Side filter; Merge loads unfiltered.
```

### `where=` with SelectedColumn inputs (`VarName("col")`)

Column selection (`RawSignal("col_a")`) sets `selected_columns` on the BaseVariable instance. The `where=` filter is applied to the preload/load of the full table first; column narrowing happens after. The combination works correctly.

### Summary table

| Input type | where= applied? | Path |
|------------|----------------|------|
| Plain `BaseVariable` | Yes | Preload bulk query or per-iteration `load()` |
| `scidb.Fixed(BaseVariable)` | Yes | Preload bulk query using fixed metadata |
| `BaseVariable("col")` | Yes | Preload bulk query, column narrowed after |
| `scidb.PathInput` | N/A | PathInput is not a DB load |
| `scidb.Merge(...)` | **No** | Always via `merge_constituents`, no filter |
| Constant (scalar/table) | N/A | Not loaded from DB |

## What is NOT implemented

- **Version-specific filter**: The filter always uses "latest version" of the filter variable. There is no mechanism to specify a particular version of the filter variable.
- **Cross-database filters**: The filter variable must be in the same database as the target.
- **where= for Merge constituents**: Merge bypasses the filter path by design. There is no mechanism to filter individual Merge constituents via the top-level `where=`.
